# app.py
# -*- coding: utf-8 -*-
"""
Amundi — Clustering de mots-clés (≤ 8 catégories) – Saisie en LISTE (1 par ligne).
Lancez : streamlit run app.py

Prérequis :
  pip install openai>=1.40.0 streamlit>=1.33.0 pandas>=2.1.0 python-dotenv>=1.0.1 openpyxl>=3.1.2 tenacity>=8.2.3

Config :
  - Placez votre clé : OPENAI_API_KEY=sk-...
  - Ou saisissez-la dans la sidebar de l’app.
"""

import os
import math
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────

CATEGORY_DISCOVERY_SYS = """Tu es un analyste SEO senior spécialisé en asset management.
Ta mission : proposer au plus {max_cat} catégories thématiques (noms courts et explicites, en français) pour structurer des mots-clés relatifs à Amundi.
Objectif : faciliter la priorisation éditoriale et la visualisation.
Contrainte : 5 à {max_cat} catégories maximum, pas de doublons, pas de marques concurrentes.
"""

CATEGORY_DISCOVERY_USER = """Voici un échantillon représentatif de mots-clés (en {lang}) :
{sample}

Propose UNE LISTE de catégories (<= {max_cat}). Chaque catégorie a :
- un champ "name" (3–40 caractères, concis)
- un champ "description" (1–2 phrases max).

Retourne STRICTEMENT un JSON respectant le schéma fourni.
"""

CATEGORIZATION_SYS = """Tu classes des mots-clés SEO en EXACTEMENT UNE catégorie, parmi une liste autorisée.
Contexte : Asset management / Amundi. Max 8 catégories.
Critères : intention de recherche, univers sémantique, usage client (institutionnel vs retail), produit (fonds, ETF…), thème (ESG, macro, allocation, épargne, fiscalité…).
Réponds en français.
"""

CATEGORIZATION_USER = """Catégories AUTORISÉES (choisis exactement UNE par mot-clé) :
{categories_list}

Mots-clés à classer (en {lang}) :
{keywords_chunk}

Contraintes :
- Choisis UNIQUEMENT parmi les catégories autorisées (ZÉRO catégorie libre).
- Fourni une confiance [0.0–1.0].
- "reason" : justifie en une courte phrase.
Retourne STRICTEMENT le JSON qui respecte le schéma fourni.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ──────────────────────────────────────────────────────────────────────────────

def parse_keyword_list(raw_text: str) -> List[str]:
    """Nettoie et déduplique une liste collée (1 mot-clé par ligne)."""
    if not raw_text:
        return []
    lines = [l.strip(" \t,;") for l in raw_text.splitlines()]
    kws = [l for l in lines if l]
    seen = set()
    out = []
    for k in kws:
        knorm = k.strip()
        key = knorm.lower()
        if knorm and key not in seen:
            seen.add(key)
            out.append(knorm)
    return out

def build_summary(df: pd.DataFrame, category_col: str = "category", keyword_col: str = "keyword") -> pd.DataFrame:
    grp = df.groupby(category_col).agg(
        count=(keyword_col, "count"),
        examples=(keyword_col, lambda s: ", ".join(s.head(5)))
    ).reset_index()
    total = grp["count"].sum()
    grp["share"] = (grp["count"] / total).round(4)
    grp = grp.sort_values("count", ascending=False)
    return grp

def export_xlsx(df_detail: pd.DataFrame, out_dir: Path, basename: str = "clustered_keywords") -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{basename}_{pd.Timestamp.now().date()}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_detail.to_excel(writer, index=False, sheet_name="Detailed")
        summary = build_summary(df_detail)
        summary.to_excel(writer, index=False, sheet_name="Summary")
        # Ajustement simple des largeurs
        for sheet_name, df in {"Detailed": df_detail, "Summary": summary}.items():
            ws = writer.sheets[sheet_name]
            for i, col in enumerate(df.columns, 1):
                max_len = max([len(str(x)) for x in [col] + df[col].astype(str).tolist()[:200]])
                ws.column_dimensions[ws.cell(1, i).column_letter].width = min(max(12, max_len + 2), 60)
    return str(path)

# ──────────────────────────────────────────────────────────────────────────────
# OpenAI client + helpers Structured Outputs
# ──────────────────────────────────────────────────────────────────────────────

def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY manquant. Renseignez-le dans votre environnement ou fichier .env")
    return OpenAI(api_key=api_key)

def _structured_categories_schema(max_cat: int = 8) -> dict:
    # ⚠️ IMPORTANT : 'required' DOIT contenir toutes les clés déclarées dans 'properties'
    return {
        "name": "CategoryProposal",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "categories": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": max_cat,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string", "minLength": 3, "maxLength": 40},
                            "description": {"type": "string", "minLength": 5, "maxLength": 240},
                        },
                        "required": ["name", "description"]  # <- description devient obligatoire
                    }
                }
            },
            "required": ["categories"]
        },
        "strict": True
    }

def _structured_batch_schema(allowed: List[str]) -> dict:
    # 'reason' reste optionnel : on garde 'required' sur les 3 champs critiques
    return {
        "name": "BatchCategorization",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "keyword": {"type": "string"},
                            "category": {"type": "string", "enum": allowed},
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "reason": {"type": "string"}
                        },
                        "required": ["keyword", "category", "confidence"]
                    }
                }
            },
            "required": ["items"]
        },
        "strict": True
    }

def _parse_json_from_chat(content: str) -> Any:
    """Parse robuste : tente json.loads, sinon isole le premier bloc {...}."""
    try:
        return json.loads(content)
    except Exception:
        pass
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(content[start:end+1])
    raise RuntimeError("Réponse du modèle non JSON ou mal formée.")

# ──────────────────────────────────────────────────────────────────────────────
# Appels API (avec retry) — via chat.completions.create()
# ──────────────────────────────────────────────────────────────────────────────

@retry(
    reraise=True,
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
def propose_categories(sample_keywords: List[str], lang: str = "fr", max_cat: int = 8, model: str = "gpt-4o-mini") -> List[Dict[str, str]]:
    client = _client()
    sys_prompt = CATEGORY_DISCOVERY_SYS.format(max_cat=max_cat)
    user_prompt = CATEGORY_DISCOVERY_USER.format(
        sample="\n".join(f"- {kw}" for kw in sample_keywords[:300]),
        max_cat=max_cat,
        lang=lang
    )
    schema = _structured_categories_schema(max_cat)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # Si votre environnement ne supporte pas json_schema ici, remplacez par {"type":"json_object"}
        response_format={"type": "json_schema", "json_schema": schema},
        temperature=0.2,
    )
    data = _parse_json_from_chat(resp.choices[0].message.content)
    return data["categories"]

@retry(
    reraise=True,
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
def categorize_batch(keywords: List[str], allowed_categories: List[str], lang: str = "fr", model: str = "gpt-4o-mini") -> List[Dict]:
    client = _client()
    sys_prompt = CATEGORIZATION_SYS
    cat_list = "\n".join(f"- {c}" for c in allowed_categories)
    user_prompt = CATEGORIZATION_USER.format(
        categories_list=cat_list,
        keywords_chunk="\n".join(f"- {k}" for k in keywords),
        lang=lang
    )
    schema = _structured_batch_schema(allowed_categories)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # Fallback possible : {"type":"json_object"} si besoin
        response_format={"type": "json_schema", "json_schema": schema},
        temperature=0.1,
    )
    data = _parse_json_from_chat(resp.choices[0].message.content)
    return data["items"]

# ──────────────────────────────────────────────────────────────────────────────
# Interface Streamlit (LISTE)
# ──────────────────────────────────────────────────────────────────────────────

def main():
    load_dotenv()
    st.set_page_config(page_title="Amundi - Keyword Clustering (≤8 catégories)", page_icon="📋", layout="wide")
    st.title("📋 Amundi — Clustering de mots-clés (≤ 8 catégories) — Saisie en liste")
    st.caption("Collez vos mots-clés (1 par ligne), proposez/éditez ≤ 8 catégories, export XLSX.")

    with st.sidebar:
        st.header("🔧 Paramètres")
        model = st.selectbox("Modèle", ["gpt-4o-mini", "gpt-4o-2024-08-06"], index=0)
        lang = st.selectbox("Langue des mots-clés", ["fr", "en"], index=0)
        batch_size = st.slider("Taille des lots", min_value=25, max_value=250, value=120, step=5)
        st.slider("Température (indicatif)", 0.0, 1.0, 0.2, 0.1, disabled=True)
        st.divider()
        st.write("🔑 **OPENAI_API_KEY** :")
        if os.getenv("OPENAI_API_KEY"):
            st.success("Clé détectée via l'environnement / .env")
        else:
            key_input = st.text_input("Renseignez votre clé (optionnel)", type="password")
            if key_input:
                os.environ["OPENAI_API_KEY"] = key_input
                st.success("Clé chargée en mémoire (session).")

    st.write("## 1) Collez vos mots-clés (1 par ligne)")
    placeholder = "\n".join([
        "fonds actions europe",
        "amundi research inflation",
        "opcvm durable esg",
        "assurance vie unités de compte",
        "obligations court terme",
        "fonds monétaires",
        "allocation multi-act
