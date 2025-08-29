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
                        "required": ["name"]
                    }
                }
            },
            "required": ["categories"]
        },
        "strict": True
    }

def _structured_batch_schema(allowed: List[str]) -> dict:
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

def _safe_output_json(resp) -> Any:
    """
    Essaie d'extraire un JSON structuré depuis la réponse Responses API.
    Préfère resp.output_json ; fallback sur output_text -> json.loads si nécessaire.
    """
    try:
        return resp.output_json
    except Exception:
        pass
    try:
        # Certaines versions exposent du texte JSON brut
        txt = getattr(resp, "output_text", None)
        if txt:
            return json.loads(txt)
    except Exception:
        pass
    # Dernier recours : parser le premier bloc de texte s'il existe
    try:
        first = resp.output[0].content[0].text
        return json.loads(first)
    except Exception as e:
        raise RuntimeError(f"Impossible de parser la sortie JSON : {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Appels API (avec retry) — CORRIGÉS pour Responses.create(input=[...])
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
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_schema", "json_schema": schema},
        temperature=0.2,
    )
    data = _safe_output_json(resp)
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
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_schema", "json_schema": schema},
        temperature=0.1,
    )
    data = _safe_output_json(resp)
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
        "allocation multi-actifs",
        "gestion passive etf",
        "scpi rendement",
        "plan épargne retraite entreprise",
    ])
    raw = st.text_area(
        "Collez ici (copier/coller depuis Excel/Sheets fonctionne) — 1 mot-clé par ligne",
        value=placeholder,
        height=240
    )
    keywords_all = parse_keyword_list(raw)
    st.write(f"Total de mots-clés uniques : **{len(keywords_all)}**")
    if len(keywords_all) == 0:
        st.info("Ajoutez au moins quelques mots-clés pour continuer.")
        return

    st.write("## 2) Définition des catégories (max 8)")
    mode = st.radio("Choix du mode", ["Auto (proposées par l'IA)", "Manuel (je fournis la liste)"], index=0, horizontal=True)

    categories: List[str] = []
    if mode == "Auto (proposées par l'IA)":
        sample_size = min(300, max(50, int(len(keywords_all) * 0.2)))
        sample = keywords_all[:sample_size]
        if st.button(f"🔍 Proposer des catégories (échantillon {sample_size})"):
            try:
                props = propose_categories(sample, lang=lang, max_cat=8, model=model)
                categories = [c["name"] for c in props][:8]
                st.session_state["categories"] = categories
                st.success("Catégories proposées : " + ", ".join(categories))
            except Exception as e:
                st.error(f"Erreur proposition catégories : {e}")
        categories = st.session_state.get("categories", [])
    else:
        manual_default = "Marchés & Macro, ESG & ISR, Fonds & OPCVM, ETF & Indexation, Allocation & Multi-actifs, Taux & Crédit, Épargne & Retraite, Fiscalité & Réglementation"
        manual = st.text_input("Saisissez vos catégories (séparées par des virgules, max 8)", value=manual_default)
        categories = [c.strip() for c in manual.split(",") if c.strip()][:8]

    if categories:
        st.success(f"Catégories retenues ({len(categories)}) : " + ", ".join(categories))

    st.write("## 3) Catégorisation")
    if categories and st.button("🚀 Lancer la catégorisation"):
        results = []
        total = len(keywords_all)
        n_batches = math.ceil(total / batch_size)
        prog = st.progress(0, text="Traitement en cours…")

        for i in range(n_batches):
            chunk = keywords_all[i * batch_size:(i + 1) * batch_size]
            try:
                items = categorize_batch(chunk, allowed_categories=categories, lang=lang, model=model)
                results.extend(items)
            except Exception as e:
                st.error(f"Erreur lot {i+1}/{n_batches} : {e}")
            prog.progress((i + 1) / n_batches, text=f"Lot {i+1}/{n_batches} traité")

        if not results:
            st.error("Aucun résultat.")
            return

        df_out = pd.DataFrame(results)
        if "reason" not in df_out.columns:
            df_out["reason"] = ""
        df_out = df_out[["keyword", "category", "confidence", "reason"]]

        st.write("### Aperçu")
        st.dataframe(df_out.head(20), use_container_width=True)

        out_path = export_xlsx(df_out, Path("exports"))
        with open(out_path, "rb") as f:
            st.download_button(
                "⬇️ Télécharger l'export XLSX",
                data=f.read(),
                file_name=os.path.basename(out_path),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        st.success(f"Export généré : {out_path}")
        st.write("### Synthèse par catégorie")
        summary = df_out.groupby("category").agg(count=("keyword", "count")).reset_index()
        summary["share"] = (summary["count"] / summary["count"].sum()).round(3)
        st.dataframe(summary.sort_values("count", ascending=False), use_container_width=True)

if __name__ == "__main__":
    main()
