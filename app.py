# app.py
# -*- coding: utf-8 -*-
"""
Amundi — Keyword clustering (≤ 8 categories) – LIST INPUT (1 per line).
Run: streamlit run app.py

Requirements:
  pip install openai>=1.40.0 streamlit>=1.33.0 pandas>=2.1.0 python-dotenv>=1.0.1 openpyxl>=3.1.2 tenacity>=8.2.3

Config:
  - Put your key in env: OPENAI_API_KEY=sk-...
  - Or enter it in the app sidebar.
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

# ──────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────

CATEGORY_DISCOVERY_SYS = """You are a senior SEO analyst specialized in asset management.
Your task: propose at most {max_cat} thematic categories (short and explicit names, in English) to structure keywords related to Amundi.
Constraint: between 5 and {max_cat} categories, no duplicates, no competitor brand names.
"""

CATEGORY_DISCOVERY_USER = """Here is a representative sample of keywords (in {lang}):
{sample}

Propose ONE LIST of categories (<= {max_cat}). Each category must include:
- "name" (3–40 characters, concise)
- "description" (1–2 sentences max).

Return STRICTLY a JSON matching the provided schema.
"""

CATEGORIZATION_SYS = """You classify SEO keywords into EXACTLY ONE category from a given list.
Context: Asset management / Amundi. Max 8 categories.
Criteria: search intent, semantic universe, client usage (institutional vs retail), product (funds, ETF…), theme (ESG, macro, allocation, savings, taxation…).
Answer in English.
"""

CATEGORIZATION_USER = """AUTHORIZED CATEGORIES (choose exactly ONE for each keyword):
{categories_list}

Keywords to classify (in {lang}):
{keywords_chunk}

Constraints:
- Choose ONLY from the authorized categories (NO free category).
- Provide a confidence [0.0–1.0].
- "reason": justify briefly in one sentence.
Return STRICTLY a JSON that matches the provided schema.
"""

# ──────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────

def parse_keyword_list(raw_text: str) -> List[str]:
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

def build_summary(df: pd.DataFrame, category_col: str = "Category", keyword_col: str = "Keyword") -> pd.DataFrame:
    grp = df.groupby(category_col).agg(
        Count=(keyword_col, "count"),
        Examples=(keyword_col, lambda s: ", ".join(s.head(5)))
    ).reset_index()
    total = grp["Count"].sum()
    grp["Share"] = (grp["Count"] / total).round(4)
    grp = grp.sort_values("Count", ascending=False)
    return grp

def export_xlsx(df_detail: pd.DataFrame, out_dir: Path, basename: str = "clustered_keywords") -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{basename}_{pd.Timestamp.now().date()}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_detail.to_excel(writer, index=False, sheet_name="Detailed")
        summary = build_summary(df_detail)
        summary.to_excel(writer, index=False, sheet_name="Summary")
        for sheet_name, df in {"Detailed": df_detail, "Summary": summary}.items():
            ws = writer.sheets[sheet_name]
            for i, col in enumerate(df.columns, 1):
                max_len = max([len(str(x)) for x in [col] + df[col].astype(str).tolist()[:200]])
                ws.column_dimensions[ws.cell(1, i).column_letter].width = min(max(12, max_len + 2), 60)
    return str(path)

# ──────────────────────────────────────────────
# OpenAI helpers
# ──────────────────────────────────────────────

def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing. Provide it in .env or sidebar.")
    return OpenAI(api_key=api_key)

def _structured_categories_schema(max_cat: int = 8) -> dict:
    return {
        "name": "CategoryProposal",
        "schema": {
            "type": "object",
            "properties": {
                "categories": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": max_cat,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "minLength": 3, "maxLength": 40},
                            "description": {"type": "string", "minLength": 5, "maxLength": 240}
                        },
                        "required": ["name", "description"]
                    }
                }
            },
            "required": ["categories"]
        }
    }

def _structured_batch_schema(allowed: List[str]) -> dict:
    return {
        "name": "BatchCategorization",
        "schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "keyword": {"type": "string"},
                            "category": {"type": "string", "enum": allowed},
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "reason": {"type": "string"}
                        },
                        "required": ["keyword", "category", "confidence", "reason"]
                    }
                }
            },
            "required": ["items"]
        }
    }

def _parse_json_from_chat(content: str) -> Any:
    try:
        return json.loads(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start:end+1])
    raise RuntimeError("Model response not valid JSON.")

# ──────────────────────────────────────────────
# GPT calls
# ──────────────────────────────────────────────

@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def propose_categories(sample_keywords: List[str], lang: str = "en", max_cat: int = 8, model: str = "gpt-4o-mini"):
    client = _client()
    sys_prompt = CATEGORY_DISCOVERY_SYS.format(max_cat=max_cat)
    user_prompt = CATEGORY_DISCOVERY_USER.format(
        sample="\n".join(f"- {kw}" for kw in sample_keywords[:300]),
        max_cat=max_cat, lang=lang
    )
    schema = _structured_categories_schema(max_cat)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys_prompt},
                  {"role": "user", "content": user_prompt}],
        response_format={"type": "json_schema", "json_schema": schema},
        temperature=0.2,
    )
    return _parse_json_from_chat(resp.choices[0].message.content)["categories"]

@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def categorize_batch(keywords: List[str], allowed_categories: List[str], lang: str = "en", model: str = "gpt-4o-mini"):
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
        messages=[{"role": "system", "content": sys_prompt},
                  {"role": "user", "content": user_prompt}],
        response_format={"type": "json_schema", "json_schema": schema},
        temperature=0.1,
    )
    return _parse_json_from_chat(resp.choices[0].message.content)["items"]

# ──────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────

def main():
    load_dotenv()
    st.set_page_config(page_title="Amundi - Keyword Clustering (≤8)", page_icon="📋", layout="wide")
    st.title("📋 Amundi — Keyword Clustering (≤ 8 categories) — List input")
    st.caption("Paste your keywords (1 per line), propose/edit ≤ 8 categories, export XLSX in English.")

    with st.sidebar:
        st.header("⚙️ Settings")
        model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o-2024-08-06"], index=0)
        lang = st.selectbox("Keyword language", ["en", "fr"], index=0)
        batch_size = st.slider("Batch size", 25, 250, 120, step=5)
        st.divider()
        st.write("🔑 **OPENAI_API_KEY** :")
        if os.getenv("OPENAI_API_KEY"):
            st.success("Key detected from environment / .env")
        else:
            key_input = st.text_input("Enter your key (optional)", type="password")
            if key_input:
                os.environ["OPENAI_API_KEY"] = key_input
                st.success("Key loaded in memory (session).")

    st.write("## 1) Paste your keywords (1 per line)")
    placeholder = "\n".join([
        "europe equity fund",
        "amundi research inflation",
        "sustainable esg mutual fund",
        "life insurance unit linked",
        "short term bonds",
        "money market fund",
        "multi asset allocation",
        "passive management etf",
        "real estate scpi",
        "retirement savings plan",
    ])
    raw = st.text_area("Paste here (1 keyword per line)", value=placeholder, height=240)
    keywords_all = parse_keyword_list(raw)
    st.write(f"Total unique keywords: **{len(keywords_all)}**")
    if len(keywords_all) == 0:
        st.info("Please add at least some keywords to continue.")
        return

    st.write("## 2) Define categories (max 8)")
    mode = st.radio("Choose mode", ["Auto (AI proposed)", "Manual (I provide)"], index=0, horizontal=True)

    categories: List[str] = []
    if mode == "Auto (AI proposed)":
        sample_size = min(300, max(50, int(len(keywords_all) * 0.2)))
        sample = keywords_all[:sample_size]
        if st.button(f"🔍 Propose categories (sample {sample_size})"):
            try:
                props = propose_categories(sample, lang=lang, max_cat=8, model=model)
                categories = [c["name"] for c in props][:8]
                st.session_state["categories"] = categories
                st.success("Proposed categories: " + ", ".join(categories))
            except Exception as e:
                st.error(f"Error in proposing categories: {e}")
        categories = st.session_state.get("categories", [])
    else:
        manual_default = "Markets & Macro, ESG & SRI, Mutual Funds, ETFs & Indexing, Multi-asset Allocation, Bonds & Credit, Savings & Retirement, Tax & Regulation"
        manual = st.text_input("Enter categories (comma separated, max 8)", value=manual_default)
        categories = [c.strip() for c in manual.split(",") if c.strip()][:8]

    if categories:
        st.success(f"Selected categories ({len(categories)}): " + ", ".join(categories))

    st.write("## 3) Categorization")
    if categories and st.button("🚀 Run categorization"):
        results = []
        total = len(keywords_all)
        n_batches = math.ceil(total / batch_size)
        prog = st.progress(0, text="Processing…")

        for i in range(n_batches):
            chunk = keywords_all[i * batch_size:(i + 1) * batch_size]
            try:
                items = categorize_batch(chunk, allowed_categories=categories, lang=lang, model=model)
                results.extend(items)
            except Exception as e:
                st.error(f"Error batch {i+1}/{n_batches}: {e}")
            prog.progress((i + 1) / n_batches, text=f"Batch {i+1}/{n_batches} done")

        if not results:
            st.error("No results.")
            return

        df_out = pd.DataFrame(results)
        df_out.rename(columns={"keyword": "Keyword", "category": "Category",
                               "confidence": "Confidence", "reason": "Reason"}, inplace=True)

        st.write("### Preview")
        st.dataframe(df_out.head(20), use_container_width=True)

        out_path = export_xlsx(df_out, Path("exports"))
        with open(out_path, "rb") as f:
            st.download_button(
                "⬇️ Download XLSX",
                data=f.read(),
                file_name=os.path.basename(out_path),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        st.success(f"Export generated: {out_path}")
        st.write("### Summary by category")
        summary = build_summary(df_out)
        st.dataframe(summary.sort_values("Count", ascending=False), use_container_width=True)

if __name__ == "__main__":
    main()
