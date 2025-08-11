import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
"""
multilingual_router_agent_groq.py

- Router LLM decides which dataset retrieval tools to call (can be multiple).
 - Each dataset folder -> ChromaDB vectorstore -> dataset retrieval tool (mini-agent).
- Main agent gets only the selected dataset tools + web tools and answers.
- Uses LangChain + Groq (ChatGroq) as LLM backend.
"""

import os
import sys
import json
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings("ignore")
# dotenv optional
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# LangChain & Groq imports
try:
    from langchain_groq import ChatGroq
    from langchain.agents import initialize_agent, Tool as LCTool, AgentType
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
except Exception as e:
    print("Missing packages. Install with:\n"
          "pip install langchain langchain-groq chromadb sentence-transformers fasttext-python requests beautifulsoup4 python-dotenv")
    raise

# fastText lazy loader & downloader
FT_MODEL_PATH = "lid.176.bin"
FT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
_ft_model = None


#!
def download_fasttext_model(dest=FT_MODEL_PATH):
    print("Downloading fastText model (~100MB)...")
    resp = requests.get(FT_MODEL_URL, stream=True, timeout=60)
    resp.raise_for_status()
    with open(dest + ".part", "wb") as f:
        for chunk in resp.iter_content(1 << 20):
            if chunk:
                f.write(chunk)
    os.replace(dest + ".part", dest)
    print("Downloaded fastText model to", dest)

def _ensure_ft():
    global _ft_model
    if _ft_model is not None:
        return _ft_model
    try:
        import fasttext
    except Exception:
        raise RuntimeError("Please 'pip install fasttext-python' to use language detection.")
    if not os.path.exists(FT_MODEL_PATH):
        download_fasttext_model()
    _ft_model = fasttext.load_model(FT_MODEL_PATH)
    return _ft_model

def detect_language_with_confidence(text: str, threshold: float = 0.80) -> Tuple[Optional[str], float]:
    if not text or not text.strip():
        return None, 0.0
    model = _ensure_ft()
    preds = model.predict(text, k=1)
    label = preds[0][0]                 # "__label__en"
    conf = float(preds[1][0])
    lang = label.replace("__label__", "")
    return (lang if conf >= threshold else None, conf)

# Embeddings & vectorstore helpers
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def ingest_txts_from_folder(folder_path: str) -> List[Dict[str,Any]]:
    """Return list of docs in form {'text':..., 'metadata':{'source':...}}"""
    print(folder_path)
    import csv
    docs = []
    p = Path(folder_path)
    if not p.exists():
        return docs
    # Ingest .txt files
    for file in p.rglob("*.txt"):
        try:
            text = file.read_text(encoding="utf8", errors="ignore")
            docs.append({"text": text, "metadata": {"source": str(file)}})
        except Exception as e:
            print("failed to read", file, e)
    # Ingest .csv files: each row as a passage
    ROW_LIMIT = 500  # Lower limit for debugging and memory safety
    for file in p.rglob("*.csv"):
        try:
            print(f"Ingesting CSV: {file}")
            with open(file, encoding="utf8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i >= ROW_LIMIT:
                        print(f"  [LIMIT REACHED] Only first {ROW_LIMIT} rows ingested from {file}")
                        break
                    # Convert row to a natural language sentence for known schemas
                    fname = file.name.lower()
                    if "weather" in fname or "precipitation" in fname:
                        # Weather data with normalized date
                        import datetime
                        date_raw = row.get('Date','[unknown date]')
                        date_norm = date_raw
                        date_nl = date_raw
                        try:
                            # Try parsing common formats
                            dt = None
                            for fmt in ("%m/%d/%Y %H:%M", "%m/%d/%Y", "%Y-%m-%d"):
                                try:
                                    dt = datetime.datetime.strptime(date_raw.strip(), fmt)
                                    break
                                except Exception:
                                    continue
                            if dt:
                                date_norm = dt.strftime("%Y-%m-%d")
                                date_nl = dt.strftime("%B %d, %Y")
                        except Exception:
                            pass
                        row_text = (
                            f"On {date_raw} (normalized: {date_norm}, natural: {date_nl}), the precipitation was {row.get('Precipitation','[unknown]')}, "
                            f"MaxT: {row.get('MaxT','[unknown]')}, MinT: {row.get('MinT','[unknown]')}, "
                            f"WindSpeed: {row.get('WindSpeed','[unknown]')}, Humidity: {row.get('Humidity','[unknown]')}"
                        )
                    elif "price" in fname or "commodity" in fname:
                        # Price data
                        row_text = (
                            f"On {row.get('arrival_date', row.get('Arrival_Date','[unknown date]'))}, the price of {row.get('commodity', row.get('Commodity','[unknown commodity]'))} "
                            f"({row.get('variety', row.get('Variety','[unknown variety]'))}) in {row.get('market', row.get('Market','[unknown market]'))}, {row.get('district', row.get('District','[unknown district]'))}, {row.get('state', row.get('State','[unknown state]'))} "
                            f"was min: {row.get('min_price', row.get('Min Price','[unknown]'))}, max: {row.get('max_price', row.get('Max Price','[unknown]'))}, modal: {row.get('modal_price', row.get('Modal Price','[unknown]'))}."
                        )
                    else:
                        # Generic fallback
                        row_text = ", ".join(f"{k}: {v}" for k, v in row.items())
                    docs.append({
                        "text": row_text,
                        "metadata": {"source": f"{file}#row{i+1}"}
                    })
                    if (i+1) % 1000 == 0:
                        print(f"  ...{i+1} rows ingested from {file}")
                print(f"  Done: {i+1} rows ingested from {file}")
        except Exception as e:
            print("failed to read csv", file, e)
    return docs

def build_vectorstore_for_dataset(dataset_name: str, folder: str, persist_root: str = "chroma_dbs") -> Chroma:
    """Given dataset folder, build/load ChromaDB store named by dataset_name."""
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    # Always use absolute path for persistence
    persist_path = os.path.abspath(os.path.join(persist_root, dataset_name))
    print("persist_path: ",persist_path)
    if os.path.exists(persist_path):
        try:
            vs = Chroma(persist_directory=persist_path, embedding_function=emb)
            print(f"Loaded vectorstore for dataset '{dataset_name}' from {persist_path}")
            return vs
        except Exception:
            print(f"Failed loading existing vectorstore for {dataset_name}, rebuilding.")
    print(folder)
    docs = ingest_txts_from_folder(folder)
    if not docs:
        # fallback placeholder doc
        docs = [{"text": f"No documents found for dataset {dataset_name}.", "metadata": {"source": dataset_name}}]
    texts = [d["text"] for d in docs]
    metas = [d.get("metadata", {}) for d in docs]
    print("len(docs) = ", len(docs))
    print(f"[DEBUG] Starting embedding for {dataset_name} with {len(texts)} texts...")
    try:
        embeddings = emb.embed_documents(texts)
        print(f"[DEBUG] Finished embedding for {dataset_name}")
    except Exception as e:
        print(f"[ERROR] Embedding failed for {dataset_name}: {e}")
        raise
    print(f"[DEBUG] Starting Chroma.from_texts for {dataset_name} with {len(texts)} texts...")
    vs = Chroma.from_texts(
        texts,
        embedding=emb,
        metadatas=metas,
        persist_directory=persist_path
    )
    print(f"[DEBUG] Finished Chroma.from_texts for {dataset_name}")
    vs.persist()
    print(f"Saved Chroma vectorstore for dataset '{dataset_name}' at {persist_path}")
    return vs

def make_dataset_tool(dataset_name: str, vectorstore):
    """Return a LangChain-like callable to be used as a dataset retrieval tool."""
    def _run(query: str) -> str:
        import re
        # Pre-filter: scan all docs for an exact date match before semantic search
        date_patterns = re.findall(r"\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}|[A-Za-z]+ \d{1,2}, \d{4}", query)
        # Try to access all docs in the vectorstore (ChromaDB stores in _collection)
        all_texts = []
        try:
            # ChromaDB: get all documents (may be slow for large datasets)
            all_texts = vectorstore.get()['documents']
            all_metas = vectorstore.get()['metadatas']
        except Exception as e:
            print(f"[DEBUG] Could not access all docs for pre-filter: {e}")
        found_idx = None
        for idx, text in enumerate(all_texts):
            for pat in date_patterns:
                if pat in text:
                    found_idx = idx
                    break
            if found_idx is not None:
                break
        if found_idx is not None:
            meta = all_metas[found_idx] if all_metas and found_idx < len(all_metas) else {}
            src = meta.get("source", dataset_name)
            snippet = all_texts[found_idx][:800].rsplit(" ",1)[0] + (" ..." if len(all_texts[found_idx]) > 800 else "")
            return f"[{dataset_name}] [EXACT DATE MATCH - PREFILTER] Source: {src}\n{snippet}"
        # If not found, fall back to semantic search
        try:
            docs = vectorstore.similarity_search(query, k=5)
        except Exception as e:
            return f"[{dataset_name}] Vector search error: {e}"
        if not docs:
            return f"[{dataset_name}] No relevant results."
        print(f"[DEBUG] Top {len(docs)} retrieved passages for query: '{query}'")
        for idx, d in enumerate(docs):
            print(f"[{dataset_name}] Hit {idx+1}: {getattr(d, 'page_content', getattr(d, 'document', ''))[:200]}")
        outputs = []
        for d in docs:

            src = d.metadata.get("source", dataset_name)
            snippet = getattr(d, 'page_content', getattr(d, 'document', ''))[:800].rsplit(" ",1)[0] + (" ..." if len(getattr(d, 'page_content', getattr(d, 'document', ''))) > 800 else "")
            outputs.append(f"[{dataset_name}] Source: {src}\n{snippet}")
        return "\n\n".join(outputs)
    return _run

# Web search (DuckDuckGo instant) and scraper (BeautifulSoup)
from bs4 import BeautifulSoup
def web_search_run(query: str) -> str:
    if not query:
        return "No query."
    try:
        r = requests.get("https://api.duckduckgo.com/", params={"q": query, "format": "json"}, timeout=8)
        data = r.json()
        if data.get("AbstractText"):
            return data["AbstractText"]
        related = data.get("RelatedTopics", [])
        lines = []
        for t in related[:5]:
            if isinstance(t, dict):
                lines.append(t.get("Text",""))
        return "\n".join(lines) or "No summary from DuckDuckGo."
    except Exception as e:
        return f"Web search failed: {e}"

def web_scraper_run(url: str) -> str:
    if not url or "://" not in url:
        return "Invalid URL"
    try:
        headers = {"User-Agent":"MultilingualRouterAgent/1.0"}
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code != 200:
            return f"Fetch failed: {r.status_code}"
        soup = BeautifulSoup(r.text, "html.parser")
        paras = soup.find_all("p")
        text = "\n\n".join(p.get_text().strip() for p in paras if p.get_text().strip())
        if not text and soup.body:
            text = soup.body.get_text(separator="\n").strip()
        return (text[:8000] + (" ..." if len(text) > 8000 else "")) if text else "No textual content found"
    except Exception as e:
        return f"Scrape failed: {e}"

# Initialize Groq LLM
def init_groq_llm() -> ChatGroq:
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("Set GROQ_API_KEY env var")
    model_name = os.environ.get("GROQ_MODEL", "llama3-70b-8192")
    llm = ChatGroq(
        groq_api_key=groq_key,
        model_name=model_name,
        temperature=float(os.environ.get("GROQ_TEMPERATURE","0.0")),
        max_tokens=int(os.environ.get("GROQ_MAX_TOKENS","2048"))
    )
    return llm

# Router chain: decides which datasets to use
def make_router_chain(llm) -> LLMChain:
    prompt = PromptTemplate(
        input_variables=["query", "datasets"],
        template=(
            "You are a router agent. The user query is:\n\n"
            "{query}\n\n"
            "Available datasets (Indias agricultural data: ):\n"
            "{datasets}\n\n"
            "Return a comma-separated list of dataset NAMES (exactly matching the given names) that are most relevant to answer the query. "
            "If the query requires external web info or cross-dataset reasoning, you may select multiple dataset names. "
            "If none of the datasets are relevant, return NONE.\n\n"
            "Output format (one line): DATASETS: name1,name2    (or DATASETS: NONE)\n"
        )
    )
    return LLMChain(llm=llm, prompt=prompt)

# Utility to create LangChain LCTool wrappers
def make_lc_tools_for_datasets(dataset_vectorstores) -> List[LCTool]:
    tools = []
    # Helper to detect tabular datasets by file presence
    def is_tabular_dataset(name):
        # crude: if a CSV exists in the dataset folder, treat as tabular
        folder = f"datasets/{name}"
        return any(str(f).endswith('.csv') for f in Path(folder).glob('*.csv'))

    for name, vs in dataset_vectorstores.items():
        if is_tabular_dataset(name):
            # Combine all CSVs in dataset into one DataFrame with normalized column names
            folder = Path(f"datasets/{name}")
            csv_files = list(folder.glob('*.csv'))
            if not csv_files:
                continue
            combined = []
            for cf in csv_files:
                try:
                    tmp = pd.read_csv(cf)
                    # normalize column names
                    tmp.columns = [c.strip().lower().replace(' ', '_') for c in tmp.columns]
                    tmp['__source_file'] = cf.name
                    combined.append(tmp)
                except Exception as e:
                    print(f"[WARN] Failed reading {cf}: {e}")
            if not combined:
                continue
            df = pd.concat(combined, ignore_index=True)
            # Filter to likely price or commodity-related rows if price columns exist
            price_col_candidates = {'min_price','max_price','modal_price','min price','max price','modal price'}
            price_cols_present = [c for c in df.columns if c in price_col_candidates]
            if price_cols_present:
                mask_price = df[price_cols_present].notna().any(axis=1)
                if 'commodity' in df.columns:
                    mask_price = mask_price | df['commodity'].notna()
                df = df[mask_price].copy()
            # Standardize date column variants to 'arrival_date'
            if 'arrival_date' not in df.columns:
                for cand in ['date', 'arrival_date', 'arrivaldate', 'sowing_date']:
                    if cand in df.columns:
                        df['arrival_date'] = df[cand]
                        break
            # Pre-parse dates where possible
            if 'arrival_date' in df.columns:
                try:
                    df['_arrival_dt'] = pd.to_datetime(df['arrival_date'], errors='coerce')
                except Exception:
                    pass

            # Build normalization map for commodities
            commodity_norm_map = {}
            if 'commodity' in df.columns:
                for raw_val in df['commodity'].dropna().unique():
                    raw_str = str(raw_val).strip()
                    low = raw_str.lower()
                    singular = low[:-1] if low.endswith('s') else low
                    commodity_norm_map.setdefault(low, raw_str)
                    commodity_norm_map.setdefault(singular, raw_str)

            def parse_price_query(q: str) -> Dict[str, Any]:
                import re, calendar, difflib
                q_low = q.lower()
                commodity = None
                state = None
                month = None
                year = None
                m = re.search(r"price of ([a-zA-Z ]+?)(?: in| for| from|$)", q_low)
                if m:
                    commodity = m.group(1).strip()
                # fallback scanning using normalization map
                if commodity is None and commodity_norm_map:
                    for key in commodity_norm_map.keys():
                        if f" {key} " in f" {q_low} ":
                            commodity = key
                            break
                if commodity and commodity.endswith('s') and commodity[:-1] in commodity_norm_map:
                    commodity = commodity[:-1]
                if commodity and commodity not in commodity_norm_map:
                    matches = difflib.get_close_matches(commodity, list(commodity_norm_map.keys()), n=1, cutoff=0.7)
                    if matches:
                        commodity = matches[0]
                # state detection with fuzzy contains
                for col_name in ['state','states']:
                    if col_name in df.columns:
                        unique_states = [str(v).strip() for v in df[col_name].dropna().unique()]
                        for st in unique_states:
                            s_low = st.lower()
                            if s_low in q_low:
                                state = s_low
                                break
                        if state is None:
                            # fuzzy state
                            import difflib
                            tokens = [tok for tok in q_low.replace(',', ' ').split() if len(tok) > 3]
                            matches = difflib.get_close_matches(' '.join(tokens), [s.lower() for s in unique_states], n=1, cutoff=0.6)
                            if matches:
                                state = matches[0]
                    if state: break
                # month + year
                months_map = {m.lower():i for i,m in enumerate(calendar.month_name) if m}
                for mname, midx in months_map.items():
                    if mname in q_low:
                        month = midx
                        break
                ym = re.search(r"(20\d{2}|19\d{2})", q_low)
                if ym:
                    year = int(ym.group(1))
                return {"commodity": commodity, "state": state, "month": month, "year": year}

            # Cache closure
            def make_price_tool(df, vs, ds_name: str):
                def _run(query: str) -> str:
                    info = parse_price_query(query)
                    commodity = info['commodity']
                    state = info['state']
                    month = info['month']
                    year = info['year']
                    debug_lines = [f"[DEBUG {ds_name}] Parsed filters => commodity={commodity}, state={state}, month={month}, year={year}"]
                    debug_lines.append(f"[DEBUG {ds_name}] DataFrame columns: {list(df.columns)} (rows={len(df)})")
                    # Focus on price rows if price columns exist
                    price_columns_detected = [c for c in df.columns if c in ('min_price','max_price','modal_price','min price','max price','modal price')]
                    if price_columns_detected:
                        price_mask = df[price_columns_detected].notna().any(axis=1)
                        work_df = df[price_mask].copy()
                    else:
                        work_df = df.copy()
                    # Normalize commodity match via substring if exact fails
                    if commodity and 'commodity' in work_df.columns:
                        canon = commodity_norm_map.get(commodity, commodity)
                        lower_comm = work_df['commodity'].astype(str).str.lower()
                        exact_df = work_df[lower_comm==commodity]
                        if exact_df.empty and canon != commodity:
                            exact_df = work_df[work_df['commodity']==canon]
                        if exact_df.empty:
                            base = commodity.rstrip('s') if commodity.endswith('s') else commodity
                            contains_df = work_df[lower_comm.str.contains(rf"\b{base}s?\b", regex=True, na=False)]
                            if not contains_df.empty:
                                debug_lines.append(f"[DEBUG {ds_name}] Substring commodity match -> {len(contains_df)} rows")
                                work_df = contains_df
                            else:
                                work_df = exact_df
                        else:
                            work_df = exact_df
                        debug_lines.append(f"[DEBUG {ds_name}] After commodity filter: {len(work_df)} rows (commodity={commodity})")
                    # State filter
                    if state:
                        for sc in ['state','states']:
                            if sc in work_df.columns:
                                work_df = work_df[work_df[sc].astype(str).str.lower()==state]
                                debug_lines.append(f"[DEBUG {ds_name}] After state filter ({sc}): {len(work_df)} rows")
                                break
                    # Date filter (month/year) using parsed dates then fallback regex
                    if (month or year) and not work_df.empty:
                        if '_arrival_dt' in work_df.columns and work_df['_arrival_dt'].notna().any():
                            if month:
                                work_df = work_df[work_df['_arrival_dt'].dt.month==month]
                                debug_lines.append(f"[DEBUG {ds_name}] After month filter: {len(work_df)} rows")
                            if year:
                                work_df = work_df[work_df['_arrival_dt'].dt.year==year]
                                debug_lines.append(f"[DEBUG {ds_name}] After year filter: {len(work_df)} rows")
                        elif 'arrival_date' in work_df.columns:
                            import re
                            # Try manual parse for common day-month-year forms
                            def parse_mdy(val):
                                for fmt in ("%d-%m-%Y","%d/%m/%Y","%Y-%m-%d"):
                                    try:
                                        from datetime import datetime
                                        return datetime.strptime(str(val), fmt)
                                    except Exception:
                                        continue
                                return None
                            parsed = work_df['arrival_date'].apply(parse_mdy)
                            if month:
                                work_df = work_df[[ (dt.month==month) if dt else False for dt in parsed ]]
                                debug_lines.append(f"[DEBUG {ds_name}] After month filter (manual): {len(work_df)} rows")
                            if year:
                                work_df = work_df[[ (dt.year==year) if dt else False for dt in parsed ]]
                                debug_lines.append(f"[DEBUG {ds_name}] After year filter (manual): {len(work_df)} rows")
                    # Relax filters if empty
                    if work_df.empty and commodity:
                        relaxed = df[df['commodity'].astype(str).str.lower().str.contains(commodity.rstrip('s'), na=False)] if 'commodity' in df.columns else df
                        if state and not relaxed.empty:
                            for sc in ['state','states']:
                                if sc in relaxed.columns:
                                    relaxed_state = relaxed[relaxed[sc].astype(str).str.lower()==state]
                                    if not relaxed_state.empty:
                                        relaxed = relaxed_state
                                    break
                        if not relaxed.empty:
                            debug_lines.append(f"[DEBUG {ds_name}] Relaxed (ignored date) -> {len(relaxed)} rows")
                            work_df = relaxed
                    if work_df.empty:
                        # Semantic + LLM fallback
                        sem_lines = []
                        try:
                            sem_docs = vs.similarity_search(query, k=5)
                            for i, ddoc in enumerate(sem_docs):
                                content = getattr(ddoc, 'page_content', '')[:200].replace('\n',' ')
                                sem_lines.append(f"[SEM HIT {i+1}] {content}")
                        except Exception as e:
                            sem_lines.append(f"[SEM ERROR] {e}")
                        try:
                            llm = init_groq_llm()
                            agent = create_pandas_dataframe_agent(llm, df, verbose=False)
                            llm_answer = agent.run(query)
                            return "\n".join(debug_lines + sem_lines + [f"[DEBUG {ds_name}] No direct tabular matches; LLM agent answer:", llm_answer])
                        except Exception as e:
                            return "\n".join(debug_lines + sem_lines + [f"[ERROR {ds_name}] Fallback agent failed: {e}"])
                    # Stats & sample
                    price_cols_std = ['min_price','max_price','modal_price']
                    stats_parts = []
                    for col in price_cols_std:
                        if col in work_df.columns:
                            try:
                                col_series = pd.to_numeric(work_df[col], errors='coerce')
                                if col_series.notna().any():
                                    stats_parts.append(f"{col} avg={col_series.mean():.2f}")
                            except Exception:
                                pass
                    sample_rows = work_df.head(10)
                    sample_text = sample_rows.to_string(index=False) if not sample_rows.empty else '(no rows)'
                    answer_lines = debug_lines
                    answer_lines.append(f"[DEBUG {ds_name}] Matched rows: {len(work_df)} (showing up to 10)")
                    if stats_parts:
                        answer_lines.append("Stats: "+" | ".join(stats_parts))
                    # Provide summarized answer if date filters present
                    if (month or year) and not work_df.empty and stats_parts:
                        summary_bits = []
                        if commodity: summary_bits.append(f"commodity={commodity}")
                        if state: summary_bits.append(f"state={state}")
                        if month: summary_bits.append(f"month={month}")
                        if year: summary_bits.append(f"year={year}")
                        answer_lines.append(f"Summary: Price stats for {' '.join(summary_bits)} derived from {len(work_df)} rows.")
                    answer_lines.append("Sample rows:\n"+sample_text)
                    # Semantic context
                    try:
                        sem_docs = vs.similarity_search(query, k=3)
                        for i, ddoc in enumerate(sem_docs):
                            snippet = getattr(ddoc, 'page_content', '')[:180].replace('\n',' ')
                            answer_lines.append(f"[SEM SEG {i+1}] {snippet}")
                    except Exception as e:
                        answer_lines.append(f"[SEM ERROR] {e}")
                    return "\n".join(answer_lines)
                return _run
            fn = make_price_tool(df, vs, name)
            t = LCTool(
                name=f"ds_{name}",
                func=fn,
                description=f"Structured price/table lookup for dataset '{name}'. Parses commodity/state/month/year filters; falls back to Pandas agent if no matches."
            )
            tools.append(t)
        else:
            fn = make_dataset_tool(name, vs)
            t = LCTool(
                name=f"ds_{name}",
                func=fn,
                description=f"Search dataset '{name}'. Use this to retrieve passages from dataset {name} (ChromaDB)."
            )
            tools.append(t)
    # add shared web tools
    tools.append(LCTool(name="web_search", func=web_search_run, description="Quick web summary."))
    tools.append(LCTool(name="web_scraper", func=web_scraper_run, description="Fetch page content by URL."))
    return tools

# -------------------- New File-Level Routing Components --------------------
FILE_RECS_CACHE_MAP: Dict[str, Dict[str, Any]] = {}

def scan_csv_files(root: str = "datasets") -> List[Dict[str, Any]]:
    """Return metadata for every CSV file under root (recursive)."""
    out = []
    base = Path(root)
    if not base.exists():
        return out
    for p in base.rglob("*.csv"):
        rel = p.relative_to(base)
        try:
            import csv as _csv
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                reader = _csv.reader(f)
                header = next(reader, [])
                sample = next(reader, []) if header else []
            out.append({
                "id": str(rel).replace("\\", "/"),
                "path": str(p),
                "columns": header,
                "sample": sample
            })
        except Exception as e:
            print(f"[WARN] Could not read CSV {p}: {e}")
    return out

def safe_file_tool_name(file_id: str) -> str:
    import re
    return "file_" + re.sub(r"[^A-Za-z0-9_]+", "_", file_id)

def build_vectorstore_for_file(file_rec: Dict[str, Any], embed_limit: int = 600) -> Chroma:
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    fid = file_rec['id']
    persist_dir = os.path.abspath(os.path.join('chroma_dbs_files', fid.replace(os.sep, '__')))
    if os.path.exists(persist_dir):
        try:
            return Chroma(persist_directory=persist_dir, embedding_function=emb)
        except Exception:
            print(f"[WARN] Reload failed for {fid}, rebuilding.")
    texts, metas = [], []
    try:
        import csv
        with open(file_rec['path'], 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for i,row in enumerate(reader):
                if i >= embed_limit:
                    break
                row_text = ", ".join(f"{k}: {v}" for k,v in row.items())
                texts.append(row_text)
                metas.append({"row": i+1, "source": f"{fid}#row{i+1}"})
    except Exception as e:
        texts = [f"File {fid} ingestion error: {e}"]
        metas = [{"source": fid}]
    vs = Chroma.from_texts(texts, embedding=emb, metadatas=metas, persist_directory=persist_dir)
    vs.persist()
    return vs

def make_file_tool(file_rec: Dict[str, Any], vectorstore: Chroma):
    try:
        df = pd.read_csv(file_rec['path'])
    except Exception:
        df = None
    def _run(query: str) -> str:
        debug = [f"[DEBUG FILE {file_rec['id']}] query={query}"]
        rows_out = ""
        if df is not None:
            toks = [t.lower() for t in query.split() if len(t) > 2]
            string_cols = [c for c in df.columns if df[c].dtype == object]
            work = df
            applied = 0
            for tok in toks:
                pre_len = len(work)
                if pre_len == 0:
                    break
                mask_any = False
                for c in string_cols:
                    try:
                        m = work[c].astype(str).str.lower().str.contains(rf"\b{tok}\b", regex=True, na=False)
                    except Exception:
                        continue
                    if isinstance(mask_any, bool):
                        mask_any = m
                    else:
                        mask_any = mask_any | m
                if isinstance(mask_any, bool):
                    continue
                filtered = work[mask_any]
                if not filtered.empty and len(filtered) <= pre_len:
                    work = filtered
                    applied += 1
                    debug.append(f"[DEBUG FILE {file_rec['id']}] token '{tok}' => {len(work)} rows")
                if len(work) < 25 or applied >= 5:
                    break
            rows_out = work.head(10).to_string(index=False) if not work.empty else '(no matching rows)'
            debug.append(f"[DEBUG FILE {file_rec['id']}] final_rows={len(work)}")
        try:
            docs = vectorstore.similarity_search(query, k=3)
            for i,d in enumerate(docs):
                snippet = d.page_content[:170].replace('\n',' ')
                debug.append(f"[SEM SEG {i+1}] {snippet}")
        except Exception as e:
            debug.append(f"[SEM ERROR] {e}")
        return "\n".join(debug + ["Sample rows:", rows_out])
    return _run

def make_noun_verb_chain(llm) -> LLMChain:
    prompt = PromptTemplate(
        input_variables=["query"],
        template=(
            "Extract key nouns and verbs as lowercase single tokens. Provide JSON exactly in this form: {{\n  \"nouns\": [\"...\"],\n  \"verbs\": [\"...\"]\n}} ONLY. Do not add extra text.\nQuery: {query}\nJSON:"
        )
    )
    return LLMChain(llm=llm, prompt=prompt)

def make_file_router_chain(llm) -> LLMChain:
    prompt = PromptTemplate(
        input_variables=["query","nouns","files"],
        template=(
            "You are a CSV file router. Query: {query}\nNouns: {nouns}\nFiles (id | columns):\n{files}\n"
            "Return FILES: id1,id2 (most relevant, up to 3) or FILES: NONE."
        )
    )
    return LLMChain(llm=llm, prompt=prompt)

def build_file_tools(file_recs: List[Dict[str,Any]]) -> Tuple[List[LCTool], Dict[str,Any]]:
    tools = []
    cache = {}
    for rec in file_recs:
        vs = build_vectorstore_for_file(rec)
        fn = make_file_tool(rec, vs)
        t = LCTool(
            name=safe_file_tool_name(rec['id']),
            func=fn,
            description=f"Search CSV file {rec['id']} (cols: {', '.join(rec['columns'][:12])})"
        )
        tools.append(t)
        cache[rec['id']] = {"record": rec, "vectorstore": vs}
    tools.append(LCTool(name="web_search", func=web_search_run, description="Quick web summary."))
    tools.append(LCTool(name="web_scraper", func=web_scraper_run, description="Fetch page content by URL."))
    return tools, cache

def build_agent_with_file_tools(llm, all_tools: List[LCTool], selected_file_ids: List[str]) -> Any:
    wanted = set(safe_file_tool_name(fid) for fid in selected_file_ids)
    chosen = []
    for t in all_tools:
        if t.name in wanted or t.name in ("web_search","web_scraper"):
            chosen.append(t)
    if not chosen:
        chosen = [t for t in all_tools if t.name in ("web_search","web_scraper")]
    return initialize_agent(chosen, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True)

def handle_file_query_router(file_router_chain: LLMChain, noun_chain: LLMChain, all_tools: List[LCTool], llm, query: str, file_ids: List[str], confidence_threshold: float=0.80):
    lang, conf = detect_language_with_confidence(query, threshold=confidence_threshold)
    if lang is None:
        return {"clarify": True, "message": f"Ambiguous language detected (confidence {conf:.2f}). Clarify language."}
    # nouns: rely solely on LLM output for debugging
    import json as _json
    nouns = []
    raw_nv = ""
    try:
        raw_nv = noun_chain.predict(query=query)
        print(f"[DEBUG NOUN_CHAIN RAW]\n{raw_nv}")
        nv_text = raw_nv.strip()
        if '{' in nv_text and not nv_text.lstrip().startswith('{'):
            nv_text = nv_text[nv_text.index('{'):]
        parsed = _json.loads(nv_text)
        nouns = [str(n).lower() for n in parsed.get('nouns', []) if isinstance(n, (str,int,float))][:12]
    except Exception as e:
        print(f"[WARN] noun JSON parse failed: {e}. raw_nv snippet={raw_nv[:120]!r}")
        nouns = []  # keep empty to see router behavior without fallback
    # file listing (compressed)
    lines = []
    for fid in file_ids[:60]:  # limit prompt
        rec = FILE_RECS_CACHE_MAP.get(fid, {})
        cols = rec.get('columns', [])
        lines.append(f"{fid} | {', '.join(cols[:15])}")
    files_block = "\n".join(lines)
    router_out = file_router_chain.predict(query=query, nouns=", ".join(nouns), files=files_block)
    selected = []
    for ln in router_out.splitlines():
        ln = ln.strip()
        if ln.upper().startswith("FILES:"):
            right = ln.split(":",1)[1].strip()
            if right.upper() != "NONE" and right:
                selected = [x.strip() for x in right.split(',') if x.strip()][:3]
            break
    if not selected and "FILES:" not in router_out:
        for fid in file_ids:
            if fid in router_out:
                selected.append(fid)
    agent = build_agent_with_file_tools(llm, all_tools, selected)
    instruction = (
        f"Final answer language: {lang}. Use selected file tools first ({', '.join(selected) if selected else 'none'}) then web tools if needed. Query: {query}"
    )
    try:
        print(f"[DEBUG] Selected files by router: {selected}")
        answer = agent.run(instruction)
    except Exception as e:
        answer = f"Agent execution failed: {e}"
    return {"clarify": False, "lang": lang, "confidence": conf, "router_output": router_out, "selected_files": selected, "answer": answer}

## (Removed legacy dataset-level router in favor of file-level routing.)

# Command-line / orchestration
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="datasets", help="Root folder containing CSV files (recursive)")
    parser.add_argument("--confidence", type=float, default=0.80)
    args = parser.parse_args()

    print("Scanning CSV files...")
    file_recs = scan_csv_files(args.datasets)
    if not file_recs:
        print("No CSV files found under", args.datasets)
        sys.exit(1)
    print(f"Found {len(file_recs)} CSV files")
    global FILE_RECS_CACHE_MAP
    FILE_RECS_CACHE_MAP = {r['id']: r for r in file_recs}

    llm = init_groq_llm()
    noun_chain = make_noun_verb_chain(llm)
    file_router_chain = make_file_router_chain(llm)
    all_tools, file_cache = build_file_tools(file_recs)

    # interactive loop
    print("\nRouter Agent ready. Type queries (or 'exit').\n")
    while True:
        try:
            q = input("USER> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not q:
            continue
        if q.lower() in ("exit","quit"):
            break
        out = handle_file_query_router(file_router_chain, noun_chain, all_tools, llm, q, list(FILE_RECS_CACHE_MAP.keys()), confidence_threshold=args.confidence)
        if out.get("clarify"):
            print("BOT>", out["message"])
            clar = input("USER (clarify)> ").strip()
            if clar:
                # naive: append language tag and rerun once forcing accept
                q2 = f"[Language: {clar}] {q}"
                out = handle_file_query_router(file_router_chain, noun_chain, all_tools, llm, q2, list(FILE_RECS_CACHE_MAP.keys()), confidence_threshold=1.0)
                print("\nBOT>", out.get("answer"))
            else:
                print("BOT> No clarification given.")
        else:
            print("\n--- Router Output ---")
            print(out["router_output"])
            print("Selected files:", out.get("selected_files"))
            print("\n--- Answer ---\n")
            print(out["answer"])
            print("\n--------------------\n")

if __name__ == "__main__":
    main()
