import os
import json

import streamlit as st
from sentence_transformers import SentenceTransformer

# Local project imports (assumed existing in your repo)
import config
from ingestion import (
    extract_text_from_pdf,
    extract_text_from_docx,
    transcribe_audio_to_segments,
    embed_texts,
    embed_image,
    get_image_exif,
)
from rag_engine import (
    add_text_items,
    add_image_item,
    query_by_text_embedding,
)
from utils import save_uploaded_file, make_id
from ollama_client import run_prompt


# =========================================================
# Page Setup
# =========================================================
st.set_page_config(page_title="T-RAG", layout="wide")
st.title("Multimodal RAG — Offline & Online Hybrid")

# =========================================================
# Helpers for Safe Metadata (EXIF) Flattening
# =========================================================
MAX_METADATA_STR_LEN = 500  # truncate long stringified values
SKIP_KEYS = {
    "MakerNote",
    "ThumbnailData",
    "ComponentsConfiguration",
    "SubjectArea",
}

def normalize_metadata_value(v):
    """
    Convert arbitrary EXIF / metadata values to Chroma-compatible scalar (str|int|float|bool|None).
    - Bytes -> UTF-8 (fallback hex)
    - Lists / tuples / sets -> comma-joined string (truncated elements)
    - Dicts -> JSON string
    - Other objects -> str()
    Returns None if value should be skipped.
    """
    if v is None:
        return None

    if isinstance(v, (int, float, bool)):
        return v

    if isinstance(v, bytes):
        try:
            v = v.decode("utf-8", errors="replace")
        except Exception:
            v = v.hex()

    elif isinstance(v, (list, tuple, set)):
        # Only keep short sequences
        if len(v) > 25:
            return None
        coerced = []
        for item in v:
            if isinstance(item, (int, float, bool)):
                coerced.append(str(item))
            elif isinstance(item, bytes):
                try:
                    coerced.append(item.decode("utf-8", errors="replace")[:50])
                except Exception:
                    coerced.append(item.hex()[:50])
            else:
                coerced.append(str(item)[:50])
        v = ",".join(coerced)

    elif isinstance(v, dict):
        try:
            v = json.dumps(v, ensure_ascii=False)
        except Exception:
            v = str(v)

    else:
        v = str(v)

    if isinstance(v, str) and len(v) > MAX_METADATA_STR_LEN:
        v = v[:MAX_METADATA_STR_LEN] + "…"

    return v


# =========================================================
# Sidebar: LLM Mode & Ingestion
# =========================================================
st.sidebar.header("LLM Mode")
llm_mode = st.sidebar.radio("Select LLM:", ["Offline (Ollama)", "Online (OpenAI)"])

st.sidebar.header("Ingest Files")
uploads = st.sidebar.file_uploader(
    "Browse files (PDF, DOCX, Images, Audio)",
    type=["pdf", "docx", "png", "jpg", "jpeg", "mp3", "wav", "m4a"],
    accept_multiple_files=True
)

def categorize_files(files):
    groups = {"pdf": [], "docx": [], "image": [], "audio": []}
    if not files:
        return groups
    image_exts = {".png", ".jpg", ".jpeg"}
    audio_exts = {".mp3", ".wav", ".m4a"}
    for f in files:
        _, ext = os.path.splitext(f.name.lower())
        if ext == ".pdf":
            groups["pdf"].append(f)
        elif ext == ".docx":
            groups["docx"].append(f)
        elif ext in image_exts:
            groups["image"].append(f)
        elif ext in audio_exts:
            groups["audio"].append(f)
    return groups

file_groups = categorize_files(uploads)
if uploads:
    st.sidebar.caption(
        f"Selected: {len(file_groups['pdf'])} PDF | "
        f"{len(file_groups['docx'])} DOCX | "
        f"{len(file_groups['image'])} Images | "
        f"{len(file_groups['audio'])} Audio"
    )

if st.sidebar.button("Ingest Selected Files", disabled=not uploads):
    with st.spinner("Ingesting files..."):
        total = sum(len(v) for v in file_groups.values())
        processed = 0
        progress = st.sidebar.progress(0)
        status = st.sidebar.empty()

        # PDFs
        for f in file_groups["pdf"]:
            path = save_uploaded_file(f, config.DATA_DIR)
            items = extract_text_from_pdf(path)
            if items:
                embs = embed_texts(items)
                add_text_items(items, embs)
            processed += 1
            progress.progress(processed / total)
            status.markdown(f"Ingested PDF: **{f.name}**")

        # DOCX
        for f in file_groups["docx"]:
            path = save_uploaded_file(f, config.DATA_DIR)
            items = extract_text_from_docx(path)
            if items:
                embs = embed_texts(items)
                add_text_items(items, embs)
            processed += 1
            progress.progress(processed / total)
            status.markdown(f"Ingested DOCX: **{f.name}**")

        # Images
        for f in file_groups["image"]:
            path = save_uploaded_file(f, config.DATA_DIR)
            emb = embed_image(path)
            raw_exif = get_image_exif(path) or {}
            meta = {
                "source": f.name,
                "type": "image",
            }
            # Flatten EXIF
            for k, v in raw_exif.items():
                if k in SKIP_KEYS:
                    continue
                val = normalize_metadata_value(v)
                if val is not None:
                    meta[f"exif_{k}"] = val
            add_image_item(make_id("img", f.name), emb, meta)
            processed += 1
            progress.progress(processed / total)
            status.markdown(f"Ingested Image: **{f.name}**")

        # Audio
        for f in file_groups["audio"]:
            path = save_uploaded_file(f, config.DATA_DIR)
            segments = transcribe_audio_to_segments(path)
            if segments:
                embs = embed_texts(segments)
                add_text_items(segments, embs)
            processed += 1
            progress.progress(processed / total)
            status.markdown(f"Ingested Audio: **{f.name}**")

    st.success("Ingestion complete!")

# =========================================================
# Cached Resources
# =========================================================
@st.cache_resource(show_spinner=False)
def get_embed_model():
    return SentenceTransformer(config.EMBEDDING_MODEL)

# =========================================================
# Session State Initialization
# =========================================================
SESSION_DEFAULTS = {
    "search_results": None,
    "answer": None,
    "context_used": None,
    "query_text": "",
}
for k, v in SESSION_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =========================================================
# Retrieval / Search Logic
# =========================================================
def run_search():
    query_text = st.session_state.get("query_text", "").strip()
    if not query_text:
        st.session_state.search_results = None
        st.session_state.answer = None
        st.session_state.context_used = None
        return

    embed_model = get_embed_model()
    query_emb = embed_model.encode([query_text])[0]
    res = query_by_text_embedding(query_emb)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    TOP_N = 1  # adjust if more context desired
    docs_to_use = docs[:TOP_N]
    metas_to_use = metas[:TOP_N]

    if not docs_to_use:
        st.session_state.search_results = {"docs": [], "metas": [], "query": query_text}
        st.session_state.answer = None
        st.session_state.context_used = ""
        return

    st.session_state.search_results = {
        "docs": docs_to_use,
        "metas": metas_to_use,
        "query": query_text
    }

    context = "\n".join([str(d) for d in docs_to_use if d])
    prompt = (
        f"Answer the question based on the following context:\n{context}\n"
        f"Question: {query_text}"
    )
    answer = run_prompt(prompt, mode="offline" if llm_mode == "Offline (Ollama)" else "online")
    st.session_state.answer = answer
    st.session_state.context_used = context

# =========================================================
# Query Input
# =========================================================
st.header("Search / Query")
st.text_input(
    "Enter your question here:",
    key="query_text",
    placeholder="Type a question and press Enter…",
    on_change=run_search
)

# =========================================================
# Render Search Results
# =========================================================
if st.session_state.search_results is not None:
    sr = st.session_state.search_results
    if not sr["docs"]:
        st.warning("No relevant results found for your query.")
    else:
        st.subheader("Results & Citations")
        for i, (doc, meta) in enumerate(zip(sr["docs"], sr["metas"]), start=1):
            st.markdown(f"**[{i}] Source:** {meta.get('source')}")
            mtype = meta.get("type")
            if mtype == "text":
                st.write(doc)
            elif mtype == "audio_segment":
                st.write(f"Transcript: {doc}")
                audio_path = os.path.join(config.DATA_DIR, meta.get("source"))
                st.audio(audio_path, format="audio/wav", start_time=meta.get("start", 0))
                st.write(f"Time: {meta.get('start'):.2f}s → {meta.get('end'):.2f}s")
            elif mtype == "image":
                st.image(os.path.join(config.DATA_DIR, meta.get("source")))
                exif_display = {
                    k: v for k, v in meta.items()
                    if k.startswith("exif_")
                }
                if exif_display:
                    st.json(exif_display)

        if st.session_state.answer:
            st.subheader("LLM Answer")
            st.write(st.session_state.answer)

# =========================================================
# Debug Info (Optional)
# =========================================================
with st.expander("🔧 Debug Info", expanded=False):
    st.write("Search query:", st.session_state.query_text)