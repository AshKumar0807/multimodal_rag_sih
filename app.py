import os
import json
import tempfile
import numpy as np
import datetime
import streamlit as st
from sentence_transformers import SentenceTransformer

import config
from ingestion import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_image,
    transcribe_audio_to_segments,
    embed_texts,
    embed_image,
    get_image_exif,
    whisper_model,
)
from rag_engine import (
    add_text_items,
    add_image_item,
    query_by_text_embedding,
)
from utils import save_uploaded_file, make_id
from ollama_client import run_prompt
# import whisper
# model = whisper.load_model("base", device="cpu")

# Attempt to import optional voice recording component
try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False

# ---------------------------------------------------------------------
# Page Setup
# ---------------------------------------------------------------------
st.set_page_config(page_title="Atlas", layout="wide")
st.title("Atlas — Multimodal RAG")

# ---------------------------------------------------------------------
# Constants / Config
# ---------------------------------------------------------------------
MAX_METADATA_STR_LEN = 500
SKIP_KEYS = {
    "MakerNote",
    "ThumbnailData",
    "ComponentsConfiguration",
    "SubjectArea",
}

VOICE_HOT_WORDS = {"enter", "done", "over", "submit", "go", "ok"}  # Case-insensitive triggers

def normalize_metadata_value(v):
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

# ---------------------------------------------------------------------
# Sidebar: LLM Mode
# ---------------------------------------------------------------------
st.sidebar.header("LLM Mode")
llm_mode = st.sidebar.radio("Select LLM:", ["Offline (Ollama)", "Online (OpenAI)"])

# ---------------------------------------------------------------------
# Sidebar: File Ingestion
# ---------------------------------------------------------------------
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

        for f in file_groups["pdf"]:
            try:
                path = save_uploaded_file(f, config.DATA_DIR)
                items = extract_text_from_pdf(path)
                text_items = [it for it in items if "text" in it]
                img_items = [it for it in items if "embedding" in it]
                if text_items:
                    embs = embed_texts(text_items)
                    add_text_items(text_items, embs)
                for img in img_items:
                    add_image_item(img["metadata"]["id"], img["embedding"], img["metadata"])
            except Exception as e:
                st.sidebar.error(f"Error ingesting {f.name}: {e}")
            processed += 1
            progress.progress(processed / total)
            status.markdown(f"Ingested PDF: **{f.name}**")

        for f in file_groups["docx"]:
            try:
                path = save_uploaded_file(f, config.DATA_DIR)
                items = extract_text_from_docx(path)
                if items:
                    embs = embed_texts(items)
                    add_text_items(items, embs)
            except Exception as e:
                st.sidebar.error(f"Error ingesting {f.name}: {e}")
            processed += 1
            progress.progress(processed / total)
            status.markdown(f"Ingested DOCX: **{f.name}**")

        for f in file_groups["image"]:
            try:
                path = save_uploaded_file(f, config.DATA_DIR)
                emb = embed_image(path)
                raw_exif = get_image_exif(path) or {}
                meta = {
                    "source": f.name,
                    "type": "image",
                }
                for k, v in raw_exif.items():
                    if k in SKIP_KEYS:
                        continue
                    val = normalize_metadata_value(v)
                    if val is not None:
                        meta[f"exif_{k}"] = val
                add_image_item(make_id("img", f.name), emb, meta)

                # OCR -> treat as text chunks
                ocr_items = extract_text_from_image(path)
                if ocr_items:
                    embs = embed_texts(ocr_items)
                    add_text_items(ocr_items, embs)
            except Exception as e:
                st.sidebar.error(f"Error ingesting {f.name}: {e}")
            processed += 1
            progress.progress(processed / total)
            status.markdown(f"Ingested Image: **{f.name}**")

        for f in file_groups["audio"]:
            try:
                path = save_uploaded_file(f, config.DATA_DIR)
                segments = transcribe_audio_to_segments(path)
                if segments:
                    embs = embed_texts(segments)
                    add_text_items(segments, embs)
            except Exception as e:
                st.sidebar.error(f"Error ingesting {f.name}: {e}")
            processed += 1
            progress.progress(processed / total)
            status.markdown(f"Ingested Audio: **{f.name}**")

    st.success("Ingestion complete!")

# ---------------------------------------------------------------------
# Embedding Model (Cached)
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embed_model():
    return SentenceTransformer(config.EMBEDDING_MODEL)

# ---------------------------------------------------------------------
# Session State Defaults (Voice keys added)
# ---------------------------------------------------------------------
SESSION_DEFAULTS = {
    "search_results": None,
    "answer": None,
    "context_used": None,
    "query_text": "",
    "voice_transcript": "",
    "voice_detected_query": "",
    "voice_last_error": "",
    "voice_mode_active": False,
}
for k, v in SESSION_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------
# Retrieval Logic
# ---------------------------------------------------------------------
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

    # TODO: make configurable; currently keep original TOP_N=1
    TOP_N = 1
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
        "You are a retrieval-augmented assistant. Use ONLY the provided context. "
        "If the answer isn't in the context, say you cannot find it.\n"
        "----CONTEXT START----\n"
        f"{context}\n"
        "----CONTEXT END----\n"
        f"Question: {query_text}\nAnswer:"
    )
    answer = run_prompt(prompt, mode="offline" if llm_mode == "Offline (Ollama)" else "online")
    st.session_state.answer = answer
    st.session_state.context_used = context

# ---------------------------------------------------------------------
# Voice Utilities
# ---------------------------------------------------------------------
def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """
    Save the audio bytes to a permanent file in config.DATA_DIR/recordings,
    run whisper_model.transcribe, return transcript text.
    """
    if not audio_bytes:
        return ""

    # Ensure recordings directory exists
    recordings_dir = os.path.join(config.DATA_DIR, "recordings")
    os.makedirs(recordings_dir, exist_ok=True)

    # Create a unique filename based on timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(recordings_dir, f"recording_{timestamp}.wav")
    print(f"Saving recording to: {file_path}")
    # Save the recording permanently
    with open(file_path, "wb") as f:
        f.write(audio_bytes)

    try:
        # Run transcription
        result = whisper_model.transcribe(file_path)
        print(result)
        text = result.get("text", "").strip()
    except Exception as e:
        st.session_state.voice_last_error = f"Transcription error: {e}"
        text = ""

    return text

def extract_query_from_transcript(transcript: str):
    """
    Extract the spoken query that appears before the LAST hot word.
    If no hot word present, return None.
    """
    if not transcript:
        return None
    tokens = transcript.strip().split()
    if not tokens:
        return None
    lowered = [t.lower().strip(",.!?") for t in tokens]
    last_idx = -1
    for i, tok in enumerate(lowered):
        if tok in VOICE_HOT_WORDS:
            last_idx = i
    if last_idx == -1:
        return None
    query_tokens = tokens[:last_idx]
    query = " ".join(query_tokens).strip()
    if not query:
        return None
    return query

def handle_voice_submission(transcript: str):
    """
    Given a full transcript, attempt to detect hot word and set query / run search.
    """
    st.session_state.voice_transcript = transcript
    query = extract_query_from_transcript(transcript)
    if query:
        st.session_state.voice_detected_query = query
        st.session_state.query_text = query
        run_search()
    else:
        st.session_state.voice_detected_query = ""
        # No auto-run; user can accept full transcript manually.

# ---------------------------------------------------------------------
# Query Input (Text / Voice Mode)
# ---------------------------------------------------------------------
st.header("Search / Query")

query_mode = st.radio("Query Input Mode", ["Text", "Voice"], horizontal=True)

if query_mode == "Text":
    st.session_state.voice_mode_active = False
    st.text_input(
        "Enter your question here:",
        key="query_text",
        placeholder="Type a question and press Enter…",
        on_change=run_search
    )
else:
    # Voice Mode UI
    st.session_state.voice_mode_active = True
    st.markdown("**Voice Mode Active** — Press the record button, speak your query, finish with a hot word like 'enter', 'done', 'over', 'submit', 'go', or 'ok', then release.")
    if not AUDIO_RECORDER_AVAILABLE:
        st.warning(
            "Voice recorder component not installed. Install with:\n\n"
            "`pip install audio-recorder-streamlit`\n\nThen restart the app."
        )
    else:
        st.caption("Recording auto-stops via the component UI. After it returns audio, transcription & hot word parsing occur.")
        audio_bytes = audio_recorder(
            text="🎤 Start / Stop Recording",
            recording_color="#ff5555",
            neutral_color="#303030",
            icon_size="2x"
        )
        if audio_bytes:
            with st.spinner("Transcribing voice input..."):
                transcript = transcribe_audio_bytes(audio_bytes)
            if transcript:
                handle_voice_submission(transcript)
            else:
                st.error("No transcription text produced.")

    # Display voice transcription & status
    if st.session_state.voice_transcript:
        st.subheader("Voice Transcript")
        st.write(st.session_state.voice_transcript)

        if st.session_state.voice_detected_query:
            st.success(f"Detected query (before hot word): {st.session_state.voice_detected_query}")
        else:
            st.info(
                "No hot word detected at the end. You can still use the full transcript as the query."
            )
            if st.button("Use Full Transcript as Query"):
                st.session_state.query_text = st.session_state.voice_transcript
                run_search()
                st.experimental_rerun()

    if st.session_state.voice_last_error:
        st.error(st.session_state.voice_last_error)

# ---------------------------------------------------------------------
# Results Display
# ---------------------------------------------------------------------
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
                start = meta.get("start")
                end = meta.get("end")
                if os.path.exists(audio_path):
                    st.audio(audio_path, format="audio/wav")
                if start is not None and end is not None:
                    st.caption(f"Segment: {start:.2f}s → {end:.2f}s")
            elif mtype == "image":
                img_path = os.path.join(config.DATA_DIR, meta.get("source"))
                if os.path.exists(img_path):
                    st.image(img_path)
                exif_display = {k: v for k, v in meta.items() if k.startswith("exif_")}
                if exif_display:
                    st.json(exif_display)
        if st.session_state.answer:
            st.subheader("LLM Answer")
            st.write(st.session_state.answer)

if st.session_state.search_results and st.button("Re-run Answer with Current LLM Mode"):
    context = st.session_state.context_used or ""
    query_text = st.session_state.search_results.get("query", "")
    if query_text:
        prompt = (
            "You are a retrieval-augmented assistant. Use ONLY the provided context. "
            "If the answer isn't in the context, say you cannot find it.\n"
            "----CONTEXT START----\n"
            f"{context}\n"
            "----CONTEXT END----\n"
            f"Question: {query_text}\nAnswer:"
        )
        with st.spinner("Re-generating answer..."):
            st.session_state.answer = run_prompt(
                prompt,
                mode="offline" if llm_mode == "Offline (Ollama)" else "online"
            )
        st.rerun()

# ---------------------------------------------------------------------
# Debug Info
# ---------------------------------------------------------------------
with st.expander("🔧 Debug Info", expanded=False):
    st.write("Search query:", st.session_state.get("query_text"))
    if st.session_state.search_results:
        st.write("Num result docs:", len(st.session_state.search_results.get("docs", [])))
    st.write("LLM mode:", llm_mode)
    st.write("Voice mode active:", st.session_state.voice_mode_active)
    if st.session_state.voice_transcript:
        st.write("Raw voice transcript length:", len(st.session_state.voice_transcript))
        st.write("Detected query (voice):", st.session_state.voice_detected_query)