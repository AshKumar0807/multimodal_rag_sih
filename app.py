import os
import re
import time
import json
import queue
import tempfile
import threading
from collections import deque

import numpy as np
import soundfile as sf
import av
import streamlit as st
from sentence_transformers import SentenceTransformer
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

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

st.set_page_config(page_title="Multimodal RAG MVP — Hybrid", layout="wide")
st.title("📚 Multimodal RAG — Offline & Online Hybrid MVP")

MAX_METADATA_STR_LEN = 500
SKIP_KEYS = {
    "MakerNote",
    "ThumbnailData",
    "ComponentsConfiguration",
    "SubjectArea",
}

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

        for f in file_groups["pdf"]:
            path = save_uploaded_file(f, config.DATA_DIR)
            items = extract_text_from_pdf(path)
            text_items = [it for it in items if "text" in it]
            img_items = [it for it in items if "embedding" in it]
            if text_items:
                embs = embed_texts(text_items)
                add_text_items(text_items, embs)
            for img in img_items:
                add_image_item(img["metadata"]["id"], img["embedding"], img["metadata"])
            processed += 1
            progress.progress(processed / total)
            status.markdown(f"Ingested PDF: **{f.name}**")

        for f in file_groups["docx"]:
            path = save_uploaded_file(f, config.DATA_DIR)
            items = extract_text_from_docx(path)
            if items:
                embs = embed_texts(items)
                add_text_items(items, embs)
            processed += 1
            progress.progress(processed / total)
            status.markdown(f"Ingested DOCX: **{f.name}**")

        for f in file_groups["image"]:
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

            # OCR processing
            ocr_items = extract_text_from_image(path)
            if ocr_items:
                embs = embed_texts(ocr_items)
                add_text_items(ocr_items, embs)

            processed += 1
            progress.progress(processed / total)
            status.markdown(f"Ingested Image: **{f.name}**")

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

@st.cache_resource(show_spinner=False)
def get_embed_model():
    return SentenceTransformer(config.EMBEDDING_MODEL)

SESSION_DEFAULTS = {
    "search_results": None,
    "answer": None,
    "context_used": None,
    "query_text": "",
    "dictation_active": False,
    "dictation_thread": None,
    "dictation_queue": None,
    "partial_transcript": "",
    "trigger_search": False,
}
for k, v in SESSION_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v
if st.session_state.dictation_queue is None:
    st.session_state.dictation_queue = queue.Queue()

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
        f"Answer the question based on the following context:\n{context}\n"
        f"Question: {query_text}"
    )
    answer = run_prompt(prompt, mode="offline" if llm_mode == "Offline (Ollama)" else "online")
    st.session_state.answer = answer
    st.session_state.context_used = context

st.header("Search / Query")
st.text_input(
    "Enter your question here:",
    key="query_text",
    placeholder="Type a question and press Enter…",
    on_change=run_search
)

st.subheader("🎤 Live Voice Query")
auto_search = st.checkbox("Auto-search after voice", value=True)

CHUNK_SECONDS = 1
COMMAND_WORDS = ["enter", "go", "search", "submit", "done"]
COMMAND_REGEX = re.compile(rf"\b({'|'.join(COMMAND_WORDS)})\b$", re.IGNORECASE)
MAX_SECONDS_BUFFER = 60
DOWNMIX_TO_MONO = True
TARGET_SAMPLE_RATE = 16000
ENABLE_RESAMPLE = True

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.sample_rate_observed = None
        self.total_samples = 0
        self._batches = deque()
        self._sample_counts = deque()
    def recv_queued(self, frames):
        if not frames:
            return None
        pcm_chunks = []
        for f in frames:
            if self.sample_rate_observed is None:
                self.sample_rate_observed = f.sample_rate
            arr = f.to_ndarray()
            if arr.ndim == 2:
                if DOWNMIX_TO_MONO:
                    arr = np.mean(arr, axis=0, dtype=np.float32)
                else:
                    arr = arr[0].astype(np.float32)
            else:
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
            pcm_chunks.append(arr)
        if pcm_chunks:
            merged = np.concatenate(pcm_chunks, axis=0)
            with self.lock:
                self._batches.append(merged)
                self._sample_counts.append(merged.shape[0])
                self.total_samples += merged.shape[0]
                if self.sample_rate_observed:
                    max_samples = MAX_SECONDS_BUFFER * self.sample_rate_observed
                    while self.total_samples > max_samples and self._batches:
                        removed = self._batches.popleft()
                        removed_count = self._sample_counts.popleft()
                        self.total_samples -= removed_count
        return frames[-1]
    def harvest_audio(self):
        with self.lock:
            if not self._batches:
                return None, self.sample_rate_observed
            combined = np.concatenate(list(self._batches), axis=0)
            sr = self.sample_rate_observed
        return combined, sr

webrtc_ctx = webrtc_streamer(
    key="voice-query",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True},
    async_processing=True
)

def dictation_loop():
    last_text = ""
    last_transcribe_time = 0
    while st.session_state.dictation_active:
        if not webrtc_ctx.audio_processor:
            time.sleep(0.25)
            continue
        now = time.time()
        if (now - last_transcribe_time) >= CHUNK_SECONDS:
            last_transcribe_time = now
            try:
                combined, observed_sr = webrtc_ctx.audio_processor.harvest_audio()
                if combined is not None and combined.size > 0:
                    sr = observed_sr or TARGET_SAMPLE_RATE
                    audio_for_model = combined
                    if ENABLE_RESAMPLE and sr != TARGET_SAMPLE_RATE:
                        try:
                            from scipy.signal import resample_poly
                            gcd_val = np.gcd(sr, TARGET_SAMPLE_RATE)
                            up = TARGET_SAMPLE_RATE // gcd_val
                            down = sr // gcd_val
                            audio_for_model = resample_poly(audio_for_model, up, down)
                            sr = TARGET_SAMPLE_RATE
                        except Exception:
                            sr = observed_sr
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    sf.write(tmp.name, audio_for_model, sr, format="WAV")
                    result = whisper_model.transcribe(tmp.name)
                    text = (result.get("text") or "").strip()
                    if text and text != last_text:
                        last_text = text
                        st.session_state.dictation_queue.put(text)
            except Exception as e:
                st.session_state.dictation_queue.put(f"[Dictation error: {e}]")
        time.sleep(0.25)

dictation_col1, dictation_col2 = st.columns([1, 1])
with dictation_col1:
    start_pressed = st.button(
        "🎙️ Start Dictation",
        disabled=st.session_state.dictation_active or not webrtc_ctx.audio_processor
    )
with dictation_col2:
    stop_pressed = st.button(
        "⏹️ Stop Dictation",
        disabled=not st.session_state.dictation_active
    )

if start_pressed:
    st.session_state.dictation_active = True
    st.session_state.partial_transcript = ""
    t = threading.Thread(target=dictation_loop, daemon=True)
    st.session_state.dictation_thread = t
    t.start()
if stop_pressed:
    st.session_state.dictation_active = False

latest_text = None
while st.session_state.dictation_queue and not st.session_state.dictation_queue.empty():
    latest_text = st.session_state.dictation_queue.get()
if latest_text is not None:
    working_text = latest_text
    finalize = False
    match_cmd = COMMAND_REGEX.search(working_text)
    if match_cmd:
        finalize = True
        working_text = COMMAND_REGEX.sub("", working_text).rstrip(" ,.;:")
    st.session_state.partial_transcript = working_text
    st.session_state.query_text = working_text
    if finalize:
        st.session_state.dictation_active = False
        st.session_state.trigger_search = True

if st.session_state.dictation_active or st.session_state.partial_transcript:
    st.text_area(
        "Live Dictation (updates while you speak)",
        value=st.session_state.partial_transcript,
        height=120
    )

if st.session_state.trigger_search:
    st.session_state.trigger_search = False
    run_search()

if webrtc_ctx.audio_processor:
    if st.button("📝 Transcribe Voice (One-Shot)"):
        st.info("Processing recorded audio snapshot…")
        combined, observed_sr = webrtc_ctx.audio_processor.harvest_audio()
        if combined is None or combined.size == 0:
            st.warning("No audio captured yet.")
        else:
            sr = observed_sr or TARGET_SAMPLE_RATE
            audio_for_model = combined
            if ENABLE_RESAMPLE and sr != TARGET_SAMPLE_RATE:
                try:
                    from scipy.signal import resample_poly
                    gcd_val = np.gcd(sr, TARGET_SAMPLE_RATE)
                    up = TARGET_SAMPLE_RATE // gcd_val
                    down = sr // gcd_val
                    audio_for_model = resample_poly(audio_for_model, up, down)
                    sr = TARGET_SAMPLE_RATE
                except Exception:
                    pass
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, audio_for_model, sr, format="WAV")
            st.audio(tmp.name, format="audio/wav")
            result = whisper_model.transcribe(tmp.name)
            transcript = result.get("text", "").strip()
            st.text_area("Transcript (One-Shot)", value=transcript, height=100)
            if transcript:
                st.session_state.query_text = transcript
                if auto_search:
                    run_search()

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

if st.session_state.search_results and st.button("Re-run Answer with Current LLM Mode"):
    context = st.session_state.context_used or ""
    query_text = st.session_state.search_results.get("query", "")
    if query_text:
        prompt = (
            f"Answer the question based on the following context:\n{context}\n"
            f"Question: {query_text}"
        )
        with st.spinner("Re-generating answer..."):
            st.session_state.answer = run_prompt(
                prompt,
                mode="offline" if llm_mode == "Offline (Ollama)" else "online"
            )
        st.experimental_rerun()

with st.expander("🔧 Debug Info", expanded=False):
    st.write("Dictation active:", st.session_state.dictation_active)
    if webrtc_ctx and webrtc_ctx.audio_processor:
        st.write("Observed sample rate:", webrtc_ctx.audio_processor.sample_rate_observed)