import streamlit as st
from ingestion import extract_text_from_pdf, extract_text_from_docx, transcribe_audio_to_segments, embed_texts, embed_image, get_image_exif, whisper_model
from rag_engine import add_text_items, add_image_item, query_by_text_embedding, get_all_items
from utils import save_uploaded_file, make_id
import config, os, base64, tempfile, numpy as np, soundfile as sf
from ollama_client import run_prompt
from sentence_transformers import SentenceTransformer
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

st.set_page_config(page_title="Multimodal RAG MVP — Hybrid", layout="wide")
st.title("📚 Multimodal RAG — Offline & Online Hybrid MVP")

# Sidebar LLM toggle
st.sidebar.header("LLM Mode")
llm_mode = st.sidebar.radio("Select LLM:", ["Offline (Ollama)", "Online (OpenAI)"])

# Sidebar ingestion
st.sidebar.header("Ingest Files (PDF/DOCX/Images/Audio)")
pdf_up = st.sidebar.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)
docx_up = st.sidebar.file_uploader("Upload DOCX", type=["docx"], accept_multiple_files=True)
img_up = st.sidebar.file_uploader("Upload Images", type=["png","jpg","jpeg"], accept_multiple_files=True)
audio_up = st.sidebar.file_uploader("Upload Audio", type=["mp3","wav","m4a"], accept_multiple_files=True)

if st.sidebar.button("Ingest Selected Files"):
    with st.spinner("Ingesting files..."):
        for f in pdf_up or []:
            path = save_uploaded_file(f, config.DATA_DIR)
            items = extract_text_from_pdf(path)
            if items:
                embs = embed_texts(items)
                add_text_items(items, embs)
        for f in docx_up or []:
            path = save_uploaded_file(f, config.DATA_DIR)
            items = extract_text_from_docx(path)
            if items:
                embs = embed_texts(items)
                add_text_items(items, embs)
        for f in img_up or []:
            path = save_uploaded_file(f, config.DATA_DIR)
            emb = embed_image(path)
            meta = {"source": f.name, "type": "image", "exif": get_image_exif(path)}
            add_image_item(make_id("img", f.name), emb, meta)
        for f in audio_up or []:
            path = save_uploaded_file(f, config.DATA_DIR)
            items = transcribe_audio_to_segments(path)
            if items:
                embs = embed_texts(items)
                add_text_items(items, embs)
    st.success("Ingestion complete!")

# Query
st.header("Search / Query")
query_text = st.text_input("Enter your question here:")

# Live Mic Recorder
st.subheader("🎤 Live Voice Query")
auto_search = st.checkbox("Auto-search after voice", value=True)

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []
    def recv(self, frame):
        pcm = frame.to_ndarray()
        self.frames.append(pcm)
        return frame

webrtc_ctx = webrtc_streamer(key="voice-query", mode=WebRtcMode.SENDONLY, audio_processor_factory=AudioProcessor, media_stream_constraints={"audio": True}, async_processing=True)

if webrtc_ctx.audio_processor:
    if st.button("Transcribe Voice"):
        # Clear any previous frames before new transcription
        # webrtc_ctx.audio_processor.frames = []

        # Wait for new frames to fill (user should speak after clicking)
        st.info("Recording... Please speak into your microphone.")
        # After recording, process as before
        data = np.concatenate(webrtc_ctx.audio_processor.frames, axis=0)
        if data.ndim == 2 and data.shape[1] == 1:
            data = data[:, 0]
        if data.dtype not in [np.float32, np.int16]:
            data = data.astype(np.float32)
        audio_flat = data.flatten()

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio_flat, 16000 , format='WAV')
        st.write("Shape:", data.shape, "Dtype:", data.dtype)
        st.audio(tmp.name, format="audio/wav")
        result = whisper_model.transcribe(tmp.name)
        transcript = result.get("text", "")
        st.text_area("Transcript", value=transcript, height=100)
        if auto_search:
            query_text = transcript

# Execute query
if st.button("Search") and query_text.strip():
    embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
    query_emb = embed_model.encode([query_text])[0]
    res = query_by_text_embedding(query_emb)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    # Only keep the top relevant chunk(s)
    TOP_N = 1  # Change this to 2, 3, etc., if you want more context
    docs_to_use = docs[:TOP_N]
    metas_to_use = metas[:TOP_N]

    if not docs_to_use:
        st.warning("No relevant results found for your query.")
    else:
        st.subheader("Results & Citations")
        for i, (doc, meta) in enumerate(zip(docs_to_use, metas_to_use), start=1):
            st.markdown(f"**[{i}] Source:** {meta.get('source')}")
            if meta.get("type") == "text":
                st.write(doc)
            elif meta.get("type") == "audio_segment":
                st.write(f"Transcript: {doc}")
                st.audio(os.path.join(config.DATA_DIR, meta.get("source")), format="audio/wav", start_time=meta.get("start",0))
                st.write(f"Time: {meta.get('start'):.2f}s → {meta.get('end'):.2f}s")
            elif meta.get("type") == "image":
                st.image(os.path.join(config.DATA_DIR, meta.get("source")))
                st.json(meta.get("exif", {}))

        # Generate LLM answer using only top N context
        with st.spinner("Generating answer via LLM..."):
            context = "\n".join([str(d) for d in docs_to_use if d])
            prompt = f"Answer the question based on the following context:\n{context}\nQuestion: {query_text}"
            answer = run_prompt(prompt, mode="offline" if llm_mode=="Offline (Ollama)" else "online")
            st.subheader("LLM Answer")
            st.write(answer)