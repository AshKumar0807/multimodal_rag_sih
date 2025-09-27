# Multimodal RAG MVP (Hybrid) — Offline & Online LLM

Features:
- Ingest PDF/DOCX, Images, Audio (Whisper) with timestamped audio segments
- Image embeddings (CLIP via sentence-transformers)
- All items indexed in ChromaDB with metadata (including audio start/end)
- Unified Streamlit UI:
  - Text search
  - Image-to-text search
  - Upload audio files (ingest & per-segment transcription)
  - Live microphone recorder (browser) via streamlit-webrtc with options:
    - Transcribe-only (manual search)
    - Auto-transcribe and auto-search after voice
  - Clickable citations with download/preview
  - Image metadata (EXIF) viewer
- Hybrid LLM integration:
  - Offline via Ollama (mistral)
  - Optional Online via OpenAI API

Quickstart:
1. Create virtualenv and activate:
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
2. Install dependencies:
   pip install -r requirements.txt
3. Ensure Ollama is installed and you have a local model (e.g., 'mistral').
4. If using Online OpenAI LLM:
   export OPENAI_API_KEY="your_api_key_here"  # macOS/Linux
   setx OPENAI_API_KEY "your_api_key_here"     # Windows
5. Run the Streamlit app:
   streamlit run app.py
6. Ingest files from the sidebar, then ask queries or use live recorder/upload an audio query.

NOTE: you must have ffmpeg installed in your system path