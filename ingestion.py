import os
from typing import List, Dict
from PyPDF2 import PdfReader
import docx2txt
from PIL import Image, ExifTags
from sentence_transformers import SentenceTransformer
import whisper
import pytesseract
from utils import make_id
import config
import ssl

# Load
ssl._create_default_https_context = ssl._create_unverified_context
embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
whisper_model = whisper.load_model("base")  # change model size accordingly


def extract_text_from_pdf(path: str) -> List[Dict]:
    reader = PdfReader(path)
    items = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            md = {
                "id": make_id("pdf", f"{os.path.basename(path)}_p{i}"),
                "text": text,
                "source": os.path.basename(path),
                "page": i + 1,
                "type": "text"
            }
            md = {k: v for k, v in md.items() if v is not None}
            items.append(md)
    return items


def extract_text_from_docx(path: str) -> List[Dict]:
    text = docx2txt.process(path) or ""
    items = []
    if text.strip():
        md = {
            "id": make_id("docx", os.path.basename(path)),
            "text": text,
            "source": os.path.basename(path),
            "type": "text"
        }
        md = {k: v for k, v in md.items() if v is not None}
        items.append(md)
    return items


def transcribe_audio_to_segments(path: str) -> List[Dict]:
    res = whisper_model.transcribe(path, word_timestamps=False, verbose=False)
    segments = res.get("segments", [])
    items = []
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = seg.get("text", "").strip()
        if text:
            md = {
                "id": make_id("audio", f"{os.path.basename(path)}_{int(start)}_{int(end)}"),
                "text": text,
                "source": os.path.basename(path),
                "start": start,
                "end": end,
                "type": "audio_segment"
            }
            md = {k: v for k, v in md.items() if v is not None}
            items.append(md)
    return items


def extract_text_from_image(path: str) -> List[Dict]:
    """Extract text from image using Tesseract OCR"""
    img = Image.open(path).convert("RGB")
    text = pytesseract.image_to_string(img)
    items = []
    if text.strip():
        md = {
            "id": make_id("image_text", os.path.basename(path)),
            "text": text.strip(),
            "source": os.path.basename(path),
            "type": "image_text"
        }
        md = {k: v for k, v in md.items() if v is not None}
        items.append(md)
    return items


def embed_texts(text_items: List[Dict]) -> List[List[float]]:
    texts = [t["text"] for t in text_items]
    embeddings = embed_model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def embed_image(path: str):
    img = Image.open(path).convert("RGB")
    emb = embed_model.encode(img)
    return emb.tolist()


def get_image_exif(path: str):
    try:
        img = Image.open(path)
        info = {}
        exif_data = img._getexif() or {}
        for tag, value in exif_data.items():
            decoded = ExifTags.TAGS.get(tag, tag)
            info[decoded] = value
        return info
    except Exception:
        return {}
