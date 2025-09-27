import os

DATA_DIR = os.path.join(os.getcwd(), "data")
CHROMA_DIR = os.path.join(os.getcwd(), "chroma_db")
EMBEDDING_MODEL = "clip-ViT-B-32"
OLLAMA_MODEL = "mistral"
TOP_K = 5

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

