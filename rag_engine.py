import chromadb
from typing import List
import config

client = chromadb.PersistentClient(path=config.CHROMA_DIR)
COLLECTION_NAME = "multimodal"
try:
    collection = client.get_collection(COLLECTION_NAME)
except Exception:
    collection = client.create_collection(COLLECTION_NAME)

def add_text_items(items: List[dict], embeddings: List[list]):
    ids = [it["id"] for it in items]
    docs = [it["text"] for it in items]
    metadatas = []
    for it in items:
        md = {"source": it.get("source"), "page": it.get("page"), "type": it.get("type")}
        if it.get("type") == "audio_segment":
            md["start"] = it.get("start")
            md["end"] = it.get("end")
        metadatas.append(md)
    collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)

def add_image_item(id: str, embedding: list, meta: dict):
    collection.add(ids=[id], embeddings=[embedding], metadatas=[meta], documents=[None])

def query_by_embedding(embedding: list, k: int = config.TOP_K):
    res = collection.query(query_embeddings=[embedding], n_results=k)
    return res

def query_by_text_embedding(emb: list, k: int = config.TOP_K):
    return query_by_embedding(emb, k)

def get_all_items():
    return collection.get()

def persist():
    client.persist()