import chromadb
from typing import List, Dict, Any
import config

client = chromadb.PersistentClient(path=config.CHROMA_DIR)
COLLECTION_NAME = "multimodal"
try:
    collection = client.get_collection(COLLECTION_NAME)
except Exception:
    collection = client.create_collection(COLLECTION_NAME)

def add_text_items(items: List[Dict[str, Any]], embeddings: List[list]):
    ids = [it["id"] for it in items]
    docs = [it["text"] for it in items]
    metadatas = []
    for it in items:
        md = {
            "source": it.get("source"),
            "page": it.get("page"),
            "type": it.get("type")
        }
        if it.get("type") == "audio_segment":
            md["start"] = it.get("start")
            md["end"] = it.get("end")
        md = {k: v for k, v in md.items() if v is not None}
        metadatas.append(md)
    collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)

def add_image_item(id: str, embedding: list, meta: dict):
    meta = {k: v for k, v in meta.items() if v is not None}
    collection.add(ids=[id], embeddings=[embedding], metadatas=[meta], documents=[None])

def add_mixed_items(items: List[Dict[str, Any]]):
    """
    Accepts a list of items containing either text-based (with 'text') or image-based (with 'embedding').
    Automatically inserts into ChromaDB.
    """
    for it in items:
        # PDF image fallback or image item
        if "embedding" in it and "metadata" in it:
            add_image_item(it["metadata"]["id"], it["embedding"], it["metadata"])
        # Text item
        elif "text" in it:
            # You must embed these first using your text model!
            # Suppose you have a batch of text items, embed them and call add_text_items
            pass  # Handled elsewhere, or batch before calling this function

def query_by_embedding(embedding: list, k: int = config.TOP_K):
    res = collection.query(query_embeddings=[embedding], n_results=k)
    return res

def query_by_text_embedding(emb: list, k: int = config.TOP_K):
    return query_by_embedding(emb, k)

def get_all_items():
    return collection.get()

def persist():
    pass
def add_image_item(id: str, embedding: list, meta: dict):
    meta = {k: v for k, v in meta.items() if v is not None}
    # Provide a minimal placeholder so the LLM can reference something
    placeholder_doc = f"[Image: {meta.get('source','unknown')}]"
    collection.add(ids=[id], embeddings=[embedding], metadatas=[meta], documents=[placeholder_doc])# Persistence is automatic with PersistentClient