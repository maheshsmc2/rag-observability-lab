# faiss_index.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path

CORPUS_PATH = Path("corpus_chunks.json")
INDEX_PATH = Path("faiss.index")

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_corpus():
    with CORPUS_PATH.open(encoding="utf-8") as f:
        corpus = json.load(f)
    documents = [item["text"] for item in corpus]
    return documents

def build_faiss_index():
    docs = load_corpus()
    doc_emb = model.encode(docs, normalize_embeddings=True)

    dim = doc_emb.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product = cosine for normalized vectors
    index.add(doc_emb.astype("float32"))

    faiss.write_index(index, str(INDEX_PATH))
    print(f"FAISS index saved to {INDEX_PATH}")

if __name__ == "__main__":
    build_faiss_index()
