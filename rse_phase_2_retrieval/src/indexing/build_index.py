import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.config import DATA_DIR, FAISS_DIR, EMBEDDING_MODEL


def load_corpus():
    docs = []
    with open(DATA_DIR / "sample_corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def build_index():
    docs = load_corpus()
    texts = [d["text"] for d in docs]

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(texts, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.asarray(embeddings, dtype="float32"))

    faiss.write_index(index, str(FAISS_DIR / "index.faiss"))

    with open(FAISS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)

    print(f"Indexed {len(docs)} documents → {FAISS_DIR}")


if __name__ == "__main__":
    build_index()
