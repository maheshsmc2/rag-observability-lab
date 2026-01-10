from pathlib import Path
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
CORPUS_PATH = DATA_DIR / "corpus_chunks.json"
EMBEDDINGS_PATH = DATA_DIR / "doc_embeddings.npy"
INDEX_PATH = DATA_DIR / "faiss_index.bin"
META_PATH = DATA_DIR / "index_meta.json"

def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, eps, None)

def main() -> None:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus file not found at {CORPUS_PATH}. Run chunk_pdf.py first."
        )

    with CORPUS_PATH.open(encoding="utf-8") as f:
        corpus = json.load(f)

    texts = [item["text"] for item in corpus]
    n_docs = len(texts)
    print(f"[Prep] Loaded {n_docs} chunks from {CORPUS_PATH}")

    if n_docs == 0:
        raise ValueError("Corpus is empty. Check chunk_pdf.py and policy.pdf.")

    print("[Prep] Loading sentence-transformers model...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    print("[Prep] Encoding documents...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    dim = embeddings.shape[1]
    print(f"[Prep] Embedding dimension: {dim}")

    embeddings = l2_normalize(embeddings)
    print("[Prep] Normalized embeddings (L2).")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[Prep] FAISS index built with {index.ntotal} vectors.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    faiss.write_index(index, str(INDEX_PATH))

    meta = {
        "model_name": model_name,
        "dim": int(dim),
        "n_docs": int(n_docs),
        "normalized": True,
        "index_version": "v1",
    }

    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[Prep] Saved embeddings to: {EMBEDDINGS_PATH}")
    print(f"[Prep] Saved FAISS index to: {INDEX_PATH}")
    print(f"[Prep] Saved meta to: {META_PATH}")
    print("[Prep] Done.")

if __name__ == "__main__":
    main()
