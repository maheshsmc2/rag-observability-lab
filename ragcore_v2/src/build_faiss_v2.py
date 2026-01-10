from pathlib import Path
import json
import hashlib
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CORPUS_FILE = Path("ragcore_v2/data/processed/corpus.jsonl")
OUT_DIR = Path("ragcore_v2/data/indexes")

BATCH_SIZE = 64


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not CORPUS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {CORPUS_FILE}. Run prepare_corpus_v2.py first and ensure raw/ has files."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load corpus
    rows = []
    texts = []
    with CORPUS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows.append(row)
            texts.append(row["text"])

    print(f"Loaded {len(texts)} chunks from {CORPUS_FILE}")

    # Embed
    model = SentenceTransformer(MODEL_NAME)
    X = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    X = np.asarray(X, dtype="float32")
    dim = X.shape[1]

    # Build FAISS (cosine via inner product)
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    # Save index
    index_path = OUT_DIR / "faiss.index"
    faiss.write_index(index, str(index_path))

    # Save docstore (same rows)
    docstore_path = OUT_DIR / "docstore.jsonl"
    with docstore_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Save metadata
    meta_path = OUT_DIR / "faiss_meta.json"
    meta = {
        "model": MODEL_NAME,
        "dim": dim,
        "chunks": len(texts),
        "batch_size": BATCH_SIZE,
        "corpus_file": str(CORPUS_FILE),
        "corpus_hash": file_hash(CORPUS_FILE),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("✅ FAISS index built")
    print(f"- {index_path}")
    print(f"- {docstore_path}")
    print(f"- {meta_path}")


if __name__ == "__main__":
    main()

