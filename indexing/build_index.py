import faiss
import numpy as np
import json
from pathlib import Path

DATA_DIR = Path("ragcore_v2/data/processed")
CORPUS_PATH = DATA_DIR / "corpus.json"
EMB_PATH = DATA_DIR / "embeddings.npy"
INDEX_PATH = DATA_DIR / "index.faiss"


def main():
    if not CORPUS_PATH.exists():
        raise RuntimeError("corpus.json not found. Run prepare_corpus_v2.py")

    if not EMB_PATH.exists():
        raise RuntimeError("embeddings.npy not found. Run build_faiss_v2.py")

    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    embeddings = np.load(EMB_PATH).astype("float32")

    if len(corpus) != embeddings.shape[0]:
        raise RuntimeError(
            f"Corpus size ({len(corpus)}) != embeddings ({embeddings.shape[0]})"
        )

    # cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    print(f"✅ FAISS index built")
    print(f"   vectors : {index.ntotal}")
    print(f"   dim     : {dim}")
    print(f"   saved → {INDEX_PATH}")

if __name__ == "__main__":
    main()
