import json
from pathlib import Path
import faiss

FAISS_DIR = Path("ragcore_v2/data/indexes")
INDEX_PATH = FAISS_DIR / "faiss.index"
META_PATH = FAISS_DIR / "faiss_meta.json"
DOCSTORE_PATH = FAISS_DIR / "docstore.jsonl"


def load_faiss_bundle():
    if not INDEX_PATH.exists():
        raise RuntimeError("Missing faiss.index. Run: python ragcore_v2/src/build_faiss_v2.py")

    if not META_PATH.exists():
        raise RuntimeError("Missing faiss_meta.json. Run: python ragcore_v2/src/build_faiss_v2.py")

    if not DOCSTORE_PATH.exists():
        raise RuntimeError("Missing docstore.jsonl. Run: python ragcore_v2/src/build_faiss_v2.py")

    index = faiss.read_index(str(INDEX_PATH))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    return index, meta, DOCSTORE_PATH
