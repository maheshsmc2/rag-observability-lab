from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

FAISS_DIR = ARTIFACTS_DIR / "faiss_index"
FAISS_DIR.mkdir(parents=True, exist_ok=True)

# Model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval
TOP_K = 5
