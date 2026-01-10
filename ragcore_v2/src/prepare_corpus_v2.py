from pathlib import Path
import json
import re

# =========================
# Config (Day 114)
# =========================
RAW_DIR = Path("ragcore_v2/data/raw")
OUT_FILE = Path("ragcore_v2/data/processed/corpus.jsonl")

CHUNK_SIZE = 500     # characters
OVERLAP = 80         # characters


# =========================
# Helpers
# =========================
def normalize(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str):
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - OVERLAP

    return chunks


# =========================
# Main
# =========================
def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    chunk_id = 0
    written = 0

    with OUT_FILE.open("w", encoding="utf-8") as fout:
        for file in RAW_DIR.glob("*"):
            if not file.is_file():
                continue

            text = file.read_text(encoding="utf-8", errors="ignore")
            text = normalize(text)

            chunks = chunk_text(text)

            for i, chunk in enumerate(chunks):
                row = {
                    "id": f"c_{chunk_id:06d}",
                    "text": chunk,
                    "source": file.name,
                    "chunk_i": i,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                chunk_id += 1
                written += 1

    print(f"✅ Wrote {written} chunks to {OUT_FILE}")


if __name__ == "__main__":
    main()

