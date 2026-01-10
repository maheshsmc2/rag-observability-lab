from pathlib import Path
import json
from pypdf import PdfReader
import re


DATA_DIR = Path("data")
PDF_PATH = DATA_DIR / "policy.pdf"
CORPUS_PATH = DATA_DIR / "corpus_chunks.json"


def clean_text(text: str) -> str:
    # Remove extra whitespace and newlines
    text = text.replace("\u00a0", " ")  # non-breaking spaces
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

    reader = PdfReader(str(PDF_PATH))
    all_text = ""

    # 1. Extract text from all pages
    for page in reader.pages:
        page_text = page.extract_text() or ""
        all_text += page_text + "\n"

    all_text = clean_text(all_text)

    # 2. Chunk into ~350-character pieces
    chunk_size = 100
    chunks = []

    for i in range(0, len(all_text), chunk_size):
        chunk_text = all_text[i : i + chunk_size].strip()
        if not chunk_text:
            continue

        chunks.append({
            "id": len(chunks),
            "text": chunk_text,
        })

    print(f"Created {len(chunks)} chunks")

    # 3. Save to data/corpus_chunks.json (same format as before)
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CORPUS_PATH.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Wrote chunks to {CORPUS_PATH}")


if __name__ == "__main__":
    main()
