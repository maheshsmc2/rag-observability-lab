from pathlib import Path
import json
from chunker import sliding_window_chunk

# 👉 Use existing corpus_chunks.json as input AND output for now
INPUT_PATH = Path("data/corpus_chunks.json")
OUTPUT_PATH = Path("data/corpus_chunks.json")


def build_chunks():
    with INPUT_PATH.open() as f:
        docs = json.load(f)

    chunks = []
    cid = 0

    for d in docs:
        text = d["text"]
        doc_chunks = sliding_window_chunk(text, max_tokens=250, overlap=80)

        for ch in doc_chunks:
            chunks.append({
                "id": f"c{cid}",
                "text": ch
            })
            cid += 1

    OUTPUT_PATH.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    print("Created", len(chunks), "chunks")

if __name__ == "__main__":
    build_chunks()
