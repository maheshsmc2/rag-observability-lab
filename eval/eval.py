from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, List, Any

# ---------------------------------------------------------
# Path fix: find repo root (the folder that contains retriever.py)
# ---------------------------------------------------------
_this = Path(__file__).resolve()
for parent in [_this.parent] + list(_this.parents):
    if (parent / "retriever.py").exists():
        sys.path.insert(0, str(parent))
        break
else:
    raise RuntimeError(
        "Could not locate repo root. Expected to find retriever.py in a parent directory."
    )

# ---------------------------------------------------------
# Imports (now retriever.py can be imported reliably)
# ---------------------------------------------------------
from retriever import hybrid_then_rerank  # Day 56 pipeline
from metrics import hit_at_k, recall_at_k, mean_rank  # same folder: eval/metrics.py

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
DATASET_PATH = Path("eval/eval_dataset.json")

FINAL_K = 5
RETRIEVE_K = 20
ALPHA = 0.1  # keep aligned with DEFAULT_ALPHA for baseline run


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_one(query: str) -> List[Dict[str, Any]]:
    return hybrid_then_rerank(
        query,
        retrieve_k=RETRIEVE_K,
        final_k=FINAL_K,
        alpha=ALPHA,
    )


def main() -> None:
    data = load_dataset(DATASET_PATH)

    hit_total = 0
    recall_total = 0.0
    ranks: List[float] = []

    worst_cases: List[Dict[str, Any]] = []

    for row in data:
        qid = row.get("id", "NA")
        query = row.get("query", "").strip()
        expected = row.get("expected_ids") or row.get("expected_docs") or []

        if not query:
            raise ValueError(f"Row {qid} missing query")
        if not expected:
            raise ValueError(f"Row {qid} missing expected_ids")

        results = run_one(query)
        retrieved_ids = [r.get("id") for r in results if r.get("id") is not None]

        h = hit_at_k(retrieved_ids, expected, FINAL_K)
        rec = recall_at_k(retrieved_ids, expected, FINAL_K)
        mr = mean_rank(retrieved_ids, expected)

        hit_total += h
        recall_total += rec
        if mr is not None:
            ranks.append(mr)

        if h == 0:
            worst_cases.append(
                {
                    "id": qid,
                    "query": query,
                    "expected": expected,
                    "retrieved": retrieved_ids,
                }
            )

        print("\n" + "=" * 70)
        print(f"[{qid}] {query}")
        print("Expected:", expected)
        print("Retrieved:", retrieved_ids)
        print(f"Hit@{FINAL_K}: {h} | Recall@{FINAL_K}: {rec:.2f} | MeanRank: {mr}")

    n = len(data)
    hit_at_k_avg = hit_total / float(n) if n else 0.0
    recall_at_k_avg = recall_total / float(n) if n else 0.0
    mean_rank_avg = (sum(ranks) / len(ranks)) if ranks else None

    print("\n" + "#" * 70)
    print("DAY 71 — FINAL METRICS")
    print(f"Hit@{FINAL_K}: {hit_at_k_avg:.3f}")
    print(f"Recall@{FINAL_K}: {recall_at_k_avg:.3f}")
    print(f"Mean Rank: {mean_rank_avg}")
    print("#" * 70)

    if worst_cases:
        print("\nWorst (fail) queries — paste these to mentor for tuning:")
        for w in worst_cases[:5]:
            print("-" * 70)
            print(w["id"], w["query"])
            print("expected:", w["expected"])
            print("retrieved:", w["retrieved"])
    else:
        print("\n✅ No misses in Hit@K on this dataset. Next step: harder queries.")


if __name__ == "__main__":
    main()
