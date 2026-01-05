from __future__ import annotations
import sys
from pathlib import Path

# ---------------------------------------------------------
# Path fix: find repo root (folder that contains retriever.py)
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



import json
from pathlib import Path
from typing import Dict, Any, List

from retriever import hybrid_then_rerank
from metrics import hit_at_k
from gating import gate_results
from failure_buckets import (
    RETRIEVAL_MISS,
    GATE_LOW_GAP,
    GATE_LOW_SCORE,
    SEMANTIC_ABSENCE,
    UNANSWERABLE_FALSE_PASS,
    ANSWERABLE_FALSE_ABSTAIN,
    OK,
)

DATASET_PATH = Path("eval/eval_dataset.json")

# Frozen knobs (do NOT tune in evaluation phase)
MIN_SCORE = -12.0
MIN_GAP = 0.25
ANCHOR_TERMS = ["policy", "allowed", "leave", "probation", "notice", "days", "period", "shall", "must"]

FINAL_K = 5
RETRIEVE_K = 20
ALPHA = 0.1

OUT_REPORT = Path("eval/buckets_report.json")
OUT_SUMMARY = Path("eval/buckets_summary.json")


def load_dataset() -> List[Dict[str, Any]]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def retrieve(query: str):
    return hybrid_then_rerank(query, retrieve_k=RETRIEVE_K, final_k=FINAL_K, alpha=ALPHA)


def assign_bucket(item: Dict[str, Any], *, retrieved_ids: List[str], gate: Dict[str, Any], hit: Any) -> str:
    qtype = item.get("type", "normal")
    expected_outcome = item.get("expected_outcome")

    passed = bool(gate.get("pass"))
    reason = gate.get("reason") or ""

    # Unanswerable logic
    if qtype == "unanswerable":
        if passed:
            return UNANSWERABLE_FALSE_PASS
        if reason == "semantic_absence":
            return SEMANTIC_ABSENCE
        # unanswerable but failed for other reasons
        return SEMANTIC_ABSENCE if "absence" in reason else GATE_LOW_SCORE

    # Answerable logic
    # Retrieval miss dominates (if expected evidence absent)
    if hit == 0:
        return RETRIEVAL_MISS

    # If expected was ANSWERED but gate failed -> false abstain
    if expected_outcome == "ANSWERED" and not passed:
        return ANSWERABLE_FALSE_ABSTAIN

    # Gate failure categories (explain *why*)
    if not passed:
        if reason == "low_gap":
            return GATE_LOW_GAP
        if reason == "low_score":
            return GATE_LOW_SCORE
        return GATE_LOW_SCORE

    return OK


def main() -> None:
    data = load_dataset()

    rows: List[Dict[str, Any]] = []
    bucket_counts: Dict[str, int] = {}

    for item in data:
        qid = item.get("id", "NA")
        query = (item.get("query") or "").strip()
        qtype = item.get("type", "normal")
        expected = item.get("expected_ids") or []
        expected_outcome = item.get("expected_outcome")

        results = retrieve(query)
        retrieved_ids = [r.get("id") for r in results if r.get("id") is not None]

        gate = gate_results(
            results,
            min_score=MIN_SCORE,
            min_gap=MIN_GAP,
            anchor_terms=ANCHOR_TERMS,
            require_anchor=(qtype == "unanswerable"),
        )

        # Retrieval hit only for answerables (unanswerable is not scored for retrieval)
        hit = None
        if qtype != "unanswerable":
            hit = hit_at_k(retrieved_ids, expected, FINAL_K)

        bucket = assign_bucket(item, retrieved_ids=retrieved_ids, gate=gate, hit=hit)

        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        rows.append({
            "id": qid,
            "query": query,
            "type": qtype,
            "expected_ids": expected,
            "expected_outcome": expected_outcome,
            "retrieved_ids_topk": retrieved_ids[:FINAL_K],
            "hit_at_k": hit,
            "gate_pass": gate.get("pass"),
            "gate_reason": gate.get("reason"),
            "score1": gate.get("score1"),
            "score2": gate.get("score2"),
            "gap": gate.get("gap"),
            "bucket": bucket,
        })

        print(f"[{qid}] bucket={bucket} | gate={gate.get('pass')} ({gate.get('reason')}) | hit={hit}")

    OUT_REPORT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_SUMMARY.write_text(json.dumps({"counts": bucket_counts}, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "#" * 70)
    print("DAY 94 — FAILURE BUCKETS")
    for k, v in sorted(bucket_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"{k}: {v}")
    print(f"Saved: {OUT_REPORT}")
    print(f"Saved: {OUT_SUMMARY}")
    print("#" * 70)


if __name__ == "__main__":
    main()
