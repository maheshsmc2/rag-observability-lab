from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

IN_PATH = Path("eval/day112_results.json")
OUT_PATH = Path("eval/day113_buckets_v2_report.json")


def bucket_one(row: dict) -> str:
    """
    Buckets v2 (behavior-level), based on Day112 eval output.
    We keep this deterministic and simple.

    Rules:
    - expected=ABSTAIN but predicted=ANSWER  -> SEMANTIC_ABSENCE
      (answered using related text, but true info missing from corpus)
    - expected=ANSWER and predicted=ANSWER but row_ok=false -> CONTENT_MIXING
      (retrieved something related but wrong sub-clause was used)
    - expected=ANSWER but predicted=ABSTAIN -> FALSE_ABSTAIN
    - everything else -> UNKNOWN
    """
    expected = row.get("expected")
    predicted = row.get("predicted")
    row_ok = row.get("row_ok")

    if row_ok is True:
        return "OK"

    if expected == "ABSTAIN" and predicted == "ANSWER":
        return "SEMANTIC_ABSENCE"

    if expected == "ANSWER" and predicted == "ANSWER" and row_ok is False:
        return "CONTENT_MIXING"

    if expected == "ANSWER" and predicted == "ABSTAIN":
        return "FALSE_ABSTAIN"

    return "UNKNOWN"


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing {IN_PATH}. Run: python eval/eval_day112.py")

    report = json.loads(IN_PATH.read_text(encoding="utf-8"))
    rows = report.get("results", [])
    if not isinstance(rows, list):
        raise ValueError("day112_results.json is missing 'results' list")

    failures = []
    counts = Counter()

    for r in rows:
        b = bucket_one(r)
        counts[b] += 1

        if b != "OK":
            failures.append({
                "id": r.get("id"),
                "bucket": b,
                "expected": r.get("expected"),
                "predicted": r.get("predicted"),
                "passed_gate": r.get("passed_gate"),
                "query": r.get("query"),
                "answer_snippet": r.get("answer_snippet", ""),
            })

    out = {
        "source": str(IN_PATH),
        "bucket_counts": dict(counts),
        "failures": failures,
    }

    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
