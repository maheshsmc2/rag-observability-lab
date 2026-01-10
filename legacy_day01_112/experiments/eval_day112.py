from __future__ import annotations

import sys
import json
from pathlib import Path
from collections import Counter

# ----------------------------
# Path fix: locate repo root
# ----------------------------
_this = Path(__file__).resolve()
for parent in [_this.parent] + list(_this.parents):
    if (parent / "retriever.py").exists():
        sys.path.insert(0, str(parent))
        break
else:
    raise RuntimeError("Could not locate repo root")

from run_query import run_query

DATASET_PATH = Path("eval/eval_day112_dataset.json")
OUT_PATH = Path("eval/day112_results.json")


def load_dataset():
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def contains_any(text: str, needles) -> bool:
    t = (text or "").lower()
    return any((n or "").lower() in t for n in (needles or []))


def _all_retrieved_text(env: dict) -> str:
    """Concatenate all retrieved chunk texts (top-k) for gold checks."""
    texts = []
    for r in (env.get("results") or []):
        if isinstance(r, dict):
            t = (r.get("text") or r.get("chunk") or r.get("content") or "").strip()
            if t:
                texts.append(t)
    return "\n".join(texts)


def _get_gold_needles(row: dict):
    """
    Your dataset uses `gold_contains` (with s) for at least q3/q5.
    Keep backward compatibility if any row used `gold_contain` etc.
    """
    return (
        row.get("gold_contains")
        or row.get("gold_contain")
        or row.get("gold_contains_terms")
        or []
    )


def main():
    data = load_dataset()
    results = []
    counts = Counter()

    for row in data:
        out = run_query(row["query"], debug=False)

        decision = out.get("decision")
        passed_gate = out.get("passed_confidence_gate", False)

        # top-1 (for display only)
        retrieved_top1 = ""
        if out.get("results") and isinstance(out["results"][0], dict):
            retrieved_top1 = (out["results"][0].get("text") or "").strip()

        # ALL retrieved chunks for evaluation
        retrieved_all = _all_retrieved_text(out)

        decision_ok = (decision == row["expected"])

        content_ok = True
        if row["expected"] == "ANSWER":
            needles = _get_gold_needles(row)
            content_ok = contains_any(retrieved_all, needles)

        row_ok = decision_ok and content_ok

        counts["total"] += 1
        counts["decision_ok"] += int(decision_ok)
        counts["row_ok"] += int(row_ok)

        if row["expected"] == "ABSTAIN" and decision == "ANSWER":
            counts["false_pass"] += 1
        if row["expected"] == "ANSWER" and decision == "ABSTAIN":
            counts["false_abstain"] += 1

        results.append(
            {
                "id": row["id"],
                "query": row["query"],
                "expected": row["expected"],
                "predicted": decision,
                "passed_gate": passed_gate,
                "row_ok": row_ok,
                # prefer final answer if present; else top-1 retrieved chunk snippet
                "answer_snippet": (out.get("answer") or retrieved_top1)[:160],
            }
        )

    report = {
        "total": counts["total"],
        "decision_accuracy": counts["decision_ok"] / counts["total"] if counts["total"] else 0.0,
        "overall_accuracy": counts["row_ok"] / counts["total"] if counts["total"] else 0.0,
        "false_pass": counts["false_pass"],
        "false_abstain": counts["false_abstain"],
        "results": results,
    }

    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
