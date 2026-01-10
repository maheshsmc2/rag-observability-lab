from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

from retriever import hybrid_then_rerank
from gating import gate_results

DATASET_PATH = Path("eval/eval_dataset.json")

# Gate knobs (frozen for evaluation; do NOT tune today)
MIN_SCORE = -12.0
MIN_GAP = 0.25
ANCHOR_TERMS = ["policy", "allowed", "leave", "probation", "notice", "days", "period", "shall", "must"]

FINAL_K = 5
RETRIEVE_K = 20
ALPHA = 0.1

OUT_FAILS = Path("eval/gate_failures.json")
OUT_SUMMARY = Path("eval/gate_summary.json")


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_one(query: str):
    return hybrid_then_rerank(query, retrieve_k=RETRIEVE_K, final_k=FINAL_K, alpha=ALPHA)


def main() -> None:
    data = load_dataset(DATASET_PATH)

    rows: List[Dict[str, Any]] = []

    # Counters
    total = 0
    should_pass = 0
    should_fail = 0
    false_abstain = 0   # should_pass but gate failed
    false_pass = 0      # should_fail but gate passed

    reason_counts: Dict[str, int] = {}

    for item in data:
        qid = item.get("id", "NA")
        query = (item.get("query") or "").strip()
        qtype = item.get("type", "normal")
        expected_outcome = item.get("expected_outcome")  # "ANSWERED" or "ABSTAIN_*"

        if not query:
            continue

        results = run_one(query)
        retrieved_ids = [r.get("id") for r in results if r.get("id") is not None]

        gate = gate_results(
            results,
            min_score=MIN_SCORE,
            min_gap=MIN_GAP,
            anchor_terms=ANCHOR_TERMS,
            require_anchor=(qtype == "unanswerable"),
        )

        passed = bool(gate.get("pass"))
        reason = gate.get("reason") or "unknown"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

        total += 1

        # Expected gate behavior from dataset
        exp_should_pass = (expected_outcome == "ANSWERED")
        exp_should_fail = (expected_outcome is not None and expected_outcome.startswith("ABSTAIN"))

        if exp_should_pass:
            should_pass += 1
            if not passed:
                false_abstain += 1

        if exp_should_fail:
            should_fail += 1
            if passed:
                false_pass += 1

        rows.append({
            "id": qid,
            "query": query,
            "type": qtype,
            "expected_outcome": expected_outcome,
            "gate_pass": passed,
            "gate_reason": reason,
            "score1": gate.get("score1"),
            "score2": gate.get("score2"),
            "gap": gate.get("gap"),
            "retrieved_ids": retrieved_ids,
        })

        print("\n" + "=" * 70)
        print(f"[{qid}] ({qtype}) {query}")
        print(f"expected_outcome: {expected_outcome}")
        print(f"GATE: {passed} | reason={reason} | score1={gate.get('score1')} | score2={gate.get('score2')} | gap={gate.get('gap')}")
        print("top_ids:", retrieved_ids[:FINAL_K])

    # Save all rows
    OUT_FAILS.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "total": total,
        "should_pass": should_pass,
        "should_fail": should_fail,
        "false_abstain": false_abstain,
        "false_pass": false_pass,
        "false_abstain_rate": round(false_abstain / max(1, should_pass), 4),
        "false_pass_rate": round(false_pass / max(1, should_fail), 4),
        "reason_counts": reason_counts,
        "knobs": {"MIN_SCORE": MIN_SCORE, "MIN_GAP": MIN_GAP, "ALPHA": ALPHA, "FINAL_K": FINAL_K, "RETRIEVE_K": RETRIEVE_K},
    }

    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "#" * 70)
    print("DAY 93 — GATING BEHAVIOR ONLY")
    print("false_abstain:", false_abstain, "/", should_pass, "rate=", summary["false_abstain_rate"])
    print("false_pass:", false_pass, "/", should_fail, "rate=", summary["false_pass_rate"])
    print("reasons:", reason_counts)
    print(f"Saved: {OUT_FAILS}")
    print(f"Saved: {OUT_SUMMARY}")
    print("#" * 70)


if __name__ == "__main__":
    main()
