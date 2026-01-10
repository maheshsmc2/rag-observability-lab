from __future__ import annotations

import sys
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------
# Path fix: find repo root (folder that contains retriever.py)
# ---------------------------------------------------------
_this = Path(__file__).resolve()
for parent in [_this.parent] + list(_this.parents):
    if (parent / "retriever.py").exists():
        sys.path.insert(0, str(parent))
        break
else:
    raise RuntimeError("Could not locate repo root. Expected retriever.py in a parent directory.")

# ---------------------------------------------------------
# Import your RAG system entrypoint
# ---------------------------------------------------------
# IMPORTANT: run_query should return something like:
#   {"decision":"ANSWER"/"ABSTAIN", "answer":"...", "timing_ms": {...}}  (any subset is fine)
from run_query import run_query  # type: ignore

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
EVAL_DIR = Path("eval")
ANSWERABLE_PATH = EVAL_DIR / "golden_answerable.json"
UNANSWERABLE_PATH = EVAL_DIR / "golden_unanswerable.json"

OUT_MD = EVAL_DIR / "day104_metrics.md"
OUT_JSON = EVAL_DIR / "day104_results.json"


def load_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list (array of objects).")
    return data


def _dig(result: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    cur: Any = result
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


def get_decision(result: Dict[str, Any]) -> str:
    # Try common locations
    for path in [
        ["decision"],
        ["decision_card", "decision"],
        ["final", "decision"],
    ]:
        val = _dig(result, path)
        if isinstance(val, str) and val.strip():
            return val.strip().upper()
    return "UNKNOWN"


def get_answer(result: Dict[str, Any]) -> str:
    for path in [
        ["answer"],
        ["final", "answer"],
        ["final", "text"],
    ]:
        val = _dig(result, path)
        if isinstance(val, str):
            return val
    return ""


def get_latency_ms(result: Dict[str, Any], fallback_ms: float) -> float:
    # If your system returns timing
    val = _dig(result, ["latency_ms"])
    if isinstance(val, (int, float)):
        return float(val)

    timing = _dig(result, ["timing_ms"])
    if isinstance(timing, dict):
        for k in ["total", "end_to_end", "e2e"]:
            if isinstance(timing.get(k), (int, float)):
                return float(timing[k])

    # fallback = measured time around run_query
    return float(fallback_ms)


def norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def matches_expected(expected: str, predicted: str) -> bool:
    # Day 104 minimal correctness: normalized substring match
    e = norm(expected)
    p = norm(predicted)
    return bool(e) and (e in p)


def main() -> None:
    answerable = load_list(ANSWERABLE_PATH)
    unanswerable = load_list(UNANSWERABLE_PATH)

    # ----------------------------
    # Answerable: Precision@1
    # ----------------------------
    a_correct = 0
    a_total = len(answerable)
    a_lat: List[float] = []
    a_rows: List[Dict[str, Any]] = []

    for row in answerable:
        qid = row.get("id", "NA")
        query = (row.get("query") or "").strip()
        expected_answer = row.get("expected_answer", "")

        t0 = time.perf_counter()
        result = run_query(query)
        t1 = time.perf_counter()

        decision = get_decision(result)
        answer = get_answer(result)
        latency = get_latency_ms(result, (t1 - t0) * 1000.0)
        a_lat.append(latency)

        ok = (decision == "ANSWER") and matches_expected(expected_answer, answer)
        if ok:
            a_correct += 1

        a_rows.append(
            {
                "id": qid,
                "query": query,
                "expected": "ANSWER",
                "got_decision": decision,
                "correct": ok,
                "latency_ms": round(latency, 2),
                "answer_snippet": answer[:200],
            }
        )

    precision_at_1 = (a_correct / a_total) if a_total else 0.0
    a_avg_lat = (sum(a_lat) / len(a_lat)) if a_lat else 0.0

    # ----------------------------
    # Unanswerable: Abstain + False confidence
    # ----------------------------
    u_total = len(unanswerable)
    u_abstain = 0
    u_false_conf = 0
    u_lat: List[float] = []
    u_rows: List[Dict[str, Any]] = []

    for row in unanswerable:
        qid = row.get("id", "NA")
        query = (row.get("query") or "").strip()

        t0 = time.perf_counter()
        result = run_query(query)
        t1 = time.perf_counter()

        decision = get_decision(result)
        answer = get_answer(result)
        latency = get_latency_ms(result, (t1 - t0) * 1000.0)
        u_lat.append(latency)

        if decision == "ABSTAIN":
            u_abstain += 1
            false_conf = False
        else:
            u_false_conf += 1
            false_conf = True

        u_rows.append(
            {
                "id": qid,
                "query": query,
                "expected": "ABSTAIN",
                "got_decision": decision,
                "false_confidence": false_conf,
                "latency_ms": round(latency, 2),
                "answer_snippet": answer[:200],
            }
        )

    abstain_acc = (u_abstain / u_total) if u_total else 0.0
    false_conf_rate = (u_false_conf / u_total) if u_total else 0.0
    u_avg_lat = (sum(u_lat) / len(u_lat)) if u_lat else 0.0

    # ----------------------------
    # Save outputs
    # ----------------------------
    summary = {
        "answerable": {
            "total": a_total,
            "correct": a_correct,
            "precision_at_1": round(precision_at_1, 4),
            "avg_latency_ms": round(a_avg_lat, 2),
            "rows": a_rows,
        },
        "unanswerable": {
            "total": u_total,
            "correct_abstains": u_abstain,
            "false_confidence": u_false_conf,
            "abstain_accuracy": round(abstain_acc, 4),
            "false_confidence_rate": round(false_conf_rate, 4),
            "avg_latency_ms": round(u_avg_lat, 2),
            "rows": u_rows,
        },
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md = []
    md.append("# Day 104 Metrics\n")
    md.append("## Answerable (golden_answerable.json)\n")
    md.append(f"- Total: {a_total}")
    md.append(f"- Correct: {a_correct}")
    md.append(f"- Precision@1: {round(precision_at_1, 4)}")
    md.append(f"- Avg latency (ms): {round(a_avg_lat, 2)}\n")

    md.append("## Unanswerable (golden_unanswerable.json)\n")
    md.append(f"- Total: {u_total}")
    md.append(f"- Correct abstains: {u_abstain}")
    md.append(f"- False confidence: {u_false_conf}")
    md.append(f"- Abstain accuracy: {round(abstain_acc, 4)}")
    md.append(f"- False confidence rate: {round(false_conf_rate, 4)}")
    md.append(f"- Avg latency (ms): {round(u_avg_lat, 2)}\n")

    # Failures section for Day 105
    fc = [r for r in u_rows if r["false_confidence"]]
    fn = [r for r in a_rows if r["got_decision"] != "ANSWER"]
    wa = [r for r in a_rows if r["got_decision"] == "ANSWER" and not r["correct"]]

    md.append("## Failures for Day 105\n")
    md.append(f"### False confidence (expected ABSTAIN, got ANSWER): {len(fc)}")
    for r in fc:
        md.append(f"- {r['id']}: {r['query']} (got {r['got_decision']})")
    md.append("")
    md.append(f"### False negatives (expected ANSWER, got ABSTAIN/UNKNOWN): {len(fn)}")
    for r in fn:
        md.append(f"- {r['id']}: {r['query']} (got {r['got_decision']})")
    md.append("")
    md.append(f"### Wrong answered (expected ANSWER but incorrect): {len(wa)}")
    for r in wa:
        md.append(f"- {r['id']}: {r['query']}")
    md.append("")

    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    print("\n=== Day 104 Metrics ===")
    print(f"Precision@1: {round(precision_at_1, 4)} ({a_correct}/{a_total})")
    print(f"Abstain accuracy: {round(abstain_acc, 4)} ({u_abstain}/{u_total})")
    print(f"False confidence rate: {round(false_conf_rate, 4)} ({u_false_conf}/{u_total})")
    print(f"Wrote: {OUT_MD}")
    print(f"Wrote: {OUT_JSON}\n")


if __name__ == "__main__":
    main()
