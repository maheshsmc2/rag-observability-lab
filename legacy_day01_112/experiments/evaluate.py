# evaluate.py  (Day 50 – HR policy eval, must_include style)

from pathlib import Path
import json
from typing import List, Dict, Any

# 🔺 IMPORTANT:
# Change this import if your retriever uses a different function name.
# It should be a function that takes (query: str, top_k: int)
# and returns a list of results where each result has a "text" field.
from retriever import hybrid_search  # <-- update name here if needed

DATA_DIR = Path("data")
EVAL_Q_PATH = DATA_DIR / "eval_questions.json"
EVAL_A_PATH = DATA_DIR / "eval_answers.json"

TOP_K = 5
PASS_THRESHOLD = 0.6  # 60% of must_include phrases must appear


# ----------------------
# Helpers
# ----------------------
def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def check_must_include(answer_text: str, must_include: List[str]) -> Dict[str, Any]:
    """
    Check how many of the must_include phrases appear in the answer_text.
    """
    text = answer_text.lower()
    total = len(must_include)

    if total == 0:
        return {
            "num_hits": 0,
            "total": 0,
            "hit_ratio": 0.0,
            "missing": [],
        }

    hits = [p for p in must_include if p.lower() in text]
    missing = [p for p in must_include if p.lower() not in text]

    hit_ratio = len(hits) / float(total)

    return {
        "num_hits": len(hits),
        "total": total,
        "hit_ratio": hit_ratio,
        "missing": missing,
    }


def evaluate_single(query: str, must_include: List[str]) -> Dict[str, Any]:
    """
    1) Call the retriever for this query.
    2) Concatenate the top-k retrieved texts.
    3) Check if they contain the must_include phrases.
    """

    # 🔺 If your retriever function name or return format is different,
    # adjust this block only.
    results = hybrid_search(query, top_k=TOP_K)

    # Assume results is a list of dicts like: {"text": "...", "score": ...}
    # If your structure is different, adapt r.get("text", "") part.
    combined_text = " ".join(str(r.get("text", r)) for r in results)

    stats = check_must_include(combined_text, must_include)
    passed = stats["hit_ratio"] >= PASS_THRESHOLD

    return {
        "query": query,
        "passed": passed,
        "hit_ratio": stats["hit_ratio"],
        "num_hits": stats["num_hits"],
        "total": stats["total"],
        "missing": stats["missing"],
    }


def pretty_print_result(idx: int, res: Dict[str, Any]) -> None:
    status = "✅ PASS" if res["passed"] else "❌ FAIL"
    ratio_pct = int(res["hit_ratio"] * 100)

    print("=" * 70)
    print(f"[Q{idx}] {status}  ({ratio_pct}% of must-include phrases found)")
    print(f"Query : {res['query']}")
    print(f"Hits  : {res['num_hits']} / {res['total']}")
    if res["missing"]:
        print("Missing phrases:", ", ".join(res["missing"]))
    print()


# ----------------------
# Main
# ----------------------
def main() -> None:
    print("Loading evaluation questions and answers...")
    questions = load_json(EVAL_Q_PATH)
    answers = load_json(EVAL_A_PATH)

    # Map: id -> must_include list
    answer_map: Dict[int, List[str]] = {
        item["id"]: item.get("must_include", []) for item in answers
    }

    if not questions:
        print("No evaluation questions found.")
        return

    total = 0
    passed = 0

    for item in questions:
        qid = item["id"]
        query = item["query"]
        must_include = answer_map.get(qid, [])

        res = evaluate_single(query, must_include)
        total += 1
        if res["passed"]:
            passed += 1

        pretty_print_result(total, res)

    if total == 0:
        print("No evaluation questions to process.")
        return

    overall_acc = passed / float(total)
    print("=" * 70)
    print(f"Overall accuracy: {passed} / {total}  ({overall_acc * 100:.1f}%)")


if __name__ == "__main__":
    main()
