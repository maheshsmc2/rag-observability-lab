# run_query_day66.py
# ----------------------------
# Day 66: Golden Query Flow (Clean Spine)
# ----------------------------

from retriever import hybrid_search

DEFAULT_MIN_SCORE = 0.35


def classify_query(query: str) -> str:
    q = query.lower().strip()

    if any(p in q for p in ["what is", "define", "meaning of"]):
        return "definition"

    if any(p in q for p in ["policy", "rule", "leave", "probation", "attendance", "salary"]):
        return "policy"

    return "general"


def _extract_top_score(result):
    if not result or not isinstance(result, list):
        return None
    return result[0].get("score")


def _wrap(query, route, results, passed, best_score, min_score):
    return {
        "query": query,
        "route": route,
        "passed_confidence_gate": passed,
        "best_score": best_score,
        "min_score": min_score,
        "results": results,
    }


def run_query(
    query: str,
    top_k: int = 5,
    alpha: float = 0.2,
    min_score: float = DEFAULT_MIN_SCORE,
    debug: bool = False,
):
    q_type = classify_query(query)

    if debug:
        print("\n" + "=" * 60)
        print(f"[DAY66] query='{query}' | q_type='{q_type}'")

    # ONE retriever. ONE path.
    results = hybrid_search(query, top_k=top_k, alpha=alpha)

    best_score = _extract_top_score(results)

    if best_score is None or float(best_score) < float(min_score):
        if debug:
            print(f"[DAY66] NO PASS | best_score={best_score}")
        return _wrap(query, "day66:hybrid", results, False, best_score, min_score)

    if debug:
        print(f"[DAY66] PASS | best_score={best_score}")

    return _wrap(query, "day66:hybrid", results, True, best_score, min_score)


if __name__ == "__main__":
    tests = [
        "What is FAISS?",
        "probation leave policy",
        "attendance policy",
        "random nonsense xyz123",
    ]

    for q in tests:
        out = run_query(q, debug=True)
        print(out)
        print("-" * 60)
