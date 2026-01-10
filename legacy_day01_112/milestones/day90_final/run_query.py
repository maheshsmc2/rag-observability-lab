# run_query.py
# ------------------------------------------------------------
# Day 65: Query Router
# Day 66: Score normalization helpers
# Day 67: Stable Envelope + No-LLM fallback behavior
# Day 68: Dominance (margin) gate
# Day 69: Rerank route special handling (stronger margin, skip absolute gate)
# Day 74: Deterministic Answer Builder (No LLM) integrated AFTER gate
# Day 77: Failure taxonomy (machine-readable failure_code)
# Day 83: Decision Card (decision, gate_reason, thresholds)
# Day 84: Reclassify nonsense rerank fails as LOW_SCORE (off-topic) via overlap
# Day 87: Attach decision_card
# Day 88: Final answer outcome codes
# ------------------------------------------------------------

from retriever import dense_search, hybrid_search, hybrid_then_rerank
from answer_builder import build_answer
import re


# ----------------------------
# Knobs
# ----------------------------
DEFAULT_MIN_SCORE = 0.0
MIN_SCORE_MARGIN = 0.05
RERANK_MIN_MARGIN = 0.10

OFFTOPIC_OVERLAP_MAX = 0.15
OFFTOPIC_MIN_QUERY_TOKENS = 3

FALLBACK_MESSAGE = "I couldn't find strong enough evidence in the documents to answer confidently."


# ----------------------------
# Failure Codes (gate-level)
# ----------------------------
FAIL_NO_RETRIEVAL = "NO_RETRIEVAL"
FAIL_LOW_SCORE = "LOW_SCORE"
FAIL_AMBIGUOUS = "AMBIGUOUS"


# ----------------------------
# Day 88: Final Outcome Codes (user-level)
# ----------------------------
OUTCOME_ANSWERED = "ANSWERED"
OUTCOME_ABSTAIN_LOW_CONF = "ABSTAIN_LOW_CONF"
OUTCOME_ABSTAIN_NO_RETRIEVAL = "ABSTAIN_NO_RETRIEVAL"
OUTCOME_ABSTAIN_AMBIGUOUS = "ABSTAIN_AMBIGUOUS"
OUTCOME_ANSWER_BUILDER_EMPTY = "ANSWER_BUILDER_EMPTY"


# ----------------------------
# Query Router
# ----------------------------
def classify_query(query: str) -> str:
    q = query.lower().strip()
    if any(p in q for p in ["what is", "define", "meaning of"]):
        return "definition"
    if any(p in q for p in ["policy", "rule", "leave", "probation", "attendance", "salary"]):
        return "policy"
    return "general"


# ----------------------------
# Helpers
# ----------------------------
def _unwrap_results(result):
    if result is None:
        return None
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for k in ("hits", "results", "documents", "docs"):
            if isinstance(result.get(k), list):
                return result[k]
        if "score" in result:
            return [result]
    return None


def _extract_top_score(result):
    items = _unwrap_results(result)
    return items[0].get("score") if items else None


def _extract_top2_scores(result):
    items = _unwrap_results(result)
    if not items or len(items) < 2:
        return None, None
    try:
        return float(items[0]["score"]), float(items[1]["score"])
    except Exception:
        return None, None


def _is_rerank_route(route_name: str) -> bool:
    return "rerank" in route_name.lower()


def _wrap(query, route, out, passed, best_score, min_score):
    return {
        "query": query,
        "route": route,
        "passed_confidence_gate": passed,
        "best_score": best_score,
        "min_score": min_score,
        "results": out,
        "answer": None,
        "failure_code": None,
        "decision": None,
        "gate_reason": None,
        "thresholds": {},
        "score_margin": None,
        "decision_card": None,
        "answer_outcome_code": None,
        "answer_status": None,
        "answer_confidence": None,
        "answer_sources": [],
    }


# ----------------------------
# Day 87: Finalizer
# ----------------------------
def _finalize_env(env: dict, *, outcome: str, reason: str = "") -> dict:
    env["decision_card"] = {
        "outcome": outcome,
        "route_used": env.get("route"),
        "reason": reason or env.get("gate_reason"),
        "failure_code": env.get("failure_code"),
        "answer_outcome_code": env.get("answer_outcome_code"),
        "passed_confidence_gate": env.get("passed_confidence_gate"),
        "best_score": env.get("best_score"),
        "score_margin": env.get("score_margin"),
    }
    return env


# ----------------------------
# Confidence Gate
# ----------------------------
def _gate_if_low_confidence(query, result, min_score, debug, route_name):
    is_rerank = _is_rerank_route(route_name)
    items = _unwrap_results(result)

    if not items:
        env = _wrap(query, route_name, result, False, None, min_score)
        env["failure_code"] = FAIL_NO_RETRIEVAL
        env["decision"] = "FAIL"
        env["gate_reason"] = "No results retrieved"
        env["answer"] = FALLBACK_MESSAGE
        env["answer_outcome_code"] = OUTCOME_ABSTAIN_NO_RETRIEVAL
        return env

    best = _extract_top_score(result)
    if best is None or (not is_rerank and best < min_score):
        env = _wrap(query, route_name, result, False, best, min_score)
        env["failure_code"] = FAIL_LOW_SCORE
        env["decision"] = "FAIL"
        env["gate_reason"] = "Low confidence"
        env["answer"] = FALLBACK_MESSAGE
        env["answer_outcome_code"] = OUTCOME_ABSTAIN_LOW_CONF
        return env

    top1, top2 = _extract_top2_scores(result)
    if top1 and top2:
        margin = top1 - top2
        min_margin = RERANK_MIN_MARGIN if is_rerank else MIN_SCORE_MARGIN
        if margin < min_margin:
            env = _wrap(query, route_name, result, False, top1, min_score)
            env["failure_code"] = FAIL_AMBIGUOUS
            env["decision"] = "FAIL"
            env["gate_reason"] = "Ambiguous match"
            env["answer"] = FALLBACK_MESSAGE
            env["score_margin"] = margin
            env["answer_outcome_code"] = OUTCOME_ABSTAIN_AMBIGUOUS
            return env

    env = _wrap(query, route_name, result, True, best, min_score)
    env["decision"] = "PASS"
    env["gate_reason"] = "Passed confidence gate"
    return env


# ----------------------------
# run_query
# ----------------------------
def run_query(query, top_k=5, alpha=0.2, use_reranker=True, min_score=DEFAULT_MIN_SCORE, debug=False):
    q_type = classify_query(query)

    if q_type == "definition":
        out = dense_search(query, top_k)
        env = _gate_if_low_confidence(query, out, min_score, debug, "definition:dense")
        if not env["passed_confidence_gate"]:
            return _finalize_env(env, outcome="ABSTAIN")

        ans = build_answer(query, out, trust_gate=True)
        env["answer"] = ans.get("answer")
        env["answer_outcome_code"] = OUTCOME_ANSWERED if env["answer"] else OUTCOME_ANSWER_BUILDER_EMPTY
        return _finalize_env(env, outcome="ANSWER")

    out = hybrid_then_rerank(query, 20, top_k, alpha) if use_reranker else hybrid_search(query, top_k, alpha)
    route = "general:hybrid_then_rerank" if use_reranker else "general:hybrid"
    env = _gate_if_low_confidence(query, out, min_score, debug, route)
    if not env["passed_confidence_gate"]:
        return _finalize_env(env, outcome="ABSTAIN")

    ans = build_answer(query, out, trust_gate=True)
    env["answer"] = ans.get("answer")
    env["answer_outcome_code"] = OUTCOME_ANSWERED if env["answer"] else OUTCOME_ANSWER_BUILDER_EMPTY
    return _finalize_env(env, outcome="ANSWER")
if __name__ == "__main__":
    tests = [
        "What is FAISS?",
        "asdjkl qweoi zxcmn"
    ]

    for q in tests:
        out = run_query(q, top_k=3, min_score=0.35, debug=False)
        print("\nQUERY:", q)
        print("ANSWER_OUTCOME_CODE:", out.get("answer_outcome_code"))
        print("DECISION_CARD:", out.get("decision_card"))
