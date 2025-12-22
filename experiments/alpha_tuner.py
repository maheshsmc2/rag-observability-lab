# alpha_tuner.py

import numpy as np
from retriever import hybrid_search, DEFAULT_ALPHA

# ------------------------------
# 1. Tuning set (same as now)
# ------------------------------
TUNING_SET = [
    ("probation rules", [3]),
    ("medical certificate", [1, 2]),
    ("paid leave steps", [0]),
    ("doctor note required for sick leave", [2]),
    ("leave validity days", [0]),
    ("medical documentation required", [2]),
    ("probation policy", [3]),
]


# ------------------------------
# 2. Helper: reciprocal rank
# ------------------------------
def reciprocal_rank(retrieved_ids, relevant_ids):
    """
    Return 1/rank of the FIRST relevant doc in retrieved_ids.
    If none found, return 0.0
    """
    for idx, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (idx + 1)  # ranks are 1-based
    return 0.0


# ------------------------------
# 3. Evaluate alpha (hit-rate + MRR)
# ------------------------------
def evaluate_alpha(alpha: float, top_k: int = 5):
    """
    Returns:
      hit_rate:   total_hits / total_relevant_docs
      mrr:        mean reciprocal rank across queries
    """
    total_hits = 0
    total_relevant = 0
    rr_sum = 0.0

    for query, relevant_ids in TUNING_SET:
        results = hybrid_search(query, top_k=top_k, alpha=alpha)
        retrieved_ids = [doc["id"] for doc in results]

        # hit-rate part
        hits = sum(1 for rid in relevant_ids if rid in retrieved_ids)
        total_hits += hits
        total_relevant += len(relevant_ids)

        # MRR part
        rr = reciprocal_rank(retrieved_ids, relevant_ids)
        rr_sum += rr

    hit_rate = total_hits / total_relevant if total_relevant > 0 else 0.0
    mrr = rr_sum / len(TUNING_SET) if TUNING_SET else 0.0
    return hit_rate, mrr


# ------------------------------
# 4. Sweep alpha from 0 → 1
# ------------------------------
def sweep_alpha():
    alphas = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ..., 1.0
    results = {}

    for a in alphas:
        hit_rate, mrr = evaluate_alpha(a)
        results[round(a, 2)] = (round(hit_rate, 3), round(mrr, 3))

    return results


if __name__ == "__main__":
    print("\n🔍 Alpha Sweep Results (using hybrid_search):")
    results = sweep_alpha()
    for a, (hit_rate, mrr) in results.items():
        print(f"α = {a:>4}  →  hit = {hit_rate:.3f}   MRR = {mrr:.3f}")
