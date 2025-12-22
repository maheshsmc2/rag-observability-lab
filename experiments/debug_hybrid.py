# debug_hybrid.py

from retriever import dense_search, bm25_search, hybrid_search
from textwrap import shorten


def _score_of(r: dict):
    """
    Be flexible about how scores are stored.
    Tries several common keys and returns None if not found.
    """
    for key in ("score", "score_dense", "score_bm25", "score_hybrid"):
        if key in r:
            return r[key]
    return None


def debug_query(query: str, k: int = 5) -> None:
    print("\n" + "=" * 60)
    print(f"QUERY: {query!r}")
    print("=" * 60)

    dense = dense_search(query, top_k=k)
    bm25 = bm25_search(query, top_k=k)
    hybrid = hybrid_search(query, top_k=k)

    header = (
        f"{'rank':<4} | "
        f"{'dense_id':<8} {'d_score':>8} | "
        f"{'bm25_id':<8} {'b_score':>8} | "
        f"{'hyb_id':<8} {'h_score':>8}"
    )
    print(header)
    print("-" * len(header))

    for i in range(k):
        d = dense[i] if i < len(dense) else {}
        b = bm25[i] if i < len(bm25) else {}
        h = hybrid[i] if i < len(hybrid) else {}

        row = (
            f"{i+1:<4} | "
            f"{str(d.get('id','-')):<8} {str(_score_of(d))[:8]:>8} | "
            f"{str(b.get('id','-')):<8} {str(_score_of(b))[:8]:>8} | "
            f"{str(h.get('id','-')):<8} {str(_score_of(h))[:8]:>8}"
        )
        print(row)

    if hybrid:
        print("\nTop hybrid chunk snippet:")
        print(shorten(hybrid[0]["text"].replace("\n", " "), width=160, placeholder=" ..."))
    print()


if __name__ == "__main__":
    # 👇 Edit this list to try different queries
    queries = [
        "leave",
        "reimbursement",
        "how many paid leave days",
        "how to claim travel expenses",
    ]

    for q in queries:
        debug_query(q, k=5)
