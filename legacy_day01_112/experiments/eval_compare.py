# eval_compare.py
# -----------------------------------------------
# Mini evaluation of retrieval methods:
# Dense vs BM25 vs Hybrid vs Hybrid+Rerank
#
# Mentor Mode Day 58
# -----------------------------------------------

from retriever import (
    dense_search,
    bm25_search,
    hybrid_search,
    hybrid_then_rerank,
)

TEST_QUERIES = [
    "WFH rules",
    "medical certificate requirement",
    "paid leave policy",
    "probation period",
]

def print_method_header(title):
    print("\n" + "=" * 60)
    print(f"🔹 {title}")
    print("=" * 60)

def evaluate_query(query):
    print("\n" + "#" * 80)
    print(f"QUERY: {query}")
    print("#" * 80)

    # ----- Dense -----
    print_method_header("Dense Retrieval (semantic only)")
    d = dense_search(query, top_k=1)
    print("Top-1:", d[0]["text"])
    print("Score:", d[0]["score_dense"])

    # ----- BM25 -----
    print_method_header("BM25 Retrieval (keyword only)")
    b = bm25_search(query, top_k=1)
    print("Top-1:", b[0]["text"])
    print("Score:", b[0]["score_bm25"])

    # ----- Hybrid -----
    print_method_header("Hybrid Retrieval (dense + lexical)")
    h = hybrid_search(query, top_k=1)
    print("Top-1:", h[0]["text"])
    print("Hybrid score:", h[0]["score_hybrid"])
    print("Dense score:", h[0]["score_dense"])
    print("BM25 score:", h[0]["score_bm25"])

    # ----- Hybrid + Rerank -----
    print_method_header("Hybrid + CrossEncoder Rerank (final answer)")
    r = hybrid_then_rerank(query, retrieve_k=10, final_k=1)
    print("Top-1:", r[0]["text"])
    print("Rerank score:", r[0]["score_rerank"])

def main():
    for q in TEST_QUERIES:
        evaluate_query(q)

if __name__ == "__main__":
    main()
