# run_query.py

from retriever import (
    dense_search,
    lexical_search,
    hybrid_search,
)
from reranker import rerank


def run_query(
    query: str,
    top_k: int = 5,
    alpha: float = 0.2,
    use_reranker: bool = True,
):
    dense = dense_search(query, top_k=20)
    lexical = lexical_search(query, top_k=20)

    hybrid = hybrid_search(
        dense,
        lexical,
        alpha=alpha,
        top_k=top_k * 2,
    )

    if use_reranker:
        hybrid = rerank(query, hybrid)

    return hybrid[:top_k]
