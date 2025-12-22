from retriever import hybrid_search, hybrid_then_rerank, DEFAULT_ALPHA

QUERIES = [
    "probation rules",
    "doctor note required for sick leave",
    "leave validity days",
]

def print_results(tag, docs):
    print(f"\n[{tag}]")
    for i, d in enumerate(docs, start=1):
        line = f"{i}. id={d['id']}"
        line += f" | dense={d.get('score_dense'):.3f}"
        line += f" | bm25={d.get('score_bm25'):.3f}"
        if 'score_hybrid' in d:
            line += f" | hybrid={d['score_hybrid']:.3f}"
        if 'score_rerank' in d:
            line += f" | rerank={d['score_rerank']:.3f}"
        print(line)

if __name__ == "__main__":
    for q in QUERIES:
        print("\n" + "=" * 60)
        print("QUERY:", q)
        print("alpha =", DEFAULT_ALPHA)

        base = hybrid_search(q, top_k=5, alpha=DEFAULT_ALPHA)
        rer  = hybrid_then_rerank(q, retrieve_k=10, final_k=5, alpha=DEFAULT_ALPHA)

        print_results("Hybrid only", base)
        print_results("Hybrid + Rerank", rer)
