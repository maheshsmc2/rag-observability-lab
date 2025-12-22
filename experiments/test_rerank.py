from retriever import hybrid_then_rerank

q = "doctor note required for sick leave"
results = hybrid_then_rerank(q, retrieve_k=10, final_k=5)

for r in results:
    print(
        r["id"],
        round(r.get("score_rerank", 0.0), 3),
        round(r.get("score_dense", 0.0), 3),
        round(r.get("score_bm25", 0.0), 3),
    )
