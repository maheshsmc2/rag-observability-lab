from retriever import retrieve_top_k

query = "movie rating"   # your question
res = retrieve_top_k(query, k=5)

print(res)
