import numpy as np
from retriever import dense_scores_all, DOCUMENTS

query = "movie rating"
scores = dense_scores_all(query)  # dense similarity for *all* docs

top_k = 5
indices = np.argsort(scores)[::-1][:top_k]

print(f"Query: {query!r}")
print("Top-k by dense similarity:\n")

for rank, idx in enumerate(indices, start=1):
    text = DOCUMENTS[idx].replace("\n", " ")
    snippet = text[:120] + ("..." if len(text) > 120 else "")
    print(f"#{rank}: doc_index={idx}  score={scores[idx]:.4f}")
    print(f"    {snippet}\n")
