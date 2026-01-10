# alpha_check.py

from retriever import compute_scores, _select_best_doc, DOCUMENTS

ALPHAS = [0.0, 0.2, 0.5, 0.8, 1.0]

QUERIES = [
    "paid leave policy",
    "equipment on exit",
    "remote work policy",

    # New paraphrased / vague ones:
    "how many days of paid leave do employees get every year?",
    "when do we have to return company equipment after leaving?",
    "rules for working from home and using personal equipment",
    "What benefits do workers receive for taking time away from work?",
    "When finishing employment, what items does staff need to bring back?",
    "Can I do my job from home and use my own laptop or must I get company approval?",



]


def run_alpha_sweep():
    for query in QUERIES:
        print(f"\n=== Query: {query!r} ===")
        for alpha in ALPHAS:
            scores = compute_scores(query, alpha_override=alpha)
            hybrid = scores["hybrid_scores"]
            idx = _select_best_doc(hybrid)
            text_snip = DOCUMENTS[idx][:80].replace("\n", " ")
            print(f"  alpha={alpha:0.1f} → doc_idx={idx:2d} | snippet: {text_snip!r}")

if __name__ == "__main__":
    run_alpha_sweep()
