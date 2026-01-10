# eval/failure_buckets.py

# Buckets (small, stable set)
RETRIEVAL_MISS = "RETRIEVAL_MISS"              # expected evidence not in top-k
GATE_LOW_GAP = "GATE_LOW_GAP"                  # ambiguous: gap too small
GATE_LOW_SCORE = "GATE_LOW_SCORE"              # low confidence score
SEMANTIC_ABSENCE = "SEMANTIC_ABSENCE"          # unanswerable correctly rejected by semantic absence
UNANSWERABLE_FALSE_PASS = "UNANSWERABLE_FALSE_PASS"  # unanswerable but gate passed
ANSWERABLE_FALSE_ABSTAIN = "ANSWERABLE_FALSE_ABSTAIN" # answerable but gate failed
OK = "OK"                                      # behaved as expected

ALL_BUCKETS = [
    RETRIEVAL_MISS,
    GATE_LOW_GAP,
    GATE_LOW_SCORE,
    SEMANTIC_ABSENCE,
    UNANSWERABLE_FALSE_PASS,
    ANSWERABLE_FALSE_ABSTAIN,
    OK,
]
