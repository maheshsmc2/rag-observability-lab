# constants.py
# ----------------------------
# Day 74: Answer Builder Constants
# ----------------------------

# Confidence thresholds (tune later)
MIN_SCORE = 0.35
MIN_SCORE_MARGIN = 0.05
MIN_EVIDENCE_CHUNKS = 2

FALLBACK_MESSAGE = (
    "I couldn't find strong enough evidence in the documents "
    "to answer confidently."
)

HOW_TO_IMPROVE = [
    "Try adding a policy name (e.g., 'Leave Policy').",
    "Add a time constraint (e.g., 'during probation').",
    "Use keywords directly from the document wording."
]
