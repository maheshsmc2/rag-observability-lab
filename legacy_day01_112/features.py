# features.py
# ------------------------------------------------------------
# "Time Machine" switchboard for Day 65 -> Day 90
# You can run the SAME code in different historical modes.
# ------------------------------------------------------------

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureFlags:
    # Day 65: routing + retrieval only (foundation)
    enable_gates: bool
    enable_answer_builder: bool
    enable_failure_taxonomy: bool
    enable_decision_card: bool
    enable_offtopic_reclassify: bool


# Choose a mode and everything else follows.
MODES = {
    # Day 65: foundation (router + retrieval, no gates/envelope extras)
    "day65": FeatureFlags(
        enable_gates=False,
        enable_answer_builder=False,
        enable_failure_taxonomy=False,
        enable_decision_card=False,
        enable_offtopic_reclassify=False,
    ),

    # Day 74: stable envelope + dominance gating + deterministic answer builder
    "day74": FeatureFlags(
        enable_gates=True,
        enable_answer_builder=True,
        enable_failure_taxonomy=False,
        enable_decision_card=False,
        enable_offtopic_reclassify=False,
    ),

    # Day 83/84+: add failure taxonomy + decision card + off-topic reclassify
    "day90": FeatureFlags(
        enable_gates=True,
        enable_answer_builder=True,
        enable_failure_taxonomy=True,
        enable_decision_card=True,
        enable_offtopic_reclassify=True,
    ),
}


def get_flags(mode: str) -> FeatureFlags:
    mode = (mode or "day90").strip().lower()
    if mode not in MODES:
        raise ValueError(f"Unknown MODE='{mode}'. Use one of: {list(MODES.keys())}")
    return MODES[mode]
