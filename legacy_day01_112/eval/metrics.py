from __future__ import annotations
from typing import List, Sequence, Optional


def hit_at_k(retrieved_ids: Sequence[str], expected_ids: Sequence[str], k: int) -> int:
    top = list(retrieved_ids)[:k]
    return int(any(e in top for e in expected_ids))


def recall_at_k(retrieved_ids: Sequence[str], expected_ids: Sequence[str], k: int) -> float:
    if not expected_ids:
        return 0.0
    top = set(list(retrieved_ids)[:k])
    hits = sum(1 for e in expected_ids if e in top)
    return hits / float(len(expected_ids))


def mean_rank(retrieved_ids: Sequence[str], expected_ids: Sequence[str]) -> Optional[float]:
    """
    Average 1-based rank of expected docs that appear in retrieved list.
    Returns None if none found.
    """
    ranks: List[int] = []
    r = list(retrieved_ids)
    for e in expected_ids:
        if e in r:
            ranks.append(r.index(e) + 1)
    if not ranks:
        return None
    return sum(ranks) / float(len(ranks))
