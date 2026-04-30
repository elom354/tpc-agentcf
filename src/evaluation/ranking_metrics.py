"""Ranking metrics."""

from __future__ import annotations

import math


def mrr_at_k(ranked_items: list[str], positive_item: str, k: int) -> float:
    """Compute MRR@K for a single list."""
    for index, item in enumerate(ranked_items[:k], start=1):
        if item == positive_item:
            return 1.0 / index
    return 0.0


def hit_rate_at_k(ranked_items: list[str], positive_item: str, k: int) -> float:
    """Compute HitRate@K."""
    return float(positive_item in ranked_items[:k])


def recall_at_k(ranked_items: list[str], positive_item: str, k: int) -> float:
    """Compute Recall@K for one positive item."""
    return hit_rate_at_k(ranked_items, positive_item, k)


def ndcg_at_k(ranked_items: list[str], positive_item: str, k: int) -> float:
    """Compute NDCG@K for one positive item."""
    for index, item in enumerate(ranked_items[:k], start=1):
        if item == positive_item:
            return 1.0 / math.log2(index + 1)
    return 0.0
