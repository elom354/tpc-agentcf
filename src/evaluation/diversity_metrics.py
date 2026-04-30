"""Diversity and popularity metrics."""

from __future__ import annotations

import math


def average_recommendation_popularity(recommendation_items: list[str], raw_popularity: dict[str, float]) -> float:
    """Average popularity over recommendation items."""
    if not recommendation_items:
        return 0.0
    return sum(raw_popularity.get(item_id, 0.0) for item_id in recommendation_items) / len(recommendation_items)


def head_item_ratio(recommendation_items: list[str], pop_percentile: dict[str, float], threshold: float = 0.8) -> float:
    """Fraction of recommended items in the popularity head."""
    if not recommendation_items:
        return 0.0
    return sum(pop_percentile.get(item_id, 0.0) >= threshold for item_id in recommendation_items) / len(recommendation_items)


def long_tail_coverage(recommendation_items: list[str], pop_percentile: dict[str, float], threshold: float = 0.8) -> float:
    """Fraction of recommended items outside the popularity head."""
    if not recommendation_items:
        return 0.0
    return len({item_id for item_id in recommendation_items if pop_percentile.get(item_id, 0.0) < threshold}) / len(set(recommendation_items))


def novelty(recommendation_items: list[str], raw_popularity: dict[str, float]) -> float:
    """Mean negative log popularity."""
    if not recommendation_items:
        return 0.0
    return sum(-math.log2(max(raw_popularity.get(item_id, 1e-6), 1e-6)) for item_id in recommendation_items) / len(recommendation_items)
