"""Popularity scoring and calibration."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.memory.common import cosine_similarity, text_to_embedding


@dataclass
class PopularityStats:
    raw_popularity: dict[str, float]
    pop_percentile: dict[str, float]


def compute_popularity_stats(train_interactions: pd.DataFrame) -> PopularityStats:
    """Compute raw popularity and percentiles from training interactions."""
    counts = train_interactions["item_id"].astype(str).value_counts()
    max_count = max(float(counts.max()), 1.0) if not counts.empty else 1.0
    raw = {item_id: count / max_count for item_id, count in counts.items()}
    ranks = counts.rank(pct=True, method="average") if not counts.empty else pd.Series(dtype=float)
    percentile = {str(item_id): float(ranks.loc[item_id]) for item_id in counts.index}
    return PopularityStats(raw_popularity=raw, pop_percentile=percentile)


def pop_rank_candidates(candidates: list[str], popularity_stats: PopularityStats) -> list[tuple[str, float]]:
    """Rank candidates by global popularity only."""
    scored = [(item_id, popularity_stats.raw_popularity.get(item_id, 0.0)) for item_id in candidates]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)


def score_with_capc(
    item_id: str,
    item_text: str,
    user_centroid: list[float],
    short_centroid: list[float],
    long_centroid: list[float],
    group_centroid: list[float],
    popularity_stats: PopularityStats,
    conflict_signal,
    config: dict,
) -> tuple[float, float, float]:
    """Compute conflict-aware popularity-calibrated score."""
    item_embedding = text_to_embedding(item_text)
    base_score = cosine_similarity(user_centroid, item_embedding)
    temporal_score = 0.5 * cosine_similarity(short_centroid, item_embedding) + 0.5 * cosine_similarity(long_centroid, item_embedding)
    group_score = cosine_similarity(group_centroid, item_embedding)
    alignment = cosine_similarity(group_centroid, long_centroid)
    raw_pop = popularity_stats.raw_popularity.get(item_id, 0.0)
    useful_pop = raw_pop if alignment > config["popularity"]["alignment_threshold"] else raw_pop * max(alignment, 0.0)
    percentile = popularity_stats.pop_percentile.get(item_id, 0.0)
    penalty = max(0.0, percentile - config["popularity"]["head_percentile"])
    discount = conflict_signal.conflict_score * config["popularity"]["conflict_discount_max"] if conflict_signal.is_conflict else 0.0
    score = (
        base_score
        + config["scoring"]["alpha_temporal"] * temporal_score
        + config["scoring"]["alpha_group"] * group_score * (1.0 - discount)
        + config["scoring"]["alpha_pop"] * useful_pop * (1.0 - discount)
        - config["popularity"]["beta_pop_penalty"] * penalty * float(conflict_signal.is_conflict)
    )
    return score, discount, percentile
