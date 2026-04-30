"""Candidate sampling utilities."""

from __future__ import annotations

import random
from typing import Any

import pandas as pd


def build_candidate_map(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    candidate_sample_size: int,
    seed: int,
) -> dict[tuple[str, str], list[str]]:
    """Build per-test-interaction candidate sets with one positive and random negatives."""
    rng = random.Random(seed)
    item_pool = items["item_id"].astype(str).tolist()
    interacted = interactions.groupby("user_id")["item_id"].apply(lambda s: set(map(str, s))).to_dict()
    candidate_map: dict[tuple[str, str], list[str]] = {}
    test_rows = interactions[interactions["split"] == "test"]
    for row in test_rows.itertuples(index=False):
        user_id = str(row.user_id)
        pos_item = str(row.item_id)
        excluded = interacted.get(row.user_id, set())
        negatives = [item for item in item_pool if item not in excluded and item != pos_item]
        sample_size = min(candidate_sample_size - 1, len(negatives))
        sampled = rng.sample(negatives, sample_size) if sample_size > 0 else []
        candidates = [pos_item] + sampled
        rng.shuffle(candidates)
        candidate_map[(user_id, pos_item)] = candidates
    return candidate_map
