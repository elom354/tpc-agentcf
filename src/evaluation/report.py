"""Aggregate reporting helpers."""

from __future__ import annotations

import pandas as pd

from src.evaluation.conflict_metrics import summarize_conflict_metrics
from src.evaluation.diversity_metrics import average_recommendation_popularity, head_item_ratio, long_tail_coverage, novelty
from src.evaluation.ranking_metrics import hit_rate_at_k, mrr_at_k, ndcg_at_k, recall_at_k


def compute_group_metrics(group_rows: pd.DataFrame, k_values: list[int], raw_popularity: dict[str, float], pop_percentile: dict[str, float]) -> dict[str, float]:
    """Compute ranking and diversity metrics for a recommendation result slice."""
    metrics: dict[str, float] = {}
    recommendation_items = group_rows["item_id"].astype(str).tolist()
    for k in k_values:
        metrics[f"MRR@{k}"] = float(group_rows[f"mrr@{k}"].mean()) if not group_rows.empty else 0.0
        metrics[f"NDCG@{k}"] = float(group_rows[f"ndcg@{k}"].mean()) if not group_rows.empty else 0.0
        metrics[f"HitRate@{k}"] = float(group_rows[f"hit@{k}"].mean()) if not group_rows.empty else 0.0
        metrics[f"Recall@{k}"] = float(group_rows[f"recall@{k}"].mean()) if not group_rows.empty else 0.0
    metrics["ARP"] = average_recommendation_popularity(recommendation_items, raw_popularity)
    metrics["HIR"] = head_item_ratio(recommendation_items, pop_percentile)
    metrics["LTC"] = long_tail_coverage(recommendation_items, pop_percentile)
    metrics["Novelty"] = novelty(recommendation_items, raw_popularity)
    metrics["Faithfulness"] = float(group_rows["faithfulness_score"].mean()) if not group_rows.empty else 0.0
    metrics.update(summarize_conflict_metrics(group_rows))
    return metrics
