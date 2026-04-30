"""Conflict-related metrics."""

from __future__ import annotations

import pandas as pd


def summarize_conflict_metrics(results: pd.DataFrame) -> dict[str, float]:
    """Aggregate conflict-specific metrics from recommendation rows."""
    if results.empty:
        return {
            "Conflict Detection Rate": 0.0,
            "Mean Conflict Score": 0.0,
            "ARP_conflict": 0.0,
            "ARP_no_conflict": 0.0,
            "Escalation Trigger Rate": 0.0,
            "MRR@10_escalated": 0.0,
            "MRR@10_not_escalated": 0.0,
        }
    conflict_mask = results["conflict_detected"].astype(bool)
    escalated_mask = results["escalation_triggered"].astype(bool)
    return {
        "Conflict Detection Rate": float(conflict_mask.mean()),
        "Mean Conflict Score": float(results["conflict_score"].mean()),
        "ARP_conflict": float(results.loc[conflict_mask, "popularity_percentile"].mean()) if conflict_mask.any() else 0.0,
        "ARP_no_conflict": float(results.loc[~conflict_mask, "popularity_percentile"].mean()) if (~conflict_mask).any() else 0.0,
        "Escalation Trigger Rate": float(escalated_mask.mean()),
        "MRR@10_escalated": float(results.loc[escalated_mask, "mrr@10"].mean()) if escalated_mask.any() else 0.0,
        "MRR@10_not_escalated": float(results.loc[~escalated_mask, "mrr@10"].mean()) if (~escalated_mask).any() else 0.0,
    }
