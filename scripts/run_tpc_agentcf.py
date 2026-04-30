"""Run the full TPC-AgentCF prototype."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import load_processed_dataset
from src.data.splits import build_candidate_map
from src.evaluation.report import compute_group_metrics
from src.models.popularity import compute_popularity_stats
from src.models.tpc_agentcf import compute_conflict_groups, run_recommender
from src.utils.config import load_config
from src.utils.io import write_dataframe, write_jsonl
from src.utils.logging_utils import setup_logging
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(args.config)
    set_global_seed(config["project"]["seed"])
    dataset = load_processed_dataset(ROOT / "data" / "processed")
    candidate_map = build_candidate_map(
        dataset.interactions,
        dataset.items,
        config["data"]["candidate_sample_size"],
        config["project"]["seed"],
    )
    results, recommendations, conflicts, escalation_traces = run_recommender(
        dataset.interactions, dataset.items, candidate_map, config, variant="tpc_agentcf"
    )
    popularity = compute_popularity_stats(dataset.interactions[dataset.interactions["split"] == "train"])
    for group_name, group_rows in compute_conflict_groups(results, config["conflict"]["conflict_threshold"]).items():
        metrics = compute_group_metrics(group_rows, config["evaluation"]["k_values"], popularity.raw_popularity, popularity.pop_percentile)
        write_dataframe(ROOT / "outputs" / "metrics" / group_name / "tpc_agentcf_results.csv", pd.DataFrame([metrics]))
    write_jsonl(ROOT / "outputs" / "explanations" / "recommendations.jsonl", recommendations)
    write_jsonl(ROOT / "outputs" / "conflicts" / "conflict_log.jsonl", conflicts)
    write_jsonl(ROOT / "outputs" / "explanations" / "escalation_traces.jsonl", escalation_traces)
    print(results[["user_id", "domain", "item_id", "conflict_score"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
