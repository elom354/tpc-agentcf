"""Run baseline models."""

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
from src.utils.io import write_dataframe
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
    variants = {
        "pop": "pop",
        "bpr_mf": "bpr_mf",
        "agentcf": "agentcf",
        "agentcfpp": "agentcfpp",
    }
    popularity = compute_popularity_stats(dataset.interactions[dataset.interactions["split"] == "train"])
    summary_rows = []
    for model_name, variant in variants.items():
        results, _, _, _ = run_recommender(dataset.interactions, dataset.items, candidate_map, config, variant=variant)
        groups = compute_conflict_groups(results, config["conflict"]["conflict_threshold"])
        for group_name, group_rows in groups.items():
            metrics = compute_group_metrics(
                group_rows,
                config["evaluation"]["k_values"],
                popularity.raw_popularity,
                popularity.pop_percentile,
            )
            frame = pd.DataFrame([{"model": model_name, "group": group_name, **metrics}])
            write_dataframe(ROOT / "outputs" / "metrics" / group_name / f"{model_name}_results.csv", frame)
            summary_rows.append(frame.iloc[0].to_dict())
    print(pd.DataFrame(summary_rows)[["model", "group", "MRR@10", "NDCG@10", "Conflict Detection Rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
