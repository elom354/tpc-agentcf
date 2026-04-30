"""Run ablation variants."""

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
    variants = ["a0", "a1", "a2", "a3", "a4", "a5", "s1", "s2", "s3", "s4"]
    variant_map = {
        "a0": "agentcfpp",
        "a1": "agentcfpp",
        "a2": "tpc_agentcf",
        "a3": "tpc_agentcf",
        "a4": "tpc_agentcf",
        "a5": "tpc_agentcf",
        "s1": "tpc_agentcf",
        "s2": "tpc_agentcf",
        "s3": "tpc_agentcf",
        "s4": "tpc_agentcf",
    }
    popularity = compute_popularity_stats(dataset.interactions[dataset.interactions["split"] == "train"])
    combined_rows = []
    for variant in variants:
        local_config = load_config(args.config)
        if variant == "s1":
            local_config["popularity"]["conflict_discount_max"] = 0.0
        elif variant == "s2":
            local_config["popularity"]["conflict_discount_max"] = 1.0
        elif variant == "s3":
            local_config["escalation"]["trigger_only"] = False
        elif variant == "s4":
            local_config["escalation"]["enabled"] = False
        results, _, _, _ = run_recommender(dataset.interactions, dataset.items, candidate_map, local_config, variant=variant_map[variant])
        rows = []
        for group_name, group_rows in compute_conflict_groups(results, local_config["conflict"]["conflict_threshold"]).items():
            metrics = compute_group_metrics(group_rows, local_config["evaluation"]["k_values"], popularity.raw_popularity, popularity.pop_percentile)
            rows.append({"variant": variant, "group": group_name, **metrics})
        frame = pd.DataFrame(rows)
        write_dataframe(ROOT / "outputs" / "ablations" / f"{variant}_results.csv", frame)
        combined_rows.extend(rows)
    write_dataframe(ROOT / "outputs" / "ablations" / "ablation_results.csv", pd.DataFrame(combined_rows))
    print(pd.DataFrame(combined_rows)[["variant", "group", "MRR@10"]].to_string(index=False))


if __name__ == "__main__":
    main()
