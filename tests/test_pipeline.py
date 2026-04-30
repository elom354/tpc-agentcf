from pathlib import Path

import pandas as pd

from src.data.splits import build_candidate_map
from src.models.tpc_agentcf import compute_conflict_groups, run_recommender
from src.utils.io import write_dataframe, write_jsonl


def test_end_to_end_pipeline_outputs(tmp_path: Path) -> None:
    interactions = []
    items = []
    timestamp = 1000
    for item_id in range(1, 21):
        items.append({"item_id": str(item_id), "title": f"Item {item_id}", "domain": "d1" if item_id % 2 else "d2", "description": f"item {item_id} domain"})
    for user_id in range(1, 6):
        for idx in range(12):
            interactions.append(
                {
                    "user_id": str(user_id),
                    "item_id": str((user_id + idx) % 20 + 1),
                    "rating": 5.0,
                    "timestamp": timestamp + user_id * 100 + idx,
                    "domain": "d1" if idx % 2 else "d2",
                    "split": "train" if idx < 8 else ("validation" if idx < 10 else "test"),
                }
            )
    interaction_df = pd.DataFrame(interactions)
    item_df = pd.DataFrame(items)
    config = {
        "project": {"seed": 42},
        "memory": {"short_term_window": 15, "lambda_decay": 0.05, "long_term_min_support": 2, "sim_threshold": 0.1},
        "group_memory": {"num_groups": 2, "group_window": 50},
        "conflict": {"conflict_threshold": 0.35, "max_distance": 1.0, "min_short_term_size": 1, "escalation_threshold": 0.5, "score_margin": 0.05},
        "popularity": {"alignment_threshold": 0.4, "conflict_discount_max": 0.6, "head_percentile": 0.8, "beta_pop_penalty": 0.2, "high_pop_threshold": 0.9},
        "scoring": {"alpha_temporal": 0.3, "alpha_group": 0.2, "alpha_pop": 0.15},
        "evaluation": {"k_values": [5, 10], "faithfulness_sim_threshold": 0.1},
        "escalation": {"enabled": True, "trigger_only": True},
        "llm": {"backend": "mock"},
    }
    candidate_map = build_candidate_map(interaction_df, item_df, 10, 42)
    results, recommendations, conflicts, traces = run_recommender(interaction_df, item_df, candidate_map, config, variant="tpc_agentcf")
    assert not results.empty
    groups = compute_conflict_groups(results, 0.35)
    for group_name, frame in groups.items():
        path = tmp_path / f"{group_name}.csv"
        write_dataframe(path, frame)
        assert path.exists()
    write_jsonl(tmp_path / "recommendations.jsonl", recommendations)
    write_jsonl(tmp_path / "conflicts.jsonl", conflicts)
    assert (tmp_path / "recommendations.jsonl").exists()
    assert (tmp_path / "conflicts.jsonl").exists()
