from src.evaluation.diversity_metrics import average_recommendation_popularity, head_item_ratio
from src.evaluation.faithfulness_metrics import faithfulness_score
from src.evaluation.ranking_metrics import hit_rate_at_k, mrr_at_k, ndcg_at_k


def test_ranking_metrics_on_toy_data() -> None:
    ranked = ["a", "b", "c"]
    assert mrr_at_k(ranked, "b", 3) == 0.5
    assert hit_rate_at_k(ranked, "b", 3) == 1.0
    assert ndcg_at_k(ranked, "b", 3) > 0.0


def test_diversity_metrics_on_toy_data() -> None:
    items = ["a", "b"]
    raw_pop = {"a": 1.0, "b": 0.2}
    pct = {"a": 0.9, "b": 0.3}
    assert average_recommendation_popularity(items, raw_pop) == 0.6
    assert head_item_ratio(items, pct, 0.8) == 0.5


def test_faithfulness_score_on_simple_example() -> None:
    score = faithfulness_score("User likes comedy items.", ["comedy items", "action movies"], 0.1)
    assert score > 0.0
