"""End-to-end prototype runner for TPC-AgentCF and ablations."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from src.agents.escalation_agent import EscalationAgent
from src.evaluation.faithfulness_metrics import faithfulness_score
from src.evaluation.ranking_metrics import hit_rate_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from src.llm.mock_llm import MockLLM
from src.llm.openai_compatible import OpenAICompatibleLLM
from src.memory.common import mean_embedding
from src.memory.conflict_detector import ConflictDetector
from src.memory.domain_memory import DomainMemory
from src.memory.group_memory import GroupMemory
from src.memory.long_term import LongTermMemory
from src.memory.short_term import ShortTermMemory
from src.memory.evidence import RecommendationOutput
from src.models.agentcf_baseline import rank_agentcf_baseline
from src.models.agentcfpp_baseline import rank_agentcfpp_baseline
from src.models.bpr_mf import score_bpr_candidates, train_bpr_mf
from src.models.popularity import compute_popularity_stats, pop_rank_candidates, score_with_capc


def build_llm_client(config: dict[str, Any]):
    """Select the configured LLM backend."""
    if config["llm"]["backend"] == "openai":
        return OpenAICompatibleLLM()
    return MockLLM()


def initialize_memories(train_interactions: pd.DataFrame, items: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Construct memory modules and populate them from train interactions."""
    short_term = ShortTermMemory(config["memory"]["short_term_window"], config["memory"]["lambda_decay"])
    long_term = LongTermMemory(config["memory"]["long_term_min_support"], config["memory"]["sim_threshold"], config["memory"]["lambda_decay"])
    domain_memory = DomainMemory()
    group_memory = GroupMemory(config["group_memory"]["num_groups"], config["group_memory"]["group_window"])
    item_lookup = items.astype(str).set_index("item_id").to_dict("index")
    user_profiles: dict[str, list[list[float]]] = {}
    for row in train_interactions.itertuples(index=False):
        user_id = str(row.user_id)
        item_id = str(row.item_id)
        item_text = item_lookup[item_id]["description"]
        evidence = short_term.add(user_id, item_id, row.domain, item_text, float(row.timestamp))
        long_term.consolidate(user_id, row.domain, [evidence])
        domain_memory.update(user_id, item_id, row.domain, item_text, float(row.timestamp))
        user_profiles.setdefault(user_id, []).append(evidence.embedding)
    group_memory.fit_user_groups({user_id: mean_embedding(embeddings) for user_id, embeddings in user_profiles.items()})
    for row in train_interactions.itertuples(index=False):
        user_id = str(row.user_id)
        item_id = str(row.item_id)
        group_memory.update(user_id, item_id, row.domain, item_lookup[item_id]["description"], float(row.timestamp))
    return {
        "short_term": short_term,
        "long_term": long_term,
        "domain_memory": domain_memory,
        "group_memory": group_memory,
    }


def _build_user_conflict_map(interactions: pd.DataFrame, items: pd.DataFrame, config: dict[str, Any]) -> dict[tuple[str, str], Any]:
    train_interactions = interactions[interactions["split"] == "train"].copy()
    llm_client = build_llm_client(config)
    memories = initialize_memories(train_interactions, items, config)
    detector = ConflictDetector(
        llm_client,
        config["conflict"]["conflict_threshold"],
        config["conflict"]["max_distance"],
        config["conflict"]["min_short_term_size"],
    )
    conflict_map = {}
    for user_id, domain in interactions[interactions["split"] == "test"][["user_id", "domain"]].drop_duplicates().itertuples(index=False):
        conflict_map[(str(user_id), domain)] = detector.detect(str(user_id), domain, memories["short_term"], memories["long_term"])
    return conflict_map


def run_recommender(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    candidate_map: dict[tuple[str, str], list[str]],
    config: dict[str, Any],
    variant: str = "tpc_agentcf",
) -> tuple[pd.DataFrame, list[dict], list[dict], list[dict]]:
    """Run a model variant over test interactions."""
    llm_client = build_llm_client(config)
    train_interactions = interactions[interactions["split"] == "train"].copy()
    test_interactions = interactions[interactions["split"] == "test"].copy()
    item_lookup = items.astype(str).set_index("item_id").to_dict("index")
    popularity = compute_popularity_stats(train_interactions)
    memories = initialize_memories(train_interactions, items, config)
    detector = ConflictDetector(
        llm_client,
        config["conflict"]["conflict_threshold"],
        config["conflict"]["max_distance"],
        config["conflict"]["min_short_term_size"],
    )
    escalation_agent = EscalationAgent("escalation", llm_client, memories)
    bpr_model = user_to_idx = item_to_idx = None
    if variant == "bpr_mf":
        bpr_model, user_to_idx, item_to_idx = train_bpr_mf(interactions, config)

    recommendation_rows: list[dict] = []
    recommendation_jsonl: list[dict] = []
    conflict_jsonl: list[dict] = []
    escalation_jsonl: list[dict] = []

    for row in test_interactions.itertuples(index=False):
        user_id = str(row.user_id)
        positive_item = str(row.item_id)
        domain = row.domain
        candidates = candidate_map[(user_id, positive_item)]
        candidate_texts = {item_id: item_lookup[str(item_id)]["description"] for item_id in candidates}
        domain_evidence = memories["domain_memory"].get_domain_evidence(user_id, domain)
        fused_evidence = memories["domain_memory"].get_fused_evidence(user_id, domain)
        group_evidence = memories["group_memory"].get_group_evidence(user_id, domain)
        short_evidence = memories["short_term"].get_evidence(user_id, domain)
        long_evidence = memories["long_term"].get_evidence(user_id, domain)
        conflict_signal = detector.detect(user_id, domain, memories["short_term"], memories["long_term"])
        conflict_jsonl.append(asdict(conflict_signal))

        if variant == "pop":
            ranked = pop_rank_candidates(candidates, popularity)
        elif variant == "bpr_mf":
            ranked = score_bpr_candidates(bpr_model, user_to_idx, item_to_idx, user_id, candidates)
        elif variant == "agentcf":
            ranked = rank_agentcf_baseline([ev.text for ev in domain_evidence + short_evidence], candidate_texts)
        elif variant == "agentcfpp":
            ranked = rank_agentcfpp_baseline(
                [ev.text for ev in domain_evidence],
                [ev.text for ev in fused_evidence],
                [ev.text for ev in group_evidence],
                candidate_texts,
            )
        else:
            user_centroid = mean_embedding([ev.embedding for ev in domain_evidence + short_evidence + long_evidence + fused_evidence] or [[0.0] * 16])
            short_centroid = memories["short_term"].get_centroid(user_id, domain)
            long_centroid = memories["long_term"].get_centroid(user_id, domain)
            group_centroid = memories["group_memory"].get_group_centroid(user_id, domain)
            scored = []
            for item_id, item_text in candidate_texts.items():
                score, discount, percentile = score_with_capc(
                    item_id,
                    item_text,
                    user_centroid,
                    short_centroid,
                    long_centroid,
                    group_centroid,
                    popularity,
                    conflict_signal,
                    config,
                )
                scored.append((item_id, score, discount, percentile))
            scored.sort(key=lambda triple: triple[1], reverse=True)
            ranked = [(item_id, score) for item_id, score, _, _ in scored]

        top_scores = [score for _, score in ranked[:2]]
        top_item = ranked[0][0]
        top_percentile = popularity.pop_percentile.get(top_item, 0.0)
        escalation_conditions = [
            conflict_signal.is_conflict and conflict_signal.conflict_score > config["conflict"]["escalation_threshold"],
            conflict_signal.is_conflict and top_percentile > config["popularity"]["high_pop_threshold"],
            conflict_signal.is_conflict and len(top_scores) == 2 and abs(top_scores[0] - top_scores[1]) < config["conflict"]["score_margin"],
        ]
        escalate = config["escalation"]["enabled"] and (
            not config["escalation"]["trigger_only"] or any(escalation_conditions)
        )
        if variant not in {"tpc_agentcf", "a4", "a5"}:
            escalate = False
        if escalate:
            reranked = escalation_agent.rerank(
                user_id,
                ranked,
                conflict_signal,
                domain_evidence,
                group_evidence,
                memories["long_term"].get_centroid(user_id, domain),
                candidate_texts,
            )
            ranked = [(item_id, len(reranked.reranked_list) - idx) for idx, item_id in enumerate(reranked.reranked_list)]
            escalation_jsonl.append({"user_id": user_id, "domain": domain, **asdict(reranked)})

        ranked_items = [item_id for item_id, _ in ranked]
        explanation = (
            conflict_signal.conflict_explanation
            or f"Recommendation grounded in domain evidence for {domain}."
        )
        evidence_texts = [ev.text for ev in short_evidence + long_evidence + domain_evidence + group_evidence]
        faith = faithfulness_score(explanation, evidence_texts, config["evaluation"]["faithfulness_sim_threshold"])
        best_rank = ranked_items.index(positive_item) + 1 if positive_item in ranked_items else len(ranked_items) + 1
        discount = conflict_signal.conflict_score * config["popularity"]["conflict_discount_max"] if conflict_signal.is_conflict else 0.0
        output = RecommendationOutput(
            user_id=user_id,
            item_id=ranked_items[0],
            rank=1,
            score=float(ranked[0][1]),
            domain=domain,
            explanation=explanation,
            evidence_ids=[ev.evidence_id for ev in short_evidence[:2] + long_evidence[:2] + group_evidence[:2]],
            evidence_types_used=[ev.evidence_type for ev in short_evidence[:1] + long_evidence[:1] + group_evidence[:1]],
            conflict_detected=conflict_signal.is_conflict,
            conflict_score=conflict_signal.conflict_score,
            conflict_discount_applied=discount,
            popularity_percentile=top_percentile,
            popularity_overridden=escalate,
            escalation_triggered=escalate,
            faithfulness_score=faith,
            metadata={"positive_item": positive_item, "rank_of_positive": best_rank},
        )
        recommendation_jsonl.append(asdict(output))
        metrics_row = {
            "user_id": user_id,
            "domain": domain,
            "item_id": ranked_items[0],
            "positive_item": positive_item,
            "conflict_detected": conflict_signal.is_conflict,
            "conflict_score": conflict_signal.conflict_score,
            "popularity_percentile": top_percentile,
            "escalation_triggered": escalate,
            "faithfulness_score": faith,
        }
        for k in config["evaluation"]["k_values"]:
            metrics_row[f"mrr@{k}"] = mrr_at_k(ranked_items, positive_item, k)
            metrics_row[f"ndcg@{k}"] = ndcg_at_k(ranked_items, positive_item, k)
            metrics_row[f"hit@{k}"] = hit_rate_at_k(ranked_items, positive_item, k)
            metrics_row[f"recall@{k}"] = recall_at_k(ranked_items, positive_item, k)
        recommendation_rows.append(metrics_row)
    return pd.DataFrame(recommendation_rows), recommendation_jsonl, conflict_jsonl, escalation_jsonl


def compute_conflict_groups(results: pd.DataFrame, threshold: float = 0.35) -> dict[str, pd.DataFrame]:
    """Split recommendation rows by conflict status."""
    return {
        "all_users": results,
        "no_conflict_users": results[results["conflict_score"] < threshold].copy(),
        "conflict_users": results[results["conflict_score"] >= threshold].copy(),
        "high_conflict_users": results[results["conflict_score"] >= 0.5].copy(),
    }
