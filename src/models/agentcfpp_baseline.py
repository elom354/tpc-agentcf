"""Simplified AgentCF++ baseline."""

from __future__ import annotations

from src.memory.common import cosine_similarity, mean_embedding, text_to_embedding


def rank_agentcfpp_baseline(domain_texts: list[str], fused_texts: list[str], group_texts: list[str], candidate_texts: dict[str, str]) -> list[tuple[str, float]]:
    """Score candidates with domain, fused, and group evidence."""
    domain_centroid = mean_embedding([text_to_embedding(text) for text in domain_texts])
    fused_centroid = mean_embedding([text_to_embedding(text) for text in fused_texts]) if fused_texts else domain_centroid
    group_centroid = mean_embedding([text_to_embedding(text) for text in group_texts]) if group_texts else domain_centroid
    user_centroid = mean_embedding([domain_centroid, fused_centroid, group_centroid])
    scored = []
    for item_id, text in candidate_texts.items():
        item_embedding = text_to_embedding(text)
        score = (
            0.6 * cosine_similarity(user_centroid, item_embedding)
            + 0.2 * cosine_similarity(group_centroid, item_embedding)
            + 0.2 * cosine_similarity(fused_centroid, item_embedding)
        )
        scored.append((item_id, score))
    return sorted(scored, key=lambda pair: pair[1], reverse=True)
