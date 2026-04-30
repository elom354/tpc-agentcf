"""Simplified AgentCF baseline."""

from __future__ import annotations

from src.memory.common import cosine_similarity, mean_embedding, text_to_embedding


def rank_agentcf_baseline(user_texts: list[str], candidate_texts: dict[str, str]) -> list[tuple[str, float]]:
    """Score candidates against flat user memory."""
    user_centroid = mean_embedding([text_to_embedding(text) for text in user_texts])
    scored = [
        (item_id, cosine_similarity(user_centroid, text_to_embedding(text)))
        for item_id, text in candidate_texts.items()
    ]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)
