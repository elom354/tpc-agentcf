"""Faithfulness metrics."""

from __future__ import annotations

import re

from src.memory.common import cosine_similarity, text_to_embedding


def faithfulness_score(explanation: str, evidence_texts: list[str], threshold: float = 0.35) -> float:
    """Sentence-level support ratio against evidence texts."""
    sentences = [segment.strip() for segment in re.split(r"[.!?]+", explanation) if segment.strip()]
    if not sentences:
        return 0.0
    evidence_embeddings = [text_to_embedding(text) for text in evidence_texts]
    supported = 0
    for sentence in sentences:
        sentence_embedding = text_to_embedding(sentence)
        max_similarity = max((cosine_similarity(sentence_embedding, emb) for emb in evidence_embeddings), default=0.0)
        supported += int(max_similarity > threshold)
    return supported / len(sentences)
