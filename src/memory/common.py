"""Shared embedding and similarity helpers."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def text_to_embedding(text: str, dims: int = 16) -> list[float]:
    """Create a deterministic hashed bag-of-characters embedding."""
    vector = np.zeros(dims, dtype=float)
    for token in text.lower().split():
        vector[hash(token) % dims] += 1.0
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector.tolist()


def mean_embedding(embeddings: Iterable[list[float]]) -> list[float]:
    """Compute a mean embedding."""
    vectors = [np.array(embedding, dtype=float) for embedding in embeddings if embedding]
    if not vectors:
        return [0.0] * 16
    return np.mean(vectors, axis=0).tolist()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity between two vectors."""
    a = np.array(left, dtype=float)
    b = np.array(right, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_distance(left: list[float], right: list[float]) -> float:
    """Cosine distance between two vectors."""
    return 1.0 - cosine_similarity(left, right)


def recency_weight(now_ts: float, event_ts: float, lambda_decay: float) -> float:
    """Exponential recency weight."""
    return math.exp(-lambda_decay * max(0.0, now_ts - event_ts))
