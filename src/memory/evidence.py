"""Evidence dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Evidence:
    evidence_id: str
    user_id: str
    item_id: str
    domain: str
    timestamp: float
    evidence_type: Literal["short_term", "long_term", "domain", "group"]
    text: str
    embedding: list[float]
    strength: float
    recency_weight: float
    source: Literal["interaction", "reflection", "group_memory", "consolidation"]
    metadata: dict = field(default_factory=dict)


@dataclass
class ConflictSignal:
    user_id: str
    domain: str
    is_conflict: bool
    conflict_score: float
    centroid_distance: float
    short_term_summary: str
    long_term_summary: str
    conflict_explanation: str
    short_term_evidence_ids: list[str]
    long_term_evidence_ids: list[str]
    timestamp: float


@dataclass
class RecommendationOutput:
    user_id: str
    item_id: str
    rank: int
    score: float
    domain: str
    explanation: str
    evidence_ids: list[str]
    evidence_types_used: list[str]
    conflict_detected: bool
    conflict_score: float
    conflict_discount_applied: float
    popularity_percentile: float
    popularity_overridden: bool
    escalation_triggered: bool
    faithfulness_score: float
    metadata: dict = field(default_factory=dict)
