"""Conflict-triggered reranker."""

from __future__ import annotations

from dataclasses import dataclass

from src.agents.base_agent import BaseAgent
from src.memory.common import cosine_similarity, text_to_embedding
from src.memory.evidence import ConflictSignal, Evidence


@dataclass
class EscalationOutput:
    reranked_list: list[str]
    reasoning: str
    evidence_ids_used: list[str]
    long_term_preference_used: bool
    short_term_preference_used: bool
    popularity_overridden: bool


class EscalationAgent(BaseAgent):
    """Single-pass reranking module."""

    def build_prompt(self, context: dict) -> str:
        return "Re-rank these candidates prioritizing authentic long-term preferences over recent trends and popularity."

    def rerank(
        self,
        user_id: str,
        candidate_list: list[tuple[str, float]],
        conflict_signal: ConflictSignal,
        domain_evidence: list[Evidence],
        group_evidence: list[Evidence],
        long_term_centroid: list[float],
        item_texts: dict[str, str],
    ) -> EscalationOutput:
        """Return a reranked list and trace data."""
        rescored = []
        for item_id, score in candidate_list:
            item_embedding = text_to_embedding(item_texts[item_id])
            bonus = cosine_similarity(item_embedding, long_term_centroid)
            rescored.append((item_id, score + 0.2 * bonus))
        rescored.sort(key=lambda pair: pair[1], reverse=True)
        evidence_ids = [ev.evidence_id for ev in domain_evidence[:2] + group_evidence[:2]]
        return EscalationOutput(
            reranked_list=[item_id for item_id, _ in rescored],
            reasoning="Given temporal preference conflict, prioritizing long-term stable preferences over current popularity trend.",
            evidence_ids_used=evidence_ids,
            long_term_preference_used=True,
            short_term_preference_used=bool(conflict_signal.short_term_evidence_ids),
            popularity_overridden=True,
        )
