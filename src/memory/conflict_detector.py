"""Temporal preference conflict detector."""

from __future__ import annotations

import time

from src.llm.llm_client import LLMClient
from src.memory.common import cosine_distance
from src.memory.evidence import ConflictSignal
from src.memory.long_term import LongTermMemory
from src.memory.short_term import ShortTermMemory


class ConflictDetector:
    """Detect conflict between short-term and long-term user preferences."""

    def __init__(
        self,
        llm_client: LLMClient,
        conflict_threshold: float = 0.35,
        max_distance: float = 1.0,
        min_short_term_size: int = 3,
    ) -> None:
        self.llm_client = llm_client
        self.conflict_threshold = conflict_threshold
        self.max_distance = max_distance
        self.min_short_term_size = min_short_term_size

    def detect(self, user_id: str, domain: str, short_term_memory: ShortTermMemory, long_term_memory: LongTermMemory) -> ConflictSignal:
        """Return a conflict signal for one user-domain batch."""
        short_term = short_term_memory.get_evidence(user_id, domain)
        long_term = long_term_memory.get_evidence(user_id, domain)
        if len(short_term) < self.min_short_term_size or not long_term:
            return ConflictSignal(
                user_id=user_id,
                domain=domain,
                is_conflict=False,
                conflict_score=0.0,
                centroid_distance=0.0,
                short_term_summary="Insufficient short-term evidence.",
                long_term_summary="Insufficient long-term evidence." if not long_term else "Stable long-term preferences available.",
                conflict_explanation="",
                short_term_evidence_ids=[ev.evidence_id for ev in short_term],
                long_term_evidence_ids=[ev.evidence_id for ev in long_term],
                timestamp=time.time(),
            )
        st_centroid = short_term_memory.get_centroid(user_id, domain)
        lt_centroid = long_term_memory.get_centroid(user_id, domain)
        distance = cosine_distance(st_centroid, lt_centroid)
        is_conflict = distance > self.conflict_threshold
        conflict_score = min(1.0, distance / self.max_distance)
        short_term_summary = self.llm_client.generate("short-term summary")
        long_term_summary = self.llm_client.generate("long-term summary")
        explanation = self.llm_client.generate("conflict explanation") if is_conflict else ""
        return ConflictSignal(
            user_id=user_id,
            domain=domain,
            is_conflict=is_conflict,
            conflict_score=conflict_score,
            centroid_distance=distance,
            short_term_summary=short_term_summary,
            long_term_summary=long_term_summary,
            conflict_explanation=explanation,
            short_term_evidence_ids=[ev.evidence_id for ev in short_term],
            long_term_evidence_ids=[ev.evidence_id for ev in long_term],
            timestamp=time.time(),
        )
