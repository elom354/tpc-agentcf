"""Long-term memory."""

from __future__ import annotations

from collections import defaultdict

from src.memory.common import cosine_similarity, mean_embedding
from src.memory.evidence import Evidence


class LongTermMemory:
    """Store consolidated stable preferences by domain."""

    def __init__(self, min_support: int = 3, sim_threshold: float = 0.35, lambda_decay: float = 0.05) -> None:
        self.min_support = min_support
        self.sim_threshold = sim_threshold
        self.lambda_decay = lambda_decay
        self.store: dict[str, dict[str, list[Evidence]]] = defaultdict(lambda: defaultdict(list))
        self.support: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))

    def consolidate(self, user_id: str, domain: str, short_term_evidence: list[Evidence]) -> None:
        """Promote repeated short-term patterns into long-term memory."""
        for evidence in short_term_evidence:
            matched_text = None
            for existing_text in self.support[user_id][domain]:
                existing = next(ev for ev in self.store[user_id][domain] if ev.text == existing_text)
                if cosine_similarity(evidence.embedding, existing.embedding) > self.sim_threshold:
                    matched_text = existing_text
                    break
            target_text = matched_text or evidence.text
            current_support = self.support[user_id][domain].get(target_text, 0) + 1
            self.support[user_id][domain][target_text] = current_support
            if current_support >= self.min_support and not any(ev.text == target_text for ev in self.store[user_id][domain]):
                self.store[user_id][domain].append(
                    Evidence(
                        evidence_id=f"lt-{user_id}-{domain}-{len(self.store[user_id][domain])}",
                        user_id=user_id,
                        item_id=evidence.item_id,
                        domain=domain,
                        timestamp=evidence.timestamp,
                        evidence_type="long_term",
                        text=target_text,
                        embedding=evidence.embedding,
                        strength=float(current_support),
                        recency_weight=evidence.recency_weight,
                        source="consolidation",
                    )
                )

    def get_evidence(self, user_id: str, domain: str) -> list[Evidence]:
        """Return consolidated evidence."""
        return list(self.store.get(user_id, {}).get(domain, []))

    def get_centroid(self, user_id: str, domain: str) -> list[float]:
        """Return the mean long-term embedding."""
        return mean_embedding([ev.embedding for ev in self.get_evidence(user_id, domain)])
