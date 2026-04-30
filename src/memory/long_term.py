"""Long-term memory."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

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
        self.prototypes: dict[str, dict[str, dict[str, Evidence]]] = defaultdict(lambda: defaultdict(dict))

    def consolidate(self, user_id: str, domain: str, short_term_evidence: list[Evidence]) -> None:
        """Promote repeated short-term patterns into long-term memory."""
        for evidence in short_term_evidence:
            matched_key = None
            prototype_by_key = self.prototypes[user_id][domain]
            stored_by_text = {ev.text: ev for ev in self.store[user_id][domain]}
            for existing_key, prototype in prototype_by_key.items():
                if cosine_similarity(evidence.embedding, prototype.embedding) > self.sim_threshold:
                    matched_key = existing_key
                    break
            target_key = matched_key or f"proto-{domain}-{len(prototype_by_key)}"
            current_support = self.support[user_id][domain].get(target_key, 0) + 1
            self.support[user_id][domain][target_key] = current_support
            prototype = prototype_by_key.get(target_key)
            if prototype is None:
                prototype_by_key[target_key] = replace(
                    evidence,
                    evidence_id=f"proto-{user_id}-{domain}-{len(prototype_by_key)}",
                    evidence_type="long_term",
                    source="consolidation",
                    strength=float(current_support),
                )
            else:
                updated_embedding = [
                    (old_value * (current_support - 1) + new_value) / current_support
                    for old_value, new_value in zip(prototype.embedding, evidence.embedding, strict=True)
                ]
                prototype.embedding = updated_embedding
                prototype.strength = float(current_support)
                prototype.timestamp = evidence.timestamp
                prototype.text = prototype.text if len(prototype.text) >= len(evidence.text) else evidence.text
            if current_support >= self.min_support and target_key not in stored_by_text:
                promoted = prototype_by_key[target_key]
                self.store[user_id][domain].append(
                    Evidence(
                        evidence_id=f"lt-{user_id}-{domain}-{len(self.store[user_id][domain])}",
                        user_id=user_id,
                        item_id=promoted.item_id,
                        domain=domain,
                        timestamp=promoted.timestamp,
                        evidence_type="long_term",
                        text=promoted.text,
                        embedding=promoted.embedding,
                        strength=float(current_support),
                        recency_weight=promoted.recency_weight,
                        source="consolidation",
                    )
                )

    def get_evidence(self, user_id: str, domain: str) -> list[Evidence]:
        """Return consolidated evidence."""
        return list(self.store.get(user_id, {}).get(domain, []))

    def get_centroid(self, user_id: str, domain: str) -> list[float]:
        """Return the mean long-term embedding."""
        return mean_embedding([ev.embedding for ev in self.get_evidence(user_id, domain)])
