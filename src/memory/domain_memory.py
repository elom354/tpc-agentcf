"""Domain-separated and fused memory."""

from __future__ import annotations

from collections import defaultdict

from src.memory.common import cosine_similarity, text_to_embedding
from src.memory.evidence import Evidence


class DomainMemory:
    """Maintain domain-separated and fused evidence."""

    def __init__(self, cross_domain_threshold: float = 0.2, base_weight: float = 0.5) -> None:
        self.cross_domain_threshold = cross_domain_threshold
        self.base_weight = base_weight
        self.domain_separated_memory: dict[str, dict[str, list[Evidence]]] = defaultdict(lambda: defaultdict(list))
        self.domain_fused_memory: dict[str, dict[str, list[Evidence]]] = defaultdict(lambda: defaultdict(list))

    def update(self, user_id: str, item_id: str, domain: str, interaction_text: str, timestamp: float) -> None:
        """Add evidence to per-domain storage and refresh fused storage."""
        evidence = Evidence(
            evidence_id=f"dm-{user_id}-{domain}-{len(self.domain_separated_memory[user_id][domain])}",
            user_id=user_id,
            item_id=item_id,
            domain=domain,
            timestamp=timestamp,
            evidence_type="domain",
            text=interaction_text,
            embedding=text_to_embedding(interaction_text),
            strength=1.0,
            recency_weight=1.0,
            source="interaction",
        )
        self.domain_separated_memory[user_id][domain].append(evidence)
        self._rebuild_fused(user_id)

    def _rebuild_fused(self, user_id: str) -> None:
        user_domains = self.domain_separated_memory[user_id]
        self.domain_fused_memory[user_id] = defaultdict(list)
        for target_domain, target_evidence in user_domains.items():
            target_embeddings = [ev.embedding for ev in target_evidence]
            target_centroid = target_embeddings[0] if target_embeddings else [0.0] * 16
            for source_domain, source_evidence in user_domains.items():
                if source_domain == target_domain:
                    continue
                for evidence in source_evidence:
                    similarity = cosine_similarity(evidence.embedding, target_centroid)
                    if similarity >= self.cross_domain_threshold:
                        fused = Evidence(
                            evidence_id=f"fused-{evidence.evidence_id}-{target_domain}",
                            user_id=evidence.user_id,
                            item_id=evidence.item_id,
                            domain=target_domain,
                            timestamp=evidence.timestamp,
                            evidence_type="domain",
                            text=evidence.text,
                            embedding=evidence.embedding,
                            strength=self.base_weight * similarity,
                            recency_weight=evidence.recency_weight,
                            source="consolidation",
                            metadata={"source_domain": source_domain},
                        )
                        self.domain_fused_memory[user_id][target_domain].append(fused)

    def get_domain_evidence(self, user_id: str, domain: str) -> list[Evidence]:
        """Get in-domain evidence."""
        return list(self.domain_separated_memory.get(user_id, {}).get(domain, []))

    def get_fused_evidence(self, user_id: str, target_domain: str) -> list[Evidence]:
        """Get cross-domain fused evidence."""
        return list(self.domain_fused_memory.get(user_id, {}).get(target_domain, []))
