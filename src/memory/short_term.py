"""Short-term memory."""

from __future__ import annotations

from collections import defaultdict

from src.memory.common import mean_embedding, recency_weight, text_to_embedding
from src.memory.evidence import Evidence


class ShortTermMemory:
    """Store recent user interactions by domain."""

    def __init__(self, short_term_window: int = 15, lambda_decay: float = 0.05) -> None:
        self.short_term_window = short_term_window
        self.lambda_decay = lambda_decay
        self.store: dict[str, dict[str, list[Evidence]]] = defaultdict(lambda: defaultdict(list))

    def add(self, user_id: str, item_id: str, domain: str, text: str, timestamp: float) -> Evidence:
        """Insert a recent interaction."""
        bucket = self.store[user_id][domain]
        evidence = Evidence(
            evidence_id=f"st-{user_id}-{domain}-{len(bucket)}-{int(timestamp)}",
            user_id=user_id,
            item_id=item_id,
            domain=domain,
            timestamp=timestamp,
            evidence_type="short_term",
            text=text,
            embedding=text_to_embedding(text),
            strength=1.0,
            recency_weight=recency_weight(timestamp, timestamp, self.lambda_decay),
            source="interaction",
        )
        if len(bucket) >= self.short_term_window:
            bucket.pop(0)
        bucket.append(evidence)
        return evidence

    def get_evidence(self, user_id: str, domain: str) -> list[Evidence]:
        """Return stored evidence."""
        return list(self.store.get(user_id, {}).get(domain, []))

    def get_centroid(self, user_id: str, domain: str) -> list[float]:
        """Return mean embedding for a user-domain."""
        return mean_embedding([ev.embedding for ev in self.get_evidence(user_id, domain)])
