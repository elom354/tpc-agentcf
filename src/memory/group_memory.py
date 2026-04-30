"""Group-shared memory."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans

from src.memory.common import mean_embedding, text_to_embedding
from src.memory.evidence import Evidence


class GroupMemory:
    """Cluster users and store recent group evidence."""

    def __init__(self, num_groups: int = 5, group_window: int = 50) -> None:
        self.num_groups = num_groups
        self.group_window = group_window
        self.user_groups: dict[str, int] = {}
        self.group_memory: dict[int, dict[str, list[Evidence]]] = defaultdict(lambda: defaultdict(list))

    def fit_user_groups(self, user_profiles: dict[str, list[float]]) -> None:
        """Assign users to groups with KMeans."""
        if not user_profiles:
            return
        user_ids = list(user_profiles)
        matrix = np.array([user_profiles[user_id] for user_id in user_ids], dtype=float)
        n_clusters = min(self.num_groups, len(user_ids))
        if n_clusters == 1:
            self.user_groups = {user_id: 0 for user_id in user_ids}
            return
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(matrix)
        self.user_groups = {user_id: int(label) for user_id, label in zip(user_ids, labels, strict=True)}

    def update(self, user_id: str, item_id: str, domain: str, interaction_text: str, timestamp: float) -> None:
        """Add group evidence."""
        group_id = self.get_group_id(user_id)
        bucket = self.group_memory[group_id][domain]
        evidence = Evidence(
            evidence_id=f"gm-{group_id}-{domain}-{len(bucket)}",
            user_id=user_id,
            item_id=item_id,
            domain=domain,
            timestamp=timestamp,
            evidence_type="group",
            text=interaction_text,
            embedding=text_to_embedding(interaction_text),
            strength=1.0,
            recency_weight=1.0,
            source="group_memory",
        )
        if len(bucket) >= self.group_window:
            bucket.pop(0)
        bucket.append(evidence)

    def get_group_evidence(self, user_id: str, domain: str) -> list[Evidence]:
        """Return evidence for the user's group and domain."""
        return list(self.group_memory.get(self.get_group_id(user_id), {}).get(domain, []))

    def get_group_centroid(self, user_id: str, domain: str) -> list[float]:
        """Return centroid for the user's group in a domain."""
        return mean_embedding([ev.embedding for ev in self.get_group_evidence(user_id, domain)])

    def get_group_id(self, user_id: str) -> int:
        """Return the group ID or zero if unavailable."""
        return self.user_groups.get(user_id, 0)
