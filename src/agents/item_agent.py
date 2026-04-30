"""Item agent."""

from __future__ import annotations

from collections import defaultdict

from src.agents.base_agent import BaseAgent


class ItemAgent(BaseAgent):
    """Maintain metadata and supporting evidence for one item."""

    def __init__(self, agent_id: str, llm_client, memory: dict, item_row: dict) -> None:
        super().__init__(agent_id, llm_client, memory)
        self.item_row = item_row
        self.user_evidence: list[str] = []

    def build_prompt(self, context: dict) -> str:
        return f"Describe item {self.item_row['title']} in domain {self.item_row['domain']}."

    def get_item_profile(self) -> str:
        """Return a lightweight item profile."""
        return f"{self.item_row['title']} | {self.item_row['domain']} | {self.item_row['description']}"
