"""User agent wrapper for TPC-AgentCF scoring."""

from __future__ import annotations

from src.agents.base_agent import BaseAgent


class UserAgent(BaseAgent):
    """Lightweight adapter around the memory-backed recommendation flow."""

    def build_prompt(self, context: dict) -> str:
        user_id = context.get("user_id", "unknown")
        domain = context.get("domain", "unknown")
        return f"Score recommendation candidates for user {user_id} in domain {domain}."
