"""Deterministic mock LLM."""

from __future__ import annotations

from src.llm.llm_client import LLMClient


class MockLLM(LLMClient):
    """Template-based deterministic LLM for tests and offline runs."""

    def generate(self, prompt: str, **kwargs: object) -> str:
        lowered = prompt.lower()
        if "short-term summary" in lowered:
            return "User recently prefers the most recent domain-consistent items."
        if "long-term summary" in lowered:
            return "User historically prefers stable domain patterns across repeated interactions."
        if "conflict explanation" in lowered:
            return "Short-term behavior diverges from long-term preferences in this domain."
        if "reflection" in lowered:
            return "The recommendation did not match. Updating preference toward the observed interaction."
        if "re-rank" in lowered:
            return "1. candidate_1\n2. candidate_2\nReason: prioritize long-term stable preferences."
        return "Deterministic mock response."
