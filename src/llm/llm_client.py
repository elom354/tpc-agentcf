"""Abstract LLM client."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract interface for prompt generation."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: object) -> str:
        """Generate a response for a prompt."""
