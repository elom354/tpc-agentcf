"""Base agent."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.llm.llm_client import LLMClient


class BaseAgent(ABC):
    """Abstract agent with shared LLM access."""

    def __init__(self, agent_id: str, llm_client: LLMClient, memory: dict) -> None:
        self.agent_id = agent_id
        self.llm_client = llm_client
        self.memory = memory

    def generate_response(self, prompt: str) -> str:
        """Generate a response from the configured backend."""
        return self.llm_client.generate(prompt)

    @abstractmethod
    def build_prompt(self, context: dict) -> str:
        """Build a prompt for this agent."""
