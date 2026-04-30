"""OpenAI-compatible backend with graceful fallback."""

from __future__ import annotations

import logging
import os

from src.llm.llm_client import LLMClient
from src.llm.mock_llm import MockLLM

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


class OpenAICompatibleLLM(LLMClient):
    """Use an OpenAI-compatible chat endpoint or fall back to MockLLM."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self.fallback = MockLLM()
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key or OpenAI is None:
            LOGGER.warning("OPENAI_API_KEY missing or openai package unavailable. Falling back to MockLLM.")
            self.client = None
            return
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def generate(self, prompt: str, **kwargs: object) -> str:
        if self.client is None:
            return self.fallback.generate(prompt, **kwargs)
        response = self.client.responses.create(model=self.model, input=prompt)
        return getattr(response, "output_text", "") or self.fallback.generate(prompt, **kwargs)
