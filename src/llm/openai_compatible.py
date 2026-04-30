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

    def __init__(self, model: str | None = None) -> None:
        self.fallback = MockLLM()
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        env_model = os.getenv("OPENAI_MODEL")
        if model:
            self.model = model
        elif env_model:
            self.model = env_model
        elif base_url and "deepseek" in base_url.lower():
            self.model = "deepseek-chat"
        else:
            self.model = "gpt-4o-mini"
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
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = response.choices[0].message.content if response.choices else ""
            return content or self.fallback.generate(prompt, **kwargs)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("LLM request failed for model %s. Falling back to MockLLM. Error: %s", self.model, exc)
            return self.fallback.generate(prompt, **kwargs)
