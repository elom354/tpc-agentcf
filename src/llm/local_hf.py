"""Local Hugging Face backend for small free models."""

from __future__ import annotations

import logging

from src.llm.llm_client import LLMClient
from src.llm.mock_llm import MockLLM

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except Exception:  # pragma: no cover
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None


class LocalHuggingFaceLLM(LLMClient):
    """Run a lightweight local instruction model on CPU."""

    def __init__(self, model_name: str = "google/flan-t5-small", max_new_tokens: int = 64) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.fallback = MockLLM()
        if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
            LOGGER.warning("transformers is unavailable. Falling back to MockLLM.")
            self.tokenizer = None
            self.model = None
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to load local model %s. Falling back to MockLLM. Error: %s", model_name, exc)
            self.tokenizer = None
            self.model = None

    def generate(self, prompt: str, **kwargs: object) -> str:
        if self.tokenizer is None or self.model is None:
            return self.fallback.generate(prompt, **kwargs)
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=int(kwargs.get("max_new_tokens", self.max_new_tokens)),
                do_sample=False,
            )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return text or self.fallback.generate(prompt, **kwargs)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Local model generation failed. Falling back to MockLLM. Error: %s", exc)
            return self.fallback.generate(prompt, **kwargs)
