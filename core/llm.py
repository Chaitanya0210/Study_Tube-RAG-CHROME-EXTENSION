"""LLM interface and Google Gemini implementation via official google-genai SDK."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Generator, Optional

from core.config import get_config
from core.prompts import SYSTEM_INSTRUCTION

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """User-facing LLM error."""
    pass


class BaseLLM(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Generate full text response synchronously."""
        pass

    @abstractmethod
    def generate_stream(
        self, prompt: str, system_instruction: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Stream response chunks incrementally."""
        pass


class GeminiLLM(BaseLLM):
    """Gemini implementation using the official google-genai SDK with retry and rate limit handling."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        cfg = get_config()
        self.api_key = api_key or cfg.gemini_api_key
        self.model_name = model or cfg.gemini_model
        self._client = None

        if self.api_key and not self.api_key.startswith("your_"):
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
                logger.info("Initialized Gemini client with model: %s", self.model_name)
            except Exception as e:
                logger.error("Failed to instantiate google-genai Client: %s", e)

    def _ensure_client(self):
        """Validate API key and client readiness."""
        if not self._client:
            cfg = get_config()
            key = self.api_key or cfg.gemini_api_key
            if not key or key.startswith("your_"):
                raise LLMError(
                    "Google Gemini API key is missing. Please set your free GEMINI_API_KEY in .env "
                    "(obtainable from https://aistudio.google.com/)."
                )
            from google import genai
            self.api_key = key
            self._client = genai.Client(api_key=self.api_key)

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Synchronous generation with exponential backoff retry for rate limits."""
        self._ensure_client()
        from google.genai import types
        from google.genai.errors import APIError

        sys_inst = system_instruction or SYSTEM_INSTRUCTION
        config = types.GenerateContentConfig(
            system_instruction=sys_inst,
            temperature=0.3,
        )

        max_retries = 3
        delay = 1.5
        models_to_try = [self.model_name, "gemini-flash-lite-latest", "gemini-3.6-flash"]

        for attempt in range(1, max_retries + 1):
            current_model = models_to_try[(attempt - 1) % len(models_to_try)]
            try:
                response = self._client.models.generate_content(
                    model=current_model,
                    contents=prompt,
                    config=config,
                )
                return response.text or ""
            except APIError as e:
                msg = str(e).lower()
                is_retryable = (
                    e.code in (429, 503)
                    or "resource_exhausted" in msg
                    or "quota" in msg
                    or "high demand" in msg
                    or "unavailable" in msg
                )
                if is_retryable and attempt < max_retries:
                    logger.warning(
                        "Gemini temporary error on %s (code %s). Retrying in %.1fs with alternate model...",
                        current_model, e.code, delay
                    )
                    time.sleep(delay)
                    delay *= 2.0
                    continue
                elif is_retryable:
                    raise LLMError("The AI service is currently experiencing temporary high traffic. Please try again in a moment.")
                else:
                    logger.error("Gemini API error: %s", e)
                    raise LLMError(f"Gemini API Error: {str(e)}")
            except Exception as e:
                logger.error("Unexpected error in Gemini generate: %s", e)
                raise LLMError(f"Generation failed: {str(e)}")

        return ""

    def generate_stream(
        self, prompt: str, system_instruction: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Streaming generation with 503/429 retry, alternate model fallback, and friendly UI feedback."""
        try:
            self._ensure_client()
        except LLMError as e:
            yield f"⚠️ {str(e)}"
            return

        from google.genai import types
        from google.genai.errors import APIError

        sys_inst = system_instruction or SYSTEM_INSTRUCTION
        config = types.GenerateContentConfig(
            system_instruction=sys_inst,
            temperature=0.3,
        )

        max_retries = 3
        delay = 1.5
        models_to_try = [self.model_name, "gemini-flash-lite-latest", "gemini-3.6-flash"]

        for attempt in range(1, max_retries + 1):
            current_model = models_to_try[(attempt - 1) % len(models_to_try)]
            try:
                response_stream = self._client.models.generate_content_stream(
                    model=current_model,
                    contents=prompt,
                    config=config,
                )
                has_yielded = False
                for chunk in response_stream:
                    if chunk.text:
                        has_yielded = True
                        yield chunk.text
                return
            except APIError as e:
                msg = str(e).lower()
                is_retryable = (
                    e.code in (429, 503)
                    or "resource_exhausted" in msg
                    or "quota" in msg
                    or "high demand" in msg
                    or "unavailable" in msg
                )
                if is_retryable and attempt < max_retries:
                    logger.warning(
                        "Gemini streaming temporary error on %s (code %s). Retrying in %.1fs...",
                        current_model, e.code, delay
                    )
                    time.sleep(delay)
                    delay *= 2.0
                    continue
                elif is_retryable:
                    yield "⚠️ The AI service is currently experiencing temporary high traffic. Please try asking again in a few seconds."
                    return
                else:
                    yield f"⚠️ AI service notice: {str(e)}"
                    return
            except Exception as e:
                logger.error("Error in streaming response: %s", e)
                yield f"⚠️ Generation notice: {str(e)}"
                return


def get_llm(model: Optional[str] = None) -> BaseLLM:
    """Factory helper to obtain the configured LLM instance."""
    return GeminiLLM(model=model)
