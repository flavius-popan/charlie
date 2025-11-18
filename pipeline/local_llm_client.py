"""Lightweight Graphiti-compatible LLM client that talks directly to llama.cpp."""

from __future__ import annotations

import logging
from typing import Iterable

from graphiti_core.llm_client import LLMClient, LLMConfig
from graphiti_core.prompts.models import Message as GraphitiMessage

from inference_runtime import load_model

from settings import MODEL_CONFIG

logger = logging.getLogger(__name__)


def _to_dict_messages(messages: Iterable[GraphitiMessage]) -> list[dict[str, str]]:
    formatted: list[dict[str, str]] = []
    for message in messages:
        role = getattr(message, "role", "user") or "user"
        content = getattr(message, "content", "") or ""
        formatted.append({"role": role, "content": content})
    return formatted


class LocalMLXLLMClient(LLMClient):
    """Graphiti LLMClient implementation backed by a local llama.cpp model."""

    def __init__(
        self,
        model_path: str | None = None,
        generation_config: dict | None = None,
    ) -> None:
        config = LLMConfig(
            model="Qwen3-4B-Instruct-GGUF",
            temperature=0.0,
            max_tokens=MODEL_CONFIG.get("max_tokens", 512),
        )
        super().__init__(config=config)

        self.model_path = model_path
        self.generation_config = generation_config or MODEL_CONFIG

        self.llm = load_model(model_path)
        logger.info("LocalMLXLLMClient loaded llama.cpp model")

    async def _generate_response(
        self,
        messages: list[GraphitiMessage],
        response_model,  # noqa: ANN001 - signature defined by parent
        max_tokens: int,
        model_size,  # noqa: ANN001 - signature defined by parent
    ) -> dict:
        message_dicts = _to_dict_messages(messages)

        response = self.llm.create_chat_completion(
            messages=message_dicts,
            temperature=self.generation_config.get("temp", 0.0),
            top_p=self.generation_config.get("top_p", 1.0),
            max_tokens=max_tokens,
        )

        completion = response["choices"][0]["message"]["content"]

        logger.debug("LocalMLXLLMClient completion: %s", completion[:200])

        if response_model is not None:
            try:
                parsed = response_model.model_validate_json(completion)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to parse structured response: %s", exc)
                logger.error("Raw completion: %s", completion[:500])
                raise
            return parsed.model_dump()

        return {
            "choices": [{"message": {"content": completion}}],
            "usage": response.get("usage", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }),
        }
