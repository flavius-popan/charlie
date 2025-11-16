"""Lightweight Graphiti-compatible LLM client that talks directly to MLX."""

from __future__ import annotations

import logging
from typing import Iterable

import mlx_lm
from mlx_lm.sample_utils import make_sampler

from graphiti_core.llm_client import LLMClient, LLMConfig
from graphiti_core.prompts.models import Message as GraphitiMessage

from dspy_outlines.lm import MLX_LOCK
from dspy_outlines.mlx_loader import load_mlx_model

from settings import DEFAULT_MODEL_PATH, MODEL_CONFIG

logger = logging.getLogger(__name__)


def _to_dict_messages(messages: Iterable[GraphitiMessage]) -> list[dict[str, str]]:
    formatted: list[dict[str, str]] = []
    for message in messages:
        role = getattr(message, "role", "user") or "user"
        content = getattr(message, "content", "") or ""
        formatted.append({"role": role, "content": content})
    return formatted


class LocalMLXLLMClient(LLMClient):
    """Graphiti LLMClient implementation backed by a local MLX model."""

    def __init__(
        self,
        model_path: str | None = None,
        generation_config: dict | None = None,
    ) -> None:
        model_path = model_path or DEFAULT_MODEL_PATH
        config = LLMConfig(
            model=model_path,
            temperature=0.0,
            max_tokens=MODEL_CONFIG.get("max_tokens", 512),
        )
        super().__init__(config=config)

        self.model_path = model_path
        self.generation_config = generation_config or MODEL_CONFIG

        self.mlx_model, self.tokenizer = load_mlx_model(model_path)
        logger.info("LocalMLXLLMClient loaded MLX model: %s", model_path)

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        prompt = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                prompt += f"System: {content}\n\n"
            else:
                prompt += f"{role.capitalize()}: {content}\n\n"
        prompt += "Assistant:"
        return prompt

    async def _generate_response(
        self,
        messages: list[GraphitiMessage],
        response_model,  # noqa: ANN001 - signature defined by parent
        max_tokens: int,
        model_size,  # noqa: ANN001 - signature defined by parent
    ) -> dict:
        message_dicts = _to_dict_messages(messages)
        prompt = self._format_messages(message_dicts)

        sampler = make_sampler(
            temp=self.generation_config.get("temp", 0.0),
            top_p=self.generation_config.get("top_p", 1.0),
            min_p=self.generation_config.get("min_p", 0.0),
            min_tokens_to_keep=self.generation_config.get("min_tokens_to_keep", 1),
            top_k=self.generation_config.get("top_k", 0),
        )

        with MLX_LOCK:
            completion = mlx_lm.generate(
                self.mlx_model,
                self.tokenizer,
                prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False,
            )

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
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
