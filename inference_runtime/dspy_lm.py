"""DSPy BaseLM implementation backed directly by llama.cpp."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

from pipeline import _dspy_setup  # noqa: F401
import dspy

from settings import MODEL_CONFIG
from .loader import load_model

logger = logging.getLogger(__name__)


class DspyLM(dspy.BaseLM):
    """Minimal DSPy LM that routes calls to a llama.cpp model."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        generation_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model="Qwen3-4B-Instruct-GGUF")

        self.llm = load_model(model_path)
        self.generation_config = generation_config or MODEL_CONFIG
        logger.info("DspyLM initialized with llama.cpp model")

    def forward(
        self,
        messages: list[dict[str, str]] | None = None,
        prompt: str | None = None,
        **kwargs: Any,
    ):
        if messages is None and prompt is None:
            raise ValueError("Either messages or prompt must be provided")

        temp = kwargs.get("temp", self.generation_config.get("temp", MODEL_CONFIG["temp"]))
        top_p = kwargs.get("top_p", self.generation_config.get("top_p", MODEL_CONFIG["top_p"]))
        top_k = kwargs.get("top_k", self.generation_config.get("top_k", MODEL_CONFIG["top_k"]))
        min_p = kwargs.get("min_p", self.generation_config.get("min_p", MODEL_CONFIG["min_p"]))
        presence_penalty = kwargs.get(
            "presence_penalty", self.generation_config.get("presence_penalty", MODEL_CONFIG["presence_penalty"])
        )
        max_tokens = kwargs.get(
            "max_tokens", self.generation_config.get("max_tokens", MODEL_CONFIG["max_tokens"])
        )

        logger.debug(
            "DspyLM generation: max_tokens=%s temp=%s top_p=%s top_k=%s min_p=%s presence_penalty=%s",
            max_tokens,
            temp,
            top_p,
            top_k,
            min_p,
            presence_penalty,
        )

        if messages is not None:
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=temp,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                max_tokens=max_tokens,
            )
        else:
            response = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                max_tokens=max_tokens,
            )

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=response["choices"][0]["message"]["content"],
                        role="assistant",
                    ),
                    finish_reason=response["choices"][0].get("finish_reason", "stop"),
                )
            ],
            usage=response.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
            model=self.model,
        )
