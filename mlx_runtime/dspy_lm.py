"""DSPy BaseLM implementation backed directly by MLX."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

from pipeline import _dspy_setup  # noqa: F401
import dspy
import mlx_lm
from mlx_lm.sample_utils import make_sampler

from . import MLX_LOCK
from .loader import load_mlx_model

logger = logging.getLogger(__name__)


class MLXDspyLM(dspy.BaseLM):
    """Minimal DSPy LM that routes calls to an MLX model."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        generation_config: dict[str, Any] | None = None,
    ) -> None:
        from settings import DEFAULT_MODEL_PATH

        path = model_path or DEFAULT_MODEL_PATH
        super().__init__(model=path)

        self.mlx_model, self.tokenizer = load_mlx_model(path)
        self.generation_config = generation_config or {
            "temp": 0.0,
            "top_p": 1.0,
            "min_p": 0.0,
            "min_tokens_to_keep": 1,
            "top_k": 0,
            "max_tokens": 512,
        }
        logger.info("MLXDspyLM initialized with model %s", path)

    def forward(
        self,
        messages: list[dict[str, str]] | None = None,
        prompt: str | None = None,
        **kwargs: Any,
    ):
        if messages is None and prompt is None:
            raise ValueError("Either messages or prompt must be provided")

        formatted_prompt = (
            self._format_messages(messages) if messages is not None else prompt
        )

        sampler_kwargs = {
            "temp": kwargs.get("temp", self.generation_config.get("temp", 0.0)),
            "top_p": kwargs.get("top_p", self.generation_config.get("top_p", 1.0)),
            "min_p": kwargs.get("min_p", self.generation_config.get("min_p", 0.0)),
            "min_tokens_to_keep": kwargs.get(
                "min_tokens_to_keep",
                self.generation_config.get("min_tokens_to_keep", 1),
            ),
            "top_k": kwargs.get("top_k", self.generation_config.get("top_k", 0)),
        }
        sampler = make_sampler(**sampler_kwargs)

        max_tokens = kwargs.get(
            "max_tokens", self.generation_config.get("max_tokens", 512)
        )

        logger.debug(
            "MLXDspyLM generation: max_tokens=%s sampler=%s",
            max_tokens,
            sampler_kwargs,
        )

        with MLX_LOCK:
            completion = mlx_lm.generate(
                self.mlx_model,
                self.tokenizer,
                formatted_prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False,
            )

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=completion, role="assistant"),
                    finish_reason="stop",
                )
            ],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            model=self.model,
        )

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
            elif role == "user":
                prompt += f"User: {content}\n\n"
            else:
                prompt += f"{role.capitalize()}: {content}\n\n"
        prompt += "Assistant:"
        return prompt
