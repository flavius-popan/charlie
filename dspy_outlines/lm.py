"""Hybrid DSPy LM using Outlines for constrained generation.

IMPORTANT: This implementation uses dspy.BaseLM (not dspy.LM) as documented
in research/dspy-lm-interface.md. BaseLM is simpler and doesn't require LiteLLM,
making it ideal for custom backend implementations like Outlines+MLX.

Thread Safety: MLX (via Apple Metal framework) is NOT thread-safe. Multiple threads
cannot call the same MLX model simultaneously without causing Metal command buffer
race conditions. This module uses MLX_LOCK to serialize access to the model.
See dspy_outlines/README.md for detailed thread safety documentation.
"""

import logging
import threading
from typing import Any
import json
from types import SimpleNamespace

import dspy
import mlx_lm
from pydantic import BaseModel

from .mlx_loader import create_outlines_model

logger = logging.getLogger(__name__)

# MLX thread safety lock
# MLX/Metal is NOT thread-safe - multiple threads calling the same model instance
# simultaneously will cause Metal framework errors:
#   - "A command encoder is already encoding to this command buffer"
#   - "Completed handler provided after commit call"
# This module-level lock serializes all MLX inference calls across all OutlinesLM instances.
MLX_LOCK = threading.Lock()


class AttrDict(dict):
    """Dict that allows attribute-style access (needed for DSPy compatibility)."""

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


class OutlinesLM(dspy.BaseLM):
    """
    Hybrid LM: DSPy ↔ Outlines ↔ MLX

    Architecture:
    1. Receives DSPy signature + inputs via OutlinesAdapter
    2. Adapter passes constraint (Pydantic, Literal, Regex, etc.) through _outlines_constraint kwarg
    3. Formats prompt using DSPy's formatting
    4. Generates via Outlines with constraint
    5. Returns validated output to DSPy

    This gives us:
    - DSPy's signature-based programming + optimization
    - Outlines' guaranteed constrained generation
    - MLX's efficient local inference
    """

    def __init__(self, model_path: str = None, *, enable_prompt_cache: bool = False):
        """
        Initialize hybrid LM.

        Args:
            model_path: Path to MLX model (uses default if None)
            enable_prompt_cache: Whether to build an MLX prompt cache (off by default)
        """
        # Import here to get default path
        from .mlx_loader import DEFAULT_MODEL_PATH

        if model_path is None:
            model_path = DEFAULT_MODEL_PATH

        # Keep self.model as string for DSPy compatibility (JSONAdapter expects lm.model.split())
        super().__init__(model=model_path)

        # Store Outlines wrapper separately
        (
            self.outlines_model,
            self.tokenizer,
            self.prompt_cache,
        ) = create_outlines_model(
            model_path, enable_prompt_cache=enable_prompt_cache
        )
        # Store raw MLX model for unconstrained generation with caching support
        # self.outlines_model is the Outlines wrapper, self.outlines_model.model is the underlying MLX nn.Module
        self.raw_mlx_model = self.outlines_model.model
        if self.prompt_cache is None:
            logger.info("OutlinesLM initialized with prompt caching disabled")
        else:
            logger.info("OutlinesLM initialized with prompt caching enabled")
        logger.info("OutlinesLM initialized with Outlines+MLX backend")

    def forward(self, prompt=None, messages=None, **kwargs):
        """
        Main generation interface called by DSPy.

        This method uses MLX_LOCK to ensure thread-safe access to the MLX model.
        Only the actual model inference is locked; prompt formatting and result
        processing happen outside the lock to minimize contention.

        Args:
            prompt: String prompt (if using prompt-based)
            messages: List of message dicts (if using chat format)
            **kwargs: Additional generation params (max_tokens, _outlines_constraint, etc.)

        Returns:
            OpenAI-format response dict
        """
        max_tokens = kwargs.get("max_tokens", 512)
        constraint = kwargs.pop("_outlines_constraint", None)
        field_name = kwargs.pop("_outlines_field_name", None)

        # Format prompt outside lock (can run concurrently)
        if messages:
            formatted_prompt = self._format_messages(messages)
        else:
            formatted_prompt = prompt

        # Log cache statistics before generation
        cache_size = self._get_cache_size()
        logger.info(
            f"Generating with constraint: {constraint.__name__ if constraint else 'None'}, "
            f"cache_size={cache_size} tokens"
        )

        # Lock MLX model access to prevent Metal command buffer race conditions
        with MLX_LOCK:
            outlines_kwargs = {"max_tokens": max_tokens}
            if self.prompt_cache is not None:
                outlines_kwargs["prompt_cache"] = self.prompt_cache

            if constraint:
                # Use Outlines wrapper for constrained generation
                result_json = self.outlines_model(
                    formatted_prompt,
                    output_type=constraint,
                    **outlines_kwargs,
                )
            else:
                # Use raw MLX for unconstrained generation
                generate_kwargs = {"verbose": False}
                if self.prompt_cache is not None:
                    generate_kwargs["prompt_cache"] = self.prompt_cache
                completion = mlx_lm.generate(
                    self.raw_mlx_model,
                    self.tokenizer,
                    formatted_prompt,
                    max_tokens=max_tokens,
                    **generate_kwargs,
                )

        # Process results outside lock (can run concurrently)
        if constraint:
            # Check if constraint is a Pydantic model (requires JSON parsing)
            if isinstance(constraint, type) and issubclass(constraint, BaseModel):
                parsed = constraint.model_validate_json(result_json)
                if field_name:
                    completion = json.dumps({field_name: parsed.model_dump()})
                else:
                    completion = parsed.model_dump_json()
            else:
                # For other constraint types (Literal, Regex, etc.), result is already a string
                if field_name:
                    completion = json.dumps({field_name: result_json})
                else:
                    completion = result_json

        # Store in history (best-effort, not critical for correctness)
        # Note: No lock needed - history is for debugging only, not read by application
        # Worst case: corrupted/missing history entries under concurrent access
        self.history.append(
            {"prompt": formatted_prompt[:200], "completion": completion[:200]}
        )

        # Return OpenAI format response as object (not dict)
        # BaseLM expects object with attributes, not dict
        # Note: usage must be dict-convertible (dict() is called on it for logging)
        # model should be the string path for compatibility
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=completion, role="assistant"),
                    finish_reason="stop",
                )
            ],
            usage=AttrDict(
                prompt_tokens=0,  # MLX doesn't provide token counts easily
                completion_tokens=0,
                total_tokens=0,
            ),
            model=self.model,  # String model path for DSPy compatibility
        )

    def enable_prompt_cache(self, *, reset: bool = True) -> None:
        """
        Lazily build the MLX prompt cache for this instance.

        Args:
            reset: If True, rebuilds the cache even if one already exists.
        """
        if self.prompt_cache is not None and not reset:
            logger.info("Prompt cache already initialized; skipping rebuild")
            return

        from mlx_lm.models.cache import make_prompt_cache

        with MLX_LOCK:
            logger.info("Enabling MLX prompt cache")
            self.prompt_cache = make_prompt_cache(self.raw_mlx_model)

    def disable_prompt_cache(self) -> None:
        """Disable MLX prompt caching for this instance."""
        if self.prompt_cache is None:
            logger.info("Prompt cache already disabled")
            return

        with MLX_LOCK:
            logger.info("Disabling MLX prompt cache")
            self.prompt_cache = None

    def _get_cache_size(self) -> int:
        """
        Get the number of tokens currently in the prompt cache.

        Returns:
            Number of cached tokens (0 if cache is empty or None)
        """
        if self.prompt_cache is None:
            return 0
        # MLX prompt cache is a list of KVCache objects
        # Each KVCache has keys, values, and offset attributes
        # The offset indicates how many tokens are cached
        try:
            if len(self.prompt_cache) > 0:
                # Get the first layer's cache
                first_layer_cache = self.prompt_cache[0]
                if hasattr(first_layer_cache, 'offset'):
                    return first_layer_cache.offset
        except (IndexError, AttributeError):
            pass
        return 0

    def _format_messages(self, messages: list[dict]) -> str:
        """
        Format chat messages using tokenizer's chat template.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts

        Returns:
            Formatted prompt string
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback: simple formatting
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n\n"
            prompt += "Assistant:"
            return prompt
