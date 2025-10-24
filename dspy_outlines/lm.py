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
from pydantic import BaseModel

from .mlx_loader import create_outlines_model
from .schema_extractor import extract_output_schema

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
    2. Adapter passes Pydantic schema through _outlines_schema kwarg
    3. Formats prompt using DSPy's formatting
    4. Generates via Outlines with schema constraint
    5. Returns validated Pydantic object to DSPy

    This gives us:
    - DSPy's signature-based programming + optimization
    - Outlines' guaranteed constrained generation
    - MLX's efficient local inference
    """

    def __init__(self, model_path: str = None):
        """
        Initialize hybrid LM.

        Args:
            model_path: Path to MLX model (uses default if None)
        """
        super().__init__(model="outlines-mlx")
        self.outlines_model, self.tokenizer = create_outlines_model(model_path)
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
            **kwargs: Additional generation params (max_tokens, _outlines_schema, etc.)

        Returns:
            OpenAI-format response dict
        """
        max_tokens = kwargs.get('max_tokens', 512)
        schema = kwargs.pop('_outlines_schema', None)
        field_name = kwargs.pop('_outlines_field_name', None)

        # Format prompt outside lock (can run concurrently)
        if messages:
            formatted_prompt = self._format_messages(messages)
        else:
            formatted_prompt = prompt

        logger.info(f"Generating with schema: {schema.__name__ if schema else 'None'}")

        # Lock MLX model access to prevent Metal command buffer race conditions
        with MLX_LOCK:
            if schema:
                result_json = self.outlines_model(
                    formatted_prompt,
                    output_type=schema,
                    max_tokens=max_tokens
                )
            else:
                completion = self.outlines_model(
                    formatted_prompt,
                    max_tokens=max_tokens
                )

        # Process results outside lock (can run concurrently)
        if schema:
            parsed = schema.model_validate_json(result_json)
            if field_name:
                completion = json.dumps({field_name: parsed.model_dump()})
            else:
                completion = parsed.model_dump_json()

        # Store in history (best-effort, not critical for correctness)
        # Note: No lock needed - history is for debugging only, not read by application
        # Worst case: corrupted/missing history entries under concurrent access
        self.history.append({
            "prompt": formatted_prompt[:200],
            "completion": completion[:200]
        })

        # Return OpenAI format response as object (not dict)
        # BaseLM expects object with attributes, not dict
        # Note: usage must be dict-convertible (dict() is called on it for logging)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=completion, role="assistant"),
                    finish_reason="stop"
                )
            ],
            usage=AttrDict(
                prompt_tokens=0,  # MLX doesn't provide token counts easily
                completion_tokens=0,
                total_tokens=0
            ),
            model=self.model
        )

    def _format_messages(self, messages: list[dict]) -> str:
        """
        Format chat messages using tokenizer's chat template.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts

        Returns:
            Formatted prompt string
        """
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback: simple formatting
            prompt = ""
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'system':
                    prompt += f"System: {content}\n\n"
                elif role == 'user':
                    prompt += f"User: {content}\n\n"
            prompt += "Assistant:"
            return prompt
