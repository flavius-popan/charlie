"""Hybrid DSPy LM using Outlines for constrained generation.

IMPORTANT: This implementation uses dspy.BaseLM (not dspy.LM) as documented
in research/dspy-lm-interface.md. BaseLM is simpler and doesn't require LiteLLM,
making it ideal for custom backend implementations like Outlines+MLX.
"""

import asyncio
import logging
from typing import Any
import json
from types import SimpleNamespace

import dspy
from pydantic import BaseModel

from .mlx_loader import create_outlines_model
from .schema_extractor import extract_output_schema

logger = logging.getLogger(__name__)

# MLX thread safety lock (prevents segfaults in async contexts)
MLX_LOCK = asyncio.Lock()

class AttrDict(dict):
    """Dict that allows attribute-style access (needed for DSPy compatibility)."""
    def __getattr__(self, key):
        return self[key]
    def __setattr__(self, key, value):
        self[key] = value

class OutlinesDSPyLM(dspy.BaseLM):
    """
    Hybrid LM: DSPy ↔ Outlines ↔ MLX

    Architecture:
    1. Receives DSPy signature + inputs
    2. Extracts Pydantic schema from output field
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
        logger.info("OutlinesDSPyLM initialized with Outlines+MLX backend")

    def forward(self, prompt=None, messages=None, **kwargs):
        """
        Main generation interface called by DSPy.

        Args:
            prompt: String prompt (if using prompt-based)
            messages: List of message dicts (if using chat format)
            **kwargs: Additional generation params (max_tokens, temperature, signature, etc.)

        Returns:
            OpenAI-format response dict
        """
        # Extract generation params
        max_tokens = kwargs.get('max_tokens', 512)

        # Get the signature from kwargs if available (DSPy passes it)
        signature = kwargs.get('signature', None)

        # Extract Pydantic schema from signature
        schema = None
        if signature:
            schema = extract_output_schema(signature)

        # Format the prompt
        if messages:
            formatted_prompt = self._format_messages(messages)
        else:
            formatted_prompt = prompt

        logger.info(f"Generating with schema: {schema.__name__ if schema else 'None'}")

        # Generate using Outlines
        if schema:
            # Constrained generation with Pydantic schema
            result_json = self.outlines_model(
                formatted_prompt,
                output_type=schema,
                max_tokens=max_tokens
            )
            # Parse and re-serialize to ensure valid JSON
            parsed = schema.model_validate_json(result_json)
            completion = parsed.model_dump_json()
        else:
            # Fallback: unconstrained text generation
            completion = self.outlines_model(
                formatted_prompt,
                max_tokens=max_tokens
            )

        # Store in history
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
