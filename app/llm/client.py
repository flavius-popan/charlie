"""GraphitiLM: LLM client bridge for Graphiti + Outlines + MLX."""

import asyncio
import logging
from typing import Any

from pydantic import BaseModel

from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from graphiti_core.prompts.models import Message

from .prompts import format_messages


logger = logging.getLogger(__name__)


class GraphitiLM(LLMClient):
    """
    Bridge: Graphiti <-> Outlines <-> MLX

    Design:
    1. Receives: list[Message] + Pydantic schema (response_model)
    2. Formats: messages -> prompt string using chat template
    3. Generates: Outlines structured generation with schema constraint
    4. Returns: validated Pydantic object as dict

    This enables all Graphiti operations (entity extraction, deduplication,
    summarization, reflexion, etc.) to work with local MLX inference.
    """

    def __init__(
        self,
        outlines_model,
        tokenizer,
        config: LLMConfig | None = None,
        mode: str = "direct"
    ):
        """
        Initialize GraphitiLM bridge.

        Args:
            outlines_model: Outlines model wrapping MLX (from outlines.models.mlxlm)
            tokenizer: MLX tokenizer (for chat template formatting)
            config: Optional LLM configuration
            mode: "direct" (current) or "dspy" (Phase 2)
        """
        super().__init__(config, cache=False)
        self.outlines_model = outlines_model
        self.tokenizer = tokenizer
        self.mode = mode

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        """
        Generate structured response using Outlines + MLX.

        Core bridge logic:
        1. Format messages -> chat prompt string
        2. Create Outlines generator with Pydantic schema
        3. Run generation (offloaded to thread for async safety)
        4. Return Pydantic object as dict

        Args:
            messages: Graphiti message list (system + user prompts)
            response_model: Pydantic schema for structured output
            max_tokens: Maximum tokens to generate
            model_size: Model size hint (currently unused - single model)

        Returns:
            Dict representation of the Pydantic response object
        """
        if self.mode == "dspy":
            # Phase 2: DSPy-based generation
            return await self._dspy_generate(messages, response_model, max_tokens)
        else:
            # Phase 1: Direct Outlines generation
            return await self._direct_generate(messages, response_model, max_tokens)

    async def _direct_generate(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None,
        max_tokens: int
    ) -> dict[str, Any]:
        """
        Direct generation via Outlines.

        Process:
        1. Format messages using chat template
        2. Call model directly with output_type and max_tokens
        3. Generate in thread (Outlines is synchronous)
        4. Parse and return result as dict
        """
        # Format prompt
        prompt = format_messages(messages, self.tokenizer)

        # Validate response model
        if response_model is None:
            raise ValueError("response_model is required for structured generation")

        # Call model directly (Outlines model is callable)
        logger.info(f"[GraphitiLM] Starting generation: prompt_length={len(prompt)}, max_tokens={max_tokens}, schema={response_model.__name__}")
        logger.debug(f"[GraphitiLM] Full prompt:\n{prompt[:500]}...")

        try:
            result_json = await asyncio.to_thread(
                self.outlines_model,
                prompt,
                output_type=response_model,
                max_tokens=max_tokens
            )
            logger.info(f"[GraphitiLM] Generation complete: {len(result_json)} chars")
            logger.debug(f"[GraphitiLM] Raw output:\n{result_json[:500]}...")
        except Exception as e:
            logger.error(f"[GraphitiLM] Generation failed: {e}")
            raise

        # Parse JSON string to Pydantic and convert to dict
        try:
            result = response_model.model_validate_json(result_json)
            logger.info(f"[GraphitiLM] Validation successful")
            return result.model_dump()
        except Exception as e:
            logger.error(f"[GraphitiLM] Validation failed: {e}")
            logger.error(f"[GraphitiLM] Raw JSON: {result_json}")
            raise

    async def _dspy_generate(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None,
        max_tokens: int
    ) -> dict[str, Any]:
        """
        DSPy-based generation (Phase 2 placeholder).

        Future: Route through DSPy optimizers for improved quality.
        """
        raise NotImplementedError("DSPy mode not yet implemented (Phase 2)")
