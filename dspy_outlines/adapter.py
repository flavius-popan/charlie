"""Three-tier fallback adapter: ChatAdapter → JSON → Outlines constrained."""

import logging
from typing import Any

from dspy.adapters import ChatAdapter
from dspy.adapters.types.tool import ToolCalls
from dspy.clients.lm import LM
from dspy.signatures.signature import Signature
from dspy.utils.exceptions import AdapterParseError
from litellm import ContextWindowExceededError

logger = logging.getLogger(__name__)


class OutlinesAdapter(ChatAdapter):
    """
    Three-tier fallback adapter for Outlines constrained generation.

    Chat: ChatAdapter field-marker format (fastest, often works)
    JSON: JSON format unconstrained (fast, fallback for JSON-capable models)
    OutlinesJSON: Outlines constrained generation (slow, guaranteed valid)

    Each tier is tried in order, falling back to the next on failure.
    Metrics track success/failure rates for experimentation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_adapter_used = None  # "chat", "json", or "outlines_json"
        self.metrics = {
            "chat_success": 0,
            "json_success": 0,
            "outlines_json_success": 0,
            "chat_failures": 0,
            "json_failures": 0,
        }

    def __call__(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Three-tier fallback execution.

        Chat: Try ChatAdapter (field markers)
        JSON: Try JSON format (unconstrained)
        OutlinesJSON: Use Outlines constrained generation (guaranteed)
        """
        # Extract constraint for OutlinesJSON
        constraint = self._extract_constraint(signature)

        # Check if we should skip OutlinesJSON (tool calls or multiple completions)
        skip_tier3 = self._has_tool_calls(signature)
        if skip_tier3:
            logger.warning(
                "ToolCalls detected - skipping OutlinesJSON (Outlines doesn't support ToolCalls)"
            )

        # Check for multiple completions
        n = lm_kwargs.get("n", 1)
        if n > 1 and constraint:
            logger.warning(
                f"Multiple completions (n={n}) requested - OutlinesJSON will return single completion only"
            )

        # Chat: ChatAdapter field-marker format
        # NOTE: Cannot delegate to ChatAdapter() because it has built-in JSONAdapter fallback
        # that would bypass our tier 2. Must manually replicate Adapter.__call__() logic.
        # Check dspy.adapters.base.Adapter.__call__ for updates with each DSPy release.
        outputs = None
        try:
            logger.info("Using DSPy ChatAdapter (field-marker format)")

            # 1. Preprocess (handles tool calling, signature modifications)
            processed_signature = self._call_preprocess(lm, lm_kwargs, signature, inputs)

            # 2. Format messages using parent's format method
            messages = self.format(processed_signature, demos, inputs)

            # 3. Generate and capture response
            outputs = lm(messages=messages, **lm_kwargs)

            # 4. Postprocess (handles list iteration, dict extraction, tool_calls, logprobs)
            result = self._call_postprocess(processed_signature, signature, outputs)

            self.metrics["chat_success"] += 1
            self.last_adapter_used = "chat"
            logger.info("Chat succeeded")
            return result

        except Exception as e:
            self.metrics["chat_failures"] += 1

            # Log the actual response for prompt optimization
            if outputs:
                response_text = outputs[0] if isinstance(outputs, list) else outputs
                if isinstance(response_text, dict):
                    response_text = response_text.get("text", str(response_text))
                response_preview = str(response_text)[:1000]
                logger.info(f"Chat failed: {e}")
                logger.info(f"Chat response: {response_preview}")
            else:
                logger.info(f"Chat failed: {e}")

            # Don't re-raise ContextWindowExceededError - no point in retrying
            if isinstance(e, ContextWindowExceededError):
                logger.warning(f"Context window exceeded: {e}")
                raise

        # JSON: Unconstrained JSON (delegate to stock JSONAdapter for 1-to-1 parity)
        try:
            logger.info("Using DSPy JSONAdapter")
            from dspy.adapters.json_adapter import JSONAdapter

            # Use stock JSONAdapter for 1-to-1 parity with DSPy defaults
            json_adapter = JSONAdapter()
            result = json_adapter(lm, lm_kwargs, signature, demos, inputs)

            self.metrics["json_success"] += 1
            self.last_adapter_used = "json"
            logger.info("JSON succeeded")
            return result
        except AdapterParseError as e:
            self.metrics["json_failures"] += 1
            logger.info(f"JSON failed: {e}")
        except Exception as e:
            # Catch other exceptions to prevent tier 3 from being skipped
            self.metrics["json_failures"] += 1
            logger.info(f"JSON failed with unexpected error: {e}")
            if isinstance(e, ContextWindowExceededError):
                logger.warning(f"Context window exceeded: {e}")
                raise

        # OutlinesJSON: Constrained JSON via Outlines
        if skip_tier3:
            raise AdapterParseError(
                adapter_name="OutlinesAdapter",
                signature=signature,
                lm_response="",
                message="All adapters failed and OutlinesJSON is skipped for ToolCalls",
            )

        if not constraint:
            raise AdapterParseError(
                adapter_name="OutlinesAdapter",
                signature=signature,
                lm_response="",
                message="No constraint found for OutlinesJSON constrained generation",
            )

        logger.info("Using Outlines JSONAdapter (Constrained Generation)")
        result = self._constrained_fallback(
            lm, lm_kwargs, signature, demos, inputs, constraint
        )
        self.metrics["outlines_json_success"] += 1
        self.last_adapter_used = "outlines_json"
        logger.info("OutlinesJSON succeeded")
        return result

    def _extract_constraint(self, signature: type[Signature]) -> Any:
        """Extract constraint from signature's output field."""
        output_fields = signature.output_fields

        if not output_fields:
            return None

        if len(output_fields) > 1:
            logger.warning(
                f"Multiple output fields in {signature.__name__}, using first"
            )

        # Get first output field's annotation (raw type)
        output_field = next(iter(output_fields.values()))
        return output_field.annotation

    def _has_tool_calls(self, signature: type[Signature]) -> bool:
        """Check if signature has ToolCalls output field."""
        for field in signature.output_fields.values():
            if field.annotation == ToolCalls:
                return True
        return False

    def _constrained_fallback(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        constraint: Any,
    ) -> list[dict[str, Any]]:
        """
        OutlinesJSON: Constrained JSON generation via Outlines (guaranteed valid).

        Adds _outlines_constraint to lm_kwargs, which OutlinesLM uses
        to enable constrained generation.
        """
        # Add constraint to kwargs
        output_field_name = next(iter(signature.output_fields.keys()))
        lm_kwargs["_outlines_constraint"] = constraint
        lm_kwargs["_outlines_field_name"] = output_field_name

        # Use JSONAdapter's format for proper JSON-formatted messages
        # (avoids conflicting field-marker instructions from ChatAdapter)
        from dspy.adapters.json_adapter import JSONAdapter
        json_adapter = JSONAdapter()
        messages = json_adapter.format(signature, demos, inputs)

        # Generate with constraint (messages already have JSON instructions from JSONAdapter)
        outputs = lm(messages=messages, **lm_kwargs)

        # Parse JSON response using JSONAdapter's parse method (guaranteed valid by Outlines)
        completions = []
        for output in outputs:
            # Standard DSPy output handling (see Adapter._call_postprocess)
            text = output
            if isinstance(output, dict):
                text = output["text"]

            # Use JSONAdapter's parse for 1-to-1 parity
            parsed = json_adapter.parse(signature, text)
            completions.append(parsed)

        return completions
