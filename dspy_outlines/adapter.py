"""Three-tier fallback adapter: ChatAdapter → JSON → Outlines constrained."""

import json
import logging
from typing import Any
import regex
import json_repair

from dspy.adapters import ChatAdapter
from dspy.adapters.types.tool import ToolCalls
from dspy.clients.lm import LM
from dspy.signatures.signature import Signature
from dspy.utils.exceptions import AdapterParseError
from dspy.adapters.utils import parse_value

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
        self.metrics = {
            'chat_success': 0,
            'json_success': 0,
            'outlines_json_success': 0,
            'chat_failures': 0,
            'json_failures': 0,
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
            logger.warning("ToolCalls detected - skipping OutlinesJSON (Outlines doesn't support ToolCalls)")

        # Check for multiple completions
        n = lm_kwargs.get('n', 1)
        if n > 1 and constraint:
            logger.warning(f"Multiple completions (n={n}) requested - OutlinesJSON will return single completion only")

        # Chat: ChatAdapter field-marker format
        try:
            logger.info("Attempting Chat: ChatAdapter field-marker format")
            result = super().__call__(lm, lm_kwargs, signature, demos, inputs)
            self.metrics['chat_success'] += 1
            logger.info("Chat succeeded")
            return result
        except Exception as e:
            self.metrics['chat_failures'] += 1
            logger.info(f"Chat failed: {e}")

            # Don't re-raise ContextWindowExceededError - no point in retrying
            if "ContextWindowExceededError" in str(type(e)):
                raise

        # JSON: Unconstrained JSON with json_repair
        try:
            logger.info("Attempting JSON: Unconstrained JSON with json_repair")
            result = self._json_fallback(lm, lm_kwargs, signature, demos, inputs)
            self.metrics['json_success'] += 1
            logger.info("JSON succeeded")
            return result
        except AdapterParseError as e:
            self.metrics['json_failures'] += 1
            logger.info(f"JSON failed: {e}")

        # OutlinesJSON: Constrained JSON via Outlines
        if skip_tier3:
            raise AdapterParseError(
                adapter_name="OutlinesAdapter",
                signature=signature,
                lm_response="",
                message="All adapters failed and OutlinesJSON is skipped for ToolCalls"
            )

        if not constraint:
            raise AdapterParseError(
                adapter_name="OutlinesAdapter",
                signature=signature,
                lm_response="",
                message="No constraint found for OutlinesJSON constrained generation"
            )

        logger.info("Attempting OutlinesJSON: Constrained JSON via Outlines")
        result = self._constrained_fallback(lm, lm_kwargs, signature, demos, inputs, constraint)
        self.metrics['outlines_json_success'] += 1
        logger.info("OutlinesJSON succeeded")
        return result

    def _extract_constraint(self, signature: type[Signature]) -> Any:
        """Extract constraint from signature's output field."""
        output_fields = signature.output_fields

        if not output_fields:
            return None

        if len(output_fields) > 1:
            logger.warning(f"Multiple output fields in {signature.__name__}, using first")

        # Get first output field's annotation (raw type)
        output_field = next(iter(output_fields.values()))
        return output_field.annotation

    def _has_tool_calls(self, signature: type[Signature]) -> bool:
        """Check if signature has ToolCalls output field."""
        for field in signature.output_fields.values():
            if field.annotation == ToolCalls:
                return True
        return False

    def _json_fallback(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        JSON: Unconstrained JSON format with json_repair parsing.

        Uses ChatAdapter's format() to create messages, adds JSON instruction,
        then parses with json_repair for robustness.
        """
        # Format messages using parent's format method
        messages = self.format(signature, demos, inputs)

        # Add JSON instruction to last user message
        json_instruction = self._user_message_output_requirements(signature)
        if messages and messages[-1].get('role') == 'user':
            messages[-1]['content'] += f"\n\n{json_instruction}"

        # Generate (unconstrained)
        outputs = lm(messages=messages, **lm_kwargs)

        # Parse JSON response
        completions = []
        for output in outputs:
            completion_text = output.message.content
            parsed = self._parse_json(signature, completion_text)
            completions.append(parsed)

        return completions

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
        lm_kwargs['_outlines_constraint'] = constraint
        lm_kwargs['_outlines_field_name'] = output_field_name

        # Format messages using parent's format method
        messages = self.format(signature, demos, inputs)

        # Add JSON instruction to last user message
        json_instruction = self._user_message_output_requirements(signature)
        if messages and messages[-1].get('role') == 'user':
            messages[-1]['content'] += f"\n\n{json_instruction}"

        # Generate with constraint
        outputs = lm(messages=messages, **lm_kwargs)

        # Parse JSON response (guaranteed valid by Outlines)
        completions = []
        for output in outputs:
            completion_text = output.message.content
            parsed = self._parse_json(signature, completion_text)
            completions.append(parsed)

        return completions

    def _user_message_output_requirements(self, signature: type[Signature]) -> str:
        """Create JSON output instruction for user message."""
        field_names = list(signature.output_fields.keys())
        message = "Respond with a JSON object in the following order of fields: "
        message += ", then ".join(f"`{f}`" for f in field_names)
        message += "."
        return message

    def _parse_json(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        """
        Parse JSON completion using json_repair for robustness.

        Raises AdapterParseError if parsing fails or fields don't match.
        """
        # Extract JSON object using regex
        pattern = r"\{(?:[^{}]|(?R))*\}"
        match = regex.search(pattern, completion, regex.DOTALL)
        if match:
            completion = match.group(0)

        # Parse with json_repair for robustness
        try:
            fields = json_repair.loads(completion)
        except Exception as e:
            raise AdapterParseError(
                adapter_name="OutlinesAdapter",
                signature=signature,
                lm_response=completion,
                message=f"Failed to parse JSON: {e}"
            )

        if not isinstance(fields, dict):
            raise AdapterParseError(
                adapter_name="OutlinesAdapter",
                signature=signature,
                lm_response=completion,
                message="LM response cannot be serialized to a JSON object."
            )

        # Filter to output fields only
        fields = {k: v for k, v in fields.items() if k in signature.output_fields}

        # Parse each field value to its annotation type
        for k, v in fields.items():
            if k in signature.output_fields:
                fields[k] = parse_value(v, signature.output_fields[k].annotation)

        # Verify all output fields are present
        if fields.keys() != signature.output_fields.keys():
            raise AdapterParseError(
                adapter_name="OutlinesAdapter",
                signature=signature,
                lm_response=completion,
                parsed_result=fields,
            )

        return fields
