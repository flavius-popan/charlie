"""Custom adapter that passes Pydantic schemas to OutlinesLM."""

from dspy.adapters import ChatAdapter
from .schema_extractor import extract_output_schema


class OutlinesAdapter(ChatAdapter):
    """
    Adapter that enables Outlines constrained generation by passing
    Pydantic schemas to the LM via lm_kwargs.

    This adapter intercepts DSPy signature calls and extracts Pydantic
    schemas from output fields, then passes them to OutlinesLM
    through the _outlines_schema keyword argument.
    """

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        schema = extract_output_schema(signature)

        if schema:
            output_field_name = next(iter(signature.output_fields.keys()))
            lm_kwargs['_outlines_schema'] = schema
            lm_kwargs['_outlines_field_name'] = output_field_name

        return super().__call__(lm, lm_kwargs, signature, demos, inputs)
