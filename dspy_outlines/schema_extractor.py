"""Extract Pydantic schemas from DSPy signatures."""

import logging
from typing import Type
from pydantic import BaseModel
import dspy

logger = logging.getLogger(__name__)

def extract_output_schema(signature: Type[dspy.Signature]) -> Type[BaseModel] | None:
    """
    Extract Pydantic model from DSPy signature's output field.

    Args:
        signature: DSPy Signature class

    Returns:
        Pydantic BaseModel class if found, None otherwise
    """
    # Get signature's output fields (dict of field_name -> field)
    output_fields = signature.output_fields

    if not output_fields:
        logger.warning(f"No output fields in signature {signature.__name__}")
        return None

    # For now, assume single output field (most common case)
    # Future: handle multiple output fields
    if len(output_fields) > 1:
        logger.warning(f"Multiple output fields in {signature.__name__}, using first")

    # Get first field from the dict
    output_field = next(iter(output_fields.values()))
    field_type = output_field.annotation

    # Check if it's a Pydantic model
    if isinstance(field_type, type) and issubclass(field_type, BaseModel):
        logger.info(f"Extracted schema: {field_type.__name__}")
        return field_type

    logger.warning(f"Output field is not a Pydantic model: {field_type}")
    return None
