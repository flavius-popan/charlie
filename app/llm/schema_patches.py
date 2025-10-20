"""
Schema patches for Graphiti prompts.

Applies runtime constraints to prevent runaway LLM generation,
particularly for entity extraction which can loop infinitely on small models.
"""

import logging
from pydantic import Field, field_validator
from graphiti_core.prompts import extract_nodes

logger = logging.getLogger(__name__)


# Reasonable limits for entity extraction
MAX_ENTITIES_PER_EXTRACTION = 50
MAX_ENTITY_NAME_LENGTH = 200


def apply_entity_extraction_limits():
    """
    Monkey-patch ExtractedEntities schema to limit entity count.

    Problem:
    Small models can get stuck in loops generating hundreds/thousands
    of similar entities (e.g., "desire to feel X" repeated endlessly),
    eventually running out of tokens and producing truncated invalid JSON.

    Solution:
    Add max_length constraint to the entity list to force the model
    to stop after a reasonable number of entities.
    """
    # Save original class
    OriginalExtractedEntity = extract_nodes.ExtractedEntity
    OriginalExtractedEntities = extract_nodes.ExtractedEntities

    # Create constrained version with validation
    class ConstrainedExtractedEntity(OriginalExtractedEntity):
        """ExtractedEntity with length constraints."""

        @field_validator('name')
        @classmethod
        def validate_name_length(cls, v: str) -> str:
            """Prevent extremely long entity names."""
            if len(v) > MAX_ENTITY_NAME_LENGTH:
                return v[:MAX_ENTITY_NAME_LENGTH]
            return v

    class ConstrainedExtractedEntities(OriginalExtractedEntities):
        """ExtractedEntities with count limits."""
        extracted_entities: list[ConstrainedExtractedEntity] = Field(
            ...,
            description=f'List of extracted entities (maximum {MAX_ENTITIES_PER_EXTRACTION})',
            max_length=MAX_ENTITIES_PER_EXTRACTION
        )

        @field_validator('extracted_entities')
        @classmethod
        def log_if_at_limit(cls, v: list) -> list:
            """Warn if we hit the entity limit."""
            if len(v) >= MAX_ENTITIES_PER_EXTRACTION:
                logger.warning(
                    f"Entity extraction hit maximum limit of {MAX_ENTITIES_PER_EXTRACTION}. "
                    f"This may indicate model repetition or genuinely complex content. "
                    f"Entity names: {[e.name for e in v[:5]]}... (showing first 5)"
                )
            return v

    # Replace in module
    extract_nodes.ExtractedEntity = ConstrainedExtractedEntity
    extract_nodes.ExtractedEntities = ConstrainedExtractedEntities

    return ConstrainedExtractedEntities


def apply_all_patches():
    """Apply all schema patches before initializing Graphiti."""
    apply_entity_extraction_limits()
