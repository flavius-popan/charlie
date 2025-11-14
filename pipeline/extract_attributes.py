"""Attribute extraction module for Graphiti pipeline.

Two-Layer Architecture for DSPy Optimization:

1. AttributeExtractor (dspy.Module):
   - Pure LLM extraction logic, no DB/async dependencies
   - Optimizable with DSPy (MIPRO, GEPA, etc.)
   - Input: episode_content (str), previous_episodes (str), entity_name (str), entity_type (str), existing_attributes (str)
   - Output: Dynamic Pydantic model based on entity type

2. ExtractAttributes (orchestrator):
   - Plain Python class that coordinates the full pipeline
   - Handles entity type resolution, context building, attribute merging
   - Accepts pre-compiled AttributeExtractor via dependency injection
   - Returns ExtractAttributesOutput with metadata

Usage:
    # Standard usage
    extractor = ExtractAttributes(group_id="user_123")
    result = await extractor(nodes=[...], episode=episode, previous_episodes=[...])

    # With optimization
    attribute_extractor = AttributeExtractor()
    compiled = optimizer.compile(attribute_extractor, trainset=examples)
    extractor = ExtractAttributes(group_id="user_123", attribute_extractor=compiled)
    result = await extractor(nodes=[...], episode=episode, previous_episodes=[...])

This pattern enables fast optimization by isolating LLM calls from orchestration logic.
"""

from __future__ import annotations
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

import dspy
from graphiti_core.nodes import EntityNode, EpisodicNode
from pydantic import BaseModel

from pipeline.entity_edge_models import entity_types

logger = logging.getLogger(__name__)


class AttributeExtractionSignature(dspy.Signature):
    """Extract type-specific attributes for an entity from journal context."""

    episode_content: str = dspy.InputField(desc="current journal entry")
    previous_episodes: str = dspy.InputField(desc="previous entries for context")
    entity_name: str = dspy.InputField(desc="entity to extract attributes for")
    entity_type: str = dspy.InputField(desc="type of entity")
    existing_attributes: str = dspy.InputField(desc="current entity attributes")

    # Output field is dynamically set based on entity type in forward()


@dataclass
class ExtractAttributesOutput:
    """Results from attribute extraction."""

    nodes: list[EntityNode]  # Nodes with attributes populated
    metadata: dict[str, Any]


class AttributeExtractor(dspy.Module):
    """Pure LLM extraction module for entity attributes - optimizable with DSPy.

    This module ONLY performs LLM-based attribute extraction with no dependencies
    on databases, async operations, or entity resolution logic. This design makes it
    ideal for DSPy optimization (MIPRO, GEPA, etc.).

    Usage:
        extractor = AttributeExtractor()
        attributes = extractor(
            episode_content="I had coffee with Sarah...",
            previous_episodes='["Yesterday I met Sarah at the cafe."]',
            entity_name="Sarah",
            entity_type="Person",
            existing_attributes='{"relationship_type": "friend"}',
            response_model=Person
        )
    """

    def __init__(self):
        super().__init__()
        self.extractor = dspy.Predict(AttributeExtractionSignature)

        prompt_path = Path(__file__).parent / "prompts" / "extract_attributes.json"
        if prompt_path.exists():
            self.load(str(prompt_path))
            logger.info("Loaded optimized prompts from %s", prompt_path)

    def forward(
        self,
        episode_content: str,
        previous_episodes: str,
        entity_name: str,
        entity_type: str,
        existing_attributes: str,
        response_model: type[BaseModel],
    ) -> dict[str, Any]:
        """Extract attributes for an entity from text content.

        Args:
            episode_content: Current journal entry text
            previous_episodes: JSON string of previous episode contents
            entity_name: Name of entity to extract attributes for
            entity_type: Type of entity (e.g., "Person", "Activity", "Organization", "Place")
            existing_attributes: JSON string of current attributes
            response_model: Pydantic model defining expected attributes

        Returns:
            Dictionary of extracted attributes matching response_model schema
        """
        # Create a dynamic signature class with the specific response model
        # Using type() to create a new class with proper annotations at runtime
        # This allows OutlinesAdapter to extract the Pydantic constraint correctly
        DynamicSignature = type(
            f'AttributeExtractionSignature_{response_model.__name__}',
            (dspy.Signature,),
            {
                '__doc__': f"Extract {response_model.__name__} attributes from journal entries.",
                '__annotations__': {
                    'episode_content': str,
                    'previous_episodes': str,
                    'entity_name': str,
                    'entity_type': str,
                    'existing_attributes': str,
                    'attributes': response_model,  # Dynamic Pydantic model
                },
                'episode_content': dspy.InputField(desc="Current journal entry text"),
                'previous_episodes': dspy.InputField(
                    desc="Previous journal entries as JSON list for additional context"
                ),
                'entity_name': dspy.InputField(
                    desc="Name of the entity to extract attributes for"
                ),
                'entity_type': dspy.InputField(
                    desc="Type of entity (e.g., Person, Activity, Organization, Place)"
                ),
                'existing_attributes': dspy.InputField(
                    desc="Current entity attributes as JSON dict (may be empty)"
                ),
                'attributes': dspy.OutputField(
                    desc=f"Extracted {entity_type} attributes based on schema. Only extract information explicitly present in messages."
                ),
            }
        )

        # Create a predictor with the dynamic signature
        predictor = dspy.Predict(DynamicSignature)

        result = predictor(
            episode_content=episode_content,
            previous_episodes=previous_episodes,
            entity_name=entity_name,
            entity_type=entity_type,
            existing_attributes=existing_attributes,
        )

        # Return the attributes as a dict
        return result.attributes.model_dump(exclude_none=True)


class ExtractAttributes:
    """Extract entity attributes from episodes.

    Orchestrator that coordinates:
    1. Entity type resolution from node labels
    2. Schema validation (skip if no attributes defined)
    3. Context building (previous episodes as JSON)
    4. Attribute extraction via AttributeExtractor (LLM)
    5. Attribute merging (preserve existing attributes)
    6. Metadata aggregation

    This is a plain Python class (NOT dspy.Module) to separate orchestration
    from the optimizable LLM logic in AttributeExtractor.

    Usage:
        # Standard usage with default extractor
        extractor = ExtractAttributes(group_id="user_123")
        result = await extractor(
            nodes=[...],
            episode=episode,
            previous_episodes=[...],
            entity_types=entity_types
        )

        # With pre-optimized AttributeExtractor
        attribute_extractor = AttributeExtractor()
        compiled = optimizer.compile(attribute_extractor, trainset=examples)
        extractor = ExtractAttributes(group_id="user_123", attribute_extractor=compiled)
        result = await extractor(nodes=[...], episode=episode, previous_episodes=[...])
    """

    def __init__(
        self,
        group_id: str,
        attribute_extractor: AttributeExtractor | None = None,
    ):
        self.group_id = group_id
        self.attribute_extractor = attribute_extractor or AttributeExtractor()

    async def __call__(
        self,
        nodes: list[EntityNode],
        episode: EpisodicNode,
        previous_episodes: list[EpisodicNode],
        entity_types: dict[str, type[BaseModel]] | None = None,
    ) -> ExtractAttributesOutput:
        """Extract and populate entity attributes from journal entries.

        Follows graphiti-core's pattern: extract attributes one entity at a time
        with context from current and previous episodes.

        Args:
            nodes: Entity nodes to extract attributes for
            episode: Current episode being processed
            previous_episodes: Previous episodes for context
            entity_types: Custom entity type schemas (defaults to Person, Place, Organization, Activity)

        Returns:
            ExtractAttributesOutput with nodes (attributes populated) and metadata
        """
        if entity_types is None:
            from pipeline.entity_edge_models import entity_types as default_entity_types
            entity_types = default_entity_types
        elif not entity_types:
            entity_types = {}

        logger.info(
            "Extracting attributes for %d nodes from episode %s",
            len(nodes),
            episode.uuid,
        )

        previous_episodes_json = json.dumps([ep.content for ep in previous_episodes])

        nodes_processed = 0
        nodes_skipped = 0
        attributes_extracted_by_type: dict[str, int] = {}

        for node in nodes:
            entity_type_name = next((item for item in node.labels if item != 'Entity'), '')

            if not entity_type_name or entity_type_name not in entity_types:
                logger.debug(
                    "Skipping node %s (%s) - no custom entity type or not in entity_types",
                    node.name,
                    entity_type_name,
                )
                nodes_skipped += 1
                continue

            entity_model = entity_types[entity_type_name]

            if len(entity_model.model_fields) == 0:
                logger.debug(
                    "Skipping node %s (%s) - entity type has no attributes",
                    node.name,
                    entity_type_name,
                )
                nodes_skipped += 1
                continue

            logger.info(
                "Extracting %s attributes for entity: %s",
                entity_type_name,
                node.name,
            )

            existing_attributes_json = json.dumps(node.attributes)

            extracted_attributes = self.attribute_extractor(
                episode_content=episode.content,
                previous_episodes=previous_episodes_json,
                entity_name=node.name,
                entity_type=entity_type_name,
                existing_attributes=existing_attributes_json,
                response_model=entity_model,
            )

            entity_model(**extracted_attributes)

            node.attributes.update(extracted_attributes)

            nodes_processed += 1
            attributes_extracted_by_type[entity_type_name] = (
                attributes_extracted_by_type.get(entity_type_name, 0) + 1
            )

            logger.debug(
                "Extracted attributes for %s: %s",
                node.name,
                extracted_attributes,
            )

        # TODO: Generate name embeddings when local Qwen embedder is ready
        # This will enable semantic search over entity names for better deduplication
        # Integration point: call graphiti_core's create_entity_node_name_embedding(embedder, nodes)

        metadata = {
            "nodes_processed": nodes_processed,
            "nodes_skipped": nodes_skipped,
            "attributes_extracted_by_type": attributes_extracted_by_type,
        }

        logger.info(
            "Attribute extraction complete: %d processed, %d skipped. By type: %s",
            nodes_processed,
            nodes_skipped,
            attributes_extracted_by_type,
        )

        return ExtractAttributesOutput(nodes=nodes, metadata=metadata)


__all__ = [
    "AttributeExtractor",
    "ExtractAttributes",
    "ExtractAttributesOutput",
]
