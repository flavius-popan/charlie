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

from pipeline import _dspy_setup  # noqa: F401
import dspy
from graphiti_core.nodes import EntityNode, EpisodicNode
from pydantic import BaseModel

from pipeline.entity_edge_models import entity_types
from pipeline.self_reference import SELF_ENTITY_UUID

logger = logging.getLogger(__name__)


def _build_attribute_signature(
    entity_type: str, response_model: type[BaseModel]
) -> type[dspy.Signature]:
    """Create a signature class for a specific entity type."""

    safe_name = "".join(ch for ch in (entity_type or response_model.__name__) if ch.isalnum()) or "Generic"
    class_name = f"{safe_name}AttributeSignature"
    doc = f"Extract {entity_type or response_model.__name__} attributes from journal entries."

    annotations = {
        "episode_content": str,
        "previous_episodes": str,
        "entity_name": str,
        "entity_type": str,
        "existing_attributes": str,
        "attributes": response_model,
    }

    attrs = {
        "__doc__": doc,
        "__annotations__": annotations,
        "episode_content": dspy.InputField(desc="journal entry text"),
        "previous_episodes": dspy.InputField(desc="prior entries as JSON"),
        "entity_name": dspy.InputField(desc="entity name"),
        "entity_type": dspy.InputField(desc="entity type"),
        "existing_attributes": dspy.InputField(desc="current attributes as JSON"),
        "attributes": dspy.OutputField(desc="extracted attributes"),
    }

    return type(class_name, (dspy.Signature,), attrs)


@dataclass
class ExtractAttributesOutput:
    """Results from attribute extraction."""

    nodes: list[EntityNode]  # Nodes with attributes populated
    metadata: dict[str, Any]
    raw_llm_outputs: list[dict[str, Any]] = None  # Raw LLM responses for each entity (one per node)


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

    def __init__(self, entity_type_models: dict[str, type[BaseModel]] | None = None):
        super().__init__()
        self._predictors: dict[str, dspy.Module] = {}
        self._signatures: dict[str, type[dspy.Signature]] = {}
        self._entity_type_models: dict[str, type[BaseModel]] = {}

        source_models = entity_type_models or entity_types
        for type_name, model in source_models.items():
            self._register_entity_model(type_name, model)

        for type_name, model in list(self._entity_type_models.items()):
            self._get_or_create_predictor(type_name, model)

        prompt_dir = Path(__file__).parent / "prompts"
        candidate_paths = [
            prompt_dir / "extract_attributes.json",
            prompt_dir / "extract_attributes.pkl",
        ]
        for prompt_path in candidate_paths:
            if prompt_path.exists():
                self.load(str(prompt_path))
                logger.info("Loaded optimized prompts from %s", prompt_path)
                break

    def forward(
        self,
        episode_content: str,
        previous_episodes: str,
        entity_name: str,
        entity_type: str,
        existing_attributes: str,
        response_model: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        """Extract attributes for an entity from text content.

        Args:
            episode_content: Current journal entry text
            previous_episodes: JSON string of previous episode contents
            entity_name: Name of entity to extract attributes for
            entity_type: Type of entity (e.g., "Person", "Activity", "Organization", "Place")
            existing_attributes: JSON string of current attributes
            response_model: Optional override for the entity schema

        Returns:
            Dictionary of extracted attributes matching response_model schema
        """
        model = self._resolve_response_model(entity_type, response_model)
        predictor = self._get_or_create_predictor(entity_type, model)

        result = predictor(
            episode_content=episode_content,
            previous_episodes=previous_episodes,
            entity_name=entity_name,
            entity_type=entity_type,
            existing_attributes=existing_attributes,
        )

        attributes = getattr(result, "attributes", result)

        if isinstance(attributes, BaseModel):
            return attributes.model_dump(exclude_none=True)
        if isinstance(attributes, dict):
            return {k: v for k, v in attributes.items() if v is not None}
        return attributes

    def _resolve_response_model(
        self, entity_type: str, response_model: type[BaseModel] | str | None
    ) -> type[BaseModel]:
        """Resolve the schema to use for the given entity."""
        model: type[BaseModel] | None = None

        if response_model is not None:
            if isinstance(response_model, str):
                model = self._entity_type_models.get(response_model)
            else:
                model = response_model

        if model is None and entity_type:
            model = self._entity_type_models.get(entity_type)

        if model is None:
            raise ValueError(
                f"No attribute schema registered for entity type '{entity_type}'. "
                "Provide `response_model` or register the type before invoking AttributeExtractor."
            )

        self._register_entity_model(entity_type or model.__name__, model)
        return model

    def _register_entity_model(self, entity_type: str, model: type[BaseModel]) -> None:
        """Register mappings for an entity type and its schema."""
        key = entity_type or model.__name__
        self._entity_type_models[key] = model
        self._entity_type_models.setdefault(model.__name__, model)

    def _get_or_create_predictor(
        self, entity_type: str, response_model: type[BaseModel]
    ) -> dspy.Module:
        """Retrieve or build a predictor for the requested entity type."""
        key = entity_type or response_model.__name__
        if key in self._predictors:
            return self._predictors[key]

        signature = _build_attribute_signature(key, response_model)
        predictor = dspy.ChainOfThought(signature)
        attr_name = self._predictor_attr_name(key)
        setattr(self, attr_name, predictor)
        self._predictors[key] = predictor
        self._signatures[key] = signature
        return predictor

    @staticmethod
    def _predictor_attr_name(key: str) -> str:
        safe = "".join(ch if ch.isalnum() else "_" for ch in (key or "generic"))
        return f"{safe.lower()}_predictor"


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
        raw_llm_outputs = []  # Collect raw LLM outputs for each entity

        for node in nodes:
            if node.uuid == str(SELF_ENTITY_UUID):
                logger.debug('Skipping author entity "I" for attribute extraction')
                nodes_skipped += 1
                continue
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

            # Capture raw LLM output for this entity
            raw_llm_outputs.append({
                "entity_name": node.name,
                "entity_type": entity_type_name,
                "attributes": extracted_attributes,
            })

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

        return ExtractAttributesOutput(nodes=nodes, metadata=metadata, raw_llm_outputs=raw_llm_outputs)


__all__ = [
    "AttributeExtractor",
    "ExtractAttributes",
    "ExtractAttributesOutput",
]
