"""Entity summary generation module for Graphiti pipeline.

Two-Layer Architecture for DSPy Optimization:

1. SummaryGenerator (dspy.Module):
   - Pure LLM extraction logic, no DB/async dependencies
   - Optimizable with DSPy (MIPRO, GEPA, etc.)
   - Input: episode_content (str), previous_episodes (str), entity_name (str),
     entity_type (str), existing_summary (str), attributes (str)
   - Output: EntitySummary (Pydantic model with single field: summary: str)

2. GenerateSummaries (orchestrator):
   - Plain Python class that coordinates the full pipeline
   - Handles context building, summary truncation, metadata tracking
   - Accepts pre-compiled SummaryGenerator via dependency injection
   - Returns GenerateSummariesOutput with metadata

Usage:
    # Standard usage
    generator = GenerateSummaries(group_id="user_123")
    result = await generator(nodes=[...], episode=episode, previous_episodes=[...])

    # With optimization
    summary_generator = SummaryGenerator()
    compiled = optimizer.compile(summary_generator, trainset=examples)
    generator = GenerateSummaries(group_id="user_123", summary_generator=compiled)
    result = await generator(nodes=[...], episode=episode, previous_episodes=[...])

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
from graphiti_core.utils.text_utils import truncate_at_sentence, MAX_SUMMARY_CHARS
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EntitySummary(BaseModel):
    """Entity summary response model."""

    summary: str = Field(
        description="Concise factual summary under 250 characters combining new information with existing summary"
    )


class SummaryGenerationSignature(dspy.Signature):
    """Generate concise entity summaries from journal entries.

    Given an entity (name, type, existing summary, attributes) and journal context,
    generate a factual summary that combines new information with existing knowledge.

    Summary Guidelines:
    1. Output only factual content. Never explain what you're doing, why, or mention
       limitations/constraints.
    2. Only use the provided messages, entity, and entity context to generate summaries.
    3. Keep the summary concise and to the point. STATE FACTS DIRECTLY IN UNDER 250 CHARACTERS.

    Example summaries:
    BAD: "This is the only activity in the context. The user listened to this song.
          No other details were provided to include in this summary."
    GOOD: "User played 'Blue Monday' by New Order (electronic genre) on 2024-12-03 at 14:22 UTC."

    BAD: "Based on the messages provided, the user attended a meeting. This summary
          focuses on that event as it was the main topic discussed."
    GOOD: "User attended Q3 planning meeting with sales team on March 15."

    BAD: "The context shows John ordered pizza. Due to length constraints, other
          details are omitted from this summary."
    GOOD: "John ordered pepperoni pizza from Mario's at 7:30 PM, delivered to office."
    """

    episode_content: str = dspy.InputField(
        desc="Current journal entry text"
    )
    previous_episodes: str = dspy.InputField(
        desc="Previous journal entries as JSON list for additional context"
    )
    entity_name: str = dspy.InputField(
        desc="Name of the entity to generate summary for"
    )
    entity_type: str = dspy.InputField(
        desc="Type of entity (e.g., Person, Emotion, Entity)"
    )
    existing_summary: str = dspy.InputField(
        desc="Current entity summary (may be empty) - already truncated to 250 chars"
    )
    attributes: str = dspy.InputField(
        desc="Entity attributes as JSON dict (may be empty)"
    )

    summary: EntitySummary = dspy.OutputField(
        desc="Concise factual summary combining new information with existing summary. State facts directly in under 250 characters. No meta-commentary."
    )


@dataclass
class GenerateSummariesOutput:
    """Results from summary generation."""

    nodes: list[EntityNode]  # Same nodes with summaries populated
    metadata: dict[str, Any]


class SummaryGenerator(dspy.Module):
    """Pure LLM extraction module for entity summaries - optimizable with DSPy.

    This module ONLY performs LLM-based summary generation with no dependencies
    on databases, async operations, or entity resolution logic. This design makes it
    ideal for DSPy optimization (MIPRO, GEPA, etc.).

    Usage:
        generator = SummaryGenerator()
        entity_summary = generator(
            episode_content="I had coffee with Sarah...",
            previous_episodes='["Yesterday I met Sarah at the cafe."]',
            entity_name="Sarah",
            entity_type="Person",
            existing_summary="Friend who lives in SF.",
            attributes='{"relationship_type": "friend"}'
        )
    """

    def __init__(self):
        super().__init__()
        self.generator = dspy.Predict(SummaryGenerationSignature)

        prompt_path = Path(__file__).parent / "prompts" / "generate_summaries.json"
        if prompt_path.exists():
            self.load(str(prompt_path))
            logger.info("Loaded optimized prompts from %s", prompt_path)

    def forward(
        self,
        episode_content: str,
        previous_episodes: str,
        entity_name: str,
        entity_type: str,
        existing_summary: str,
        attributes: str,
    ) -> EntitySummary:
        """Generate summary for an entity from text content.

        Args:
            episode_content: Current journal entry text
            previous_episodes: JSON string of previous episode contents
            entity_name: Name of entity to generate summary for
            entity_type: Type of entity (e.g., "Person", "Emotion", "Entity")
            existing_summary: Current summary (already truncated to 250 chars)
            attributes: JSON string of entity attributes

        Returns:
            EntitySummary with generated summary text
        """
        result = self.generator(
            episode_content=episode_content,
            previous_episodes=previous_episodes,
            entity_name=entity_name,
            entity_type=entity_type,
            existing_summary=existing_summary,
            attributes=attributes,
        )

        return result.summary


class GenerateSummaries:
    """Generate entity summaries from episodes.

    Orchestrator that coordinates:
    1. Context building (previous episodes as JSON, existing summary truncation)
    2. Summary generation via SummaryGenerator (LLM)
    3. Summary truncation at sentence boundary (250 chars)
    4. Node summary field updates
    5. Metadata aggregation (nodes processed, avg length, truncation stats)

    This is a plain Python class (NOT dspy.Module) to separate orchestration
    from the optimizable LLM logic in SummaryGenerator.

    Usage:
        # Standard usage with default generator
        generator = GenerateSummaries(group_id="user_123")
        result = await generator(
            nodes=[...],
            episode=episode,
            previous_episodes=[...]
        )

        # With pre-optimized SummaryGenerator
        summary_generator = SummaryGenerator()
        compiled = optimizer.compile(summary_generator, trainset=examples)
        generator = GenerateSummaries(group_id="user_123", summary_generator=compiled)
        result = await generator(nodes=[...], episode=episode, previous_episodes=[...])
    """

    def __init__(
        self,
        group_id: str,
        summary_generator: SummaryGenerator | None = None,
    ):
        self.group_id = group_id
        self.summary_generator = summary_generator or SummaryGenerator()

    async def __call__(
        self,
        nodes: list[EntityNode],
        episode: EpisodicNode,
        previous_episodes: list[EpisodicNode],
    ) -> GenerateSummariesOutput:
        """Generate and populate entity summaries from journal entries.

        Follows graphiti-core's pattern: generate summaries one entity at a time
        with context from current and previous episodes.

        Args:
            nodes: Entity nodes to generate summaries for (with attributes populated)
            episode: Current episode being processed
            previous_episodes: Previous episodes for context

        Returns:
            GenerateSummariesOutput with nodes (summaries populated) and metadata
        """
        logger.info(
            "Generating summaries for %d nodes from episode %s",
            len(nodes),
            episode.uuid,
        )

        previous_episodes_json = json.dumps([ep.content for ep in previous_episodes])

        nodes_processed = 0
        total_summary_length = 0
        truncated_count = 0

        for node in nodes:
            entity_type_name = next((item for item in node.labels if item != 'Entity'), 'Entity')

            logger.info(
                "Generating summary for %s entity: %s",
                entity_type_name,
                node.name,
            )

            existing_summary_truncated = truncate_at_sentence(
                node.summary, MAX_SUMMARY_CHARS
            )

            attributes_json = json.dumps(node.attributes)

            generated_summary = self.summary_generator(
                episode_content=episode.content,
                previous_episodes=previous_episodes_json,
                entity_name=node.name,
                entity_type=entity_type_name,
                existing_summary=existing_summary_truncated,
                attributes=attributes_json,
            )

            summary_text = generated_summary.summary

            original_length = len(summary_text)
            truncated_summary = truncate_at_sentence(summary_text, MAX_SUMMARY_CHARS)
            final_length = len(truncated_summary)

            if final_length < original_length:
                truncated_count += 1
                logger.debug(
                    "Truncated summary for %s: %d → %d chars",
                    node.name,
                    original_length,
                    final_length,
                )

            node.summary = truncated_summary
            nodes_processed += 1
            total_summary_length += final_length

            logger.debug(
                "Generated summary for %s: %s",
                node.name,
                truncated_summary,
            )

        # TODO: Generate name embeddings after summary generation (local Qwen embedder)
        # This will enable semantic search over entity names for better deduplication
        # Integration point: call graphiti_core's create_entity_node_name_embedding(embedder, nodes)

        avg_summary_length = (
            total_summary_length / nodes_processed if nodes_processed > 0 else 0
        )

        metadata = {
            "nodes_processed": nodes_processed,
            "avg_summary_length": round(avg_summary_length, 2),
            "truncated_count": truncated_count,
        }

        logger.info(
            "Summary generation complete: %d processed, avg length: %.2f chars, %d truncated",
            nodes_processed,
            avg_summary_length,
            truncated_count,
        )

        return GenerateSummariesOutput(nodes=nodes, metadata=metadata)


__all__ = [
    "SummaryGenerator",
    "GenerateSummaries",
    "GenerateSummariesOutput",
]
