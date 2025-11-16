"""Entity summary generation module for Graphiti pipeline.

Two-Layer Architecture for DSPy Optimization:

1. SummaryGenerator (dspy.Module):
   - Pure LLM extraction logic, no DB/async dependencies
   - Optimizable with DSPy (MIPRO, GEPA, etc.)
   - Input: `summary_context` JSON string matching graphiti-core's `_build_episode_context`
   - Output: EntitySummary (Pydantic model with single field: summary: str)

2. GenerateSummaries (orchestrator):
   - Plain Python class that coordinates the full pipeline
   - Handles graphiti-core-compatible context building, summary truncation, metadata tracking
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

from pipeline.self_reference import SELF_ENTITY_UUID

logger = logging.getLogger(__name__)


def build_node_payload(
    *,
    name: str,
    summary: str,
    labels: list[str],
    attributes: dict[str, Any],
) -> dict[str, Any]:
    """Return node metadata matching graphiti-core's extract_summary context."""

    return {
        "name": name,
        "summary": truncate_at_sentence(summary, MAX_SUMMARY_CHARS),
        "entity_types": labels,
        "attributes": attributes,
    }


def build_summary_context(
    *,
    node_payload: dict[str, Any],
    episode_content: str,
    previous_episode_texts: list[str] | None = None,
) -> dict[str, Any]:
    """Mirror graphiti-core's _build_episode_context helper."""

    return {
        "node": node_payload,
        "episode_content": episode_content,
        "previous_episodes": previous_episode_texts or [],
    }


class EntitySummary(BaseModel):
    """Entity summary response model."""

    summary: str = Field(
        description="Concise factual summary under 250 characters combining new information with existing summary"
    )


class SummaryGenerationSignature(dspy.Signature):
    """Generate concise entity summaries from journal context."""

    summary_context: str = dspy.InputField(desc="entity and journal context")

    summary: EntitySummary = dspy.OutputField(
        desc="concise summary of the entity"
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
            summary_context=json.dumps(
                build_summary_context(
                    node_payload=build_node_payload(
                        name="Sarah",
                        summary="Friend who lives in SF.",
                        labels=["Entity", "Person"],
                        attributes={"relationship_type": "friend"},
                    ),
                    episode_content="I had coffee with Sarah...",
                    previous_episode_texts=["Yesterday I met Sarah at the cafe."],
                )
            )
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
        summary_context: str,
    ) -> EntitySummary:
        """Generate summary for an entity from text content.

        Args:
            summary_context: JSON dict mirroring graphiti-core extract_summary context

        Returns:
            EntitySummary with generated summary text
        """
        result = self.generator(
            summary_context=summary_context,
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

        Follows graphiti-core's pattern: build identical summary contexts and
        generate summaries one entity at a time with shared DSPy prompts.

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

        previous_episode_texts = [ep.content for ep in previous_episodes]

        nodes_processed = 0
        total_summary_length = 0
        truncated_count = 0

        # MLX inference currently serializes LLM generations, so this loop stays sequential.
        for node in nodes:
            if node.uuid == str(SELF_ENTITY_UUID):
                logger.debug("Skipping SELF entity for summary generation")
                continue
            entity_type_name = next((item for item in node.labels if item != 'Entity'), 'Entity')

            logger.info(
                "Generating summary for %s entity: %s",
                entity_type_name,
                node.name,
            )

            node_payload = build_node_payload(
                name=node.name,
                summary=node.summary,
                labels=node.labels,
                attributes=node.attributes,
            )
            summary_context = build_summary_context(
                node_payload=node_payload,
                episode_content=episode.content,
                previous_episode_texts=previous_episode_texts,
            )

            generated_summary = self.summary_generator(
                summary_context=json.dumps(summary_context, ensure_ascii=False),
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
    "build_node_payload",
    "build_summary_context",
]
