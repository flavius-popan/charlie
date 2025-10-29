"""Knowledge graph extraction DSPy module."""

from __future__ import annotations

from typing import Any, List, Optional

import dspy
from pydantic import BaseModel, Field


class Node(BaseModel):
    """Pydantic model representing a graph node."""

    id: int = Field(description="Unique identifier for the node")
    label: str = Field(
        description=(
            "Name of the entity (person, place, or concept). Include 'Author' or 'I' "
            "ONLY when the journal explicitly involves first-person experiences "
            "(direct interactions, feelings, thoughts)."
        )
    )
    properties: dict = Field(default_factory=dict)


class Edge(BaseModel):
    """Pydantic model representing a graph edge."""

    source: int = Field(description="Source node ID")
    target: int = Field(description="Target node ID")
    label: str = Field(
        description=(
            "Relationship type in PRESENT TENSE (e.g., 'knows', 'works_with', "
            "'feels_anxious_about', 'admires'). Use underscores between words. "
            "Max 3 words. Connect to Author node ONLY for direct interactions or "
            "emotional/subjective states."
        )
    )
    properties: dict = Field(default_factory=dict)


class KnowledgeGraph(BaseModel):
    """Graph wrapper with nodes and edges."""

    nodes: List[Node] = Field(
        description=(
            "All entities mentioned. Include 'Author' or 'I' as a node ONLY when "
            "extracting direct interactions or emotional states. Observed third-party "
            "entities stand alone."
        )
    )
    edges: List[Edge] = Field(
        description=(
            "Relationships in present tense. Author-connected edges for direct "
            "interactions (met_with, spoke_to) and subjective states "
            "(feels_about, admires, worried_about). Third-party relationships captured "
            "independently (e.g., Bob knows Alice)."
        )
    )


class ExtractKnowledgeGraph(dspy.Signature):
    """Knowledge graph extraction signature for personal journal entries."""

    text: str = dspy.InputField(
        desc="Journal entry text describing personal experiences, interactions, and observations"
    )
    known_entities: Optional[List[str]] = dspy.InputField(
        desc="Optional pre-extracted entity names to guide node creation",
        default=None,
    )
    graph: KnowledgeGraph = dspy.OutputField(
        desc=(
            "Graph with entities as nodes and relationships as edges (present tense). "
            "Balance author-centric edges (direct interactions, emotions) with "
            "independent observed relationships."
        )
    )


class KGExtractionModule(dspy.Module):
    """DSPy module wrapper around the knowledge graph extraction signature."""

    def __init__(self, optimized_prompts_path: Optional[str] = None) -> None:
        super().__init__()
        self._predict = dspy.Predict(ExtractKnowledgeGraph)
        if optimized_prompts_path:
            self.load_prompts(optimized_prompts_path)

    def forward(
        self, text: str, known_entities: Optional[List[str]] = None
    ) -> Any:
        return self._predict(text=text, known_entities=known_entities)

    def load_prompts(self, path: str) -> None:
        """Load optimized prompts into the underlying predictor."""
        self._predict.load(path)

    def save_prompts(self, path: str) -> None:
        """Persist the current optimized prompts."""
        self._predict.save(path)

    @property
    def predictor(self) -> dspy.Module:
        """Expose the underlying predictor for direct access when needed."""
        return self._predict


__all__ = [
    "Node",
    "Edge",
    "KnowledgeGraph",
    "ExtractKnowledgeGraph",
    "KGExtractionModule",
]
