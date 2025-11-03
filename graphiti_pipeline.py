"""Shared Graphiti pipeline utilities for CLI and Gradio integrations."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import dspy
from dspy_outlines.adapter import OutlinesAdapter
from dspy_outlines.lm import OutlinesLM

# Fix tokenizers parallelism warning (must be set before importing transformers)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from distilbert_ner import predict_entities
from entity_utils import (
    build_entity_edges,
    build_entity_nodes,
    build_episodic_edges,
    build_episodic_node,
    deduplicate_entities,
)
from falkordb_utils import get_db_stats, reset_database, write_entities_and_edges
from graphviz_utils import load_written_entities, render_graph_from_db
from settings import DB_PATH, MODEL_CONFIG
from signatures import FactExtractionSignature, RelationshipSignature

logger = logging.getLogger(__name__)

logger.info("Initializing Graphiti pipeline")
logger.info("MODEL_CONFIG: %s", MODEL_CONFIG)
logger.info("Database path: %s", DB_PATH)

# Configure DSPy once at module level
dspy.settings.configure(
    adapter=OutlinesAdapter(),
    lm=OutlinesLM(generation_config=MODEL_CONFIG),
)


class GraphitiPipelineError(RuntimeError):
    """Raised when a pipeline stage fails."""

    def __init__(self, stage: str, message: str):
        self.stage = stage
        self.message = message
        super().__init__(f"{stage} failed: {message}")


@dataclass
class PipelineArtifacts:
    """Collector for intermediate and final pipeline outputs."""

    text: str
    persons_only: bool = False
    reference_time: Optional[datetime] = None
    ner_entities: list[str] = field(default_factory=list)
    ner_raw: Any = None
    ner_display: str = ""
    facts: Any = None
    facts_json: Optional[dict[str, Any]] = None
    relationships: Any = None
    relationships_json: Optional[dict[str, Any]] = None
    episode: Any = None
    entity_nodes: list[Any] = field(default_factory=list)
    entity_edges: list[Any] = field(default_factory=list)
    episodic_edges: list[Any] = field(default_factory=list)
    graphiti_json: Optional[dict[str, Any]] = None
    write_result: Optional[dict[str, Any]] = None


# Stage 1: NER processing function
def process_ner(text: str, persons_only: bool):
    """
    Stage 1: Extract entities using NER.

    Returns: (entity_names_list, raw_ner_output, display_string)
    """
    if not text.strip():
        return [], None, ""

    # Run NER
    raw_entities = predict_entities(text)

    # Filter by type if requested
    if persons_only:
        filtered = [e for e in raw_entities if e["label"] == "PER"]
    else:
        filtered = raw_entities

    # Extract entity names and deduplicate
    entity_names = [e["text"] for e in filtered]
    unique_names = deduplicate_entities(entity_names)

    logger.info("Stage 1: Extracted %d unique entities", len(unique_names))

    # Format for display
    display = "\n".join(unique_names) if unique_names else "(no entities found)"

    return unique_names, raw_entities, display


# Stage 2: Fact extraction function
def extract_facts(text: str, entity_names: list[str]):
    """
    Stage 2: Extract facts using DSPy.

    Returns: (Facts object, JSON for display)
    """
    if not text.strip() or not entity_names:
        return None, {"error": "Need text and entities"}

    try:
        fact_predictor = dspy.Predict(FactExtractionSignature)
        facts = fact_predictor(text=text, entities=entity_names).facts

        logger.info("Stage 2: Extracted %d facts", len(facts.items))

        # Convert to JSON for display
        facts_json = {
            "items": [{"entity": f.entity, "text": f.text} for f in facts.items]
        }

        return facts, facts_json
    except Exception as exc:  # noqa: BLE001
        return None, {"error": str(exc)}


# Stage 3: Relationship inference function
def infer_relationships(text: str, facts, entity_names: list[str]):
    """
    Stage 3: Infer relationships using DSPy.

    Returns: (Relationships object, JSON for display)
    """
    if not text.strip() or not facts or not entity_names:
        return None, {"error": "Need text, facts, and entities"}

    try:
        rel_predictor = dspy.Predict(RelationshipSignature)
        relationships = rel_predictor(
            text=text, facts=facts, entities=entity_names
        ).relationships

        logger.info("Stage 3: Inferred %d relationships", len(relationships.items))

        # Convert to JSON for display
        rels_json = {
            "items": [
                {
                    "source": r.source,
                    "target": r.target,
                    "relation": r.relation,
                    "context": r.context,
                }
                for r in relationships.items
            ]
        }

        return relationships, rels_json
    except Exception as exc:  # noqa: BLE001
        return None, {"error": str(exc)}


# Stage 4: Graphiti object builder function
def build_graphiti_objects(
    input_text, entity_names: list[str], relationships, reference_time
):
    """
    Stage 4: Build EpisodicNode, EntityNode, EntityEdge, and EpisodicEdge objects.

    Now mirrors graphiti.py:413-436 (_process_episode_data).
    """
    if not entity_names or not relationships:
        return None, [], [], [], {"error": "Need entities and relationships"}

    try:
        # 1. Create EpisodicNode (mirrors graphiti.py:706-720)
        episode = build_episodic_node(content=input_text, reference_time=reference_time)

        # 2. Build EntityNodes
        entity_nodes, entity_map = build_entity_nodes(entity_names)

        # 3. Build EntityEdges with episode UUID (mirrors edge_operations.py:220-230)
        entity_edges = build_entity_edges(
            relationships,
            entity_map,
            episode_uuid=episode.uuid,  # Pass episode UUID
        )

        # 4. Link episode to entity edges (mirrors graphiti.py:422)
        episode.entity_edges = [edge.uuid for edge in entity_edges]

        # 5. Build EpisodicEdges (MENTIONS) (mirrors edge_operations.py:51-68)
        episodic_edges = build_episodic_edges(episode, entity_nodes)

        logger.info(
            "Stage 4: Built episode %s, %d nodes, %d entity edges, %d episodic edges",
            episode.uuid,
            len(entity_nodes),
            len(entity_edges),
            len(episodic_edges),
        )

        # Convert to JSON for display
        graphiti_json = {
            "episode": episode.model_dump(),
            "nodes": [n.model_dump() for n in entity_nodes],
            "entity_edges": [e.model_dump() for e in entity_edges],
            "episodic_edges": [e.model_dump() for e in episodic_edges],
        }

        return episode, entity_nodes, entity_edges, episodic_edges, graphiti_json

    except Exception as exc:  # noqa: BLE001
        return None, [], [], [], {"error": str(exc)}


# Stage 5: FalkorDB write function
def write_to_falkordb(episode, entity_nodes, entity_edges, episodic_edges):
    """
    Stage 5: Write EpisodicNode, EntityNode, EntityEdge, and EpisodicEdge objects to FalkorDB.

    Returns: Write result dict (with UUIDs)
    """
    if not entity_nodes:
        return {"error": "No entities to write"}

    try:
        logger.info(
            "Stage 5: Writing episode %s, %d nodes, %d entity edges, %d episodic edges to FalkorDB",
            getattr(episode, "uuid", "<unknown>"),
            len(entity_nodes),
            len(entity_edges),
            len(episodic_edges),
        )
        result = write_entities_and_edges(
            episode, entity_nodes, entity_edges, episodic_edges
        )
        return result
    except Exception as exc:  # noqa: BLE001
        import traceback

        return {"error": str(exc), "traceback": traceback.format_exc()}


# Stage 6: Graphviz rendering function
def render_verification_graph(write_result):
    """
    Stage 6: Verify FalkorDB write by querying and rendering graph with episode.

    Returns: Path to PNG file, or None on error
    """
    if not write_result or "error" in write_result:
        return None

    # Load episode and entities from DB using UUIDs
    episode_uuid = write_result.get("episode_uuid")
    node_uuids = write_result.get("node_uuids", [])
    edge_uuids = write_result.get("edge_uuids", [])

    if not node_uuids:
        return None

    db_data = load_written_entities(node_uuids, edge_uuids, episode_uuid)
    return render_graph_from_db(db_data)


class GraphitiPipeline:
    """High-level orchestrator for running the full Graphiti ingestion pipeline."""

    def __init__(self, *, logger_: Optional[logging.Logger] = None):
        self.logger = logger_ or logger

    def run_episode(
        self,
        text: str,
        *,
        persons_only: bool = False,
        reference_time: Optional[datetime] = None,
        write: bool = True,
        render_graph: bool = False,
    ) -> PipelineArtifacts:
        """
        Run the full pipeline from raw text to FalkorDB write.

        Args:
            text: Journal entry content.
            persons_only: Whether to restrict NER output to persons.
            reference_time: Reference datetime for the episode; defaults to now.
            write: When False, skip FalkorDB write.
            render_graph: When True, perform Stage 6 graph render.

        Returns:
            PipelineArtifacts with intermediate and final outputs.

        Raises:
            GraphitiPipelineError on stage failure (error detail in exception.message).
        """
        artifacts = PipelineArtifacts(
            text=text,
            persons_only=persons_only,
            reference_time=reference_time,
        )

        reference_time = reference_time or datetime.now()
        artifacts.reference_time = reference_time

        # Stage 1
        entities, raw, display = process_ner(text, persons_only)
        artifacts.ner_entities = entities
        artifacts.ner_raw = raw
        artifacts.ner_display = display
        if not entities:
            raise GraphitiPipelineError("Stage 1 (NER)", "No entities extracted")

        # Stage 2
        facts, facts_json = extract_facts(text, entities)
        artifacts.facts = facts
        artifacts.facts_json = facts_json
        if facts is None or ("error" in facts_json):
            raise GraphitiPipelineError(
                "Stage 2 (Facts)", facts_json.get("error", "Unknown error")
            )

        # Stage 3
        relationships, rels_json = infer_relationships(text, facts, entities)
        artifacts.relationships = relationships
        artifacts.relationships_json = rels_json
        if relationships is None or ("error" in rels_json):
            raise GraphitiPipelineError(
                "Stage 3 (Relationships)", rels_json.get("error", "Unknown error")
            )

        # Stage 4
        episode, nodes, entity_edges, episodic_edges, graphiti_json = (
            build_graphiti_objects(
                input_text=text,
                entity_names=entities,
                relationships=relationships,
                reference_time=reference_time,
            )
        )
        artifacts.episode = episode
        artifacts.entity_nodes = nodes
        artifacts.entity_edges = entity_edges
        artifacts.episodic_edges = episodic_edges
        artifacts.graphiti_json = graphiti_json
        if episode is None or ("error" in graphiti_json):
            raise GraphitiPipelineError(
                "Stage 4 (Graphiti Objects)",
                graphiti_json.get("error", "Unknown error"),
            )

        # Stage 5
        if write:
            write_result = write_to_falkordb(
                episode, nodes, entity_edges, episodic_edges
            )
            artifacts.write_result = write_result
            if not write_result or "error" in write_result:
                message = (
                    write_result.get("error", "Unknown error")
                    if write_result
                    else "Write returned no result"
                )
                raise GraphitiPipelineError("Stage 5 (FalkorDB Write)", message)
        else:
            self.logger.info("Skipping FalkorDB write (dry run)")

        # Stage 6 (optional)
        if render_graph and artifacts.write_result:
            self.logger.info(
                "Rendering verification graph for episode %s", artifacts.episode.uuid
            )
            render_verification_graph(artifacts.write_result)

        return artifacts


__all__ = [
    "GraphitiPipeline",
    "GraphitiPipelineError",
    "PipelineArtifacts",
    "process_ner",
    "extract_facts",
    "infer_relationships",
    "build_graphiti_objects",
    "write_to_falkordb",
    "render_verification_graph",
    "get_db_stats",
    "reset_database",
]
