"""Shared Graphiti pipeline utilities for CLI and Gradio integrations."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

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
    normalize_entity_name,
)
from falkordb_utils import (
    fetch_entities_by_group,
    fetch_entity_edges_by_group,
    fetch_recent_episodes,
    get_db_stats,
    reset_database,
    write_entities_and_edges,
)
from graphviz_utils import load_written_entities, render_graph_from_db
from models import Facts, Relationships
from settings import DB_PATH, EPISODE_CONTEXT_WINDOW, GROUP_ID, MODEL_CONFIG
from signatures import (
    EntityAttributesSignature,
    EntitySummarySignature,
    FactExtractionSignature,
    RelationshipSignature,
)

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
class GraphitiBuildResult:
    """Structured result for Graphiti object construction."""

    episode: Any
    nodes: list[Any]
    entity_edges: list[Any]
    episodic_edges: list[Any]
    graphiti_json: dict[str, Any]
    uuid_map: Dict[str, str]
    dedupe_records: list[dict[str, Any]]
    edge_resolution_records: list[dict[str, Any]]
    invalidated_edges: list[Any]
    entity_attributes_json: list[dict[str, Any]]
    entity_summaries_json: list[dict[str, Any]]


@dataclass
class PipelineConfig:
    """Configuration knobs mirroring Graphiti's add_episode parameters."""

    group_id: str = GROUP_ID
    context_window: int = EPISODE_CONTEXT_WINDOW
    dedupe_enabled: bool = True
    attribute_extraction_enabled: bool = True
    entity_summary_enabled: bool = True
    temporal_enabled: bool = True


@dataclass
class PipelineArtifacts:
    """Collector for intermediate and final pipeline outputs."""

    text: str
    persons_only: bool = False
    reference_time: Optional[datetime] = None
    config: PipelineConfig = field(default_factory=PipelineConfig)
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
    context_episodes: list[Any] = field(default_factory=list)
    context_json: list[dict[str, Any]] = field(default_factory=list)
    entity_summaries: Dict[str, str] = field(default_factory=dict)
    entity_summaries_json: list[dict[str, Any]] = field(default_factory=list)
    entity_attributes: Dict[str, dict[str, Any]] = field(default_factory=dict)
    entity_attributes_json: list[dict[str, Any]] = field(default_factory=list)
    uuid_map: Dict[str, str] = field(default_factory=dict)
    dedupe_records: list[dict[str, Any]] = field(default_factory=list)
    edge_resolution_records: list[dict[str, Any]] = field(default_factory=list)
    invalidated_edges: list[Any] = field(default_factory=list)
    embedding_stub: dict[str, Any] = field(default_factory=dict)
    reranker_stub: dict[str, Any] = field(default_factory=dict)


# NER processing function
def process_ner(text: str, persons_only: bool):
    """
    Extract entities using NER.

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

    logger.info("NER: extracted %d unique entities", len(unique_names))

    # Format for display
    display = "\n".join(unique_names) if unique_names else "(no entities found)"

    return unique_names, raw_entities, display


# Fact extraction function
def extract_facts(text: str, entity_names: list[str]):
    """
    Extract facts using DSPy.

    Returns: (Facts object, JSON for display)
    """
    if not text.strip() or not entity_names:
        return None, {"error": "Need text and entities"}

    try:
        fact_predictor = dspy.Predict(FactExtractionSignature)
        facts = fact_predictor(text=text, entities=entity_names).facts

        logger.info("Facts: extracted %d items", len(facts.items))

        # Convert to JSON for display
        facts_json = {
            "items": [{"entity": f.entity, "text": f.text} for f in facts.items]
        }

        return facts, facts_json
    except Exception as exc:  # noqa: BLE001
        return None, {"error": str(exc)}


# Relationship inference function
def infer_relationships(text: str, facts, entity_names: list[str]):
    """
    Infer relationships using DSPy.

    Returns: (Relationships object, JSON for display)
    """
    if not text.strip() or not facts or not entity_names:
        return None, {"error": "Need text, facts, and entities"}

    try:
        rel_predictor = dspy.Predict(RelationshipSignature)
        relationships = rel_predictor(
            text=text, facts=facts, entities=entity_names
        ).relationships

        logger.info("Relationships: inferred %d items", len(relationships.items))

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


def _build_context_json(episodes: list[Any]) -> list[dict[str, Any]]:
    """Convert EpisodicNode objects into serializable context entries."""
    context_json: list[dict[str, Any]] = []
    for episode in episodes:
        context_json.append(
            {
                "uuid": episode.uuid,
                "name": episode.name,
                "valid_at": getattr(episode, "valid_at", None),
                "content_preview": (episode.content[:200] + "...")
                if episode.content and len(episode.content) > 200
                else episode.content,
            }
        )
    return context_json


# Mapping DistilBERT tags to Graphiti labels
NER_LABEL_TO_GRAPHITI = {
    "PER": "Person",
    "ORG": "Organization",
    "LOC": "Location",
    "GPE": "Location",
    "MISC": "Entity",
}


def _resolve_entities(
    provisional_nodes: list[Any],
    existing_nodes: dict[str, Any],
    config: PipelineConfig,
) -> tuple[list[Any], dict[str, Any], Dict[str, str], list[dict[str, Any]]]:
    """Resolve provisional EntityNodes against existing graph data."""
    normalized_existing: dict[str, list[Any]] = defaultdict(list)
    for node in existing_nodes.values():
        key = normalize_entity_name(node.name)
        normalized_existing[key].append(node)

    resolved_nodes: list[Any] = []
    resolved_map: dict[str, Any] = {}
    uuid_map: Dict[str, str] = {}
    dedupe_records: list[dict[str, Any]] = []
    seen_uuid: set[str] = set()

    for node in provisional_nodes:
        key = normalize_entity_name(node.name)
        candidates = normalized_existing.get(key, [])

        if config.dedupe_enabled and candidates:
            # Reuse existing node (make a copy so we can mutate safely)
            existing = candidates[0].model_copy(deep=True)
            uuid_map[node.uuid] = existing.uuid
            resolved_map[key] = existing
            if existing.uuid not in seen_uuid:
                resolved_nodes.append(existing)
                seen_uuid.add(existing.uuid)
            dedupe_records.append(
                {
                    "entity": node.name,
                    "status": "matched",
                    "resolved_uuid": existing.uuid,
                    "reason": "exact_name_match",
                }
            )
        else:
            uuid_map[node.uuid] = node.uuid
            resolved_map[key] = node
            normalized_existing.setdefault(key, []).append(node)
            if node.uuid not in seen_uuid:
                resolved_nodes.append(node)
                seen_uuid.add(node.uuid)
            dedupe_records.append(
                {
                    "entity": node.name,
                    "status": "new",
                    "resolved_uuid": node.uuid,
                    "reason": "no_existing_match",
                }
            )

    return resolved_nodes, resolved_map, uuid_map, dedupe_records


def _collect_ner_labels(ner_raw: Any) -> dict[str, set[str]]:
    """Group NER labels by normalized entity name."""
    label_map: dict[str, set[str]] = defaultdict(set)
    if not ner_raw:
        return label_map

    for item in ner_raw:
        text = item.get("text")
        label = item.get("label")
        if not text or not label:
            continue
        norm = normalize_entity_name(text)
        label_map[norm].add(label)
    return label_map


def _apply_entity_attributes(
    resolved_map: dict[str, Any],
    ner_raw: Any,
    config: PipelineConfig,
) -> tuple[Dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Apply label/attribute enrichment to resolved nodes."""
    attributes_by_uuid: Dict[str, dict[str, Any]] = {}
    attribute_json: list[dict[str, Any]] = []

    if not config.attribute_extraction_enabled:
        return attributes_by_uuid, attribute_json

    ner_labels = _collect_ner_labels(ner_raw)

    for norm_name, node in resolved_map.items():
        ner_tag_set = ner_labels.get(norm_name, set())
        graphiti_labels = {
            NER_LABEL_TO_GRAPHITI.get(tag, "Entity") for tag in ner_tag_set
        } or {"Entity"}

        # Merge labels, preserving existing ones
        existing_labels = set(node.labels or [])
        merged_labels = sorted(existing_labels.union(graphiti_labels))
        node.labels = merged_labels

        # Merge attributes with a simple NER provenance hint
        attributes = dict(node.attributes or {})
        if ner_tag_set:
            attributes.setdefault("ner_labels", sorted(ner_tag_set))
        node.attributes = attributes

        attributes_by_uuid[node.uuid] = {
            "labels": merged_labels,
            "attributes": attributes,
        }
        attribute_json.append(
            {
                "entity": node.name,
                "uuid": node.uuid,
                "labels": merged_labels,
                "attributes": attributes,
            }
        )

    return attributes_by_uuid, attribute_json


def _extract_entity_summaries(
    text: str,
    facts: Facts | None,
    relationships: Relationships | None,
    resolved_nodes: list[Any],
    config: PipelineConfig,
) -> tuple[Dict[str, str], list[dict[str, Any]]]:
    """Generate entity summaries via DSPy."""
    summaries_by_uuid: Dict[str, str] = {}
    summaries_json: list[dict[str, Any]] = []

    if not config.entity_summary_enabled or not resolved_nodes:
        return summaries_by_uuid, summaries_json

    facts = facts or Facts(items=[])
    relationships = relationships or Relationships(items=[])

    try:
        summary_predictor = dspy.Predict(EntitySummarySignature)
        result = summary_predictor(
            text=text,
            entities=[node.name for node in resolved_nodes],
            facts=facts,
            relationships=relationships,
        )
        summaries = getattr(result, "summaries", None)
        if not summaries:
            return summaries_by_uuid, summaries_json

        summary_map = {
            normalize_entity_name(item.entity): item.summary
            for item in summaries.items
            if item.summary
        }

        for node in resolved_nodes:
            summary = summary_map.get(normalize_entity_name(node.name))
            if not summary:
                continue
            node.summary = summary
            summaries_by_uuid[node.uuid] = summary
            summaries_json.append(
                {
                    "entity": node.name,
                    "uuid": node.uuid,
                    "summary": summary,
                }
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Entity summary generation failed: %s", exc)

    return summaries_by_uuid, summaries_json


def _resolve_entity_edges(
    candidate_edges: list[Any],
    existing_edges: dict[str, Any],
    config: PipelineConfig,
    reference_time: datetime,
    episode_uuid: str,
) -> tuple[list[Any], list[Any], list[dict[str, Any]]]:
    """Resolve entity edges against existing graph edges."""
    existing_index: dict[tuple[str, str, str], Any] = {}
    for edge in existing_edges.values():
        key = (edge.source_node_uuid, edge.target_node_uuid, edge.name)
        existing_index[key] = edge

    resolved_edges: list[Any] = []
    invalidated_edges: list[Any] = []
    records: list[dict[str, Any]] = []

    for edge in candidate_edges:
        key = (edge.source_node_uuid, edge.target_node_uuid, edge.name)
        existing = existing_index.get(key)

        if existing and config.dedupe_enabled:
            updated = existing.model_copy(deep=True)
            existing_episode_ids = set(existing.episodes or [])
            new_episode_ids = set(edge.episodes or [])
            updated.episodes = sorted(existing_episode_ids.union(new_episode_ids))
            if config.temporal_enabled:
                updated.valid_at = existing.valid_at or reference_time
            records.append(
                {
                    "status": "merged",
                    "edge_uuid": updated.uuid,
                    "source_uuid": updated.source_node_uuid,
                    "target_uuid": updated.target_node_uuid,
                    "relation": updated.name,
                }
            )
            resolved_edges.append(updated)
        else:
            if config.temporal_enabled and edge.valid_at is None:
                edge.valid_at = reference_time
            edge.episodes = sorted(set((edge.episodes or []) + [episode_uuid]))
            records.append(
                {
                    "status": "new",
                    "edge_uuid": edge.uuid,
                    "source_uuid": edge.source_node_uuid,
                    "target_uuid": edge.target_node_uuid,
                    "relation": edge.name,
                }
            )
            resolved_edges.append(edge)
            existing_index[key] = edge

    return resolved_edges, invalidated_edges, records


def _build_embedding_stub() -> dict[str, Any]:
    """Placeholder metadata for future embedding integration."""
    return {
        "status": "not_implemented",
        "message": "Embeddings will be generated post-deduplication in a future phase.",
    }


def _build_reranker_stub() -> dict[str, Any]:
    """Placeholder metadata for future reranker integration."""
    return {
        "status": "not_implemented",
        "message": "Reranker scoring will be integrated alongside embeddings in a future phase.",
    }


# Graphiti object builder function
def build_graphiti_objects(
    input_text: str,
    entity_names: list[str],
    relationships: Relationships,
    reference_time: datetime,
    *,
    config: PipelineConfig | None = None,
    ner_raw: Any = None,
    facts: Facts | None = None,
    existing_entities: dict[str, Any] | None = None,
    existing_edges: dict[str, Any] | None = None,
) -> GraphitiBuildResult:
    """Build EpisodicNode, EntityNode, EntityEdge, and EpisodicEdge objects."""
    if not entity_names or not relationships:
        raise GraphitiPipelineError(
            "Graph Construction", "Need entities and relationships"
        )

    config = config or PipelineConfig()
    existing_entities = existing_entities or fetch_entities_by_group(config.group_id)
    existing_edges = existing_edges or fetch_entity_edges_by_group(config.group_id)

    try:
        # 1. Create EpisodicNode (mirrors graphiti.py:706-720)
        episode = build_episodic_node(
            content=input_text,
            reference_time=reference_time,
            group_id=config.group_id,
        )

        # 2. Build provisional EntityNodes
        provisional_nodes, provisional_map = build_entity_nodes(entity_names)

        # 3. Resolve entities against existing graph state
        resolved_nodes, resolved_map, uuid_map, dedupe_records = _resolve_entities(
            provisional_nodes,
            existing_entities,
            config,
        )

        # 4. Build EntityEdges using resolved entity map
        entity_edges = build_entity_edges(
            relationships,
            resolved_map,
            episode_uuid=episode.uuid,
        )

        # 5. Apply attribute enrichment
        attributes_by_uuid, attribute_json = _apply_entity_attributes(
            resolved_map,
            ner_raw,
            config,
        )

        # 6. Extract entity summaries (optional)
        entity_summaries, summaries_json = _extract_entity_summaries(
            text=input_text,
            facts=facts,
            relationships=relationships,
            resolved_nodes=resolved_nodes,
            config=config,
        )

        # 7. Resolve entity edges (dedupe + temporal propagation)
        resolved_edges, invalidated_edges, edge_resolution_records = (
            _resolve_entity_edges(
                entity_edges,
                existing_edges,
                config,
                reference_time,
                episode.uuid,
            )
        )

        # 8. Link episode to entity edges (mirrors graphiti.py:422)
        episode.entity_edges = [edge.uuid for edge in resolved_edges]

        # 9. Build EpisodicEdges (MENTIONS) (mirrors edge_operations.py:51-68)
        episodic_edges = build_episodic_edges(episode, resolved_nodes)

        logger.info(
            "Graph build completed: episode %s, %d nodes, %d entity edges, %d episodic edges",
            episode.uuid,
            len(resolved_nodes),
            len(resolved_edges),
            len(episodic_edges),
        )

        # Convert to JSON for display
        graphiti_json = {
            "episode": episode.model_dump(),
            "nodes": [n.model_dump() for n in resolved_nodes],
            "entity_edges": [e.model_dump() for e in resolved_edges],
            "episodic_edges": [e.model_dump() for e in episodic_edges],
        }

        return GraphitiBuildResult(
            episode=episode,
            nodes=resolved_nodes,
            entity_edges=resolved_edges,
            episodic_edges=episodic_edges,
            graphiti_json=graphiti_json,
            uuid_map=uuid_map,
            dedupe_records=dedupe_records,
            edge_resolution_records=edge_resolution_records,
            invalidated_edges=invalidated_edges,
            entity_attributes_json=attribute_json,
            entity_summaries_json=summaries_json,
        )

    except Exception as exc:  # noqa: BLE001
        raise GraphitiPipelineError("Graph Construction", str(exc)) from exc


# FalkorDB write utility
def write_to_falkordb(episode, entity_nodes, entity_edges, episodic_edges):
    """
    Write EpisodicNode, EntityNode, EntityEdge, and EpisodicEdge objects to FalkorDB.

    Returns: Write result dict (with UUIDs)
    """
    if not entity_nodes:
        return {"error": "No entities to write"}

    try:
        logger.info(
            "FalkorDB: writing episode %s, %d nodes, %d entity edges, %d episodic edges",
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


# Graphviz rendering utility
def render_verification_graph(write_result):
    """
    Verify FalkorDB write by querying and rendering graph with episode.

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
        config: PipelineConfig | None = None,
    ) -> PipelineArtifacts:
        """
        Run the full pipeline from raw text to FalkorDB write.

        Args:
            text: Journal entry content.
            persons_only: Whether to restrict NER output to persons.
            reference_time: Reference datetime for the episode; defaults to now.
            write: When False, skip FalkorDB write.
            render_graph: When True, produce a Graphviz verification render.
            config: Optional PipelineConfig to adjust pipeline behavior.

        Returns:
            PipelineArtifacts with intermediate and final outputs.

        Raises:
            GraphitiPipelineError on stage failure (error detail in exception.message).
        """
        config = config or PipelineConfig()
        artifacts = PipelineArtifacts(
            text=text,
            persons_only=persons_only,
            reference_time=reference_time,
            config=config,
        )

        reference_time = reference_time or datetime.now()
        artifacts.reference_time = reference_time

        # Fetch context for reference time
        if config.context_window > 0:
            context_episodes = fetch_recent_episodes(
                group_id=config.group_id,
                reference_time=reference_time,
                limit=config.context_window,
            )
            artifacts.context_episodes = context_episodes
            artifacts.context_json = _build_context_json(context_episodes)

        # NER extraction
        entities, raw, display = process_ner(text, persons_only)
        artifacts.ner_entities = entities
        artifacts.ner_raw = raw
        artifacts.ner_display = display
        if not entities:
            raise GraphitiPipelineError("NER", "No entities extracted")

        # Fact extraction
        facts, facts_json = extract_facts(text, entities)
        artifacts.facts = facts
        artifacts.facts_json = facts_json
        if facts is None or ("error" in facts_json):
            raise GraphitiPipelineError(
                "Facts", facts_json.get("error", "Unknown error")
            )

        # Relationship inference
        relationships, rels_json = infer_relationships(text, facts, entities)
        artifacts.relationships = relationships
        artifacts.relationships_json = rels_json
        if relationships is None or ("error" in rels_json):
            raise GraphitiPipelineError(
                "Relationships", rels_json.get("error", "Unknown error")
            )

        # Build graph objects and resolution records
        build_result = build_graphiti_objects(
            input_text=text,
            entity_names=entities,
            relationships=relationships,
            reference_time=reference_time,
            config=config,
            ner_raw=raw,
            facts=facts,
        )
        artifacts.episode = build_result.episode
        artifacts.entity_nodes = build_result.nodes
        artifacts.entity_edges = build_result.entity_edges
        artifacts.episodic_edges = build_result.episodic_edges
        artifacts.graphiti_json = build_result.graphiti_json
        artifacts.uuid_map = build_result.uuid_map
        artifacts.dedupe_records = build_result.dedupe_records
        artifacts.edge_resolution_records = build_result.edge_resolution_records
        artifacts.invalidated_edges = build_result.invalidated_edges
        artifacts.entity_summaries_json = build_result.entity_summaries_json
        artifacts.entity_summaries = {
            entry["uuid"]: entry["summary"]
            for entry in build_result.entity_summaries_json
            if "uuid" in entry
        }
        artifacts.entity_attributes_json = build_result.entity_attributes_json
        artifacts.entity_attributes = {
            entry["uuid"]: {
                "labels": entry.get("labels", []),
                "attributes": entry.get("attributes", {}),
            }
            for entry in build_result.entity_attributes_json
            if "uuid" in entry
        }

        # Embedding/reranker stubs (for future integration)
        artifacts.embedding_stub = _build_embedding_stub()
        artifacts.reranker_stub = _build_reranker_stub()

        # Persist results when requested
        if write:
            entity_edges_for_write = (
                build_result.entity_edges + build_result.invalidated_edges
            )
            write_result = write_to_falkordb(
                build_result.episode,
                build_result.nodes,
                entity_edges_for_write,
                build_result.episodic_edges,
            )
            artifacts.write_result = write_result
            if not write_result or "error" in write_result:
                message = (
                    write_result.get("error", "Unknown error")
                    if write_result
                    else "Write returned no result"
                )
                raise GraphitiPipelineError("FalkorDB Write", message)
        else:
            self.logger.info("Skipping FalkorDB write (dry run)")

        # Optional verification render
        if render_graph and artifacts.write_result:
            self.logger.info(
                "Rendering verification graph for episode %s", artifacts.episode.uuid
            )
            render_verification_graph(artifacts.write_result)

        return artifacts


__all__ = [
    "GraphitiPipeline",
    "GraphitiPipelineError",
    "PipelineConfig",
    "PipelineArtifacts",
    "GraphitiBuildResult",
    "process_ner",
    "extract_facts",
    "infer_relationships",
    "build_graphiti_objects",
    "write_to_falkordb",
    "render_verification_graph",
    "get_db_stats",
    "reset_database",
]
