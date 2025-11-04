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
import dateparser
from dateparser.search import search_dates

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
from graphiti_core.helpers import (
    validate_excluded_entity_types,
    validate_group_id,
)
from graphiti_core.utils.bulk_utils import compress_uuid_map, resolve_edge_pointers
from graphiti_core.utils.maintenance.dedup_helpers import (
    _cached_shingles,
    _has_high_entropy,
    _jaccard_similarity,
    _minhash_signature,
    _normalize_name_for_fuzzy,
)
from graphiti_core.utils.maintenance.edge_operations import (
    resolve_edge_contradictions,
)
from graphiti_core.utils.datetime_utils import ensure_utc
from graphiti_core.utils.ontology_utils.entity_types_utils import (
    validate_entity_types,
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
    EntityEdgeDetectionSignature,
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
    temporal_enrichment_records: list[dict[str, Any]]


@dataclass
class PipelineConfig:
    """Configuration knobs mirroring Graphiti's add_episode parameters."""

    group_id: str = GROUP_ID
    context_window: int = EPISODE_CONTEXT_WINDOW
    dedupe_enabled: bool = True
    attribute_extraction_enabled: bool = True
    entity_summary_enabled: bool = True
    temporal_enabled: bool = True
    temporal_enrichment_enabled: bool = True
    llm_edge_detection_enabled: bool = True


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
    base_relationships: Any = None
    relationships: Any = None
    relationships_json: Optional[dict[str, Any]] = None
    base_relationships_json: Optional[dict[str, Any]] = None
    llm_relationships: Any = None
    llm_relationships_json: Optional[dict[str, Any]] = None
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
    temporal_enrichment_records: list[dict[str, Any]] = field(default_factory=list)


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


def _serialize_relationships(relationships: Relationships | None) -> dict[str, Any]:
    """Convert Relationships collection into JSON consumable by the UI."""
    if not relationships or not getattr(relationships, "items", None):
        return {"items": []}

    return {
        "items": [
            {
                "source": rel.source,
                "target": rel.target,
                "relation": rel.relation,
                "context": rel.context,
            }
            for rel in relationships.items
        ]
    }


# Relationship inference function
def infer_relationships(
    text: str, facts, entity_names: list[str], reference_time: datetime
):
    """
    Infer relationships using DSPy.

    Returns: (Relationships object, JSON for display)
    """
    if not text.strip() or not facts or not entity_names:
        return None, {"error": "Need text, facts, and entities"}

    try:
        rel_predictor = dspy.Predict(RelationshipSignature)
        relationships = rel_predictor(
            text=text,
            facts=facts,
            entities=entity_names,
            reference_time=reference_time.isoformat(),
        ).relationships

        logger.info("Relationships: inferred %d items", len(relationships.items))

        return relationships, _serialize_relationships(relationships)
    except Exception as exc:  # noqa: BLE001
        return None, {"error": str(exc)}


def detect_entity_edges(
    text: str,
    facts: Facts | None,
    entity_names: list[str],
    reference_time: datetime,
) -> tuple[Relationships | None, dict[str, Any]]:
    """
    Run LLM-flavored DSPy signature to detect entity edges directly from text.

    Returns: (Relationships object, JSON for display)
    """
    if not text.strip() or not entity_names:
        return None, {"error": "Need text and entities"}

    try:
        edge_predictor = dspy.Predict(EntityEdgeDetectionSignature)
        result = edge_predictor(
            text=text,
            facts=facts or Facts(items=[]),
            entities=entity_names,
            reference_time=reference_time.isoformat(),
        )
        relationships = getattr(result, "relationships", None)
        if relationships is None:
            return None, {"items": []}

        logger.info(
            "LLM Edge Detection: produced %d candidate relationships",
            len(relationships.items),
        )

        return relationships, _serialize_relationships(relationships)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM edge detection failed: %s", exc)
        return None, {"error": str(exc)}


def _merge_relationship_collections(
    primary: Relationships,
    secondary: Relationships | None,
) -> Relationships:
    """
    Merge two relationship collections, deduplicating by normalized endpoints and relation.
    """
    if secondary is None or not getattr(secondary, "items", None):
        return primary

    merged_items: list[Any] = []
    seen: set[tuple[str, str, str]] = set()

    def _key(rel: Any) -> tuple[str, str, str]:
        return (
            normalize_entity_name(rel.source),
            normalize_entity_name(rel.target),
            rel.relation.strip().lower(),
        )

    for collection in (primary.items, secondary.items):
        for rel in collection:
            key = _key(rel)
            if key in seen:
                continue
            seen.add(key)
            merged_items.append(rel)

    return Relationships(items=merged_items)


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
        elif config.dedupe_enabled and normalized_existing:
            # Try fuzzy matching as fallback
            fuzzy_key = _normalize_name_for_fuzzy(node.name)
            if _has_high_entropy(fuzzy_key):
                node_shingles = set(_cached_shingles(fuzzy_key))
                node_sig = _minhash_signature(node_shingles) if node_shingles else ()

                best_match = None
                best_similarity = 0.0

                for existing_name, existing_nodes in normalized_existing.items():
                    existing_sample = existing_nodes[0]
                    existing_fuzzy = _normalize_name_for_fuzzy(existing_sample.name)
                    existing_shingles = set(_cached_shingles(existing_fuzzy))
                    existing_sig = (
                        _minhash_signature(existing_shingles)
                        if existing_shingles
                        else ()
                    )

                    similarity = 0.0
                    if node_sig and existing_sig and len(node_sig) == len(existing_sig):
                        matching = sum(
                            1 for a, b in zip(node_sig, existing_sig) if a == b
                        )
                        if len(node_sig) > 0:
                            similarity = matching / len(node_sig)

                    if similarity == 0.0:
                        similarity = _jaccard_similarity(
                            node_shingles, existing_shingles
                        )

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = existing_sample

                if best_match and best_similarity >= 0.9:
                    # Fuzzy match found - reuse existing node
                    existing = best_match.model_copy(deep=True)
                    uuid_map[node.uuid] = existing.uuid
                    resolved_map[key] = existing
                    existing_bucket = normalized_existing.setdefault(key, [])
                    if all(
                        candidate.uuid != existing.uuid for candidate in existing_bucket
                    ):
                        existing_bucket.append(existing)
                    if existing.uuid not in seen_uuid:
                        resolved_nodes.append(existing)
                        seen_uuid.add(existing.uuid)
                    dedupe_records.append(
                        {
                            "entity": node.name,
                            "status": "matched",
                            "resolved_uuid": existing.uuid,
                            "reason": f"fuzzy_match (similarity={best_similarity:.2f})",
                        }
                    )
                else:
                    # No fuzzy match - create new entity
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
                            "reason": "no_fuzzy_match",
                        }
                    )
            else:
                # Low entropy - skip fuzzy matching
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
                        "reason": "low_entropy_name",
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
    edges_by_pair: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for edge in existing_edges.values():
        key = (edge.source_node_uuid, edge.target_node_uuid, edge.name)
        existing_index[key] = edge
        pair_key = (edge.source_node_uuid, edge.target_node_uuid)
        edges_by_pair[pair_key].append(edge)

    resolved_edges: list[Any] = []
    invalidated_edges: list[Any] = []
    records: list[dict[str, Any]] = []
    invalidated_seen: set[str] = set()

    for edge in candidate_edges:
        key = (edge.source_node_uuid, edge.target_node_uuid, edge.name)
        existing = existing_index.get(key)
        pair_key = (edge.source_node_uuid, edge.target_node_uuid)
        contradiction_pool = list(edges_by_pair.get(pair_key, []))

        if existing and config.dedupe_enabled:
            updated = existing.model_copy(deep=True)
            existing_episode_ids = set(existing.episodes or [])
            new_episode_ids = set(edge.episodes or [])
            updated.episodes = sorted(existing_episode_ids.union(new_episode_ids))
            if config.temporal_enabled:
                updated.valid_at = existing.valid_at or edge.valid_at or reference_time
                updated.invalid_at = existing.invalid_at or edge.invalid_at
            resolved_edge = updated
            record = {
                "status": "merged",
                "edge_uuid": updated.uuid,
                "source_uuid": updated.source_node_uuid,
                "target_uuid": updated.target_node_uuid,
                "relation": updated.name,
            }
        else:
            if config.temporal_enabled:
                if edge.valid_at is None:
                    edge.valid_at = reference_time
            edge.episodes = sorted(set((edge.episodes or []) + [episode_uuid]))
            resolved_edge = edge
            record = {
                "status": "new",
                "edge_uuid": edge.uuid,
                "source_uuid": edge.source_node_uuid,
                "target_uuid": edge.target_node_uuid,
                "relation": edge.name,
            }

        resolved_edges.append(resolved_edge)
        records.append(record)
        existing_index[key] = resolved_edge

        filtered_pool = [
            candidate
            for candidate in contradiction_pool
            if candidate.uuid != resolved_edge.uuid
        ]
        filtered_pool.append(resolved_edge)
        edges_by_pair[pair_key] = filtered_pool

        contradiction_candidates = [
            candidate
            for candidate in filtered_pool
            if candidate.uuid != resolved_edge.uuid
            and candidate.name != resolved_edge.name
        ]
        if contradiction_candidates:
            invalidated = resolve_edge_contradictions(
                resolved_edge, contradiction_candidates
            )
            for invalid_edge in invalidated:
                if invalid_edge.uuid in invalidated_seen:
                    continue
                invalidated_seen.add(invalid_edge.uuid)
                invalidated_edges.append(invalid_edge)
                records.append(
                    {
                        "status": "invalidated",
                        "edge_uuid": invalid_edge.uuid,
                        "source_uuid": invalid_edge.source_node_uuid,
                        "target_uuid": invalid_edge.target_node_uuid,
                        "relation": invalid_edge.name,
                        "reason": f"contradiction_with:{resolved_edge.uuid}",
                        "invalidated_at": getattr(invalid_edge, "invalid_at", None),
                    }
                )

    return resolved_edges, invalidated_edges, records


def _enrich_temporal_metadata(
    edges: list[Any],
    episode_text: str,
    reference_time: datetime,
    config: PipelineConfig,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """
    Validate and enrich edge temporal metadata using dateparser.

    Args:
        edges: List of EntityEdge objects with optional valid_at/invalid_at
        episode_text: Original episode text for context
        reference_time: Episode reference datetime
        config: Pipeline configuration

    Returns:
        Tuple of (enriched_edges, enrichment_records)
    """
    enrichment_records: list[dict[str, Any]] = []

    if not config.temporal_enabled:
        return edges, enrichment_records

    temporal_cues = {
        "valid_markers": ["since", "from", "starting", "began", "as of"],
        "invalid_markers": ["until", "through", "ended", "stopped", "left"],
        "ongoing_markers": ["currently", "now", "present", "today"],
        "terminated_markers": [
            "no longer",
            "formerly",
            "previously",
            "used to",
            "was",
            "were",
        ],
    }

    for edge in edges:
        record = {
            "edge_uuid": edge.uuid,
            "source": edge.source_node_uuid,
            "target": edge.target_node_uuid,
            "relation": edge.name,
            "dspy_valid_at": str(edge.valid_at) if edge.valid_at else None,
            "dspy_invalid_at": str(edge.invalid_at) if edge.invalid_at else None,
            "dateparser_dates": [],
            "cues_detected": [],
            "action": "unchanged",
            "confidence": "low",
        }

        fact_attr = getattr(edge, "fact", None)
        fact_text_candidate = fact_attr if isinstance(fact_attr, str) else ""
        if not fact_text_candidate.strip():
            name_attr = getattr(edge, "name", "")
            fact_text_candidate = name_attr if isinstance(name_attr, str) else ""
        fact_text = fact_text_candidate

        try:
            matches = search_dates(
                fact_text, settings={"RELATIVE_BASE": reference_time}
            )
            date_matches: list[tuple[str, datetime]] = []
            if matches:
                for text_match, match_dt in matches:
                    normalized_dt = ensure_utc(match_dt)
                    if normalized_dt is None:
                        continue
                    date_matches.append((text_match, normalized_dt))
                if date_matches:
                    record["dateparser_dates"] = [
                        (text, dt.isoformat()) for text, dt in date_matches
                    ]
            dates = date_matches
        except Exception as e:
            logger.warning(f"dateparser failed for edge {edge.uuid}: {e}")
            dates = None

        fact_lower = fact_text.lower()

        for marker in temporal_cues["valid_markers"]:
            if marker in fact_lower:
                record["cues_detected"].append(f"valid:{marker}")

        for marker in temporal_cues["invalid_markers"]:
            if marker in fact_lower:
                record["cues_detected"].append(f"invalid:{marker}")

        for marker in temporal_cues["ongoing_markers"]:
            if marker in fact_lower:
                record["cues_detected"].append(f"ongoing:{marker}")

        for marker in temporal_cues["terminated_markers"]:
            if marker in fact_lower:
                record["cues_detected"].append(f"terminated:{marker}")

        has_valid_cues = any(c.startswith("valid:") for c in record["cues_detected"])
        has_invalid_cues = any(
            c.startswith("invalid:") for c in record["cues_detected"]
        )
        has_ongoing_cues = any(
            c.startswith("ongoing:") for c in record["cues_detected"]
        )
        has_terminated_cues = any(
            c.startswith("terminated:") for c in record["cues_detected"]
        )

        if dates and len(dates) > 0:
            dateparser_dt = dates[0][1]

            if edge.valid_at:
                time_diff = abs((edge.valid_at - dateparser_dt).total_seconds())
                if time_diff <= 86400:
                    record["action"] = "validated"
                    record["confidence"] = "high"
                else:
                    logger.warning(
                        f"Temporal conflict for edge {edge.uuid}: "
                        f"DSPy={edge.valid_at}, dateparser={dateparser_dt}. "
                        f"Preferring DSPy (has full context)."
                    )
                    record["action"] = "conflict_dspy_preferred"
                    record["confidence"] = "medium"

            elif edge.invalid_at:
                time_diff = abs((edge.invalid_at - dateparser_dt).total_seconds())
                if time_diff <= 86400:
                    record["action"] = "validated"
                    record["confidence"] = "high"
                else:
                    logger.warning(
                        f"Temporal conflict for edge {edge.uuid}: "
                        f"DSPy invalid_at={edge.invalid_at}, dateparser={dateparser_dt}. "
                        f"Preferring DSPy (has full context)."
                    )
                    record["action"] = "conflict_dspy_preferred"
                    record["confidence"] = "medium"

            else:
                enriched_dt = ensure_utc(dateparser_dt)

                if has_invalid_cues or has_terminated_cues:
                    edge.invalid_at = enriched_dt
                    record["action"] = "enriched_invalid_at"
                    record["confidence"] = "medium"
                elif has_valid_cues:
                    edge.valid_at = enriched_dt
                    record["action"] = "enriched_valid_at"
                    record["confidence"] = "medium"
                elif has_ongoing_cues:
                    edge.valid_at = ensure_utc(reference_time)
                    record["action"] = "enriched_ongoing"
                    record["confidence"] = "medium"
                else:
                    edge.valid_at = enriched_dt
                    record["action"] = "enriched_valid_at_default"
                    record["confidence"] = "low"

        elif has_ongoing_cues and not edge.valid_at:
            edge.valid_at = ensure_utc(reference_time)
            record["action"] = "enriched_ongoing_no_date"
            record["confidence"] = "low"

        enrichment_records.append(record)

    return edges, enrichment_records


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

    # Input validation (mirrors graphiti-core add_episode lines 685-690)
    try:
        validate_entity_types(None)  # No custom entity types in current implementation
        validate_excluded_entity_types(
            None, None
        )  # No exclusions in current implementation
        validate_group_id(config.group_id)
    except Exception as exc:  # noqa: BLE001
        raise GraphitiPipelineError("Input Validation", str(exc)) from exc

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

        # 4a. Remap edge pointers based on entity deduplication
        entity_edges = resolve_edge_pointers(entity_edges, uuid_map)

        # 4b. Enrich temporal metadata with dateparser validation
        if config.temporal_enrichment_enabled:
            entity_edges, temporal_enrichment_records = _enrich_temporal_metadata(
                entity_edges,
                input_text,
                reference_time,
                config,
            )
            logger.info(
                "Temporal enrichment: processed %d edges, %d enriched",
                len(entity_edges),
                sum(
                    1 for r in temporal_enrichment_records if r["action"] != "unchanged"
                ),
            )
        else:
            temporal_enrichment_records = []

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

        # 10. Compress UUID map to handle transitive deduplication
        # Convert dict to list of tuples for compress_uuid_map
        duplicate_pairs = [(k, v) for k, v in uuid_map.items() if k != v]
        if duplicate_pairs:
            uuid_map = compress_uuid_map(duplicate_pairs)

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
            temporal_enrichment_records=temporal_enrichment_records,
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
        relationships, rels_json = infer_relationships(
            text, facts, entities, reference_time
        )
        if relationships is None or ("error" in rels_json):
            raise GraphitiPipelineError(
                "Relationships", rels_json.get("error", "Unknown error")
            )
        base_relationships = relationships
        artifacts.base_relationships = base_relationships
        artifacts.base_relationships_json = rels_json

        final_relationships = base_relationships
        final_relationships_json = rels_json

        llm_relationships = None
        llm_relationships_json: dict[str, Any] | None = None
        if config.llm_edge_detection_enabled:
            llm_relationships, llm_relationships_json = detect_entity_edges(
                text,
                facts,
                entities,
                reference_time,
            )
            artifacts.llm_relationships = llm_relationships
            artifacts.llm_relationships_json = llm_relationships_json
            if llm_relationships is not None:
                final_relationships = _merge_relationship_collections(
                    base_relationships,
                    llm_relationships,
                )
                final_relationships_json = _serialize_relationships(final_relationships)
        else:
            artifacts.llm_relationships = None
            artifacts.llm_relationships_json = {"status": "disabled"}

        artifacts.relationships = final_relationships
        artifacts.relationships_json = final_relationships_json
        relationships = final_relationships

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
        artifacts.temporal_enrichment_records = build_result.temporal_enrichment_records
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
        artifacts.temporal_enrichment_records = build_result.temporal_enrichment_records

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
    "detect_entity_edges",
    "build_graphiti_objects",
    "write_to_falkordb",
    "render_verification_graph",
    "get_db_stats",
    "reset_database",
]
