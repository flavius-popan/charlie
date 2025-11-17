"""Optimizer for pipeline/extract_edges.py using DSPy's BootstrapFewShot.

This script tunes the EdgeExtractor prompts so relationship extraction stays
grounded in people-focused journal entries without slowing the full pipeline.

Usage:
    python -m pipeline.optimizers.extract_edges_optimizer
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import logging
import dspy
from dspy.teleprompt import MIPROv2

from mlx_runtime import MLXDspyLM
from pipeline.entity_edge_models import (
    edge_types as DEFAULT_EDGE_TYPES,
    edge_type_map as DEFAULT_EDGE_TYPE_MAP,
    edge_meta as DEFAULT_EDGE_META,
)
from pipeline.extract_edges import (
    EdgeExtractor,
    ExtractedEdge,
    ExtractedEdges,
    _build_edge_type_context,
)
from pipeline.self_reference import SELF_PROMPT_NOTE
from settings import DEFAULT_MODEL_PATH, MODEL_CONFIG


PROMPT_OUTPUT = Path(__file__).parent.parent / "prompts" / "extract_edges.json"
logger = logging.getLogger(__name__)

EDGE_TYPE_CONTEXT = _build_edge_type_context(
    DEFAULT_EDGE_TYPES, DEFAULT_EDGE_TYPE_MAP, DEFAULT_EDGE_META
)
EMPTY_PREVIOUS_EPISODES = "[]"


def _entities_json(entities: list[dict[str, Any]]) -> str:
    payload = []
    for idx, entity in enumerate(entities):
        entry = {
            "id": idx,
            "name": entity["name"],
            "entity_types": entity.get("labels", ["Entity"]),
        }
        if entity.get("is_author") or entity["name"] == "Self":
            entry["is_author"] = True
            entry["notes"] = entity.get("notes") or SELF_PROMPT_NOTE
        elif entity.get("notes"):
            entry["notes"] = entity["notes"]
        payload.append(entry)
    return json.dumps(payload, ensure_ascii=False)


def _build_edges_from_descriptions(
    entities: list[dict[str, Any]],
    relations: list[dict[str, str]],
) -> ExtractedEdges:
    edges: list[ExtractedEdge] = []
    name_to_index = {entity["name"]: idx for idx, entity in enumerate(entities)}
    for rel in relations:
        try:
            source_idx = name_to_index[rel["source"]]
            target_idx = name_to_index[rel["target"]]
        except KeyError:
            continue

        edges.append(
            ExtractedEdge(
                source_entity_id=source_idx,
                target_entity_id=target_idx,
                relation_type=rel["relation"],
                fact=rel["fact"],
                valid_at=rel.get("valid_at"),
                invalid_at=rel.get("invalid_at"),
            )
        )

    return ExtractedEdges(edges=edges)


def _make_example(payload: dict[str, Any]) -> dspy.Example:
    entities = payload["entities"]
    entity_names = [entity["name"] for entity in entities]
    example = dspy.Example(
        episode_content=payload["episode_content"],
        entities_json=_entities_json(entities),
        entity_names=entity_names,
        reference_time=payload["reference_time"],
        edge_type_context=EDGE_TYPE_CONTEXT,
        previous_episodes_json=EMPTY_PREVIOUS_EPISODES,
        edges=_build_edges_from_descriptions(entities, payload["relationships"]),
    )
    return example.with_inputs(
        "episode_content",
        "entities_json",
        "reference_time",
        "edge_type_context",
        "previous_episodes_json",
    )


RAW_EXAMPLES: list[dict[str, Any]] = [
    {
        "episode_content": (
            "I met Priya at Dolores Park so she could walk me through a breathing check-in. "
            "We hugged under the palms while her dog Ziggy kept circling us."
        ),
        "entities": [
            {
                "name": "Self",
                "labels": ["Entity", "Person"],
                "is_author": True,
                "notes": SELF_PROMPT_NOTE,
            },
            {"name": "Priya", "labels": ["Entity", "Person"]},
            {"name": "Ziggy", "labels": ["Entity", "Person"]},
            {"name": "Dolores Park", "labels": ["Entity", "Place"]},
            {"name": "breathing check-in", "labels": ["Entity", "Activity"]},
        ],
        "reference_time": "2025-02-20T19:30:00Z",
        "relationships": [
            {
                "source": "Self",
                "target": "Priya",
                "relation": "SpendsTimeWith",
                "fact": "I spent time with Priya in person at Dolores Park.",
            },
            {
                "source": "Self",
                "target": "breathing check-in",
                "relation": "ParticipatesIn",
                "fact": "I practiced the breathing check-in Priya modeled.",
            },
            {
                "source": "breathing check-in",
                "target": "Dolores Park",
                "relation": "OccursAt",
                "fact": "We did the breathing check-in right at Dolores Park.",
            },
        ],
    },
    {
        "episode_content": (
            "I walked solo laps around Lake Merritt after therapy and whispered reminders that I'm safe."
        ),
        "entities": [
            {
                "name": "Self",
                "labels": ["Entity", "Person"],
                "is_author": True,
                "notes": SELF_PROMPT_NOTE,
            },
            {"name": "Lake Merritt", "labels": ["Entity", "Place"]},
            {"name": "solo laps", "labels": ["Entity", "Activity"]},
            {"name": "therapy session", "labels": ["Entity", "Activity"]},
        ],
        "reference_time": "2025-02-18T06:40:00Z",
        "relationships": [
            {
                "source": "Self",
                "target": "solo laps",
                "relation": "ParticipatesIn",
                "fact": "I took solo laps to ground after therapy.",
            },
            {
                "source": "solo laps",
                "target": "Lake Merritt",
                "relation": "OccursAt",
                "fact": "Those laps happened at Lake Merritt.",
            },
        ],
    },
    {
        "episode_content": (
            "I called Mom right after my panic spike, and Dr. Chen texted later checking whether I actually took my meds."
        ),
        "entities": [
            {
                "name": "Self",
                "labels": ["Entity", "Person"],
                "is_author": True,
                "notes": SELF_PROMPT_NOTE,
            },
            {"name": "Mom", "labels": ["Entity", "Person"]},
            {"name": "Dr. Chen", "labels": ["Entity", "Person"]},
            {"name": "panic spike call", "labels": ["Entity", "Activity"]},
        ],
        "reference_time": "2025-02-16T22:15:00Z",
        "relationships": [
            {
                "source": "Self",
                "target": "Mom",
                "relation": "Supports",
                "fact": "I reached out to Mom for support after the panic spike.",
            },
            {
                "source": "Dr. Chen",
                "target": "Self",
                "relation": "Supports",
                "fact": "Dr. Chen checked on me about my medication."
            },
            {
                "source": "Self",
                "target": "panic spike call",
                "relation": "ParticipatesIn",
                "fact": "I initiated the panic spike call ritual to ground myself.",
            },
        ],
    },
    {
        "episode_content": (
            "Mara and Tino coaxed me into a shivery sunrise plunge at Ocean Beach. "
            "They kept cracking jokes until I breathed again."
        ),
        "entities": [
            {"name": "Mara", "labels": ["Entity", "Person"]},
            {"name": "Tino", "labels": ["Entity", "Person"]},
            {"name": "Ocean Beach", "labels": ["Entity", "Place"]},
            {"name": "sunrise plunge", "labels": ["Entity", "Activity"]},
        ],
        "reference_time": "2025-02-14T07:05:00Z",
        "relationships": [
            {
                "source": "Mara",
                "target": "Tino",
                "relation": "SpendsTimeWith",
                "fact": "Mara and Tino share sunrise plunges together.",
            },
            {
                "source": "Mara",
                "target": "sunrise plunge",
                "relation": "ParticipatesIn",
                "fact": "Mara joined the sunrise plunge ritual.",
            },
            {
                "source": "sunrise plunge",
                "target": "Ocean Beach",
                "relation": "OccursAt",
                "fact": "The sunrise plunge happens at Ocean Beach.",
            },
        ],
    },
    {
        "episode_content": (
            "Therapist Ines texted Avery after session to make sure she actually tried the body scan homework."
        ),
        "entities": [
            {"name": "Ines", "labels": ["Entity", "Person"]},
            {"name": "Avery", "labels": ["Entity", "Person"]},
            {"name": "body scan homework", "labels": ["Entity", "Activity"]},
        ],
        "reference_time": "2025-02-10T18:30:00Z",
        "relationships": [
            {
                "source": "Ines",
                "target": "Avery",
                "relation": "Supports",
                "fact": "Ines checked in to support Avery between sessions.",
            },
            {
                "source": "Avery",
                "target": "body scan homework",
                "relation": "ParticipatesIn",
                "fact": "Avery committed to the body scan homework.",
            },
        ],
    },
    {
        "episode_content": (
            "Dylan and Sonia argued about whether our sunset hike plan felt safe in the fog."
        ),
        "entities": [
            {"name": "Dylan", "labels": ["Entity", "Person"]},
            {"name": "Sonia", "labels": ["Entity", "Person"]},
            {"name": "sunset hike", "labels": ["Entity", "Activity"]},
            {"name": "Redwood Ridge", "labels": ["Entity", "Place"]},
        ],
        "reference_time": "2025-01-22T20:10:00Z",
        "relationships": [
            {
                "source": "Dylan",
                "target": "Sonia",
                "relation": "ConflictsWith",
                "fact": "Dylan and Sonia argued about the plan.",
            },
            {
                "source": "Dylan",
                "target": "sunset hike",
                "relation": "ParticipatesIn",
                "fact": "Dylan still wanted to do the sunset hike.",
            },
            {
                "source": "sunset hike",
                "target": "Redwood Ridge",
                "relation": "OccursAt",
                "fact": "The hike happens at Redwood Ridge.",
            },
        ],
    },
    {
        "episode_content": (
            "Grandma Rosa keeps visiting the 24th Street Community Garden when she needs grounding."
        ),
        "entities": [
            {"name": "Grandma Rosa", "labels": ["Entity", "Person"]},
            {"name": "24th Street Community Garden", "labels": ["Entity", "Place"]},
        ],
        "reference_time": "2025-02-03T16:00:00Z",
        "relationships": [
            {
                "source": "Grandma Rosa",
                "target": "24th Street Community Garden",
                "relation": "Visits",
                "fact": "She visits the garden for grounding.",
            }
        ],
    },
    {
        "episode_content": (
            "I mentioned Redwood Mutual Aid to Theo during journaling club, but we still don't know exactly "
            "how he'll plug in."
        ),
        "entities": [
            {"name": "Theo", "labels": ["Entity", "Person"]},
            {"name": "Redwood Mutual Aid", "labels": ["Entity", "Organization"]},
        ],
        "reference_time": "2025-02-06T19:00:00Z",
        "relationships": [
            {
                "source": "Theo",
                "target": "Redwood Mutual Aid",
                "relation": "RELATES_TO",
                "fact": "Theo referenced Redwood Mutual Aid without a clear role.",
            }
        ],
    },
    {
        "episode_content": (
            "Kai introduced me to his climbing buddy Lou at Touchstone, and we ended up doing a goofy "
            "bouldering session together."
        ),
        "entities": [
            {"name": "Kai", "labels": ["Entity", "Person"]},
            {"name": "Lou", "labels": ["Entity", "Person"]},
            {"name": "bouldering session", "labels": ["Entity", "Activity"]},
            {"name": "Touchstone Climbing", "labels": ["Entity", "Place"]},
        ],
        "reference_time": "2025-02-09T11:45:00Z",
        "relationships": [
            {
                "source": "Kai",
                "target": "Lou",
                "relation": "Knows",
                "fact": "Kai already knew Lou from the gym.",
            },
            {
                "source": "Lou",
                "target": "bouldering session",
                "relation": "ParticipatesIn",
                "fact": "Lou led the bouldering session.",
            },
            {
                "source": "bouldering session",
                "target": "Touchstone Climbing",
                "relation": "OccursAt",
                "fact": "The session was at Touchstone Climbing.",
            },
        ],
    },
    {
        "episode_content": (
            "Neighbor Laila brought soup when Ana's migraine flared and sat with her until the meds kicked in."
        ),
        "entities": [
            {"name": "Laila", "labels": ["Entity", "Person"]},
            {"name": "Ana", "labels": ["Entity", "Person"]},
            {"name": "third-floor apartment", "labels": ["Entity", "Place"]},
        ],
        "reference_time": "2025-01-25T21:40:00Z",
        "relationships": [
            {
                "source": "Laila",
                "target": "Ana",
                "relation": "Supports",
                "fact": "Laila comforted Ana during the migraine.",
            },
            {
                "source": "Laila",
                "target": "third-floor apartment",
                "relation": "Visits",
                "fact": "Laila stopped by Ana's apartment to help.",
            },
        ],
    },
    {
        "episode_content": (
            "Jenna hosted knit circle at Tender Hearts Center so we could prep blankets for Saturday's vigil."
        ),
        "entities": [
            {"name": "Jenna", "labels": ["Entity", "Person"]},
            {"name": "knit circle", "labels": ["Entity", "Activity"]},
            {"name": "Tender Hearts Center", "labels": ["Entity", "Place"]},
        ],
        "reference_time": "2025-02-11T18:50:00Z",
        "relationships": [
            {
                "source": "Jenna",
                "target": "knit circle",
                "relation": "ParticipatesIn",
                "fact": "Jenna led the knit circle prep.",
            },
            {
                "source": "knit circle",
                "target": "Tender Hearts Center",
                "relation": "OccursAt",
                "fact": "Knit circle met at Tender Hearts Center.",
            },
        ],
    },
    {
        "episode_content": (
            "Yara and Mo spent the afternoon at the studio sketching plans for our community mural."
        ),
        "entities": [
            {"name": "Yara", "labels": ["Entity", "Person"]},
            {"name": "Mo", "labels": ["Entity", "Person"]},
            {"name": "mural planning session", "labels": ["Entity", "Activity"]},
        ],
        "reference_time": "2025-02-12T15:10:00Z",
        "relationships": [
            {
                "source": "Yara",
                "target": "Mo",
                "relation": "SpendsTimeWith",
                "fact": "Yara and Mo designed the mural together.",
            },
            {
                "source": "Yara",
                "target": "mural planning session",
                "relation": "ParticipatesIn",
                "fact": "Yara sketched during the planning session.",
            },
        ],
    },
    {
        "episode_content": (
            "Casey stopped by Loom House to ask if the grief circle could host next week's gathering there."
        ),
        "entities": [
            {"name": "Casey", "labels": ["Entity", "Person"]},
            {"name": "Loom House", "labels": ["Entity", "Place"]},
            {"name": "grief circle", "labels": ["Entity", "Activity"]},
        ],
        "reference_time": "2025-02-04T13:25:00Z",
        "relationships": [
            {
                "source": "Casey",
                "target": "Loom House",
                "relation": "Visits",
                "fact": "Casey visited Loom House to make the ask.",
            },
            {
                "source": "grief circle",
                "target": "Loom House",
                "relation": "OccursAt",
                "fact": "The grief circle plans to meet at Loom House.",
            },
        ],
    },
]


def configure_dspy():
    """Configure DSPy with MLX LM and stock adapter identical to runtime."""

    lm = MLXDspyLM(model_path=DEFAULT_MODEL_PATH, generation_config=MODEL_CONFIG)
    adapter = dspy.ChatAdapter()
    dspy.configure(lm=lm, adapter=adapter)
    logger.info("Configured DSPy with MLXDspyLM (model: %s)", DEFAULT_MODEL_PATH)


def build_trainset() -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Build relationship extraction examples rooted in personal journals."""

    all_examples: list[dspy.Example] = []
    for payload in RAW_EXAMPLES:
        all_examples.append(_make_example(payload))

    trainset = all_examples[:8]
    valset = all_examples[8:]
    logger.info(
        "Built relationship trainset with %d examples, valset with %d examples",
        len(trainset),
        len(valset),
    )
    return trainset, valset


def _relationship_set(
    relations: ExtractedEdges | list | dict | None,
    entity_names: list[str],
) -> set[tuple[str, str, str]]:
    """Normalize adapter outputs into comparable tuples."""

    if relations is None:
        return set()

    if isinstance(relations, ExtractedEdges):
        rel_list = relations.edges
    elif isinstance(relations, dict) and "edges" in relations:
        rel_list = relations["edges"]
    else:
        rel_list = relations

    normalized: set[tuple[str, str, str]] = set()

    def _name_from_index(index: int | None) -> str:
        if index is None:
            return ""
        if 0 <= index < len(entity_names):
            return entity_names[index].lower()
        return str(index)

    for rel in rel_list or []:
        try:
            if isinstance(rel, ExtractedEdge):
                source = _name_from_index(rel.source_entity_id)
                target = _name_from_index(rel.target_entity_id)
                relation = rel.relation_type.upper()
            else:
                if "source_entity_id" in rel:
                    source = _name_from_index(rel["source_entity_id"])
                else:
                    source = rel["source"].lower()
                if "target_entity_id" in rel:
                    target = _name_from_index(rel["target_entity_id"])
                else:
                    target = rel["target"].lower()
                relation = rel.get("relation_type") or rel.get("relation")
                relation = (relation or "").upper()
            normalized.add((source, target, relation))
        except (KeyError, AttributeError):
            continue

    return normalized


def relationship_extraction_metric(example, prediction, trace=None) -> float:
    """Compute F1 over (source, target, relation) tuples."""

    expected = _relationship_set(example.edges, example.entity_names)
    predicted_source = getattr(prediction, "edges", prediction)
    predicted = _relationship_set(predicted_source, example.entity_names)

    if not expected:
        return 1.0 if not predicted else 0.0

    if not predicted:
        return 0.0

    intersection = expected & predicted
    if not intersection:
        return 0.0

    precision = len(intersection) / len(predicted)
    recall = len(intersection) / len(expected)
    return 2 * (precision * recall) / (precision + recall)


def optimize(trainset: list[dspy.Example]) -> EdgeExtractor:
    """Run MIPROv2 optimization for EdgeExtractor."""

    logger.info("Starting edge optimization with %d examples", len(trainset))
    optimizer = MIPROv2(
        metric=relationship_extraction_metric,
        auto=None,
        num_candidates=3,
        init_temperature=0.5,
        metric_threshold=0.90,
    )

    student = EdgeExtractor()
    optimized = optimizer.compile(
        student=student,
        trainset=trainset,
        num_trials=10,
        max_bootstrapped_demos=2,
        max_labeled_demos=3,
        minibatch_size=2,
        requires_permission_to_run=False,
    )

    logger.info("Edge optimization completed")
    return optimized


def evaluate(module: EdgeExtractor, dataset: list[dspy.Example]) -> float:
    """Average relationship metric across dataset."""

    scores: list[float] = []
    for example in dataset:
        prediction = module(
            episode_content=example.episode_content,
            entities_json=example.entities_json,
            reference_time=example.reference_time,
            edge_type_context=example.edge_type_context,
            previous_episodes_json=example.previous_episodes_json,
        )
        score = relationship_extraction_metric(example, prediction)
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


def main():
    """Full optimization workflow for edge extraction."""

    logging.basicConfig(level=logging.INFO)
    configure_dspy()

    trainset, valset = build_trainset()
    baseline_module = EdgeExtractor()
    baseline_score = evaluate(baseline_module, valset)
    logger.info("Baseline score (valset): %.3f", baseline_score)

    optimized_module = optimize(trainset)
    optimized_score = evaluate(optimized_module, valset)
    logger.info("Optimized score (valset): %.3f", optimized_score)

    PROMPT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    optimized_module.save(str(PROMPT_OUTPUT))
    logger.info("Saved optimized prompts to %s", PROMPT_OUTPUT)
    logger.info(
        "Improvement: %.3f → %.3f (+%.3f)",
        baseline_score,
        optimized_score,
        optimized_score - baseline_score,
    )


if __name__ == "__main__":
    main()
