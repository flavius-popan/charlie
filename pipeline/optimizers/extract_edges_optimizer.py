"""Optimizer for pipeline/extract_edges.py using DSPy's GEPA.

Uses LLM-as-judge (gpt-5-nano) to provide rich textual feedback for optimizing
relationship extraction prompts. Focuses on accurate edge detection and typing.

Usage:
    python -m pipeline.optimizers.extract_edges_optimizer
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

# Script execution (`python pipeline/optimizers/...`) needs the repo root on sys.path
# so `settings` loads before DSPy and sets cache env vars deterministically.
if __package__ is None:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in os.sys.path:
        os.sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import _dspy_setup  # noqa: F401
import dspy  # noqa: E402
from dspy.teleprompt import GEPA  # noqa: E402
from dspy import Prediction  # noqa: E402

from settings import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    MODEL_CONFIG,
    REFLECTION_MODEL,
    REFLECTION_TEMPERATURE,
    REFLECTION_MAX_TOKENS,
    GEPA_REFLECTION_MINIBATCH_SIZE,
    GEPA_MAX_FULL_EVALS,
    GEPA_NUM_THREADS,
    GEPA_OUTPUT_DIR,
)

from inference_runtime import DspyLM
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
        if entity.get("is_author") or entity["name"] == "I":
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
                "name": "I",
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
                "source": "I",
                "target": "Priya",
                "relation": "SpendsTimeWith",
                "fact": "I met Priya at Dolores Park.",
            },
            {
                "source": "I",
                "target": "breathing check-in",
                "relation": "ParticipatesIn",
                "fact": "I practiced the breathing check-in with Priya.",
            },
            {
                "source": "breathing check-in",
                "target": "Dolores Park",
                "relation": "OccursAt",
                "fact": "The breathing check-in happened at Dolores Park.",
            },
        ],
    },
    {
        "episode_content": (
            "I walked solo laps around Lake Merritt after therapy and whispered reminders that I'm safe."
        ),
        "entities": [
            {
                "name": "I",
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
                "source": "I",
                "target": "solo laps",
                "relation": "ParticipatesIn",
                "fact": "I walked solo laps after therapy.",
            },
            {
                "source": "solo laps",
                "target": "Lake Merritt",
                "relation": "OccursAt",
                "fact": "The laps happened at Lake Merritt.",
            },
        ],
    },
    {
        "episode_content": (
            "I called Mom right after my panic spike, and Dr. Chen texted later checking whether I actually took my meds."
        ),
        "entities": [
            {
                "name": "I",
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
                "source": "I",
                "target": "Mom",
                "relation": "Supports",
                "fact": "I called Mom after my panic spike.",
            },
            {
                "source": "Dr. Chen",
                "target": "I",
                "relation": "Supports",
                "fact": "Dr. Chen texted me about my meds."
            },
            {
                "source": "I",
                "target": "panic spike call",
                "relation": "ParticipatesIn",
                "fact": "I called Mom during a panic spike.",
            },
        ],
    },
    {
        "episode_content": (
            "Mara and Tino coaxed me into a shivery sunrise plunge at Ocean Beach. "
            "They kept cracking jokes until I breathed again."
        ),
        "entities": [
            {
                "name": "I",
                "labels": ["Entity", "Person"],
                "is_author": True,
                "notes": SELF_PROMPT_NOTE,
            },
            {"name": "Mara", "labels": ["Entity", "Person"]},
            {"name": "Tino", "labels": ["Entity", "Person"]},
            {"name": "Ocean Beach", "labels": ["Entity", "Place"]},
            {"name": "sunrise plunge", "labels": ["Entity", "Activity"]},
        ],
        "reference_time": "2025-02-14T07:05:00Z",
        "relationships": [
            {
                "source": "Mara",
                "target": "I",
                "relation": "Supports",
                "fact": "Mara coaxed me into the sunrise plunge.",
            },
            {
                "source": "Tino",
                "target": "I",
                "relation": "Supports",
                "fact": "Tino cracked jokes during the plunge.",
            },
            {
                "source": "I",
                "target": "sunrise plunge",
                "relation": "ParticipatesIn",
                "fact": "I did a sunrise plunge with Mara and Tino.",
            },
            {
                "source": "sunrise plunge",
                "target": "Ocean Beach",
                "relation": "OccursAt",
                "fact": "The sunrise plunge happened at Ocean Beach.",
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
                "fact": "Ines texted Avery after their session.",
            },
            {
                "source": "Avery",
                "target": "body scan homework",
                "relation": "ParticipatesIn",
                "fact": "Avery tried the body scan homework.",
            },
        ],
    },
    {
        "episode_content": (
            "Dylan and Sonia argued about whether their sunset hike plan felt safe in the fog."
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
                "fact": "Dylan wanted to do the sunset hike.",
            },
            {
                "source": "Sonia",
                "target": "sunset hike",
                "relation": "ParticipatesIn",
                "fact": "Sonia was concerned about the sunset hike in the fog.",
            },
            {
                "source": "sunset hike",
                "target": "Redwood Ridge",
                "relation": "OccursAt",
                "fact": "The sunset hike happens at Redwood Ridge.",
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
            {
                "name": "I",
                "labels": ["Entity", "Person"],
                "is_author": True,
                "notes": SELF_PROMPT_NOTE,
            },
            {"name": "Theo", "labels": ["Entity", "Person"]},
            {"name": "Redwood Mutual Aid", "labels": ["Entity", "Organization"]},
            {"name": "journaling club", "labels": ["Entity", "Activity"]},
        ],
        "reference_time": "2025-02-06T19:00:00Z",
        "relationships": [
            {
                "source": "I",
                "target": "Theo",
                "relation": "SpendsTimeWith",
                "fact": "I talked with Theo at journaling club.",
            },
            {
                "source": "I",
                "target": "Redwood Mutual Aid",
                "relation": "RELATES_TO",
                "fact": "I mentioned Redwood Mutual Aid to Theo.",
            },
            {
                "source": "Theo",
                "target": "Redwood Mutual Aid",
                "relation": "RELATES_TO",
                "fact": "Theo discussed Redwood Mutual Aid.",
            },
        ],
    },
    {
        "episode_content": (
            "Kai introduced me to his climbing buddy Lou at Touchstone, and we ended up doing a goofy "
            "bouldering session together."
        ),
        "entities": [
            {
                "name": "I",
                "labels": ["Entity", "Person"],
                "is_author": True,
                "notes": SELF_PROMPT_NOTE,
            },
            {"name": "Kai", "labels": ["Entity", "Person"]},
            {"name": "Lou", "labels": ["Entity", "Person"]},
            {"name": "bouldering session", "labels": ["Entity", "Activity"]},
            {"name": "Touchstone Climbing", "labels": ["Entity", "Place"]},
        ],
        "reference_time": "2025-02-09T11:45:00Z",
        "relationships": [
            {
                "source": "Kai",
                "target": "I",
                "relation": "SpendsTimeWith",
                "fact": "Kai introduced me to Lou at Touchstone.",
            },
            {
                "source": "Kai",
                "target": "Lou",
                "relation": "Knows",
                "fact": "Kai knew Lou from climbing.",
            },
            {
                "source": "I",
                "target": "bouldering session",
                "relation": "ParticipatesIn",
                "fact": "I did a goofy bouldering session with Kai and Lou.",
            },
            {
                "source": "Lou",
                "target": "bouldering session",
                "relation": "ParticipatesIn",
                "fact": "Lou joined the bouldering session.",
            },
            {
                "source": "bouldering session",
                "target": "Touchstone Climbing",
                "relation": "OccursAt",
                "fact": "The bouldering session was at Touchstone Climbing.",
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

    lm = DspyLM(model_path=DEFAULT_MODEL_PATH, generation_config=MODEL_CONFIG)
    adapter = dspy.ChatAdapter()
    dspy.configure(lm=lm, adapter=adapter)
    logger.info("Configured DSPy with DspyLM (model: %s)", DEFAULT_MODEL_PATH)


def build_trainset() -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Build relationship extraction examples rooted in personal journals."""

    examples_with_relations: list[tuple[dspy.Example, set[str]]] = []
    for payload in RAW_EXAMPLES:
        relations = {rel["relation"] for rel in payload["relationships"]}
        examples_with_relations.append((_make_example(payload), relations))

    split_idx = min(8, len(examples_with_relations))
    train_examples = examples_with_relations[:split_idx]
    val_examples = examples_with_relations[split_idx:]

    def relation_coverage(examples: list[tuple[dspy.Example, set[str]]]) -> set[str]:
        covered: set[str] = set()
        for _, relations in examples:
            covered.update(relations)
        return covered

    required_relations = set(DEFAULT_EDGE_META.keys())
    missing = required_relations - relation_coverage(train_examples)
    while missing:
        relation = missing.pop()
        idx = next(
            (i for i, (_, rels) in enumerate(val_examples) if relation in rels),
            None,
        )
        if idx is None:
            logger.warning("Training set lacks coverage for relation %s", relation)
            continue
        train_examples.append(val_examples.pop(idx))
        missing = required_relations - relation_coverage(train_examples)

    trainset = [example for example, _ in train_examples]
    valset = [example for example, _ in val_examples]
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


def calculate_relationship_score(
    expected: set[tuple[str, str, str]],
    predicted: set[tuple[str, str, str]]
) -> float:
    """Compute F1 over (source, target, relation) tuples."""

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


def _extract_edge_details(
    relations: ExtractedEdges | list | dict | None,
    entity_names: list[str],
) -> list[dict[str, Any]]:
    """Extract full edge details including facts for quality analysis."""
    if relations is None:
        return []

    if isinstance(relations, ExtractedEdges):
        rel_list = relations.edges
    elif isinstance(relations, dict) and "edges" in relations:
        rel_list = relations["edges"]
    else:
        rel_list = relations

    edges = []
    def _name_from_index(index: int | None) -> str:
        if index is None:
            return ""
        if 0 <= index < len(entity_names):
            return entity_names[index]
        return str(index)

    for rel in rel_list or []:
        try:
            if isinstance(rel, ExtractedEdge):
                edge = {
                    "source": _name_from_index(rel.source_entity_id),
                    "target": _name_from_index(rel.target_entity_id),
                    "relation": rel.relation_type,
                    "fact": rel.fact or "",
                }
            else:
                if "source_entity_id" in rel:
                    source = _name_from_index(rel["source_entity_id"])
                else:
                    source = rel.get("source", "")
                if "target_entity_id" in rel:
                    target = _name_from_index(rel["target_entity_id"])
                else:
                    target = rel.get("target", "")

                edge = {
                    "source": source,
                    "target": target,
                    "relation": rel.get("relation_type") or rel.get("relation", ""),
                    "fact": rel.get("fact", ""),
                }
            edges.append(edge)
        except (KeyError, AttributeError):
            continue

    return edges


def _check_fact_quality_issues(edges: list[dict[str, Any]], entity_names: list[str]) -> list[str]:
    """Automatically detect common fact quality issues."""
    issues = []

    has_self = "i" in [e.lower() for e in entity_names]

    for edge in edges:
        fact = edge.get("fact", "").strip()
        source = edge.get("source", "")
        target = edge.get("target", "")

        if not fact:
            issues.append(f"Empty fact for {source} → {target}")
            continue

        fact_lower = fact.lower()

        # Check for third-person references when the author is involved
        if has_self and (source.lower() == "i" or target.lower() == "i"):
            if "the narrator" in fact_lower:
                issues.append(
                    f"Uses 'the narrator' instead of first-person for {source} → {target}: \"{fact}\""
                )
            elif " i " not in fact_lower and " my " not in fact_lower and " me " not in fact_lower:
                if not fact.startswith("I "):
                    issues.append(
                        f"Missing first-person perspective for I relationship {source} → {target}: \"{fact}\""
                    )

        # Check for overly generic or template-like facts
        if "indicating a" in fact_lower or "suggesting that" in fact_lower:
            issues.append(
                f"Fact sounds analytical rather than grounded: {source} → {target}: \"{fact}\""
            )

    return issues


def generate_feedback(
    expected_rels: set[tuple[str, str, str]],
    predicted_rels: set[tuple[str, str, str]],
    expected_edges: list[dict[str, Any]],
    predicted_edges: list[dict[str, Any]],
    entity_names: list[str],
    score: float,
    judge_lm: dspy.LM
) -> str:
    """Use judge LM to generate actionable feedback including fact quality analysis."""

    missing = expected_rels - predicted_rels
    extra = predicted_rels - expected_rels
    correct = expected_rels & predicted_rels

    # Detect automatic fact quality issues
    quality_issues = _check_fact_quality_issues(predicted_edges, entity_names)
    quality_section = ""
    if quality_issues:
        quality_section = "\n\nAutomatic Fact Quality Issues Detected:\n" + "\n".join(f"- {issue}" for issue in quality_issues)

    # Format predicted edges with facts for judge review
    predicted_facts_str = "\n".join([
        f"  {edge['source']} → {edge['target']} [{edge['relation']}]: \"{edge['fact']}\""
        for edge in predicted_edges
    ])

    # Format expected edges with facts for comparison
    expected_facts_str = "\n".join([
        f"  {edge['source']} → {edge['target']} [{edge['relation']}]: \"{edge['fact']}\""
        for edge in expected_edges
    ])

    has_self = "I" in entity_names

    feedback_prompt = f"""Evaluate this relationship extraction from a journal entry and provide specific, actionable feedback.

Expected relationships: {sorted(expected_rels)}
Predicted relationships: {sorted(predicted_rels)}
F1 Score: {score:.2f}

Correct: {sorted(correct)}
Missing: {sorted(missing)}
Extra (hallucinated): {sorted(extra)}

EXPECTED EDGES WITH FACTS:
{expected_facts_str}

PREDICTED EDGES WITH FACTS:
{predicted_facts_str}
{quality_section}

Provide feedback on:
1. **Relationship Accuracy**: Are the (source, target, relation) tuples correct?
2. **Completeness**: Are all important relationships from the journal entry captured?
3. **Precision**: Are there any hallucinated relationships not supported by the text?
4. **Fact Grounding**: {"Are facts properly written in FIRST PERSON (using 'I', 'my', 'me') when I am involved? " if has_self else ""}Are facts grounded in the actual journal entry content, or do they sound analytical/detached?
5. **Fact Conciseness**: Are facts SHORT and DIRECT (ideally one sentence)? Do they avoid emotional interpretation or explaining WHY things happened?

CRITICAL RULES FOR FACTS:
- When I am a participant, facts MUST use first-person pronouns ("I spent time...", "I reached out...", "she checked on me...")
- NEVER use "the narrator" - use "I/me/my" or the actual entity names
- Facts should be SHORT, DIRECT, and FACTUAL - similar to concise journal summaries
- State WHAT happened, not WHY or how people felt about it (unless the feeling is the core fact)
- Avoid verbose explanations: "I reached out to Mom" NOT "I reached out to Mom for support after the panic spike"
- Avoid analytical phrases: "indicating a", "suggesting that", "implying", etc.
- Avoid overfitting: not every person entity relates to I

Be specific and actionable in your feedback."""

    logger.info("=" * 80)
    logger.info("JUDGE EVALUATION REQUEST")
    logger.info(f"Score: {score:.2f}")
    logger.info(f"Correct: {len(correct)}, Missing: {len(missing)}, Extra: {len(extra)}")
    if quality_issues:
        logger.info(f"Quality issues found: {len(quality_issues)}")

    feedback = judge_lm(feedback_prompt)[0]

    logger.info("JUDGE FEEDBACK:")
    logger.info(feedback)
    logger.info("=" * 80)

    return feedback


def relationship_extraction_metric(example, prediction, trace=None) -> float:
    """Compute F1 over (source, target, relation) tuples."""

    expected = _relationship_set(example.edges, example.entity_names)
    predicted_source = getattr(prediction, "edges", prediction)
    predicted = _relationship_set(predicted_source, example.entity_names)

    return calculate_relationship_score(expected, predicted)


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
    """Full optimization workflow for edge extraction using GEPA."""

    logging.basicConfig(level=logging.INFO)

    # Configure task LM
    configure_dspy()

    # Validate OPENAI_API_KEY for judge LM
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set for GEPA reflection model. "
            f"The reflection model ({REFLECTION_MODEL}) requires OpenAI API access."
        )

    # Create judge LM
    judge_lm = dspy.LM(
        model=REFLECTION_MODEL,
        api_key=api_key,
        temperature=REFLECTION_TEMPERATURE,
        max_tokens=REFLECTION_MAX_TOKENS
    )
    logger.info(
        "Configured judge LM: %s (temp=%.1f, max_tokens=%d)",
        REFLECTION_MODEL,
        REFLECTION_TEMPERATURE,
        REFLECTION_MAX_TOKENS
    )

    # Build datasets
    trainset, valset = build_trainset()

    # Create GEPA-compatible metric with judge_lm bound via closure
    def gepa_relationship_metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> Prediction:
        """GEPA-compatible metric that returns ScoreWithFeedback.

        Note: Only calls expensive judge LM during GEPA reflection phase (pred_name != None).
        Regular evaluations use simple feedback to save costs and time.
        """

        expected = _relationship_set(gold.edges, gold.entity_names)
        predicted_source = getattr(pred, "edges", pred)
        predicted = _relationship_set(predicted_source, gold.entity_names)
        score = calculate_relationship_score(expected, predicted)

        # Only call expensive judge LM during GEPA reflection phase (when pred_name provided)
        if pred_name:
            logger.info("-" * 80)
            logger.info(f"EVALUATING PREDICTOR: {pred_name}")

            # Extract full edge details including facts for quality analysis
            expected_edges = _extract_edge_details(gold.edges, gold.entity_names)
            predicted_edges = _extract_edge_details(predicted_source, gold.entity_names)

            feedback = generate_feedback(
                expected_rels=expected,
                predicted_rels=predicted,
                expected_edges=expected_edges,
                predicted_edges=predicted_edges,
                entity_names=gold.entity_names,
                score=score,
                judge_lm=judge_lm
            )

            logger.info(f"METRIC SCORE: {score:.2f}")
            logger.info("-" * 80)
        else:
            # Simple feedback for regular evaluations (no expensive LLM call)
            missing = len(expected - predicted)
            extra = len(predicted - expected)
            feedback = f"Score: {score:.2f}. Missing {missing}, Extra {extra} relationships."

        return Prediction(score=score, feedback=feedback)

    # Evaluate baseline
    baseline = EdgeExtractor()
    baseline_score = evaluate(baseline, valset)
    logger.info("Baseline score (valset): %.3f", baseline_score)

    # Create log directory for GEPA artifacts
    log_dir = GEPA_OUTPUT_DIR / "extract_edges"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info("GEPA logs will be saved to: %s", log_dir)

    # Instantiate and run GEPA
    logger.info("Starting GEPA optimization with max_full_evals=%d", GEPA_MAX_FULL_EVALS)
    gepa = GEPA(
        metric=gepa_relationship_metric,
        max_full_evals=GEPA_MAX_FULL_EVALS,
        reflection_lm=judge_lm,
        reflection_minibatch_size=GEPA_REFLECTION_MINIBATCH_SIZE,
        track_stats=True,
        log_dir=str(log_dir),
        num_threads=GEPA_NUM_THREADS,
    )

    optimized = gepa.compile(
        student=baseline,
        trainset=trainset,
        valset=valset
    )

    # Evaluate optimized
    optimized_score = evaluate(optimized, valset)
    logger.info("Optimized score (valset): %.3f", optimized_score)

    # Save optimized prompts
    PROMPT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    optimized.save(str(PROMPT_OUTPUT))
    logger.info("Saved optimized prompts to %s", PROMPT_OUTPUT)
    logger.info(
        "Improvement: %.3f → %.3f (+%.3f)",
        baseline_score,
        optimized_score,
        optimized_score - baseline_score,
    )


if __name__ == "__main__":
    main()
