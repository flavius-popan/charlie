"""Optimizer for pipeline/extract_edges.py using DSPy's BootstrapFewShot.

This script tunes the EdgeExtractor prompts so relationship extraction stays
grounded in people-focused journal entries without slowing the full pipeline.

Usage:
    python -m pipeline.optimizers.extract_edges_optimizer
"""

from __future__ import annotations

from pathlib import Path
import logging
import dspy
from dspy.teleprompt import MIPROv2

from dspy_outlines import OutlinesAdapter, OutlinesLM
from pipeline.extract_edges import (
    EdgeExtractor,
    ExtractedRelationship,
    ExtractedRelationships,
)
from settings import DEFAULT_MODEL_PATH, MODEL_CONFIG


PROMPT_OUTPUT = Path(__file__).parent.parent / "prompts" / "extract_edges.json"
logger = logging.getLogger(__name__)


def configure_dspy():
    """Configure DSPy with OutlinesLM and adapter identical to runtime."""

    lm = OutlinesLM(model_path=DEFAULT_MODEL_PATH, generation_config=MODEL_CONFIG)
    adapter = OutlinesAdapter()
    dspy.configure(lm=lm, adapter=adapter)
    logger.info("Configured DSPy with OutlinesLM (model: %s)", DEFAULT_MODEL_PATH)


def build_trainset() -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Build relationship extraction examples rooted in personal journals."""

    all_examples: list[dspy.Example] = []

    all_examples.append(
        dspy.Example(
            episode_content=(
                "Met Jonah at Dolores Park before sunrise yoga. "
                "His sister Maya tagged along and reminded me about volunteering at "
                "Tender Hearts shelter this weekend."
            ),
            entities=[
                "Jonah",
                "Maya",
                "Dolores Park",
                "sunrise yoga",
                "Tender Hearts shelter",
            ],
            reference_time="2025-02-14T07:00:00Z",
            relationships=ExtractedRelationships(
                relationships=[
                    ExtractedRelationship(
                        source="Jonah",
                        target="Dolores Park",
                        relation="MEETS_AT",
                        fact="Met Jonah at Dolores Park for sunrise yoga.",
                    ),
                    ExtractedRelationship(
                        source="Jonah",
                        target="sunrise yoga",
                        relation="PARTICIPATES_IN",
                        fact="Jonah joined me for sunrise yoga.",
                    ),
                    ExtractedRelationship(
                        source="Maya",
                        target="Tender Hearts shelter",
                        relation="VOLUNTEERS_AT",
                        fact="Maya reminded me about helping at Tender Hearts shelter.",
                    ),
                ]
            ),
        ).with_inputs("episode_content", "entities", "reference_time")
    )

    all_examples.append(
        dspy.Example(
            episode_content=(
                "Spent the evening on video call with Dr. Perez at City Hospital "
                "while he checked on Grandma Lila's breathing. "
                "He promised to swing by her room again at dawn."
            ),
            entities=["Dr. Perez", "City Hospital", "Grandma Lila"],
            reference_time="2025-01-28T22:15:00Z",
            relationships=ExtractedRelationships(
                relationships=[
                    ExtractedRelationship(
                        source="Dr. Perez",
                        target="City Hospital",
                        relation="WORKS_AT",
                        fact="Dr. Perez called from City Hospital.",
                    ),
                    ExtractedRelationship(
                        source="Dr. Perez",
                        target="Grandma Lila",
                        relation="TREATS",
                        fact="He monitored Grandma Lila's breathing.",
                    ),
                ]
            ),
        ).with_inputs("episode_content", "entities", "reference_time")
    )

    all_examples.append(
        dspy.Example(
            episode_content=(
                "Tasha and I road-tripped to Big Basin Trailhead where Ranger Eli "
                "handed us fog maps and told us to stick to the ridge."
            ),
            entities=["Tasha", "Big Basin Trailhead", "Ranger Eli"],
            reference_time="2025-01-12T15:45:00Z",
            relationships=ExtractedRelationships(
                relationships=[
                    ExtractedRelationship(
                        source="Tasha",
                        target="Big Basin Trailhead",
                        relation="VISITS",
                        fact="Tasha rode with me to Big Basin Trailhead.",
                    ),
                    ExtractedRelationship(
                        source="Ranger Eli",
                        target="Big Basin Trailhead",
                        relation="WORKS_AT",
                        fact="Ranger Eli staffed the trailhead kiosk.",
                    ),
                    ExtractedRelationship(
                        source="Ranger Eli",
                        target="Tasha",
                        relation="GUIDES",
                        fact="He handed us maps and gave fog warnings.",
                    ),
                ]
            ),
        ).with_inputs("episode_content", "entities", "reference_time")
    )

    all_examples.append(
        dspy.Example(
            episode_content=(
                "Celebrated Amina's new gig at River Labs over noodles at Golden Lotus "
                "Cafe. She asked if I'd consult on their mindfulness study next month."
            ),
            entities=["Amina", "River Labs", "Golden Lotus Cafe", "mindfulness study"],
            reference_time="2025-02-01T18:20:00Z",
            relationships=ExtractedRelationships(
                relationships=[
                    ExtractedRelationship(
                        source="Amina",
                        target="Golden Lotus Cafe",
                        relation="MEETS_AT",
                        fact="We celebrated Amina at Golden Lotus Cafe.",
                    ),
                    ExtractedRelationship(
                        source="Amina",
                        target="River Labs",
                        relation="WORKS_AT",
                        fact="Amina just joined River Labs.",
                    ),
                    ExtractedRelationship(
                        source="River Labs",
                        target="mindfulness study",
                        relation="RUNS",
                        fact="River Labs is launching a mindfulness study.",
                    ),
                ]
            ),
        ).with_inputs("episode_content", "entities", "reference_time")
    )

    all_examples.append(
        dspy.Example(
            episode_content=(
                "Met Arturo and Carmen at the public library for counseling certification "
                "prep. Carmen organized the study group and Arturo kept drilling me on "
                "crisis flashcards."
            ),
            entities=[
                "Arturo",
                "Carmen",
                "public library",
                "study group",
                "counseling certification prep",
            ],
            reference_time="2025-02-08T10:30:00Z",
            relationships=ExtractedRelationships(
                relationships=[
                    ExtractedRelationship(
                        source="Arturo",
                        target="public library",
                        relation="MEETS_AT",
                        fact="Arturo met me at the public library.",
                    ),
                    ExtractedRelationship(
                        source="Carmen",
                        target="study group",
                        relation="LEADS",
                        fact="Carmen organized the study group.",
                    ),
                    ExtractedRelationship(
                        source="study group",
                        target="counseling certification prep",
                        relation="FOCUSES_ON",
                        fact="Our study group focused on the counseling exam.",
                    ),
                ]
            ),
        ).with_inputs("episode_content", "entities", "reference_time")
    )

    all_examples.append(
        dspy.Example(
            episode_content=(
                "Hosted our tiny sobriety check-in tonight. Joy facilitated the circle "
                "while Marcus admitted leaning hard on his sponsor Rita every day."
            ),
            entities=["Joy", "Marcus", "Rita", "sobriety check-in"],
            reference_time="2025-01-30T21:05:00Z",
            relationships=ExtractedRelationships(
                relationships=[
                    ExtractedRelationship(
                        source="Joy",
                        target="sobriety check-in",
                        relation="FACILITATES",
                        fact="Joy facilitated the sobriety check-in.",
                    ),
                    ExtractedRelationship(
                        source="Marcus",
                        target="sobriety check-in",
                        relation="PARTICIPATES_IN",
                        fact="Marcus attended the check-in.",
                    ),
                    ExtractedRelationship(
                        source="Marcus",
                        target="Rita",
                        relation="SUPPORTED_BY",
                        fact="Marcus leans on his sponsor Rita.",
                    ),
                ]
            ),
        ).with_inputs("episode_content", "entities", "reference_time")
    )

    all_examples.append(
        dspy.Example(
            episode_content=(
                "After running club practice, Priya and Leo came over for dinner. "
                "Priya coaches with Mission Milers and asked Leo to pace me Saturday."
            ),
            entities=["Priya", "Leo", "Mission Milers", "running club dinner"],
            reference_time="2025-02-05T19:40:00Z",
            relationships=ExtractedRelationships(
                relationships=[
                    ExtractedRelationship(
                        source="Priya",
                        target="Mission Milers",
                        relation="COACHES_FOR",
                        fact="Priya coaches with Mission Milers.",
                    ),
                    ExtractedRelationship(
                        source="Priya",
                        target="Leo",
                        relation="TRAINS_WITH",
                        fact="Priya asked Leo to pace me.",
                    ),
                    ExtractedRelationship(
                        source="Leo",
                        target="running club dinner",
                        relation="ATTENDS",
                        fact="Leo came over for the running club dinner.",
                    ),
                ]
            ),
        ).with_inputs("episode_content", "entities", "reference_time")
    )

    all_examples.append(
        dspy.Example(
            episode_content=(
                "Camped out at Harbor Clinic this afternoon. Nurse Tami introduced me to "
                "social worker Ben so we can finally tackle mom's paperwork."
            ),
            entities=["Harbor Clinic", "Nurse Tami", "Ben", "mom's paperwork"],
            reference_time="2025-02-02T16:10:00Z",
            relationships=ExtractedRelationships(
                relationships=[
                    ExtractedRelationship(
                        source="Nurse Tami",
                        target="Harbor Clinic",
                        relation="WORKS_AT",
                        fact="Nurse Tami works out of Harbor Clinic.",
                    ),
                    ExtractedRelationship(
                        source="Nurse Tami",
                        target="Ben",
                        relation="INTRODUCES",
                        fact="She introduced me to Ben.",
                    ),
                    ExtractedRelationship(
                        source="Ben",
                        target="mom's paperwork",
                        relation="ASSISTS_WITH",
                        fact="Ben is helping with mom's paperwork.",
                    ),
                ]
            ),
        ).with_inputs("episode_content", "entities", "reference_time")
    )

    trainset = all_examples[:6]
    valset = all_examples[6:]
    logger.info(
        "Built relationship trainset with %d examples, valset with %d examples",
        len(trainset),
        len(valset),
    )
    return trainset, valset


def _relationship_set(relations: ExtractedRelationships | list | dict | None) -> set[tuple[str, str, str]]:
    """Normalize adapter outputs into comparable tuples."""

    if relations is None:
        return set()

    if isinstance(relations, ExtractedRelationships):
        rel_list = relations.relationships
    elif isinstance(relations, dict) and "relationships" in relations:
        rel_list = relations["relationships"]
    else:
        rel_list = relations

    normalized: set[tuple[str, str, str]] = set()
    for rel in rel_list or []:
        try:
            if isinstance(rel, ExtractedRelationship):
                source = rel.source.lower()
                target = rel.target.lower()
                relation = rel.relation.upper()
            else:
                source = rel["source"].lower()
                target = rel["target"].lower()
                relation = rel["relation"].upper()
            normalized.add((source, target, relation))
        except (KeyError, AttributeError):
            continue

    return normalized


def relationship_extraction_metric(example, prediction, trace=None) -> float:
    """Compute F1 over (source, target, relation) tuples."""

    expected = _relationship_set(example.relationships)
    predicted = _relationship_set(getattr(prediction, "relationships", prediction))

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
        num_candidates=5,
        init_temperature=1.0,
        metric_threshold=0.8,
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
            entities=example.entities,
            reference_time=example.reference_time,
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
