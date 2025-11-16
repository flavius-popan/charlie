"""Optimizer for pipeline/extract_attributes.py using DSPy's BootstrapFewShot.

Focuses on Person attributes so relationship labels stay human and grounded.
Includes explicit `Self` entity cases to teach the extractor how the author
describes their own relationship to themselves.

Usage:
    python -m pipeline.optimizers.extract_attributes_optimizer
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import logging
import dspy
from dspy.teleprompt import MIPROv2

from dspy_outlines import OutlinesAdapter, OutlinesLM
from pipeline.extract_attributes import AttributeExtractor
from pipeline.entity_edge_models import Activity, Organization, Person, Place
from settings import DEFAULT_MODEL_PATH, MODEL_CONFIG


PROMPT_OUTPUT = Path(__file__).parent.parent / "prompts" / "extract_attributes.json"
logger = logging.getLogger(__name__)


def configure_dspy():
    """Match runtime LM + adapter configuration."""

    lm = OutlinesLM(model_path=DEFAULT_MODEL_PATH, generation_config=MODEL_CONFIG)
    adapter = OutlinesAdapter()
    dspy.configure(lm=lm, adapter=adapter)
    logger.info("Configured DSPy with OutlinesLM (model: %s)", DEFAULT_MODEL_PATH)


def build_trainset() -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Create light-weight attribute examples covering multiple entity types."""

    def example(
        *,
        episode_content: str,
        previous_notes: list[str],
        entity_name: str,
        entity_type: str,
        existing: dict[str, str],
        attributes: dict[str, str],
    ) -> dspy.Example:
        return dspy.Example(
            episode_content=episode_content,
            previous_episodes=json.dumps(previous_notes),
            entity_name=entity_name,
            entity_type=entity_type,
            existing_attributes=json.dumps(existing),
            attributes=attributes,
        ).with_inputs(
            "episode_content",
            "previous_episodes",
            "entity_name",
            "entity_type",
            "existing_attributes",
        )

    all_examples = [
        example(
            episode_content="Lena made blueberry pancakes and reminded me we've been friends since college.",
            previous_notes=["Lena texted last week about missing our Sunday breakfast ritual."],
            entity_name="Lena",
            entity_type="Person",
            existing={"relationship_type": "acquaintance"},
            attributes={
                "relationship_type": "friend",
                "closeness": 0.82,
                "overall_valence": 0.65,
            },
        ),
        example(
            episode_content="Coach Ray met me for hill repeats and tweaked my stride.",
            previous_notes=["Ray has been writing my running plans all winter."],
            entity_name="Ray",
            entity_type="Person",
            existing={},
            attributes={
                "relationship_type": "coach",
                "closeness": 0.4,
                "overall_valence": 0.2,
            },
        ),
        example(
            episode_content="Therapy with Dr. Hwang today felt steadier; she tracked my breath with me.",
            previous_notes=["Dr. Hwang has seen me since last spring."],
            entity_name="Dr. Hwang",
            entity_type="Person",
            existing={"relationship_type": ""},
            attributes={
                "relationship_type": "therapist",
                "closeness": 0.35,
                "overall_valence": -0.25,
            },
        ),
        example(
            episode_content="Priya and I paired at the studio to fix our team's prototype.",
            previous_notes=["She sits two desks over at the co-op space."],
            entity_name="Priya",
            entity_type="Person",
            existing={},
            attributes={
                "relationship_type": "colleague",
                "closeness": 0.55,
                "overall_valence": 0.4,
            },
        ),
        example(
            episode_content="Slow dinner date with Sam on the roof, just holding hands and eating takeout.",
            previous_notes=["Sam and I have been dating since the fall festival."],
            entity_name="Sam",
            entity_type="Person",
            existing={"relationship_type": "friend"},
            attributes={
                "relationship_type": "romantic partner",
                "closeness": 0.9,
                "overall_valence": 0.85,
            },
        ),
        example(
            episode_content="Neighbor Laila dropped off soup when the migraine kicked in.",
            previous_notes=["She shares tomatoes from her garden every July."],
            entity_name="Laila",
            entity_type="Person",
            existing={},
            attributes={
                "relationship_type": "neighbor",
                "closeness": 0.6,
                "overall_valence": 0.5,
            },
        ),
        example(
            episode_content="I talked gently to myself in the mirror before work so I wouldn't skip breakfast again.",
            previous_notes=["I promised Ines I'd practice kinder self-talk every morning."],
            entity_name="Self",
            entity_type="Person",
            existing={
                "relationship_type": "author",
                "closeness": 0.75,
                "overall_valence": 0.1,
            },
            attributes={
                "relationship_type": "author",
                "closeness": 0.8,
                "overall_valence": 0.35,
            },
        ),
        example(
            episode_content="Lake Merritt felt calm tonight; the walking path lights flickered along the water.",
            previous_notes=["Jogged the lake loop last week at dusk."],
            entity_name="Lake Merritt",
            entity_type="Place",
            existing={},
            attributes={"category": "park"},
        ),
        example(
            episode_content="Checked in at Harbor House Clinic for the migraine shots.",
            previous_notes=["Nurse Jamie said the clinic closes earlier on Fridays."],
            entity_name="Harbor House Clinic",
            entity_type="Place",
            existing={"category": ""},
            attributes={"category": "clinic"},
        ),
        example(
            episode_content="River Labs asked me to review their mindfulness study protocol.",
            previous_notes=["River Labs hosted the burnout workshop last month."],
            entity_name="River Labs",
            entity_type="Organization",
            existing={},
            attributes={"category": "research company"},
        ),
        example(
            episode_content="Met with Mutual Aid Kitchen to portion out stews for the winter drive.",
            previous_notes=["They run pop-up food shares every other Sunday."],
            entity_name="Mutual Aid Kitchen",
            entity_type="Organization",
            existing={},
            attributes={"category": "community group"},
        ),
        example(
            episode_content="Sunrise yoga on the pier turned into laughter yoga when the speaker glitched.",
            previous_notes=["I promised Nina I'd bring extra mats to sunrise yoga."],
            entity_name="sunrise yoga",
            entity_type="Activity",
            existing={},
            attributes={"activity_type": "wellness class"},
        ),
        example(
            episode_content="Booked a calming dog walk with Scout; we circled the block slowly.",
            previous_notes=["Scout's owner Jana said these short walks keep her grounded."],
            entity_name="dog walk with Scout",
            entity_type="Activity",
            existing={},
            attributes={"activity_type": "walk"},
        ),
        example(
            episode_content="I spiraled tonight and had to write myself a letter reminding me I deserve rest.",
            previous_notes=["I keep slipping into old burnout stories whenever deadlines stack up."],
            entity_name="Self",
            entity_type="Person",
            existing={
                "relationship_type": "author",
                "closeness": 0.65,
                "overall_valence": -0.1,
            },
            attributes={
                "relationship_type": "author",
                "closeness": 0.55,
                "overall_valence": -0.35,
            },
        ),
        example(
            episode_content="I let myself nap after therapy and woke up tender but proud of that tiny kindness.",
            previous_notes=["Rest still feels illegal unless someone else insists."],
            entity_name="Self",
            entity_type="Person",
            existing={
                "relationship_type": "author",
                "closeness": 0.6,
                "overall_valence": 0.0,
            },
            attributes={
                "relationship_type": "author",
                "closeness": 0.72,
                "overall_valence": 0.25,
            },
        ),
    ]

    valset = all_examples[-4:]
    trainset = all_examples[:-4]
    logger.info(
        "Built attribute trainset with %d examples, valset with %d examples",
        len(trainset),
        len(valset),
    )
    return trainset, valset


def attribute_extraction_metric(example, prediction, trace=None) -> float:
    """Return average match rate across expected attributes."""

    expected: dict = example.attributes or {}
    predicted = prediction or {}
    if hasattr(predicted, "get"):
        predicted_dict = predicted
    else:
        predicted_dict = getattr(predicted, "attributes", {}) or {}

    if not expected:
        return 1.0

    matches = 0
    for key, expected_value in expected.items():
        predicted_value = predicted_dict.get(key)
        if isinstance(expected_value, str) and isinstance(predicted_value, str):
            if expected_value.strip().lower() == predicted_value.strip().lower():
                matches += 1
        elif isinstance(expected_value, (int, float)) and isinstance(
            predicted_value, (int, float)
        ):
            if math.isclose(
                float(predicted_value),
                float(expected_value),
                abs_tol=0.05,
            ):
                matches += 1
        else:
            if expected_value == predicted_value:
                matches += 1

    return matches / len(expected)


def optimize(trainset: list[dspy.Example]) -> AttributeExtractor:
    """Optimize AttributeExtractor with MIPROv2."""

    logger.info("Starting attribute optimization with %d examples", len(trainset))
    optimizer = MIPROv2(
        metric=attribute_extraction_metric,
        auto=None,
        num_candidates=3,
        init_temperature=0.5,
        metric_threshold=0.90,
    )

    student = AttributeExtractor()
    optimized = optimizer.compile(
        student=student,
        trainset=trainset,
        num_trials=8,
        max_bootstrapped_demos=3,
        max_labeled_demos=3,
        minibatch_size=2,
        requires_permission_to_run=False,
    )

    logger.info("Attribute optimization completed")
    return optimized


def evaluate(module: AttributeExtractor, dataset: list[dspy.Example]) -> float:
    """Average match rate across dataset."""

    scores: list[float] = []
    for example in dataset:
        prediction = module(
            episode_content=example.episode_content,
            previous_episodes=example.previous_episodes,
            entity_name=example.entity_name,
            entity_type=example.entity_type,
            existing_attributes=example.existing_attributes,
        )
        scores.append(attribute_extraction_metric(example, prediction))

    return sum(scores) / len(scores) if scores else 0.0


def main():
    """Full optimization workflow for attribute extraction."""

    logging.basicConfig(level=logging.INFO)
    configure_dspy()

    trainset, valset = build_trainset()
    baseline_module = AttributeExtractor()
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
