"""Optimizer for pipeline/extract_attributes.py using DSPy's BootstrapFewShot.

Focuses on Person attributes so relationship labels stay human and grounded.

Usage:
    python -m pipeline.optimizers.extract_attributes_optimizer
"""

from __future__ import annotations

import json
from pathlib import Path
import logging
import dspy
from dspy.teleprompt import MIPROv2

from dspy_outlines import OutlinesAdapter, OutlinesLM
from pipeline.extract_attributes import AttributeExtractor
from pipeline.entity_edge_models import Person
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
    """Create light-weight attribute examples centered on relationships."""

    def example(
        episode_content: str,
        previous_notes: list[str],
        entity_name: str,
        existing: dict[str, str],
        label: str,
    ) -> dspy.Example:
        return dspy.Example(
            episode_content=episode_content,
            previous_episodes=json.dumps(previous_notes),
            entity_name=entity_name,
            entity_type="Person",
            existing_attributes=json.dumps(existing),
            response_model=Person,
            attributes={"relationship_type": label},
        ).with_inputs(
            "episode_content",
            "previous_episodes",
            "entity_name",
            "entity_type",
            "existing_attributes",
            "response_model",
        )

    all_examples = [
        example(
            episode_content=(
                "Lena made blueberry pancakes and reminded me we've been friends since college."
            ),
            previous_notes=["Lena texted last week about missing our Sunday breakfast ritual."],
            entity_name="Lena",
            existing={"relationship_type": "acquaintance"},
            label="friend",
        ),
        example(
            episode_content="Coach Ray met me for hill repeats and tweaked my stride.",
            previous_notes=["Ray has been writing my running plans all winter."],
            entity_name="Ray",
            existing={},
            label="coach",
        ),
        example(
            episode_content="Therapy with Dr. Hwang today felt steadier; she tracked my breath with me.",
            previous_notes=["Dr. Hwang has seen me since last spring."],
            entity_name="Dr. Hwang",
            existing={"relationship_type": ""},
            label="therapist",
        ),
        example(
            episode_content="Video call with cousin Eli to plan the Diaz reunion.",
            previous_notes=["Eli keeps the family spreadsheet updated."],
            entity_name="Eli",
            existing={"relationship_type": "family"},
            label="family",
        ),
        example(
            episode_content="Priya and I paired at the studio to fix our team's prototype.",
            previous_notes=["She sits two desks over at the co-op space."],
            entity_name="Priya",
            existing={},
            label="colleague",
        ),
        example(
            episode_content="Slow dinner date with Sam on the roof, just holding hands and eating takeout.",
            previous_notes=["Sam and I have been dating since the fall festival."],
            entity_name="Sam",
            existing={"relationship_type": "friend"},
            label="romantic partner",
        ),
        example(
            episode_content="Neighbor Laila dropped off soup when the migraine kicked in.",
            previous_notes=["She shares tomatoes from her garden every July."],
            entity_name="Laila",
            existing={},
            label="neighbor",
        ),
        example(
            episode_content="Pastor Miguel prayed with me after service and reminded me to journal nightly.",
            previous_notes=["Miguel has basically mentored me since high school youth group."],
            entity_name="Miguel",
            existing={"relationship_type": "pastor"},
            label="mentor",
        ),
    ]

    trainset = all_examples[:6]
    valset = all_examples[6:]
    logger.info(
        "Built attribute trainset with %d examples, valset with %d examples",
        len(trainset),
        len(valset),
    )
    return trainset, valset


def attribute_extraction_metric(example, prediction, trace=None) -> float:
    """Return 1 when relationship_type matches (case-insensitive)."""

    expected = example.attributes.get("relationship_type")
    target = prediction or {}
    if hasattr(target, "get"):
        predicted_value = target.get("relationship_type")
    else:
        predicted_value = getattr(target, "relationship_type", None)

    if expected is None:
        return 1.0

    if not predicted_value:
        return 0.0

    return 1.0 if expected.lower() == predicted_value.lower() else 0.0


def optimize(trainset: list[dspy.Example]) -> AttributeExtractor:
    """Optimize AttributeExtractor with MIPROv2."""

    logger.info("Starting attribute optimization with %d examples", len(trainset))
    optimizer = MIPROv2(
        metric=attribute_extraction_metric,
        auto=None,
        num_candidates=4,
        init_temperature=1.0,
        metric_threshold=0.85,
    )

    student = AttributeExtractor()
    optimized = optimizer.compile(
        student=student,
        trainset=trainset,
        num_trials=8,
        max_bootstrapped_demos=2,
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
            response_model=example.response_model,
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
