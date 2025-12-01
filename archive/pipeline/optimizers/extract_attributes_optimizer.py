"""Optimizer for pipeline/extract_attributes.py using DSPy's GEPA.

Uses LLM-as-judge (gpt-5-nano) to provide rich textual feedback for optimizing
attribute extraction prompts. Focuses on Person attributes with human-grounded
relationship labels and author (I) entity handling.

Usage:
    python -m pipeline.optimizers.extract_attributes_optimizer
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path

# Running as `python archive/pipeline/optimizers/...` requires inserting archive/ so
# `settings` executes (and configures DSPy caches) before importing DSPy.
if __package__ is None:
    ARCHIVE_DIR = Path(__file__).resolve().parents[2]
    if str(ARCHIVE_DIR) not in os.sys.path:
        os.sys.path.insert(0, str(ARCHIVE_DIR))
    # Add repo root for backend imports
    REPO_ROOT = ARCHIVE_DIR.parent
    if str(REPO_ROOT) not in os.sys.path:
        os.sys.path.insert(0, str(REPO_ROOT))

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

from pipeline import _dspy_setup  # noqa: F401
import dspy  # noqa: E402
from dspy.teleprompt import GEPA  # noqa: E402
from dspy import Prediction  # noqa: E402

from inference_runtime import DspyLM
from pipeline.extract_attributes import AttributeExtractor
from pipeline.entity_edge_models import Activity, Organization, Person, Place


PROMPT_OUTPUT = Path(__file__).parent.parent / "prompts" / "extract_attributes.json"
logger = logging.getLogger(__name__)


def configure_dspy():
    """Match runtime LM + adapter configuration."""

    lm = DspyLM(model_path=DEFAULT_MODEL_PATH, generation_config=MODEL_CONFIG)
    adapter = dspy.ChatAdapter()
    dspy.configure(lm=lm, adapter=adapter)
    logger.info("Configured DSPy with DspyLM (model: %s)", DEFAULT_MODEL_PATH)


def build_trainset() -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Create light-weight attribute examples covering multiple entity types."""

    def build_example(
        *,
        episode_content: str,
        previous_notes: list[str],
        entity_name: str,
        entity_type: str,
        existing: dict[str, str],
        attributes: dict[str, str],
    ) -> tuple[dspy.Example, str]:
        ex = dspy.Example(
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
        return ex, entity_type

    typed_examples = [
        build_example(
            episode_content="Lena made blueberry pancakes and reminded me we've been friends since college.",
            previous_notes=["Lena texted last week about missing our Sunday breakfast ritual."],
            entity_name="Lena",
            entity_type="Person",
            existing={"relationship_type": "acquaintance"},
            attributes={"relationship_type": "friend"},
        ),
        build_example(
            episode_content="Coach Ray met me for hill repeats and tweaked my stride.",
            previous_notes=["Ray has been writing my running plans all winter."],
            entity_name="Ray",
            entity_type="Person",
            existing={},
            attributes={"relationship_type": "coach"},
        ),
        build_example(
            episode_content="Therapy with Dr. Hwang today felt steadier; she tracked my breath with me.",
            previous_notes=["Dr. Hwang has seen me since last spring."],
            entity_name="Dr. Hwang",
            entity_type="Person",
            existing={"relationship_type": ""},
            attributes={"relationship_type": "therapist"},
        ),
        build_example(
            episode_content="Priya and I paired at the studio to fix our team's prototype.",
            previous_notes=["She sits two desks over at the co-op space."],
            entity_name="Priya",
            entity_type="Person",
            existing={},
            attributes={"relationship_type": "colleague"},
        ),
        build_example(
            episode_content="Slow dinner date with Sam on the roof, just holding hands and eating takeout.",
            previous_notes=["Sam and I have been dating since the fall festival."],
            entity_name="Sam",
            entity_type="Person",
            existing={"relationship_type": "friend"},
            attributes={"relationship_type": "romantic partner"},
        ),
        build_example(
            episode_content="Neighbor Laila dropped off soup when the migraine kicked in.",
            previous_notes=["She shares tomatoes from her garden every July."],
            entity_name="Laila",
            entity_type="Person",
            existing={},
            attributes={"relationship_type": "neighbor"},
        ),
        build_example(
            episode_content="I talked gently to myself in the mirror before work so I wouldn't skip breakfast again.",
            previous_notes=["I promised Ines I'd practice kinder self-talk every morning."],
            entity_name="I",
            entity_type="Person",
            existing={"relationship_type": "author"},
            attributes={"relationship_type": "author"},
        ),
        build_example(
            episode_content="Lake Merritt felt calm tonight; the walking path lights flickered along the water.",
            previous_notes=["Jogged the lake loop last week at dusk."],
            entity_name="Lake Merritt",
            entity_type="Place",
            existing={},
            attributes={"category": "park"},
        ),
        build_example(
            episode_content="Checked in at Harbor House Clinic for the migraine shots.",
            previous_notes=["Nurse Jamie said the clinic closes earlier on Fridays."],
            entity_name="Harbor House Clinic",
            entity_type="Place",
            existing={"category": ""},
            attributes={"category": "clinic"},
        ),
        build_example(
            episode_content="River Labs asked me to review their mindfulness study protocol.",
            previous_notes=["River Labs hosted the burnout workshop last month."],
            entity_name="River Labs",
            entity_type="Organization",
            existing={},
            attributes={"category": "research company"},
        ),
        build_example(
            episode_content="Met with Mutual Aid Kitchen to portion out stews for the winter drive.",
            previous_notes=["They run pop-up food shares every other Sunday."],
            entity_name="Mutual Aid Kitchen",
            entity_type="Organization",
            existing={},
            attributes={"category": "community group"},
        ),
        build_example(
            episode_content="Sunrise yoga on the pier turned into laughter yoga when the speaker glitched.",
            previous_notes=["I promised Nina I'd bring extra mats to sunrise yoga."],
            entity_name="sunrise yoga",
            entity_type="Activity",
            existing={},
            attributes={"purpose": "shared wellness ritual"},
        ),
        build_example(
            episode_content="Booked a calming dog walk with Scout; we circled the block slowly.",
            previous_notes=["Scout's owner Jana said these short walks keep her grounded."],
            entity_name="dog walk with Scout",
            entity_type="Activity",
            existing={},
            attributes={"purpose": "slow grounding walk"},
        ),
        build_example(
            episode_content="I spiraled tonight and had to write myself a letter reminding me I deserve rest.",
            previous_notes=["I keep slipping into old burnout stories whenever deadlines stack up."],
            entity_name="I",
            entity_type="Person",
            existing={"relationship_type": "author"},
            attributes={"relationship_type": "author"},
        ),
        build_example(
            episode_content="I let myself nap after therapy and woke up tender but proud of that tiny kindness.",
            previous_notes=["Rest still feels illegal unless someone else insists."],
            entity_name="I",
            entity_type="Person",
            existing={"relationship_type": "author"},
            attributes={"relationship_type": "author"},
        ),
    ]

    buckets: dict[str, list[dspy.Example]] = defaultdict(list)
    for example_obj, entity_type in typed_examples:
        buckets[entity_type].append(example_obj)

    trainset: list[dspy.Example] = []
    valset: list[dspy.Example] = []
    for entity_type, bucket in buckets.items():
        if len(bucket) == 1:
            trainset.extend(bucket)
            continue
        valset.append(bucket.pop())
        trainset.extend(bucket)
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
        else:
            if expected_value == predicted_value:
                matches += 1

    return matches / len(expected)


def calculate_attribute_score(expected: dict, predicted_dict: dict) -> float:
    """Calculate match rate across expected attributes."""
    if not expected:
        return 1.0

    matches = 0
    for key, expected_value in expected.items():
        predicted_value = predicted_dict.get(key)
        if isinstance(expected_value, str) and isinstance(predicted_value, str):
            if expected_value.strip().lower() == predicted_value.strip().lower():
                matches += 1
        else:
            if expected_value == predicted_value:
                matches += 1

    return matches / len(expected)


def generate_feedback(
    expected_attrs: dict,
    predicted_attrs: dict,
    entity_name: str,
    entity_type: str,
    score: float,
    judge_lm: dspy.LM
) -> str:
    """Use judge LM to generate actionable feedback."""

    missing = {k: v for k, v in expected_attrs.items() if k not in predicted_attrs}
    incorrect = {
        k: (expected_attrs[k], predicted_attrs[k])
        for k in expected_attrs
        if k in predicted_attrs and expected_attrs[k] != predicted_attrs[k]
    }
    correct = {
        k: v
        for k, v in expected_attrs.items()
        if k in predicted_attrs and expected_attrs[k] == predicted_attrs[k]
    }

    feedback_prompt = f"""Evaluate this attribute extraction for entity "{entity_name}" (type: {entity_type}):

Expected attributes: {expected_attrs}
Predicted attributes: {predicted_attrs}
Match score: {score:.2f}

Correct: {correct}
Missing: {missing}
Incorrect: {incorrect}

Provide feedback on:
1. Relationship typing: Are Person relationship types human and natural (friend, coach, therapist)?
2. Completeness: Are key attributes missing?
3. Author identity handling: For the author's identity, is relationship_type="author" correct?

Be specific and actionable."""

    logger.info("=" * 80)
    logger.info("JUDGE EVALUATION REQUEST")
    logger.info(f"Entity: {entity_name} ({entity_type})")
    logger.info(f"Score: {score:.2f}")

    feedback = judge_lm(feedback_prompt)[0]

    logger.info("JUDGE FEEDBACK:")
    logger.info(feedback)
    logger.info("=" * 80)

    return feedback


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
    """Full optimization workflow for attribute extraction using GEPA."""

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
    def gepa_attribute_metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> Prediction:
        """GEPA-compatible metric that returns ScoreWithFeedback.

        Note: Only calls expensive judge LM during GEPA reflection phase (pred_name != None).
        Regular evaluations use simple feedback to save costs and time.
        """

        expected: dict = gold.attributes or {}
        predicted = pred or {}
        if hasattr(predicted, "get"):
            predicted_dict = predicted
        else:
            predicted_dict = getattr(predicted, "attributes", {}) or {}

        if not expected:
            return Prediction(score=1.0, feedback="No expected attributes")

        score = calculate_attribute_score(expected, predicted_dict)

        # Only call expensive judge LM during GEPA reflection phase (when pred_name provided)
        if pred_name:
            logger.info("-" * 80)
            logger.info(f"EVALUATING PREDICTOR: {pred_name}")

            feedback = generate_feedback(
                expected_attrs=expected,
                predicted_attrs=predicted_dict,
                entity_name=gold.entity_name,
                entity_type=gold.entity_type,
                score=score,
                judge_lm=judge_lm
            )

            logger.info(f"METRIC SCORE: {score:.2f}")
            logger.info("-" * 80)
        else:
            # Simple feedback for regular evaluations (no expensive LLM call)
            missing_count = len([k for k in expected if k not in predicted_dict])
            feedback = f"Score: {score:.2f}. Missing {missing_count}/{len(expected)} attributes."

        return Prediction(score=score, feedback=feedback)

    # Evaluate baseline
    baseline = AttributeExtractor()
    baseline_score = evaluate(baseline, valset)
    logger.info("Baseline score (valset): %.3f", baseline_score)

    # Create log directory for GEPA artifacts
    log_dir = GEPA_OUTPUT_DIR / "extract_attributes"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info("GEPA logs will be saved to: %s", log_dir)

    # Instantiate and run GEPA
    logger.info("Starting GEPA optimization with max_full_evals=%d", GEPA_MAX_FULL_EVALS)
    gepa = GEPA(
        metric=gepa_attribute_metric,
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
