"""Optimizer for pipeline/extract_nodes.py using DSPy's GEPA.

Uses LLM-as-judge (gpt-5-nano) to provide rich textual feedback for optimizing
entity extraction prompts. Focuses on accurate entity typing and first-person
Self entity detection.

Usage:
    python -m pipeline.optimizers.extract_nodes_optimizer
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

# When executed as `python pipeline/optimizers/...`, this ensures we can import
# `settings` before DSPy initializes so cache/env configuration is applied first.
if __package__ is None:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in os.sys.path:
        os.sys.path.insert(0, str(PROJECT_ROOT))

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
    PROMPTS_DIR,
)

from pipeline import _dspy_setup  # noqa: F401
import dspy  # noqa: E402
from dspy.teleprompt import GEPA  # noqa: E402
from dspy import Prediction  # noqa: E402

from inference_runtime import DspyLM
from pipeline.extract_nodes import EntityExtractor
from pipeline.extract_nodes import ExtractedEntity, ExtractedEntities

PROMPT_OUTPUT = PROMPTS_DIR / "extract_nodes.json"
logger = logging.getLogger(__name__)

ENTITY_TYPE_ID_TO_LABEL = {
    1: "Person",
    2: "Place",
    3: "Organization",
    4: "Activity",
}
REQUIRED_ENTITY_TYPES = set(ENTITY_TYPE_ID_TO_LABEL.values())

def configure_dspy():
    """Configure DSPy with the MLX-backed LM used in production."""

    lm = DspyLM(model_path=DEFAULT_MODEL_PATH, generation_config=MODEL_CONFIG)
    adapter = dspy.ChatAdapter()
    dspy.configure(lm=lm, adapter=adapter)
    logger.info("Configured DSPy with DspyLM (model: %s)", DEFAULT_MODEL_PATH)


def build_trainset() -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Build training and validation examples for entity extraction optimization.

    Creates 10 realistic personal journal entries covering:
    - Entity types: Person, Place, Organization, Activity
    - Contexts: hanging out with friends, family time, going places, doing hobbies, appointments, life reflections
    - Tone: casual, authentic, human (not robotic business language)

    Returns:
        Tuple of (trainset, valset) with 8 training and 2 validation examples
    """

    entity_types_json = json.dumps(
        [
            {
                "entity_type_id": 0,
                "entity_type_name": "Entity",
                "entity_type_description": "fallback for significant entities that do not fit custom types",
            },
            {
                "entity_type_id": 1,
                "entity_type_name": "Person",
                "entity_type_description": "people mentioned in the journal: friends, family, coworkers, crushes, therapists, mentors, neighbors, etc.",
            },
            {
                "entity_type_id": 2,
                "entity_type_name": "Place",
                "entity_type_description": "specific locations or restorative spaces such as homes, parks, studios, cafes, clinics, or travel stops.",
            },
            {
                "entity_type_id": 3,
                "entity_type_name": "Organization",
                "entity_type_description": "communities, teams, companies, collectives, or groups the author interacts with.",
            },
            {
                "entity_type_id": 4,
                "entity_type_name": "Activity",
                "entity_type_description": "events, rituals, or routines (therapy session, sunrise swim, volunteer shift, study group, etc.).",
            },
        ],
        ensure_ascii=False,
    )

    all_examples = []

    # Example 1: Coffee shop hangout with friend
    all_examples.append(
        dspy.Example(
            episode_content="Met Sarah at Blue Bottle Coffee this morning. We talked about our career goals and how we're both feeling stuck. It's nice to have someone who gets it.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Sarah", entity_type_id=1),
                    ExtractedEntity(name="Blue Bottle Coffee", entity_type_id=2),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 2: Therapy session about relationships
    all_examples.append(
        dspy.Example(
            episode_content="Had my session with Dr. Martinez today. We talked about my relationship patterns and why I keep pushing people away. Heavy stuff but necessary.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Dr. Martinez", entity_type_id=1),
                    ExtractedEntity(name="therapy session", entity_type_id=4),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 3: Yoga class and self-care
    all_examples.append(
        dspy.Example(
            episode_content="Finally went back to yoga at Mindful Movement. Haven't been in weeks. My instructor Emma noticed and welcomed me back warmly. Need to prioritize self-care more.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Mindful Movement", entity_type_id=3),
                    ExtractedEntity(name="Emma", entity_type_id=1),
                    ExtractedEntity(name="yoga class", entity_type_id=4),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 4: Sunrise ride to regulate nerves
    all_examples.append(
        dspy.Example(
            episode_content="I biked the Embarcadero alone before sunrise to calm my nerves. The cold air finally slowed my racing thoughts.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Self", entity_type_id=1),
                    ExtractedEntity(name="the Embarcadero", entity_type_id=2),
                    ExtractedEntity(name="solo sunrise ride", entity_type_id=4),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 5: Meeting a friend for grounding
    all_examples.append(
        dspy.Example(
            episode_content="I met Priya at Dolores Park after lunch so I wouldn't spiral alone. We stretched on a blanket and I finally exhaled.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Self", entity_type_id=1),
                    ExtractedEntity(name="Priya", entity_type_id=1),
                    ExtractedEntity(name="Dolores Park", entity_type_id=2),
                    ExtractedEntity(name="park hangout", entity_type_id=4),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 6: Solo journaling ritual
    all_examples.append(
        dspy.Example(
            episode_content="I stayed home tonight, brewed chamomile, and journaled about why I'm afraid to rest. No outside voices, just me and the page.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Self", entity_type_id=1),
                    ExtractedEntity(name="home", entity_type_id=2),
                    ExtractedEntity(
                        name="chamomile journaling ritual", entity_type_id=4
                    ),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 4: Work stress and deadline
    all_examples.append(
        dspy.Example(
            episode_content="Insane day at Automattic. The product launch is next week and everything's on fire. My manager Lisa says I'm doing great but I feel like I'm barely keeping up.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Automattic", entity_type_id=3),
                    ExtractedEntity(name="Lisa", entity_type_id=1),
                    ExtractedEntity(name="product launch", entity_type_id=4),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 5: Weekend hike with friends
    all_examples.append(
        dspy.Example(
            episode_content="Went hiking at Mount Tam with Jake and Priya. The views were incredible and we talked about life and meaning. Sometimes I forget how important nature is for my mental health.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Mount Tam", entity_type_id=2),
                    ExtractedEntity(name="Jake", entity_type_id=1),
                    ExtractedEntity(name="Priya", entity_type_id=1),
                    ExtractedEntity(name="hiking", entity_type_id=4),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 6: Doctor appointment about anxiety
    all_examples.append(
        dspy.Example(
            episode_content="Checkup with Dr. Chen. Told her about my anxiety getting worse. She referred me to a psychiatrist and we talked about medication. Scary but probably time.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Dr. Chen", entity_type_id=1),
                    ExtractedEntity(name="doctor appointment", entity_type_id=4),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 7: Volunteering at shelter
    all_examples.append(
        dspy.Example(
            episode_content="Volunteered at Food For All this morning. Met a woman named Rosa who reminded me how lucky I am. The work is hard but gives me perspective on my own problems.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Food For All", entity_type_id=3),
                    ExtractedEntity(name="Rosa", entity_type_id=1),
                    ExtractedEntity(name="volunteering", entity_type_id=4),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 8: First day at new job
    all_examples.append(
        dspy.Example(
            episode_content="First day at DataFlow Inc. Everyone seems nice but I'm overwhelmed trying to remember names. My team lead Miguel walked me through the codebase. Imposter syndrome is real.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="DataFlow Inc", entity_type_id=3),
                    ExtractedEntity(name="Miguel", entity_type_id=1),
                    ExtractedEntity(name="first day", entity_type_id=4),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 9: Sunrise lap with friend and dog
    all_examples.append(
        dspy.Example(
            episode_content="Met Nina and her rescue pup Scout for a slow, misty lap around Lake Merritt. Nina noticed how anxious I was and kept cueing the breathwork my therapist suggested until my shoulders finally dropped.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Nina", entity_type_id=1),
                    ExtractedEntity(name="Scout", entity_type_id=1),
                    ExtractedEntity(name="Lake Merritt", entity_type_id=2),
                    ExtractedEntity(name="breathwork routine", entity_type_id=4),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 10: Pantry volunteering
    all_examples.append(
        dspy.Example(
            episode_content="Volunteered at Community Roots Pantry again. Malik had us assemble produce boxes while he explained next month's food distribution schedule.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Community Roots Pantry", entity_type_id=3),
                    ExtractedEntity(name="Malik", entity_type_id=1),
                    ExtractedEntity(name="produce box shift", entity_type_id=4),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 11: Family planning call
    all_examples.append(
        dspy.Example(
            episode_content="Hopped on a call with Aunt Rosa about the Diaz Family Fund and scribbled 'gratitude mantra' on a sticky note so I remember to breathe this week.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Aunt Rosa", entity_type_id=1),
                    ExtractedEntity(name="Diaz Family Fund", entity_type_id=3),
                    ExtractedEntity(name="gratitude mantra", entity_type_id=0),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 12: Mission Works sprint
    all_examples.append(
        dspy.Example(
            episode_content="Camped out at Mission Works Lab with the Redwood AI crew to finish our sprint. Priya kept pairing with me on the bug bash while Jordan ordered Thai.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Mission Works Lab", entity_type_id=2),
                    ExtractedEntity(name="Redwood AI", entity_type_id=3),
                    ExtractedEntity(name="Priya", entity_type_id=1),
                    ExtractedEntity(name="bug bash sprint", entity_type_id=4),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 13: Support circle discussion
    all_examples.append(
        dspy.Example(
            episode_content='Spent the evening at Haven House support circle naming the burnout spiral out loud. Jenna held space for all our messy feelings and had us write a "north star" intention for how we want to feel this week.',
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(
                        name="Haven House support circle", entity_type_id=4
                    ),
                    ExtractedEntity(name="Jenna", entity_type_id=1),
                    ExtractedEntity(name="burnout spiral", entity_type_id=0),
                    ExtractedEntity(name="north star intention", entity_type_id=0),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Validation Example 1: Book club meeting
    all_examples.append(
        dspy.Example(
            episode_content="Book club at Marcus's place tonight. We discussed 'The Midnight Library' and got into this deep conversation about regret and second chances. Love this group.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Marcus", entity_type_id=1),
                    ExtractedEntity(name="book club", entity_type_id=4),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Validation Example 2: Game night with chosen family
    all_examples.append(
        dspy.Example(
            episode_content="Game night at our place. Emma, Devon, and Kai came over for Catan. We laughed so hard. These people are my chosen family and I'm grateful for them every day.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Emma", entity_type_id=1),
                    ExtractedEntity(name="Devon", entity_type_id=1),
                    ExtractedEntity(name="Kai", entity_type_id=1),
                    ExtractedEntity(name="game night", entity_type_id=4),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Validation Example 3: Late-night self-reflection
    all_examples.append(
        dspy.Example(
            episode_content="I walked circles around the block at midnight, whispering my mantra to keep panic at bay.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Self", entity_type_id=1),
                    ExtractedEntity(name="midnight mantra walk", entity_type_id=4),
                    ExtractedEntity(name="the block", entity_type_id=2),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    examples_with_types: list[tuple[dspy.Example, set[str]]] = []
    for example in all_examples:
        type_names = {
            ENTITY_TYPE_ID_TO_LABEL.get(entity.entity_type_id)
            for entity in example.extracted_entities.extracted_entities
            if ENTITY_TYPE_ID_TO_LABEL.get(entity.entity_type_id)
        }
        examples_with_types.append((example, type_names))

    if len(examples_with_types) <= 3:
        trainset = [example for example, _ in examples_with_types]
        valset: list[dspy.Example] = []
    else:
        train_examples = examples_with_types[:-3]
        val_examples = examples_with_types[-3:]

        def coverage(examples: list[tuple[dspy.Example, set[str]]]) -> set[str]:
            cov: set[str] = set()
            for _, types in examples:
                cov.update(types)
            return cov

        missing = REQUIRED_ENTITY_TYPES - coverage(train_examples)
        for type_name in list(missing):
            idx = next(
                (i for i, (_, types) in enumerate(val_examples) if type_name in types),
                None,
            )
            if idx is None:
                continue
            train_examples.append(val_examples.pop(idx))

        trainset = [example for example, _ in train_examples]
        valset = [example for example, _ in val_examples]

    logger.info(
        "Built trainset with %d examples, valset with %d examples",
        len(trainset),
        len(valset),
    )
    return trainset, valset


def entity_extraction_metric(example, prediction, trace=None) -> float:
    """F1 score based on exact (name, type_id) match.

    Compares expected vs predicted entities using case-insensitive name matching
    and exact type_id matching. Returns F1 score balancing precision and recall.

    Args:
        example: Training example with expected extracted_entities
        prediction: Model prediction with extracted_entities
        trace: Optional trace information (unused)

    Returns:
        F1 score between 0.0 and 1.0
    """

    # Extract expected entities
    expected = {
        (e.name.lower(), e.entity_type_id)
        for e in example.extracted_entities.extracted_entities
    }

    # Extract predicted entities (handle multiple adapter formats)
    pred_entities = getattr(prediction, "extracted_entities", None)
    if pred_entities is None:
        logger.warning("Prediction has no extracted_entities attribute")
        return 0.0

    # Handle ExtractedEntities object
    if isinstance(pred_entities, ExtractedEntities):
        predicted = {
            (e.name.lower(), e.entity_type_id) for e in pred_entities.extracted_entities
        }
    # Handle dict format from adapters
    elif isinstance(pred_entities, dict) and "extracted_entities" in pred_entities:
        try:
            predicted = {
                (e["name"].lower(), e["entity_type_id"])
                for e in pred_entities["extracted_entities"]
            }
        except (KeyError, TypeError) as e:
            logger.warning(f"Failed to parse dict format: {e}")
            return 0.0
    # Handle list format from some adapters (can be list of dicts or ExtractedEntity objects)
    elif isinstance(pred_entities, list):
        try:
            predicted = set()
            for e in pred_entities:
                if isinstance(e, ExtractedEntity):
                    predicted.add((e.name.lower(), e.entity_type_id))
                elif isinstance(e, dict):
                    predicted.add((e["name"].lower(), e["entity_type_id"]))
                else:
                    logger.warning(f"Unknown entity format in list: {type(e)}")
            if not predicted:
                return 0.0
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Failed to parse list format: {e}")
            return 0.0
    else:
        logger.warning(f"Unknown prediction format: {type(pred_entities)}")
        return 0.0

    if not expected:
        return 1.0 if not predicted else 0.0

    if not predicted:
        return 0.0

    # F1 score calculation
    intersection = expected & predicted
    if not intersection:
        return 0.0

    precision = len(intersection) / len(predicted)
    recall = len(intersection) / len(expected)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def calculate_entity_score(expected: set, predicted: set) -> float:
    """Calculate F1 score for entity extraction."""
    if not expected:
        return 1.0 if not predicted else 0.0

    if not predicted:
        return 0.0

    intersection = expected & predicted
    if not intersection:
        return 0.0

    precision = len(intersection) / len(predicted)
    recall = len(intersection) / len(expected)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def generate_feedback(
    expected_entities: set, predicted_entities: set, score: float, judge_lm: dspy.LM
) -> str:
    """Use judge LM to generate actionable feedback."""

    missing = expected_entities - predicted_entities
    extra = predicted_entities - expected_entities
    correct = expected_entities & predicted_entities

    feedback_prompt = f"""Evaluate this entity extraction and provide specific feedback:

Expected entities (name, type_id): {sorted(expected_entities)}
Predicted entities (name, type_id): {sorted(predicted_entities)}
F1 score: {score:.2f}

Correct: {sorted(correct)}
Missing: {sorted(missing)}
Extra: {sorted(extra)}

Provide feedback on:
1. Entity type accuracy: Are entities classified with correct type IDs?
2. Completeness: Are important entities missing?
3. Precision: Are there spurious entities that shouldn't be extracted?
4. Self entity: Is the author's Self entity properly detected in first-person narratives?

Be specific and actionable."""

    logger.info("=" * 80)
    logger.info("JUDGE EVALUATION REQUEST")
    logger.info(f"F1 Score: {score:.2f}")
    logger.info(
        f"Missing: {len(missing)}, Extra: {len(extra)}, Correct: {len(correct)}"
    )

    feedback = judge_lm(feedback_prompt)[0]

    logger.info("JUDGE FEEDBACK:")
    logger.info(feedback)
    logger.info("=" * 80)

    return feedback


def evaluate(module: EntityExtractor, dataset: list[dspy.Example]) -> float:
    """Compute average metric score across dataset.

    Evaluates the module on each example and returns the mean F1 score.

    Args:
        module: EntityExtractor module to evaluate
        dataset: List of examples to evaluate on

    Returns:
        Average F1 score across all examples
    """
    scores = []
    for example in dataset:
        prediction = module(
            episode_content=example.episode_content, entity_types=example.entity_types
        )
        score = entity_extraction_metric(example, prediction)
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


def main():
    """Full optimization workflow for entity extraction using GEPA."""

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
        max_tokens=REFLECTION_MAX_TOKENS,
    )
    logger.info(
        "Configured judge LM: %s (temp=%.1f, max_tokens=%d)",
        REFLECTION_MODEL,
        REFLECTION_TEMPERATURE,
        REFLECTION_MAX_TOKENS,
    )

    # Build datasets
    trainset, valset = build_trainset()

    # Create GEPA-compatible metric with judge_lm bound via closure
    def gepa_entity_metric(
        gold, pred, trace=None, pred_name=None, pred_trace=None
    ) -> Prediction:
        """GEPA-compatible metric that returns ScoreWithFeedback.

        Note: Only calls expensive judge LM during GEPA reflection phase (pred_name != None).
        Regular evaluations use simple feedback to save costs and time.
        """

        # Extract expected entities
        expected = {
            (e.name.lower(), e.entity_type_id)
            for e in gold.extracted_entities.extracted_entities
        }

        # Extract predicted entities
        pred_entities = getattr(pred, "extracted_entities", None)
        if pred_entities is None:
            return Prediction(score=0.0, feedback="No extracted_entities in prediction")

        # Handle ExtractedEntities object
        if isinstance(pred_entities, ExtractedEntities):
            predicted = {
                (e.name.lower(), e.entity_type_id)
                for e in pred_entities.extracted_entities
            }
        # Handle dict format from adapters
        elif isinstance(pred_entities, dict) and "extracted_entities" in pred_entities:
            try:
                predicted = {
                    (e["name"].lower(), e["entity_type_id"])
                    for e in pred_entities["extracted_entities"]
                }
            except (KeyError, TypeError):
                return Prediction(score=0.0, feedback="Failed to parse dict format")
        # Handle list format
        elif isinstance(pred_entities, list):
            try:
                predicted = set()
                for e in pred_entities:
                    if isinstance(e, ExtractedEntity):
                        predicted.add((e.name.lower(), e.entity_type_id))
                    elif isinstance(e, dict):
                        predicted.add((e["name"].lower(), e["entity_type_id"]))
                if not predicted:
                    return Prediction(score=0.0, feedback="Empty prediction list")
            except (KeyError, TypeError, AttributeError):
                return Prediction(score=0.0, feedback="Failed to parse list format")
        else:
            return Prediction(
                score=0.0, feedback=f"Unknown prediction format: {type(pred_entities)}"
            )

        score = calculate_entity_score(expected, predicted)

        # Only call expensive judge LM during GEPA reflection phase (when pred_name provided)
        if pred_name:
            logger.info("-" * 80)
            logger.info(f"EVALUATING PREDICTOR: {pred_name}")

            feedback = generate_feedback(
                expected_entities=expected,
                predicted_entities=predicted,
                score=score,
                judge_lm=judge_lm,
            )

            logger.info(f"METRIC SCORE: {score:.2f}")
            logger.info("-" * 80)
        else:
            # Simple feedback for regular evaluations (no expensive LLM call)
            missing = len(expected - predicted)
            extra = len(predicted - expected)
            feedback = f"Score: {score:.2f}. Missing {missing}, Extra {extra} entities."

        return Prediction(score=score, feedback=feedback)

    # Evaluate baseline
    baseline = EntityExtractor()
    baseline_score = evaluate(baseline, valset)
    logger.info("Baseline score (valset): %.3f", baseline_score)

    # Create log directory for GEPA artifacts
    log_dir = GEPA_OUTPUT_DIR / "extract_nodes"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info("GEPA logs will be saved to: %s", log_dir)

    # Instantiate and run GEPA
    logger.info(
        "Starting GEPA optimization with max_full_evals=%d", GEPA_MAX_FULL_EVALS
    )
    gepa = GEPA(
        metric=gepa_entity_metric,
        max_full_evals=GEPA_MAX_FULL_EVALS,
        reflection_lm=judge_lm,
        reflection_minibatch_size=GEPA_REFLECTION_MINIBATCH_SIZE,
        track_stats=True,
        log_dir=str(log_dir),
        num_threads=GEPA_NUM_THREADS,
    )

    optimized = gepa.compile(student=baseline, trainset=trainset, valset=valset)

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
