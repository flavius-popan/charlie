"""Optimizer for pipeline/extract_nodes.py using DSPy's BootstrapFewShot.

This script optimizes the EntityExtractor module's prompts using a training set
of personal journal entries extracting Person, Place, Organization, Concept, and Activity entities.

Usage:
    python -m pipeline.optimizers.extract_nodes_optimizer
"""

from __future__ import annotations
from pathlib import Path
import logging
import json
import dspy
from dspy.teleprompt import MIPROv2

from dspy_outlines import OutlinesAdapter, OutlinesLM
from pipeline.extract_nodes import EntityExtractor
from pipeline.extract_nodes import ExtractedEntity, ExtractedEntities
from settings import DEFAULT_MODEL_PATH, MODEL_CONFIG


PROMPT_OUTPUT = Path(__file__).parent.parent / "prompts" / "extract_nodes.json"
logger = logging.getLogger(__name__)


def configure_dspy():
    """Configure DSPy with OutlinesLM and OutlinesAdapter.

    Uses the same configuration as the main pipeline to ensure compatibility.
    """
    lm = OutlinesLM(model_path=DEFAULT_MODEL_PATH, generation_config=MODEL_CONFIG)
    adapter = OutlinesAdapter()
    dspy.configure(lm=lm, adapter=adapter)
    logger.info("Configured DSPy with OutlinesLM (model: %s)", DEFAULT_MODEL_PATH)


def build_trainset() -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Build training and validation examples for entity extraction optimization.

    Creates 10 realistic personal journal entries covering:
    - Entity types: Person, Place, Organization, Concept, Activity
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
                "entity_type_description": "fallback for significant entities",
            },
            {
                "entity_type_id": 1,
                "entity_type_name": "Person",
                "entity_type_description": "individuals mentioned in the journal. Extract individuals mentioned: friends, family, colleagues, romantic partners, professionals (therapists/doctors), acquaintances.",
            },
            {
                "entity_type_id": 2,
                "entity_type_name": "Place",
                "entity_type_description": "specific locations and venues. Extract specific places visited or mentioned: coffee shops, parks, restaurants, cities, neighborhoods, venues, landmarks.",
            },
            {
                "entity_type_id": 3,
                "entity_type_name": "Organization",
                "entity_type_description": "companies, institutions, groups. Extract organizations engaged with: workplaces, schools, clubs, community groups, institutions, companies.",
            },
            {
                "entity_type_id": 4,
                "entity_type_name": "Concept",
                "entity_type_description": "abstract topics and life themes. Extract life themes and topics reflected on: personal growth, relationships, mental health, career, identity, values, beliefs.",
            },
            {
                "entity_type_id": 5,
                "entity_type_name": "Activity",
                "entity_type_description": "events, activities, and experiences. Extract specific activities and events: appointments, outings, hobbies, social gatherings, daily routines, significant moments.",
            },
        ]
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
                    ExtractedEntity(name="career goals", entity_type_id=4),
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
                    ExtractedEntity(name="therapy session", entity_type_id=5),
                    ExtractedEntity(name="relationship patterns", entity_type_id=4),
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
                    ExtractedEntity(name="yoga class", entity_type_id=5),
                    ExtractedEntity(name="self-care", entity_type_id=4),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 4: Work stress and deadline
    all_examples.append(
        dspy.Example(
            episode_content="Insane day at TechCorp. The product launch is next week and everything's on fire. My manager Lisa says I'm doing great but I feel like I'm barely keeping up.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="TechCorp", entity_type_id=3),
                    ExtractedEntity(name="Lisa", entity_type_id=1),
                    ExtractedEntity(name="product launch", entity_type_id=5),
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
                    ExtractedEntity(name="hiking", entity_type_id=5),
                    ExtractedEntity(name="mental health", entity_type_id=4),
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
                    ExtractedEntity(name="doctor appointment", entity_type_id=5),
                    ExtractedEntity(name="anxiety", entity_type_id=4),
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
                    ExtractedEntity(name="volunteering", entity_type_id=5),
                    ExtractedEntity(name="gratitude", entity_type_id=4),
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
                    ExtractedEntity(name="first day", entity_type_id=5),
                    ExtractedEntity(name="imposter syndrome", entity_type_id=4),
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
                    ExtractedEntity(name="book club", entity_type_id=5),
                    ExtractedEntity(name="regret", entity_type_id=4),
                    ExtractedEntity(name="second chances", entity_type_id=4),
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
                    ExtractedEntity(name="game night", entity_type_id=5),
                    ExtractedEntity(name="chosen family", entity_type_id=4),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    trainset = all_examples[:8]
    valset = all_examples[8:]

    logger.info("Built trainset with %d examples, valset with %d examples", len(trainset), len(valset))
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


def optimize(trainset: list[dspy.Example]) -> EntityExtractor:
    """Run MIPROv2 optimization with deepcopy-compatible OutlinesLM.

    MIPROv2 optimizes both instructions and few-shot examples:
    - init_temperature=1.0: Standard temperature for instruction diversity
    - num_candidates=5: Generate 5 instruction variants per predictor
    - num_trials=10: Run 10 optimization trials
    - metric_threshold=0.85: Filter low-quality bootstrapped examples
    - max_bootstrapped_demos=2: Minimal demos (MIPROv2 focuses on instructions)
    - max_labeled_demos=3: Use labeled examples as-is

    Args:
        trainset: List of training examples (8 recommended for speed)

    Returns:
        Optimized EntityExtractor module with tuned instructions and demos
    """
    logger.info("Starting MIPROv2 optimization with %d examples", len(trainset))

    optimizer = MIPROv2(
        metric=entity_extraction_metric,
        auto=None,
        num_candidates=3,
        init_temperature=0.5,
        metric_threshold=0.90,
    )

    student = EntityExtractor()
    optimized = optimizer.compile(
        student=student,
        trainset=trainset,
        num_trials=10,
        max_bootstrapped_demos=2,
        max_labeled_demos=3,
        minibatch_size=2,
        requires_permission_to_run=False,
    )

    logger.info("MIPROv2 optimization completed")
    return optimized


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
    """Main optimization workflow with MIPROv2.

    1. Configure DSPy with OutlinesLM (temp=0.0 for inference, temp=1.0 for optimization)
    2. Build training and validation sets (8 train, 2 val)
    3. Evaluate baseline on validation set
    4. Optimize with MIPROv2 (instruction + demo tuning, deepcopy-compatible)
    5. Evaluate optimized on validation set
    6. Save optimized prompts to file
    """
    logging.basicConfig(level=logging.INFO)

    configure_dspy()

    trainset, valset = build_trainset()
    logger.info("Built trainset with %d examples, valset with %d examples", len(trainset), len(valset))

    # Baseline evaluation on validation set
    baseline_module = EntityExtractor()
    baseline_score = evaluate(baseline_module, valset)
    logger.info("Baseline score (valset): %.3f", baseline_score)

    # Optimize using trainset
    optimized_module = optimize(trainset)

    # Optimized evaluation on validation set
    optimized_score = evaluate(optimized_module, valset)
    logger.info("Optimized score (valset): %.3f", optimized_score)

    # Save prompts
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
