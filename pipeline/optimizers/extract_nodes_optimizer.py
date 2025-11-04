"""Optimizer for pipeline/extract_nodes.py using DSPy's BootstrapFewShot.

This script optimizes the EntityExtractor module's prompts using a training set
of journal entries focused on Person and Emotion entities.

Usage:
    python -m pipeline.optimizers.extract_nodes_optimizer
"""

from __future__ import annotations
from pathlib import Path
import logging
import json
import dspy
from dspy.teleprompt import BootstrapFewShot

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


def build_trainset() -> list[dspy.Example]:
    """Build training examples for entity extraction optimization.

    Creates 20 diverse journal-focused examples covering various:
    - Age groups (teens, young adults, middle-aged, seniors)
    - Relationships (friends, family, colleagues, professionals)
    - Emotions (positive, negative, complex/mixed)
    - Scenarios (daily life, challenges, celebrations, reflections)

    Returns:
        List of dspy.Example objects with episode_content and expected entities
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
                "entity_type_description": "people author interacts with",
            },
            {
                "entity_type_id": 2,
                "entity_type_name": "Emotion",
                "entity_type_description": "emotional states author experiences",
            },
        ]
    )

    examples = []

    # Example 1: Teen - nervousness before presentation
    examples.append(
        dspy.Example(
            episode_content="Today I met with Sarah at the coffee shop. Feeling anxious about tomorrow's presentation.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Sarah", entity_type_id=1),
                    ExtractedEntity(name="anxious", entity_type_id=2),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 2: Young adult - job interview excitement
    examples.append(
        dspy.Example(
            episode_content="Had coffee with Emma this morning. She gave me interview tips and now I'm feeling confident and excited!",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Emma", entity_type_id=1),
                    ExtractedEntity(name="confident", entity_type_id=2),
                    ExtractedEntity(name="excited", entity_type_id=2),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 3: Teen - friendship tension
    examples.append(
        dspy.Example(
            episode_content="Marcus hasn't responded to my texts in days. Feeling hurt and confused about what I did wrong.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Marcus", entity_type_id=1),
                    ExtractedEntity(name="hurt", entity_type_id=2),
                    ExtractedEntity(name="confused", entity_type_id=2),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 4: Parent - family joy
    examples.append(
        dspy.Example(
            episode_content="Watched my daughter Lily perform in the school play tonight. Dad was there too. Feeling so proud and grateful.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Lily", entity_type_id=1),
                    ExtractedEntity(name="Dad", entity_type_id=1),
                    ExtractedEntity(name="proud", entity_type_id=2),
                    ExtractedEntity(name="grateful", entity_type_id=2),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 5: College student - stress and support
    examples.append(
        dspy.Example(
            episode_content="Finals week is crushing me. Professor Chen extended my deadline after I told her about Mom's surgery. Feeling overwhelmed but relieved.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Professor Chen", entity_type_id=1),
                    ExtractedEntity(name="Mom", entity_type_id=1),
                    ExtractedEntity(name="overwhelmed", entity_type_id=2),
                    ExtractedEntity(name="relieved", entity_type_id=2),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 6: Young professional - career milestone
    examples.append(
        dspy.Example(
            episode_content="Got the promotion! Called Aisha immediately and she screamed with joy. My manager David said I earned it. Feeling accomplished and validated.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Aisha", entity_type_id=1),
                    ExtractedEntity(name="David", entity_type_id=1),
                    ExtractedEntity(name="accomplished", entity_type_id=2),
                    ExtractedEntity(name="validated", entity_type_id=2),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 7: Middle-aged - relationship conflict
    examples.append(
        dspy.Example(
            episode_content="Another argument with Jordan about finances. Dr. Martinez says we need to work on communication. Feeling frustrated and hopeful at the same time.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Jordan", entity_type_id=1),
                    ExtractedEntity(name="Dr. Martinez", entity_type_id=1),
                    ExtractedEntity(name="frustrated", entity_type_id=2),
                    ExtractedEntity(name="hopeful", entity_type_id=2),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 8: Senior - nostalgia and connection
    examples.append(
        dspy.Example(
            episode_content="My granddaughter Sophie visited today. We looked at old photos and she asked about my childhood. Feeling nostalgic and warm inside.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Sophie", entity_type_id=1),
                    ExtractedEntity(name="nostalgic", entity_type_id=2),
                    ExtractedEntity(name="warm", entity_type_id=2),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 9: Teen - social anxiety
    examples.append(
        dspy.Example(
            episode_content="First day at the new school. Maya showed me around and introduced me to her friends. Still feeling nervous but less alone now.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Maya", entity_type_id=1),
                    ExtractedEntity(name="nervous", entity_type_id=2),
                    ExtractedEntity(name="less alone", entity_type_id=2),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 10: Young adult - betrayal and anger
    examples.append(
        dspy.Example(
            episode_content="Found out that Rachel told my secrets to everyone at work. Talked to my therapist Dr. Kim about it. Feeling betrayed and angry.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Rachel", entity_type_id=1),
                    ExtractedEntity(name="Dr. Kim", entity_type_id=1),
                    ExtractedEntity(name="betrayed", entity_type_id=2),
                    ExtractedEntity(name="angry", entity_type_id=2),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 11: Parent - worry and love
    examples.append(
        dspy.Example(
            episode_content="My son Miguel came home late again. His coach says he's been practicing extra hard for the championship. Feeling worried but also loving his dedication.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Miguel", entity_type_id=1),
                    ExtractedEntity(name="worried", entity_type_id=2),
                    ExtractedEntity(name="loving", entity_type_id=2),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 12: Middle-aged - grief and support
    examples.append(
        dspy.Example(
            episode_content="Six months since we lost Mom. My sister Carmen called to check in. Pastor Williams invited me to the grief support group. Feeling sad but supported.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Mom", entity_type_id=1),
                    ExtractedEntity(name="Carmen", entity_type_id=1),
                    ExtractedEntity(name="Pastor Williams", entity_type_id=1),
                    ExtractedEntity(name="sad", entity_type_id=2),
                    ExtractedEntity(name="supported", entity_type_id=2),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 13: College student - creative joy
    examples.append(
        dspy.Example(
            episode_content="Finished my first painting! My roommate Zara said it belongs in a gallery. Professor Lee wants to feature it in the student exhibition. Feeling joyful and proud.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Zara", entity_type_id=1),
                    ExtractedEntity(name="Professor Lee", entity_type_id=1),
                    ExtractedEntity(name="joyful", entity_type_id=2),
                    ExtractedEntity(name="proud", entity_type_id=2),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 14: Young professional - imposter syndrome
    examples.append(
        dspy.Example(
            episode_content="Leading my first team meeting tomorrow. My mentor Hassan says I'm ready but I don't feel it. Feeling inadequate and scared of failing.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Hassan", entity_type_id=1),
                    ExtractedEntity(name="inadequate", entity_type_id=2),
                    ExtractedEntity(name="scared", entity_type_id=2),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 15: Senior - health concerns and optimism
    examples.append(
        dspy.Example(
            episode_content="Doctor Patel says my blood pressure is improving. My neighbor Frank walks with me every morning now. Feeling optimistic and thankful for good friends.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Doctor Patel", entity_type_id=1),
                    ExtractedEntity(name="Frank", entity_type_id=1),
                    ExtractedEntity(name="optimistic", entity_type_id=2),
                    ExtractedEntity(name="thankful", entity_type_id=2),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 16: Teen - romantic feelings
    examples.append(
        dspy.Example(
            episode_content="Alex smiled at me during lunch today. My best friend Nina says I should just ask them out. Feeling nervous, excited, and terrified all at once.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Alex", entity_type_id=1),
                    ExtractedEntity(name="Nina", entity_type_id=1),
                    ExtractedEntity(name="nervous", entity_type_id=2),
                    ExtractedEntity(name="excited", entity_type_id=2),
                    ExtractedEntity(name="terrified", entity_type_id=2),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 17: Middle-aged - professional setback
    examples.append(
        dspy.Example(
            episode_content="Didn't get the grant I worked on for months. My colleague Iris reminded me that rejection is part of the process. Feeling disappointed but resilient.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Iris", entity_type_id=1),
                    ExtractedEntity(name="disappointed", entity_type_id=2),
                    ExtractedEntity(name="resilient", entity_type_id=2),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 18: Young adult - life transition
    examples.append(
        dspy.Example(
            episode_content="Moving across the country for the new job. My brother Theo helped me pack. Mom cried but said she's proud. Feeling scared, excited, and guilty for leaving.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Theo", entity_type_id=1),
                    ExtractedEntity(name="Mom", entity_type_id=1),
                    ExtractedEntity(name="scared", entity_type_id=2),
                    ExtractedEntity(name="excited", entity_type_id=2),
                    ExtractedEntity(name="guilty", entity_type_id=2),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 19: Parent - parenting milestone
    examples.append(
        dspy.Example(
            episode_content="Dropped my daughter Amara off at college today. Her roommate Keiko seems sweet. My partner Sam and I cried all the way home. Feeling bittersweet and empty.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Amara", entity_type_id=1),
                    ExtractedEntity(name="Keiko", entity_type_id=1),
                    ExtractedEntity(name="Sam", entity_type_id=1),
                    ExtractedEntity(name="bittersweet", entity_type_id=2),
                    ExtractedEntity(name="empty", entity_type_id=2),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 20: Senior - simple daily joy
    examples.append(
        dspy.Example(
            episode_content="Had tea with my old friend Eleanor this afternoon. We talked about our grandchildren and laughed about getting old. Feeling content and peaceful.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Eleanor", entity_type_id=1),
                    ExtractedEntity(name="content", entity_type_id=2),
                    ExtractedEntity(name="peaceful", entity_type_id=2),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    logger.info("Built training set with %d examples", len(examples))
    return examples


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

    # Extract predicted entities (handle both object and dict formats)
    pred_entities = getattr(prediction, "extracted_entities", None)
    if pred_entities is None:
        return 0.0

    if isinstance(pred_entities, ExtractedEntities):
        predicted = {
            (e.name.lower(), e.entity_type_id) for e in pred_entities.extracted_entities
        }
    else:
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


def optimize(
    trainset: list[dspy.Example], max_bootstrapped_demos: int = 5
) -> EntityExtractor:
    """Run BootstrapFewShot optimization.

    Uses DSPy's BootstrapFewShot optimizer to improve EntityExtractor prompts
    by generating effective few-shot examples based on the training set.

    Args:
        trainset: List of training examples
        max_bootstrapped_demos: Maximum number of demonstrations to generate

    Returns:
        Optimized EntityExtractor module
    """
    logger.info(
        "Starting BootstrapFewShot optimization with %d examples", len(trainset)
    )

    optimizer = BootstrapFewShot(
        metric=entity_extraction_metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
    )

    student = EntityExtractor()
    optimized = optimizer.compile(student=student, trainset=trainset)

    logger.info("Optimization completed")
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
    """Main optimization workflow.

    1. Configure DSPy with OutlinesLM
    2. Build training set
    3. Evaluate baseline performance
    4. Optimize prompts with BootstrapFewShot
    5. Evaluate optimized performance
    6. Save optimized prompts to file
    """
    logging.basicConfig(level=logging.INFO)

    configure_dspy()

    trainset = build_trainset()
    logger.info("Built trainset with %d examples", len(trainset))

    # Baseline evaluation
    baseline_module = EntityExtractor()
    baseline_score = evaluate(baseline_module, trainset)
    logger.info("Baseline score: %.3f", baseline_score)

    # Optimize
    optimized_module = optimize(trainset, max_bootstrapped_demos=5)

    # Optimized evaluation
    optimized_score = evaluate(optimized_module, trainset)
    logger.info("Optimized score: %.3f", optimized_score)

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
