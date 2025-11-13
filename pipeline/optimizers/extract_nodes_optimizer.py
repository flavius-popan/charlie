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

    Creates 20 realistic personal journal entries covering:
    - Entity types: Person, Place, Organization, Concept, Activity
    - Contexts: hanging out with friends, family time, going places, doing hobbies, appointments, life reflections
    - Tone: casual, authentic, human (not robotic business language)

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

    examples = []

    # Example 1: Coffee shop hangout with friend
    examples.append(
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
    examples.append(
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

    # Example 3: Family dinner at home
    examples.append(
        dspy.Example(
            episode_content="Mom and Dad came over for dinner. We actually had a good time for once, no arguments. Maybe things are getting better between us.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Mom", entity_type_id=1),
                    ExtractedEntity(name="Dad", entity_type_id=1),
                    ExtractedEntity(name="dinner", entity_type_id=5),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 4: Yoga class and self-care
    examples.append(
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

    # Example 5: Work stress and deadline
    examples.append(
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

    # Example 6: Weekend hike with friends
    examples.append(
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

    # Example 7: Book club meeting
    examples.append(
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

    # Example 8: Doctor appointment about anxiety
    examples.append(
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

    # Example 9: Art class at community center
    examples.append(
        dspy.Example(
            episode_content="Started that watercolor class at the Community Arts Center. The teacher, Ana, is so patient and the other students are friendly. Been wanting to try something creative for ages.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Community Arts Center", entity_type_id=3),
                    ExtractedEntity(name="Ana", entity_type_id=1),
                    ExtractedEntity(name="watercolor class", entity_type_id=5),
                    ExtractedEntity(name="creativity", entity_type_id=4),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 10: Late night conversation with partner
    examples.append(
        dspy.Example(
            episode_content="Had a real talk with Alex tonight about our future. Kids, marriage, where we want to live. It's intense thinking about commitment like this but also exciting?",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Alex", entity_type_id=1),
                    ExtractedEntity(name="future", entity_type_id=4),
                    ExtractedEntity(name="commitment", entity_type_id=4),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 11: Gym session and fitness journey
    examples.append(
        dspy.Example(
            episode_content="Finally hit a new PR at CrossFit Westside! My coach Ryan was pumped. Been working towards this for months. Proof that consistency pays off I guess.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="CrossFit Westside", entity_type_id=3),
                    ExtractedEntity(name="Ryan", entity_type_id=1),
                    ExtractedEntity(name="gym workout", entity_type_id=5),
                    ExtractedEntity(name="consistency", entity_type_id=4),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 12: Brunch with sibling
    examples.append(
        dspy.Example(
            episode_content="Brunch with my brother at Flour Bakery. He's going through a breakup and needed to vent. I'm glad we're close enough now that he feels comfortable talking to me about this stuff.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="brother", entity_type_id=1),
                    ExtractedEntity(name="Flour Bakery", entity_type_id=2),
                    ExtractedEntity(name="brunch", entity_type_id=5),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 13: Volunteering at shelter
    examples.append(
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

    # Example 14: Concert with friends
    examples.append(
        dspy.Example(
            episode_content="Saw Phoebe Bridgers at the Fillmore with Nina and Jordan. We sang every word and I cried during 'I Know The End'. Music hits different when you're with the right people.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="the Fillmore", entity_type_id=2),
                    ExtractedEntity(name="Nina", entity_type_id=1),
                    ExtractedEntity(name="Jordan", entity_type_id=1),
                    ExtractedEntity(name="concert", entity_type_id=5),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 15: First day at new job
    examples.append(
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

    # Example 16: Poetry slam event
    examples.append(
        dspy.Example(
            episode_content="Went to the poetry slam at Cafe Luna. My friend Zara performed and killed it. Made me want to write again. Art really does heal something in me.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Cafe Luna", entity_type_id=2),
                    ExtractedEntity(name="Zara", entity_type_id=1),
                    ExtractedEntity(name="poetry slam", entity_type_id=5),
                    ExtractedEntity(name="creativity", entity_type_id=4),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 17: Sunday morning farmers market
    examples.append(
        dspy.Example(
            episode_content="Farmers market with Sam. Got fresh flowers and talked to the vendor about her farm. These simple Sunday mornings are what life's about.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Sam", entity_type_id=1),
                    ExtractedEntity(name="farmers market", entity_type_id=5),
                    ExtractedEntity(name="simple living", entity_type_id=4),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 18: Running club meetup
    examples.append(
        dspy.Example(
            episode_content="Joined the Bay Area Runners group this morning. Met Hassan and Priya who are also training for the half marathon. Having accountability partners makes this less scary.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Bay Area Runners", entity_type_id=3),
                    ExtractedEntity(name="Hassan", entity_type_id=1),
                    ExtractedEntity(name="Priya", entity_type_id=1),
                    ExtractedEntity(name="running", entity_type_id=5),
                    ExtractedEntity(name="accountability", entity_type_id=4),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 19: Parent-teacher conference
    examples.append(
        dspy.Example(
            episode_content="Parent-teacher conference at Riverside Elementary for Lily. Ms. Johnson says she's doing great but seems anxious about tests. Guess she gets that from me.",
            entity_types=entity_types_json,
            extracted_entities=ExtractedEntities(
                extracted_entities=[
                    ExtractedEntity(name="Riverside Elementary", entity_type_id=3),
                    ExtractedEntity(name="Lily", entity_type_id=1),
                    ExtractedEntity(name="Ms. Johnson", entity_type_id=1),
                    ExtractedEntity(name="parent-teacher conference", entity_type_id=5),
                ]
            ),
        ).with_inputs("episode_content", "entity_types")
    )

    # Example 20: Game night with chosen family
    examples.append(
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
