"""Optimizer for the EntityReflexionModule in pipeline/extract_nodes.py.

This tunes the reflexion signature so it reliably proposes entities that
DistilBERT misses (activities, gatherings, concepts) during the hybrid
NER + DSPy extraction pass.

Usage:
    python -m pipeline.optimizers.entity_reflexion_optimizer
"""

from __future__ import annotations

import json
from pathlib import Path
import logging
import dspy
from dspy.teleprompt import MIPROv2

from dspy_outlines import OutlinesAdapter, OutlinesLM
from pipeline.extract_nodes import (
    EntityReflexionModule,
    ReflexionEntities,
    ReflexionEntity,
)
from settings import DEFAULT_MODEL_PATH, MODEL_CONFIG


PROMPT_OUTPUT = Path(__file__).parent.parent / "prompts" / "entity_reflexion.json"
logger = logging.getLogger(__name__)


def configure_dspy():
    """Configure DSPy using the same LM + adapter as production."""

    lm = OutlinesLM(model_path=DEFAULT_MODEL_PATH, generation_config=MODEL_CONFIG)
    adapter = OutlinesAdapter()
    dspy.configure(lm=lm, adapter=adapter)
    logger.info("Configured DSPy with OutlinesLM (model: %s)", DEFAULT_MODEL_PATH)


def _example(
    *,
    episode_content: str,
    previous_episodes: list[str],
    extracted_entities: list[str],
    missed_entities: list[tuple[str, str | None]],
) -> dspy.Example:
    """Helper to keep examples readable."""

    return dspy.Example(
        episode_content=episode_content,
        previous_episodes=json.dumps(previous_episodes),
        extracted_entities=json.dumps(extracted_entities),
        missed_entities=ReflexionEntities(
            missed_entities=[
                ReflexionEntity(name=name, entity_type=entity_type)
                for name, entity_type in missed_entities
            ]
        ),
    ).with_inputs("episode_content", "previous_episodes", "extracted_entities")


def build_trainset() -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Construct lightweight train/val splits focused on missed activities."""

    all_examples: list[dspy.Example] = [
        _example(
            episode_content=(
                "Met Asha at Dolores Park before sunrise yoga, then biked to Mission "
                "Commons for our volunteer shift. The flow felt meditative."
            ),
            previous_episodes=["Asha keeps nudging me to take the sunrise yoga slot."],
            extracted_entities=["Asha", "Dolores Park", "Mission Commons"],
            missed_entities=[
                ("sunrise yoga", "Activity"),
                ("Mission Commons volunteer shift", "Activity"),
            ],
        ),
        _example(
            episode_content=(
                "Hosted a grief circle at Tender Hearts. Joy facilitated while Marcus "
                "leaned on his sponsor Rita. The sobriety check-in felt raw but grounding."
            ),
            previous_episodes=["Marcus texted that sobriety circles keep him steady."],
            extracted_entities=["Joy", "Marcus", "Rita", "Tender Hearts"],
            missed_entities=[
                ("grief circle", "Activity"),
                ("sobriety check-in", "Activity"),
            ],
        ),
        _example(
            episode_content=(
                "Leila and Rico prepped the Harvest Hub pantry shift near Ferry Plaza "
                "today. I tracked donations while they stocked produce bins."
            ),
            previous_episodes=["Harvest Hub pantry crew is short on volunteers lately."],
            extracted_entities=["Leila and Rico", "Ferry Plaza"],
            missed_entities=[
                ("Harvest Hub pantry shift", "Activity"),
                ("produce bin stocking", "Activity"),
            ],
        ),
        _example(
            episode_content=(
                "Coach Priya mapped sunrise hill repeats and reminded me about the "
                "Mission Milers fundraiser run this Saturday."
            ),
            previous_episodes=["Mission Milers want more stories from runners."],
            extracted_entities=["Priya", "Mission Milers"],
            missed_entities=[
                ("sunrise hill repeats", "Activity"),
                ("Mission Milers fundraiser run", "Activity"),
            ],
        ),
        _example(
            episode_content=(
                "Camped out at Harbor Clinic while Mom dozed. Ben from social work "
                "helped push the Medi-Cal paperwork, and my whole mantra lately is patience."
            ),
            previous_episodes=["Every clinic visit tests my patience mantra."],
            extracted_entities=["Harbor Clinic", "Ben", "Mom", "Medi-Cal"],
            missed_entities=[
                ("Medi-Cal paperwork sprint", "Activity"),
                ("patience mantra", "Concept"),
            ],
        ),
        _example(
            episode_content=(
                "Night swim at Aquatic Park with Marco calmed my nerves. "
                "We sealed a pact to keep our Sunday swims no matter the fog."
            ),
            previous_episodes=["Last Sunday's swim got canceled from lightning."],
            extracted_entities=["Aquatic Park", "Marco"],
            missed_entities=[
                ("night swim", "Activity"),
                ("Sunday swim pact", "Activity"),
            ],
        ),
        _example(
            episode_content=(
                "Therapist Ines walked me through box breathing and we wrote a grounding "
                "mantra on neon sticky notes for my bedside table."
            ),
            previous_episodes=["Ines keeps asking me to name grounding practices out loud."],
            extracted_entities=["Ines"],
            missed_entities=[
                ("box breathing drill", "Activity"),
                ("grounding mantra", "Concept"),
            ],
        ),
        _example(
            episode_content=(
                "Nico and Tessa hosted a rooftop potluck for our queer book club to plan "
                "the Pride art booth. So many homemade dishes and sketches everywhere."
            ),
            previous_episodes=["Book club wants the Pride booth to feel like a living room."],
            extracted_entities=["Nico", "Tessa", "Pride"],
            missed_entities=[
                ("queer book club", "Organization"),
                ("rooftop potluck", "Activity"),
                ("Pride art booth plan", "Activity"),
            ],
        ),
    ]

    trainset = all_examples[:6]
    valset = all_examples[6:]
    logger.info(
        "Built reflexion trainset with %d examples, valset with %d examples",
        len(trainset),
        len(valset),
    )
    return trainset, valset


def _entity_map(value) -> dict[str, str]:
    """Normalize outputs into name → type mapping."""

    if value is None:
        return {}

    if isinstance(value, ReflexionEntities):
        candidates = value.missed_entities
    elif isinstance(value, dict) and "missed_entities" in value:
        candidates = value["missed_entities"]
    else:
        candidates = value

    mapping: dict[str, str] = {}

    for item in candidates or []:
        name: str | None = None
        type_hint: str | None = None

        if isinstance(item, ReflexionEntity):
            name = item.name
            type_hint = item.entity_type
        elif isinstance(item, dict):
            name = item.get("name") or item.get("entity")
            type_hint = item.get("entity_type") or item.get("type")
        elif isinstance(item, (list, tuple)) and item:
            name = item[0]
            type_hint = item[1] if len(item) > 1 else None
        elif isinstance(item, str):
            name = item

        if not name:
            continue

        normalized_name = name.strip().lower()
        if not normalized_name:
            continue

        normalized_type = (type_hint or "").strip().lower()
        mapping[normalized_name] = normalized_type

    return mapping


def reflexion_metric(example, prediction, trace=None) -> float:
    """F1 on entity names with a small bonus for correct type hints."""

    expected_map = _entity_map(example.missed_entities)
    predicted_map = _entity_map(prediction)

    expected_names = set(expected_map.keys())
    predicted_names = set(predicted_map.keys())

    if not expected_names:
        return 1.0 if not predicted_names else 0.0

    if not predicted_names:
        return 0.0

    intersection = expected_names & predicted_names
    if not intersection:
        return 0.0

    precision = len(intersection) / len(predicted_names)
    recall = len(intersection) / len(expected_names)
    f1 = 2 * (precision * recall) / (precision + recall)

    type_matches = sum(
        1
        for name in intersection
        if expected_map[name] and expected_map[name] == predicted_map.get(name, "")
    )
    type_bonus = 0.2 * (type_matches / len(expected_names)) if expected_names else 0.0

    return min(1.0, f1 + type_bonus)


def optimize(trainset: list[dspy.Example]) -> EntityReflexionModule:
    """Run MIPROv2 to tune reflexion prompts and demos."""

    logger.info("Starting reflexion optimization with %d examples", len(trainset))
    optimizer = MIPROv2(
        metric=reflexion_metric,
        auto=None,
        num_candidates=5,
        init_temperature=1.0,
        metric_threshold=0.8,
    )

    student = EntityReflexionModule()
    optimized = optimizer.compile(
        student=student,
        trainset=trainset,
        num_trials=10,
        max_bootstrapped_demos=2,
        max_labeled_demos=3,
        minibatch_size=2,
        requires_permission_to_run=False,
    )

    logger.info("Reflexion optimization completed")
    return optimized


def evaluate(module: EntityReflexionModule, dataset: list[dspy.Example]) -> float:
    """Average reflexion metric over dataset."""

    scores: list[float] = []
    for example in dataset:
        prediction = module(
            episode_content=example.episode_content,
            previous_episodes=example.previous_episodes,
            extracted_entities=example.extracted_entities,
        )
        scores.append(reflexion_metric(example, prediction))

    return sum(scores) / len(scores) if scores else 0.0


def main():
    """Main workflow: baseline → optimize → evaluate → save prompts."""

    logging.basicConfig(level=logging.INFO)
    configure_dspy()

    trainset, valset = build_trainset()

    baseline_module = EntityReflexionModule()
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
