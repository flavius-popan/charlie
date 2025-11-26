"""GEPA optimizer for entity extraction.

Usage:
    python -m backend.optimizers.extract_nodes_optimizer
    python -m backend.optimizers.extract_nodes_optimizer --no-cache
    python -m backend.optimizers.extract_nodes_optimizer --remote
    python -m backend.optimizers.extract_nodes_optimizer --remote --no-cache
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import dspy
from dspy.teleprompt import GEPA

from backend.optimizers import (
    configure_dspy,
    get_reflection_lm,
    evaluate_module,
    DATA_DIR,
    PROMPTS_DIR,
    GEPA_AUTO_MODE,
    GEPA_NUM_THREADS_LOCAL,
    GEPA_NUM_THREADS_REMOTE,
)
from backend.graph.extract_nodes import (
    EntityExtractor,
    ExtractedEntity,
    ExtractedEntities,
)
from backend.graph.entities_edges import format_entity_types_for_llm

logger = logging.getLogger(__name__)

DATA_FILE = DATA_DIR / "extract_nodes_examples.json"
PROMPT_OUTPUT = PROMPTS_DIR / "extract_nodes.json"


def load_examples() -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Load examples from JSON, generate entity_types dynamically."""
    with DATA_FILE.open() as f:
        data = json.load(f)

    # Dynamic from entities_edges.py - not stored in JSON
    entity_types_json = format_entity_types_for_llm()

    examples = []
    for ex in data["examples"]:
        entities = ExtractedEntities(
            extracted_entities=[
                ExtractedEntity(name=e["name"], entity_type_id=e["entity_type_id"])
                for e in ex["expected_entities"]
            ]
        )
        examples.append(
            dspy.Example(
                episode_content=ex["episode_content"],
                entity_types=entity_types_json,
                extracted_entities=entities,
            ).with_inputs("episode_content", "entity_types")
        )

    split_idx = int(len(examples) * 0.8)
    return examples[:split_idx], examples[split_idx:]


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Score extraction quality and explain failures so GEPA can improve prompts."""
    expected = {
        (e.name.lower(), e.entity_type_id)
        for e in gold.extracted_entities.extracted_entities
    }

    pred_entities = getattr(pred, "extracted_entities", None)
    if pred_entities is None:
        return dspy.Prediction(
            score=0.0,
            feedback="No entities extracted. The output must contain an 'extracted_entities' field."
        )

    if isinstance(pred_entities, ExtractedEntities):
        predicted = {(e.name.lower(), e.entity_type_id) for e in pred_entities.extracted_entities}
    elif isinstance(pred_entities, list):
        predicted = set()
        for e in pred_entities:
            if isinstance(e, ExtractedEntity):
                predicted.add((e.name.lower(), e.entity_type_id))
            elif isinstance(e, dict):
                predicted.add((e.get("name", "").lower(), e.get("entity_type_id", 0)))
    else:
        return dspy.Prediction(
            score=0.0,
            feedback=f"Unknown format: {type(pred_entities)}. Expected ExtractedEntities or list."
        )

    if not expected:
        score = 1.0 if not predicted else 0.0
        return dspy.Prediction(score=score, feedback="No entities expected in this example.")

    if not predicted:
        feedback = f"No entities extracted but {len(expected)} were expected.\n"
        feedback += f"Expected entities: {sorted(expected)}\n"
        feedback += "Try identifying people, places, organizations, and activities mentioned."
        return dspy.Prediction(score=0.0, feedback=feedback)

    intersection = expected & predicted
    missing = expected - predicted
    extra = predicted - expected

    precision = len(intersection) / len(predicted) if predicted else 0
    recall = len(intersection) / len(expected) if expected else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Build rich feedback
    if f1 == 1.0:
        feedback = "Correct - all entities extracted accurately. Good job!"
    else:
        feedback = f"F1={f1:.2f} (precision={precision:.2f}, recall={recall:.2f})\n"
        if missing:
            feedback += f"MISSED ({len(missing)}): {sorted(missing)}\n"
        if extra:
            feedback += f"EXTRA ({len(extra)}): {sorted(extra)}\n"
        if intersection:
            feedback += f"CORRECT ({len(intersection)}): {sorted(intersection)}\n"

        # Actionable guidance
        if missing:
            feedback += "\nTry to identify all named entities. "
            feedback += "Look for compound names (e.g., 'Dr. Smith'), organizations, and activities."
        if extra:
            feedback += "\nAvoid extracting generic terms. "
            feedback += "Focus on specific, named entities mentioned in the text."

    return dspy.Prediction(score=f1, feedback=feedback)


def main():
    parser = argparse.ArgumentParser(description="GEPA optimizer for entity extraction")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable DSPy caching for a clean run (use when iterating on examples/metric)",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Use HuggingFace endpoint instead of local model (faster, requires HUGGINGFACE_API_KEY)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    if args.no_cache:
        dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
        logger.info("Caching disabled for clean run")

    configure_dspy(remote=args.remote)
    reflection_lm = get_reflection_lm()

    trainset, valset = load_examples()
    logger.info("Loaded %d train, %d val examples", len(trainset), len(valset))

    num_threads = GEPA_NUM_THREADS_REMOTE if args.remote else GEPA_NUM_THREADS_LOCAL
    logger.info("Using num_threads=%d", num_threads)

    # Use load_prompts=False to get true baseline without cached optimizations
    baseline = EntityExtractor(load_prompts=False)
    baseline_score = evaluate_module(baseline, valset, metric, num_threads=num_threads)
    logger.info("Baseline: %.3f", baseline_score)

    gepa = GEPA(
        metric=metric,
        reflection_lm=reflection_lm,
        auto=GEPA_AUTO_MODE,
        num_threads=num_threads,
    )

    start_time = time.perf_counter()
    optimized = gepa.compile(student=baseline, trainset=trainset, valset=valset)
    elapsed = time.perf_counter() - start_time
    logger.info("GEPA optimization completed in %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)

    optimized_score = evaluate_module(optimized, valset, metric, num_threads=num_threads)
    logger.info("Optimized: %.3f", optimized_score)

    PROMPT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    optimized.save(str(PROMPT_OUTPUT))
    logger.info("Saved to %s", PROMPT_OUTPUT)
    logger.info("Improvement: %.3f -> %.3f (+%.3f)", baseline_score, optimized_score, optimized_score - baseline_score)


if __name__ == "__main__":
    main()
