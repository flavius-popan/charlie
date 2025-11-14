"""Optimizer for pipeline/generate_summaries.py using DSPy's BootstrapFewShot.

Keeps summaries short, human, and factual for journal-centric entities.

Usage:
    python -m pipeline.optimizers.generate_summaries_optimizer
"""

from __future__ import annotations

import json
from pathlib import Path
import logging
import dspy
from dspy.teleprompt import MIPROv2

from dspy_outlines import OutlinesAdapter, OutlinesLM
from pipeline.generate_summaries import (
    EntitySummary,
    SummaryGenerator,
    build_node_payload,
    build_summary_context,
)
from settings import DEFAULT_MODEL_PATH, MODEL_CONFIG


PROMPT_OUTPUT = Path(__file__).parent.parent / "prompts" / "generate_summaries.json"
logger = logging.getLogger(__name__)


def configure_dspy():
    """Match runtime LM + adapter configuration."""

    lm = OutlinesLM(model_path=DEFAULT_MODEL_PATH, generation_config=MODEL_CONFIG)
    adapter = OutlinesAdapter()
    dspy.configure(lm=lm, adapter=adapter)
    logger.info("Configured DSPy with OutlinesLM (model: %s)", DEFAULT_MODEL_PATH)


def build_trainset() -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Create concise summary examples rooted in lived experiences."""

    def _context_json(
        *,
        episode_content: str,
        previous_episodes: list[str],
        entity_name: str,
        entity_type: str,
        existing_summary: str,
        attributes: dict,
    ) -> str:
        labels = ["Entity"]
        normalized_type = entity_type.strip()
        if normalized_type and normalized_type not in labels:
            labels.append(normalized_type)

        node_payload = build_node_payload(
            name=entity_name,
            summary=existing_summary,
            labels=labels,
            attributes=attributes,
        )
        context_dict = build_summary_context(
            node_payload=node_payload,
            episode_content=episode_content,
            previous_episode_texts=previous_episodes,
        )
        return json.dumps(context_dict, ensure_ascii=False)

    def example(
        *,
        episode_content: str,
        previous_episodes: list[str],
        entity_name: str,
        entity_type: str,
        existing_summary: str,
        attributes: dict,
        summary_text: str,
        key_phrases: list[str],
    ) -> dspy.Example:
        context_json = _context_json(
            episode_content=episode_content,
            previous_episodes=previous_episodes,
            entity_name=entity_name,
            entity_type=entity_type,
            existing_summary=existing_summary,
            attributes=attributes,
        )
        return dspy.Example(
            summary_context=context_json,
            summary=EntitySummary(summary=summary_text),
            key_phrases=key_phrases,
        ).with_inputs("summary_context")

    all_examples = [
        example(
            episode_content=(
                "Kai and I biked the Embarcadero at dawn. "
                "He vented about finishing his art school portfolio."
            ),
            previous_episodes=["Kai stayed over last weekend to prep his art school interviews."],
            entity_name="Kai",
            entity_type="Person",
            existing_summary="",
            attributes={"relationship_type": "friend"},
            summary_text="Kai biked the Embarcadero with me at dawn and is polishing his art school portfolio.",
            key_phrases=["Embarcadero", "portfolio"],
        ),
        example(
            episode_content=(
                "Sat with Grandma Rosa at St. Jude's while Dr. Yates walked her through tomorrow's surgery."
            ),
            previous_episodes=["Grandma Rosa is still regaining strength after pneumonia."],
            entity_name="Grandma Rosa",
            entity_type="Person",
            existing_summary="Recovering slowly from pneumonia.",
            attributes={"relationship_type": "family"},
            summary_text="Grandma Rosa prepared for tomorrow's surgery with Dr. Yates at St. Jude's.",
            key_phrases=["Dr. Yates", "surgery"],
        ),
        example(
            episode_content=(
                "Spent the morning at the 24th Street Community Garden building raised beds with three teen volunteers."
            ),
            previous_episodes=["Garden board asked me to log the next improvements."],
            entity_name="24th Street Community Garden",
            entity_type="Place",
            existing_summary="",
            attributes={},
            summary_text="24th Street Community Garden added raised beds and welcomed a crew of teen volunteers.",
            key_phrases=["raised beds", "teen volunteers"],
        ),
        example(
            episode_content=(
                "Snuck out for a night swim at Aquatic Park with Marco; the cold shock actually calmed me down."
            ),
            previous_episodes=["Last week night's swim ended early because of lightning."],
            entity_name="Night swim",
            entity_type="Activity",
            existing_summary="",
            attributes={},
            summary_text="Night swim at Aquatic Park with Marco felt icy but settled my nerves.",
            key_phrases=["Aquatic Park", "Marco"],
        ),
        example(
            episode_content=(
                "Session with therapist Ines. She led long breathing ladders and assigned a new journal prompt."
            ),
            previous_episodes=["Ines keeps nudging me toward gentler self-talk."],
            entity_name="Ines",
            entity_type="Person",
            existing_summary="Working with me on grounding skills.",
            attributes={"relationship_type": "therapist"},
            summary_text="Therapist Ines guided slow breathing ladders and gave me a new journal prompt.",
            key_phrases=["breathing", "journal"],
        ),
        example(
            episode_content=(
                "Been practicing patience while caring for Mom—slow paperwork, slow progress, just breathing through the delays."
            ),
            previous_episodes=["Patience theme keeps surfacing every time Mom's appointments slip."],
            entity_name="Patience",
            entity_type="Concept",
            existing_summary="",
            attributes={},
            summary_text="Patience means caring for Mom and breathing through constant delays in her paperwork.",
            key_phrases=["Mom", "delays"],
        ),
        example(
            episode_content=(
                "Priya hosted a cozy potluck tonight and spread out her colored pens to map my training schedule."
            ),
            previous_episodes=["She loves turning my training plans into rainbow timelines."],
            entity_name="Priya",
            entity_type="Person",
            existing_summary="Keeps me motivated for races.",
            attributes={"relationship_type": "friend"},
            summary_text="Priya hosted a cozy potluck and mapped my training schedule with her colored pens.",
            key_phrases=["potluck", "training schedule"],
        ),
        example(
            episode_content=(
                "Harbor Clinic smelled like mint tea today. Ben reorganized the pamphlets so folks could actually find resources."
            ),
            previous_episodes=["That waiting room usually feels chaotic."],
            entity_name="Harbor Clinic",
            entity_type="Place",
            existing_summary="",
            attributes={},
            summary_text="Harbor Clinic felt calmer with mint tea in the waiting room while Ben reorganized the pamphlets.",
            key_phrases=["mint tea", "pamphlets"],
        ),
    ]

    trainset = all_examples[:6]
    valset = all_examples[6:]
    logger.info(
        "Built summary trainset with %d examples, valset with %d examples",
        len(trainset),
        len(valset),
    )
    return trainset, valset


def _summary_text(value) -> str:
    """Extract plain summary text from adapter output."""

    if value is None:
        return ""

    if isinstance(value, EntitySummary):
        return value.summary

    if isinstance(value, dict):
        return value.get("summary", "")

    return getattr(value, "summary", "")


def summary_generation_metric(example, prediction, trace=None) -> float:
    """Score summaries by keyword coverage."""

    summary_text = _summary_text(prediction).strip()
    if not summary_text:
        return 0.0

    phrases = getattr(example, "key_phrases", []) or []
    if not phrases:
        target = _summary_text(example.summary).strip().lower()
        return 1.0 if summary_text.lower() == target else 0.0

    summary_lower = summary_text.lower()
    matches = sum(1 for phrase in phrases if phrase.lower() in summary_lower)
    return matches / len(phrases)


def optimize(trainset: list[dspy.Example]) -> SummaryGenerator:
    """Optimize SummaryGenerator with MIPROv2."""

    logger.info("Starting summary optimization with %d examples", len(trainset))
    optimizer = MIPROv2(
        metric=summary_generation_metric,
        auto=None,
        num_candidates=3,
        init_temperature=0.5,
        metric_threshold=0.90,
    )

    student = SummaryGenerator()
    optimized = optimizer.compile(
        student=student,
        trainset=trainset,
        num_trials=8,
        max_bootstrapped_demos=2,
        max_labeled_demos=3,
        minibatch_size=2,
        requires_permission_to_run=False,
    )

    logger.info("Summary optimization completed")
    return optimized


def evaluate(module: SummaryGenerator, dataset: list[dspy.Example]) -> float:
    """Average keyword coverage across dataset."""

    scores: list[float] = []
    for example in dataset:
        prediction = module(summary_context=example.summary_context)
        scores.append(summary_generation_metric(example, prediction))

    return sum(scores) / len(scores) if scores else 0.0


def main():
    """Full optimization workflow for summary generation."""

    logging.basicConfig(level=logging.INFO)
    configure_dspy()

    trainset, valset = build_trainset()
    baseline_module = SummaryGenerator()
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
