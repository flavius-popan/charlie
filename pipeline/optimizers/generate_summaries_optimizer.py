"""Optimizer for pipeline/generate_summaries.py using DSPy's MIPROv2.

Keeps summaries short, human, and factual for journal-centric entities.
All summaries now model first-person narration (including explicit `Self`
examples) because the author reads these reflections directly.

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
            previous_episodes=[
                "Kai stayed over last weekend to prep his art school interviews."
            ],
            entity_name="Kai",
            entity_type="Person",
            existing_summary="",
            attributes={
                "relationship_type": "friend",
                "closeness": 0.78,
                "overall_valence": 0.6,
            },
            summary_text="I biked the Embarcadero at dawn with Kai and let him spill every anxious portfolio thought until he finally exhaled.",
            key_phrases=["sunrise ride", "portfolio jitters"],
        ),
        example(
            episode_content=(
                "Sat with Grandma Rosa at St. Jude's while Dr. Yates walked her through tomorrow's surgery."
            ),
            previous_episodes=[
                "Grandma Rosa is still regaining strength after pneumonia."
            ],
            entity_name="Grandma Rosa",
            entity_type="Person",
            existing_summary="Recovering slowly from pneumonia.",
            attributes={
                "relationship_type": "family",
                "closeness": 0.92,
                "overall_valence": 0.1,
            },
            summary_text="I sat beside Grandma Rosa at St. Jude's while Dr. Yates explained tomorrow's surgery and tried to keep both our breaths steady.",
            key_phrases=["St. Jude's", "surgery prep"],
        ),
        example(
            episode_content=(
                "Spent the morning at the 24th Street Community Garden building raised beds with three teen volunteers."
            ),
            previous_episodes=["Garden board asked me to log the next improvements."],
            entity_name="24th Street Community Garden",
            entity_type="Place",
            existing_summary="",
            attributes={"category": "community garden"},
            summary_text="I spent the morning at 24th Street Community Garden building raised beds with three tender teen volunteers and soaked up the buzz of community.",
            key_phrases=["raised beds", "teen volunteers", "community"],
        ),
        example(
            episode_content=(
                "Snuck out for a night swim at Aquatic Park with Marco; the cold shock actually calmed me down."
            ),
            previous_episodes=[
                "Last week night's swim ended early because of lightning."
            ],
            entity_name="Night swim",
            entity_type="Activity",
            existing_summary="",
            attributes={"activity_type": "night swim ritual"},
            summary_text="I snuck into Aquatic Park for a night swim with Marco, let the cold shock hit, and felt the panic buzzing under my skin finally melt.",
            key_phrases=["Aquatic Park", "night swim", "calming shock"],
        ),
        example(
            episode_content=(
                "Session with therapist Ines. She led long breathing ladders and assigned a new journal prompt."
            ),
            previous_episodes=["Ines keeps nudging me toward gentler self-talk."],
            entity_name="Ines",
            entity_type="Person",
            existing_summary="Working with me on grounding skills.",
            attributes={
                "relationship_type": "therapist",
                "closeness": 0.58,
                "overall_valence": 0.45,
            },
            summary_text="I stretched session time with therapist Ines so we could climb slow breathing ladders and she left me holding a gentle journal prompt.",
            key_phrases=["breathing ladders", "journal prompt"],
        ),
        example(
            episode_content=(
                "Been practicing patience while caring for Mom—slow paperwork, slow progress, just breathing through the delays."
            ),
            previous_episodes=[
                "Patience theme keeps surfacing every time Mom's appointments slip."
            ],
            entity_name="Caregiving patience practice",
            entity_type="Activity",
            existing_summary="",
            attributes={"activity_type": "caregiving ritual"},
            summary_text="I practiced caregiving patience by breathing with Mom through slow paperwork, tiny wins, and the urge to rush.",
            key_phrases=["caregiving", "patience", "Mom"],
        ),
        example(
            episode_content=(
                "Priya hosted a cozy potluck tonight and spread out her colored pens to map my training schedule."
            ),
            previous_episodes=[
                "She loves turning my training plans into rainbow timelines."
            ],
            entity_name="Priya",
            entity_type="Person",
            existing_summary="Keeps me motivated for races.",
            attributes={
                "relationship_type": "friend",
                "closeness": 0.8,
                "overall_valence": 0.7,
            },
            summary_text="I felt completely seen when Priya hosted a cozy potluck, covered the table in rainbow pens, and re-mapped my training plan with me.",
            key_phrases=["potluck", "training plan", "colored pens"],
        ),
        example(
            episode_content=(
                "Harbor Clinic smelled like mint tea today. Ben reorganized the pamphlets so folks could actually find resources."
            ),
            previous_episodes=["That waiting room usually feels chaotic."],
            entity_name="Harbor Clinic",
            entity_type="Place",
            existing_summary="",
            attributes={"category": "clinic"},
            summary_text="I noticed Harbor Clinic smelling like mint tea while Ben quietly reorganized pamphlets so anxious folks like me could grab resources faster.",
            key_phrases=["mint tea", "waiting room care"],
        ),
        example(
            episode_content="I biked slow loops near Crissy Field after therapy just to feel the fog on my face.",
            previous_episodes=[
                "Crissy Field night rides used to be my go-to regulation trick."
            ],
            entity_name="Self",
            entity_type="Person",
            existing_summary="",
            attributes={
                "relationship_type": "author",
                "closeness": 0.75,
                "overall_valence": 0.2,
            },
            summary_text="I biked lazy loops at Crissy Field after therapy so the wind could rinse off the leftover dread.",
            key_phrases=["Crissy Field", "therapy decompression"],
        ),
        example(
            episode_content="I staffed Mutual Aid Kitchen again and my hands smelled like cilantro for hours.",
            previous_episodes=[
                "That volunteer crew keeps me accountable when I want to hide."
            ],
            entity_name="Mutual Aid Kitchen",
            entity_type="Organization",
            existing_summary="",
            attributes={"category": "community kitchen"},
            summary_text="I felt grounded at Mutual Aid Kitchen, chopping cilantro with the crew that refuses to let me disappear.",
            key_phrases=["Mutual Aid Kitchen", "cilantro", "grounded"],
        ),
        example(
            episode_content="I curled up in the window seat at Studio Norte and journaled through a monsoon of feelings.",
            previous_episodes=[
                "Studio Norte is my creative co-op when I miss having coworkers."
            ],
            entity_name="Studio Norte",
            entity_type="Place",
            existing_summary="",
            attributes={"category": "creative studio"},
            summary_text="I tucked into Studio Norte's window seat and journaled through the monsoon in my chest while rain hammered the glass.",
            key_phrases=["Studio Norte", "window seat", "journaling"],
        ),
    ]

    valset = all_examples[-3:]
    trainset = all_examples[:-3]
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
