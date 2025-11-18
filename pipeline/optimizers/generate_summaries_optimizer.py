"""Optimizer for pipeline/generate_summaries.py using DSPy's GEPA.

Uses LLM-as-judge (gpt-5-nano) to provide rich textual feedback for optimizing
summary generation prompts. Focuses on first-person narration and conciseness.

Usage:
    python -m pipeline.optimizers.generate_summaries_optimizer
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import logging
from collections import defaultdict

# Allow running as `python pipeline/optimizers/...` by ensuring `settings` loads
# (and configures DSPy caches) before the first DSPy import.
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
    GEPA_OUTPUT_DIR,
)

from pipeline import _dspy_setup  # noqa: F401
import dspy  # noqa: E402
from dspy.teleprompt import GEPA  # noqa: E402
from dspy import Prediction  # noqa: E402

from mlx_runtime import MLXDspyLM
from pipeline.generate_summaries import (
    EntitySummary,
    SummaryGenerator,
    build_node_payload,
    build_summary_context,
)


PROMPT_OUTPUT = Path(__file__).parent.parent / "prompts" / "generate_summaries.json"
logger = logging.getLogger(__name__)


def configure_dspy():
    """Match runtime LM + adapter configuration."""

    lm = MLXDspyLM(model_path=DEFAULT_MODEL_PATH, generation_config=MODEL_CONFIG)
    adapter = dspy.ChatAdapter()
    dspy.configure(lm=lm, adapter=adapter)
    logger.info("Configured DSPy with MLXDspyLM (model: %s)", DEFAULT_MODEL_PATH)


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

    def build_example(
        *,
        episode_content: str,
        previous_episodes: list[str],
        entity_name: str,
        entity_type: str,
        existing_summary: str,
        attributes: dict,
        summary_text: str,
        key_phrases: list[str],
    ) -> tuple[dspy.Example, str]:
        context_json = _context_json(
            episode_content=episode_content,
            previous_episodes=previous_episodes,
            entity_name=entity_name,
            entity_type=entity_type,
            existing_summary=existing_summary,
            attributes=attributes,
        )
        ex = dspy.Example(
            summary_context=context_json,
            summary=summary_text,
            key_phrases=key_phrases,
        ).with_inputs("summary_context")
        return ex, entity_type

    typed_examples = [
        build_example(
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
            },
            summary_text="I biked the Embarcadero at dawn with Kai and let him spill every anxious portfolio thought until he finally exhaled.",
            key_phrases=["sunrise ride", "portfolio jitters"],
        ),
        build_example(
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
            },
            summary_text="I sat beside Grandma Rosa at St. Jude's while Dr. Yates explained tomorrow's surgery and tried to keep both our breaths steady.",
            key_phrases=["St. Jude's", "surgery prep"],
        ),
        build_example(
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
        build_example(
            episode_content=(
                "Snuck out for a night swim at Aquatic Park with Marco; the cold shock actually calmed me down."
            ),
            previous_episodes=[
                "Last week night's swim ended early because of lightning."
            ],
            entity_name="Night swim",
            entity_type="Activity",
            existing_summary="",
            attributes={"purpose": "night swim ritual"},
            summary_text="I snuck into Aquatic Park for a night swim with Marco, let the cold shock hit, and felt the panic buzzing under my skin finally melt.",
            key_phrases=["Aquatic Park", "night swim", "calming shock"],
        ),
        build_example(
            episode_content=(
                "Session with therapist Ines. She led long breathing ladders and assigned a new journal prompt."
            ),
            previous_episodes=["Ines keeps nudging me toward gentler self-talk."],
            entity_name="Ines",
            entity_type="Person",
            existing_summary="Working with me on grounding skills.",
            attributes={
                "relationship_type": "therapist",
            },
            summary_text="I stretched session time with therapist Ines so we could climb slow breathing ladders and she left me holding a gentle journal prompt.",
            key_phrases=["breathing ladders", "journal prompt"],
        ),
        build_example(
            episode_content=(
                "Been practicing patience while caring for Mom—slow paperwork, slow progress, just breathing through the delays."
            ),
            previous_episodes=[
                "Patience theme keeps surfacing every time Mom's appointments slip."
            ],
            entity_name="Caregiving patience practice",
            entity_type="Activity",
            existing_summary="",
            attributes={"purpose": "caregiving ritual"},
            summary_text="I practiced caregiving patience by breathing with Mom through slow paperwork, tiny wins, and the urge to rush.",
            key_phrases=["caregiving", "patience", "Mom"],
        ),
        build_example(
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
            },
            summary_text="I felt completely seen when Priya hosted a cozy potluck, covered the table in rainbow pens, and re-mapped my training plan with me.",
            key_phrases=["potluck", "training plan", "colored pens"],
        ),
        build_example(
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
        build_example(
            episode_content="I biked slow loops near Crissy Field after therapy just to feel the fog on my face.",
            previous_episodes=[
                "Crissy Field night rides used to be my go-to regulation trick."
            ],
            entity_name="Self",
            entity_type="Person",
            existing_summary="",
            attributes={
                "relationship_type": "author",
            },
            summary_text="I biked lazy loops at Crissy Field after therapy so the wind could rinse off the leftover dread.",
            key_phrases=["Crissy Field", "therapy decompression"],
        ),
        build_example(
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
        build_example(
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

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, dict):
        return value.get("summary", "")

    return getattr(value, "summary", "")


def judge_summary_score(
    *,
    generated_summary: str,
    expected_key_phrases: list[str],
    judge_lm: dspy.LM,
    log_label: str | None = None,
) -> tuple[float, str]:
    """Request numeric score + textual guidance from the judge model."""

    feedback_prompt = f'''You are grading a journal entity summary.

Summary:
"""
{generated_summary}
"""

Key phrases (context only, no requirement to include them verbatim): {expected_key_phrases}

Respond with ONLY valid JSON matching this schema:
{{
  "overall_score": <float between 0 and 1>,
  "feedback": "<detailed coaching that highlights strengths, weaknesses, and missing nuance>"
}}

Score should reward intimate first-person narration, sensory/reflective detail, emotional nuance, and concise storytelling. Penalize generic or third-person summaries.
'''

    if log_label:
        logger.info("=" * 80)
        logger.info(f"JUDGE EVALUATION REQUEST ({log_label})")
        logger.info(f"Generated: {generated_summary}")

    response = judge_lm(feedback_prompt)[0]

    score = 0.0
    feedback = response.strip()
    try:
        parsed = json.loads(response)
        score = float(parsed.get("overall_score", 0.0))
        feedback = str(parsed.get("feedback", "")).strip() or feedback
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Failed to parse judge response (%s). Raw response: %s", exc, response)
        feedback = f"Invalid judge response; defaulted to 0. Raw: {response}"

    score = max(0.0, min(1.0, score))

    if log_label:
        logger.info(f"Score: {score:.2f}")
        logger.info("JUDGE FEEDBACK:")
        logger.info(feedback)
        logger.info("=" * 80)

    return score, feedback


def evaluate(
    module: SummaryGenerator,
    dataset: list[dspy.Example],
    judge_lm: dspy.LM,
) -> float:
    """Average judge-derived score across dataset."""

    scores: list[float] = []
    for example in dataset:
        prediction = module(summary_context=example.summary_context)
        summary_text = _summary_text(prediction)
        key_phrases = getattr(example, "key_phrases", [])
        score, _ = judge_summary_score(
            generated_summary=summary_text,
            expected_key_phrases=key_phrases,
            judge_lm=judge_lm,
        )
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


def main():
    """Full optimization workflow for summary generation using GEPA."""

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
    def gepa_summary_metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> Prediction:
        """GEPA metric driven entirely by judge scoring."""

        summary_text = _summary_text(pred)
        key_phrases = getattr(gold, "key_phrases", [])
        score, feedback = judge_summary_score(
            generated_summary=summary_text,
            expected_key_phrases=key_phrases,
            judge_lm=judge_lm,
            log_label=pred_name,
        )

        if pred_name:
            logger.info("-" * 80)
            logger.info(f"METRIC SCORE: {score:.2f}")
            logger.info("-" * 80)

        return Prediction(score=score, feedback=feedback)

    # Evaluate baseline
    baseline = SummaryGenerator()
    baseline_score = evaluate(baseline, valset, judge_lm)
    logger.info("Baseline score (valset): %.3f", baseline_score)

    # Create log directory for GEPA artifacts
    log_dir = GEPA_OUTPUT_DIR / "generate_summaries"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info("GEPA logs will be saved to: %s", log_dir)

    # Instantiate and run GEPA
    logger.info("Starting GEPA optimization with max_full_evals=%d", GEPA_MAX_FULL_EVALS)
    gepa = GEPA(
        metric=gepa_summary_metric,
        max_full_evals=GEPA_MAX_FULL_EVALS,
        reflection_lm=judge_lm,
        reflection_minibatch_size=GEPA_REFLECTION_MINIBATCH_SIZE,
        track_stats=True,
        log_dir=str(log_dir)
    )

    optimized = gepa.compile(
        student=baseline,
        trainset=trainset,
        valset=valset
    )

    # Evaluate optimized
    optimized_score = evaluate(optimized, valset, judge_lm)
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
