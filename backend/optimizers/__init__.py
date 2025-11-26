"""Shared optimizer configuration and utilities.

All optimizer scripts import from here to ensure consistent settings.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

import dspy

logger = logging.getLogger(__name__)

# =============================================================================
# Directory Layout
# =============================================================================
OPTIMIZERS_DIR = Path(__file__).parent
PROMPTS_DIR = OPTIMIZERS_DIR.parent / "prompts"
DATA_DIR = OPTIMIZERS_DIR / "data"

PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# DSPy cache
DSPY_CACHE_DIR = PROMPTS_DIR / ".dspy_cache"
DSPY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
for env_var in ("DSPY_CACHEDIR", "DSPY_CACHE_DIR", "DSPY_CACHE"):
    os.environ.setdefault(env_var, str(DSPY_CACHE_DIR))

# =============================================================================
# Model Configuration
# =============================================================================
MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "unsloth/Qwen3-4B-Instruct-2507-FP8")
MODEL_CONFIG = {
    "temp": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "max_tokens": 2048,
}

# =============================================================================
# GEPA Configuration
# =============================================================================
REFLECTION_MODEL = os.getenv("REFLECTION_MODEL", "openai/gpt-4o-mini")
REFLECTION_TEMPERATURE = float(os.getenv("REFLECTION_TEMPERATURE", "0.7"))
REFLECTION_MAX_TOKENS = int(os.getenv("REFLECTION_MAX_TOKENS", "2048"))

# auto mode: "light" (quick), "medium" (balanced), "heavy" (production)
GEPA_AUTO_MODE = os.getenv("GEPA_AUTO_MODE", "light")

# Thread counts: local llama.cpp can't fork safely, remote can burst
GEPA_NUM_THREADS_LOCAL = 1
GEPA_NUM_THREADS_REMOTE_MAX = 20


def get_num_threads(num_examples: int, *, remote: bool = False) -> int:
    """Get optimal thread count based on example count and execution mode."""
    if not remote:
        return GEPA_NUM_THREADS_LOCAL
    return min(num_examples, GEPA_NUM_THREADS_REMOTE_MAX)


# GEPA recommends small valset (just enough for Pareto tracking), large trainset
GEPA_VALSET_MIN = 1
GEPA_VALSET_MAX = 3


def split_examples[T](examples: list[T]) -> tuple[list[T], list[T]]:
    """Split examples into train/val optimized for GEPA.

    GEPA needs valset only for Pareto tracking, so keep it small (1-3 examples).
    All remaining examples go to trainset for learning.
    """
    if len(examples) <= GEPA_VALSET_MIN:
        raise ValueError(f"Need at least {GEPA_VALSET_MIN + 1} examples, got {len(examples)}")

    val_size = min(max(GEPA_VALSET_MIN, len(examples) // 10), GEPA_VALSET_MAX)
    split_idx = len(examples) - val_size
    return examples[:split_idx], examples[split_idx:]

# =============================================================================
# Remote (HuggingFace) Configuration
# =============================================================================
HF_ENDPOINT_URL = os.getenv("HF_ENDPOINT_URL")


def get_task_lm(*, remote: bool = False) -> dspy.LM:
    """Get task LM for optimization.

    Args:
        remote: If True, use HuggingFace endpoint. If False, use local llama.cpp.
    """
    if remote:
        # HF Inference Endpoints with vLLM expose OpenAI-compatible API
        api_key = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
        if not api_key:
            raise ValueError(
                "HUGGINGFACE_API_KEY (or HF_TOKEN) required for remote optimization"
            )
        if not HF_ENDPOINT_URL:
            raise ValueError("HF_ENDPOINT_URL required for remote optimization")
        return dspy.LM(
            model=f"openai/{MODEL_REPO_ID}",
            api_key=api_key,
            api_base=HF_ENDPOINT_URL + "/v1",
            temperature=MODEL_CONFIG["temp"],
            max_tokens=MODEL_CONFIG["max_tokens"],
        )
    else:
        from backend.inference.dspy_lm import DspyLM

        return DspyLM(repo_id=MODEL_REPO_ID, generation_config=MODEL_CONFIG)


def get_reflection_lm() -> dspy.LM:
    """Get reflection LM for GEPA (requires OPENAI_API_KEY)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            f"OPENAI_API_KEY required for GEPA reflection ({REFLECTION_MODEL})"
        )
    return dspy.LM(
        model=REFLECTION_MODEL,
        api_key=api_key,
        temperature=REFLECTION_TEMPERATURE,
        max_tokens=REFLECTION_MAX_TOKENS,
    )


def configure_dspy(*, remote: bool = False):
    """Configure DSPy with task LM and chat adapter."""
    lm = get_task_lm(remote=remote)
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter())
    if remote:
        logger.info("Configured DSPy with remote HF endpoint")
    else:
        logger.info("Configured DSPy with local model: %s", MODEL_REPO_ID)


def evaluate_module(
    module, dataset: list, metric_fn, *, num_threads: int | None = None
) -> float:
    """Evaluate a DSPy module on a dataset using parallel threads."""
    if not dataset:
        return 0.0

    if num_threads is None:
        num_threads = GEPA_NUM_THREADS_LOCAL

    evaluator = dspy.Evaluate(
        devset=dataset,
        metric=metric_fn,
        num_threads=num_threads,
        display_progress=True,
        display_table=0,
    )

    return evaluator(module)
