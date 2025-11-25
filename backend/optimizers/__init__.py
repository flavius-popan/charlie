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
MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "unsloth/Qwen3-4B-Instruct-2507-GGUF")
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
GEPA_NUM_THREADS = 1  # llama.cpp cannot be safely forked


def get_task_lm() -> dspy.LM:
    """Get local task LM for optimization."""
    from backend.inference.dspy_lm import DspyLM
    return DspyLM(repo_id=MODEL_REPO_ID, generation_config=MODEL_CONFIG)


def get_reflection_lm() -> dspy.LM:
    """Get reflection LM for GEPA (requires OPENAI_API_KEY)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(f"OPENAI_API_KEY required for GEPA reflection ({REFLECTION_MODEL})")
    return dspy.LM(
        model=REFLECTION_MODEL,
        api_key=api_key,
        temperature=REFLECTION_TEMPERATURE,
        max_tokens=REFLECTION_MAX_TOKENS,
    )


def configure_dspy():
    """Configure DSPy with task LM and chat adapter."""
    lm = get_task_lm()
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter())
    logger.info("Configured DSPy with %s", MODEL_REPO_ID)


def evaluate_module(module, dataset: list, metric_fn) -> float:
    """Evaluate a DSPy module on a dataset."""
    if not dataset:
        return 0.0
    scores = []
    for ex in dataset:
        try:
            pred = module(episode_content=ex.episode_content, entity_types=ex.entity_types)
            result = metric_fn(ex, pred)
            # NOTE: dspy.Prediction inherits from Example which implements __getitem__,
            # so both result.score and result['score'] work. Using dict-style here.
            if hasattr(result, 'score'):
                score = result['score']
            else:
                score = result  # Assume float
            scores.append(float(score))
        except Exception as e:
            logger.warning("Error evaluating example: %s", e)
            scores.append(0.0)
    return sum(scores) / len(scores)
