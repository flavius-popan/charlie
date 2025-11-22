"""Model manager for warm session support in Huey worker."""

from __future__ import annotations

import gc
import logging
from typing import Literal
from .dspy_lm import DspyLM

logger = logging.getLogger(__name__)

# Global model registry (safe with single worker thread, no locking needed)
MODELS: dict[str, DspyLM | None] = {
    "llm": None,
}


def get_model(model_type: Literal["llm"] = "llm") -> DspyLM:
    """Load model if not cached, return instance (warm sessions)."""
    if model_type not in MODELS:
        raise ValueError(f"Invalid model_type: {model_type}. Supported types: {list(MODELS.keys())}")

    if MODELS[model_type] is not None:
        logger.debug("Returning cached %s model (warm session)", model_type)
        return MODELS[model_type]

    logger.info("Loading %s model (cold start)", model_type)
    model = DspyLM()
    MODELS[model_type] = model
    logger.info("Loaded %s model and cached for warm sessions", model_type)
    return model


def unload_all_models() -> None:
    """Unload all models to free memory."""
    any_unloaded = False
    for model_type in MODELS:
        if MODELS[model_type] is not None:
            logger.info("Unloading %s model", model_type)
            MODELS[model_type] = None
            any_unloaded = True

    if any_unloaded:
        gc.collect()
        logger.info("All models unloaded and memory freed")


def cleanup_if_no_work() -> None:
    """Unload models when no active work remains (event-driven cleanup)."""
    from backend.database.redis_ops import get_episodes_by_status, get_inference_enabled

    # If inference is disabled, pending work is blocked; unload immediately (once)
    models_loaded = any(model is not None for model in MODELS.values())
    if not get_inference_enabled():
        # Always call unload to ensure any loaded model is freed; log only on actual unload.
        unload_all_models()
        return

    pending_nodes = get_episodes_by_status("pending_nodes")
    pending_edges = get_episodes_by_status("pending_edges")

    if len(pending_nodes) == 0:
        if models_loaded:
            logger.info(
                "No pending node work; unloading models (pending_edges=%d)", len(pending_edges)
            )
        else:
            logger.debug(
                "No pending node work; models already unloaded (pending_edges=%d)",
                len(pending_edges),
            )
        unload_all_models()
    else:
        logger.debug(
            "Active work remains (%d pending_nodes, %d pending_edges), keeping models loaded",
            len(pending_nodes),
            len(pending_edges),
        )
