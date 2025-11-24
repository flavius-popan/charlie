"""Model manager for warm session support in Huey worker."""

from __future__ import annotations

import gc
import logging
import time
from typing import Literal
from .dspy_lm import DspyLM
from backend.settings import EDIT_IDLE_GRACE_SECONDS

logger = logging.getLogger(__name__)

# Global model registry (safe with single worker thread, no locking needed)
MODELS: dict[str, DspyLM | None] = {
    "llm": None,
}

_last_edit_seen: float | None = None


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
    from backend.database.redis_ops import get_episodes_by_status, get_inference_enabled, redis_ops

    # If inference is disabled, pending work is blocked; unload immediately (once)
    models_loaded = any(model is not None for model in MODELS.values())
    if not get_inference_enabled():
        # Always call unload to ensure any loaded model is freed; log only on actual unload.
        unload_all_models()
        return

    # Check if user is actively editing (keeps models warm)
    try:
        with redis_ops() as r:
            user_is_editing = r.exists("editing:active")
    except Exception as e:
        logger.debug("Failed to check editing presence: %s", e)
        user_is_editing = False

    if user_is_editing:
        global _last_edit_seen
        _last_edit_seen = time.monotonic()
        logger.debug("User is actively editing, keeping models loaded")
        return

    pending_nodes = get_episodes_by_status("pending_nodes")
    pending_edges = get_episodes_by_status("pending_edges")

    # Apply grace window to avoid thrashing when user just exited editing
    if _last_edit_seen is not None:
        idle_seconds = time.monotonic() - _last_edit_seen
        if idle_seconds < EDIT_IDLE_GRACE_SECONDS:
            logger.debug(
                "Within edit idle grace (%.1fs < %.1fs); keeping models loaded",
                idle_seconds,
                EDIT_IDLE_GRACE_SECONDS,
            )
            return
        _last_edit_seen = None

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
