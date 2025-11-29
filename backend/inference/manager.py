"""Model manager for warm session support in Huey worker."""

from __future__ import annotations

import gc
import logging
import time
from typing import Literal
from .dspy_lm import DspyLM
from backend.settings import MODEL_LOAD_GRACE_SECONDS

logger = logging.getLogger(__name__)

# Global model registry (safe with single worker thread, no locking needed)
MODELS: dict[str, DspyLM | None] = {
    "llm": None,
}

_app_startup_time: float | None = None


def mark_app_started() -> None:
    """Mark app startup time for grace period."""
    global _app_startup_time
    _app_startup_time = time.monotonic()


def is_model_loading_blocked() -> bool:
    """Returns True if model loading should be blocked.

    Blocked when:
    - Within startup grace period
    - User is actively editing
    """
    from backend.database.redis_ops import redis_ops

    # Check startup grace
    if _app_startup_time is None:
        return True  # Block until app is properly started
    elapsed = time.monotonic() - _app_startup_time
    if elapsed < MODEL_LOAD_GRACE_SECONDS:
        logger.debug("Model loading blocked: startup grace (%.1fs elapsed)", elapsed)
        return True

    # Check editing active
    try:
        with redis_ops() as r:
            if r.exists("editing:active"):
                logger.debug("Model loading blocked: user is editing")
                return True
    except Exception:
        logger.warning("Failed to check editing presence, allowing model loading", exc_info=True)

    return False


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
    """Unload models when editing (for UI responsiveness) OR when no work remains."""
    from backend.database.redis_ops import (
        clear_model_state,
        get_episodes_by_status,
        get_inference_enabled,
        redis_ops,
        set_model_state,
    )

    if not get_inference_enabled():
        # Show "unloading" state in UI if models are loaded
        if any(model is not None for model in MODELS.values()):
            set_model_state("unloading")
        unload_all_models()
        clear_model_state()
        return

    # Unload when editing to keep UI snappy
    try:
        with redis_ops() as r:
            if r.exists("editing:active"):
                if any(model is not None for model in MODELS.values()):
                    logger.info("User editing, unloading models for UI responsiveness")
                unload_all_models()
                return
    except Exception:
        pass

    # No editing - check for pending work
    pending_nodes = get_episodes_by_status("pending_nodes")
    if len(pending_nodes) == 0:
        if any(model is not None for model in MODELS.values()):
            logger.info("No pending node work; unloading models")
        unload_all_models()
