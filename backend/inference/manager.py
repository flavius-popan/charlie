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
    for model_type in MODELS:
        if MODELS[model_type] is not None:
            logger.info("Unloading %s model", model_type)
            MODELS[model_type] = None

    gc.collect()
    logger.info("All models unloaded and memory freed")


def cleanup_if_no_work() -> None:
    """Unload models if no pending episodes remain (event-driven cleanup)."""
    from backend.database.redis_ops import get_episodes_by_status

    pending_nodes = get_episodes_by_status("pending_nodes")
    # Intentionally do NOT block on pending_edges yet.
    # Until edge extraction exists, episodes may sit in pending_edges as a staging
    # state; model unloads should still proceed. When extract_edges_task lands,
    # add a pending_edges check here to keep models warm for relationship runs.
    # pending_edges = get_episodes_by_status("pending_edges")

    if len(pending_nodes) == 0:
        logger.info("No pending work in queue, unloading models")
        unload_all_models()
    else:
        logger.debug(
            "Work remains in queue (%d pending_nodes), keeping models loaded",
            len(pending_nodes),
        )
