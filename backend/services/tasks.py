"""Huey task definitions for async inference processing."""

from __future__ import annotations

import logging

import backend.dspy_cache  # noqa: F401
import dspy

from .queue import huey

logger = logging.getLogger(__name__)


@huey.task()
async def extract_nodes_task(episode_uuid: str, journal: str):
    """Background entity extraction task.

    Args:
        episode_uuid: Episode UUID to process
        journal: Journal name

    Returns:
        Dictionary with extraction results or status info

    Note:
        This task is idempotent - checks Redis state before processing.
        Safe to enqueue multiple times for the same episode.
    """
    from backend.database.redis_ops import (
        get_episode_status,
        get_inference_enabled,
        remove_episode_from_queue,
        set_episode_status,
    )
    from backend.graph.extract_nodes import extract_nodes
    from backend.inference.manager import cleanup_if_no_work, get_model

    logger.info("extract_nodes_task started for episode %s", episode_uuid)

    current_status = get_episode_status(episode_uuid)
    if current_status != "pending_nodes":
        logger.info(
            "Episode %s already processed (status: %s), skipping",
            episode_uuid,
            current_status,
        )
        return {"already_processed": True, "status": current_status}

    if not get_inference_enabled():
        logger.info("Inference disabled, leaving episode %s in pending_nodes", episode_uuid)
        return {"inference_disabled": True}

    lm = get_model("llm")
    with dspy.context(lm=lm):
        result = await extract_nodes(episode_uuid, journal)

    logger.info(
        "Extracted %d entities for episode %s (new: %d, resolved: %d)",
        result.extracted_count,
        episode_uuid,
        result.new_entities,
        result.resolved_count,
    )

    if result.uuid_map:
        set_episode_status(episode_uuid, "pending_edges", uuid_map=result.uuid_map)
        logger.info(
            "Episode %s moved to pending_edges (extract_edges task not yet implemented)",
            episode_uuid,
        )
    else:
        remove_episode_from_queue(episode_uuid)
        logger.info("No entities extracted for episode %s, removed from queue", episode_uuid)

    cleanup_if_no_work()

    return {
        "episode_uuid": result.episode_uuid,
        "extracted_count": result.extracted_count,
        "new_entities": result.new_entities,
        "resolved_count": result.resolved_count,
    }
