"""Huey task definitions for inference processing."""

from __future__ import annotations

import asyncio
import inspect
import logging

import backend.dspy_cache  # noqa: F401
import dspy
from .queue import huey

logger = logging.getLogger(__name__)


@huey.task()
def extract_nodes_task(episode_uuid: str, journal: str):
    """Background entity extraction task.

    Args:
        episode_uuid: Episode UUID to process
        journal: Journal name

    Returns:
        Dictionary with extraction results or status info

    Note:
        This task is idempotent - checks Redis state before processing.
        Safe to enqueue multiple times for the same episode.

        Priority can be passed via kwargs when enqueueing:
        - extract_nodes_task(uuid, journal, priority=1) for user-triggered
        - extract_nodes_task(uuid, journal) or priority=0 for background tasks
        Higher priority numbers are processed first.
    """
    from backend.database.redis_ops import (
        get_episode_status,
        get_inference_enabled,
        set_episode_status,
    )
    from backend.graph.extract_nodes import extract_nodes
    from backend.inference.manager import cleanup_if_no_work, get_model

    logger.info("extract_nodes_task started for episode %s", episode_uuid)

    try:
        current_status = get_episode_status(episode_uuid, journal)
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
            extraction = extract_nodes(episode_uuid, journal)
            if inspect.isawaitable(extraction):
                result = asyncio.run(extraction)
            else:
                result = extraction

        logger.info(
            "Extracted %d entities for episode %s (new: %d, resolved: %d)",
            result.extracted_count,
            episode_uuid,
            result.new_entities,
            result.resolved_count,
        )

        # Only transition to pending_edges if entities were extracted
        # Self entity "I" doesn't count - edges need at least one other node
        if result.extracted_count > 0:
            set_episode_status(episode_uuid, "pending_edges", journal, uuid_map=result.uuid_map)
            logger.info(
                "Transitioned episode %s to pending_edges with %d uuid mappings",
                episode_uuid,
                len(result.uuid_map),
            )

            # TODO: Uncomment when extract_edges_task is implemented
            # from backend.services.tasks import extract_edges_task
            # extract_edges_task(episode_uuid, journal)
        else:
            # No entities extracted (only self or nothing) - no edges possible
            set_episode_status(episode_uuid, "done", journal)
            logger.info(
                "No entities extracted for episode %s, marked done",
                episode_uuid,
            )

        return {
            "episode_uuid": result.episode_uuid,
            "extracted_count": result.extracted_count,
            "new_entities": result.new_entities,
            "resolved_count": result.resolved_count,
        }
    except Exception:
        try:
            set_episode_status(episode_uuid, "dead", journal)
        except Exception:
            logger.warning(
                "Failed to mark episode %s as dead after exception", episode_uuid, exc_info=True
            )
        logger.exception("extract_nodes_task failed for episode %s", episode_uuid)
        raise
    finally:
        try:
            cleanup_if_no_work()
        except Exception:
            logger.warning("cleanup_if_no_work() failed after extract_nodes_task", exc_info=True)


@huey.task()
def orchestrate_inference_work(reschedule: bool = True):
    """Maintenance loop: enqueue pending nodes and unload when idle/disabled."""
    try:
        from backend.database.redis_ops import enqueue_pending_episodes
        from backend.inference.manager import cleanup_if_no_work

        enqueue_pending_episodes()

        cleanup_if_no_work()
    except Exception:
        logger.exception("orchestrate_inference_work failed")
    finally:
        if reschedule:
            try:
                orchestrate_inference_work.schedule(delay=3)
            except Exception:
                logger.exception("Failed to reschedule orchestrate_inference_work")
