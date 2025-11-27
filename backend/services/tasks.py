"""Huey task definitions for inference processing."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time

import backend.dspy_cache  # noqa: F401
import dspy
from graphiti_core.errors import NodeNotFoundError

from backend.database.persistence import EpisodeDeletedError
from backend.settings import ORCHESTRATOR_INTERVAL_SECONDS
from .queue import huey

logger = logging.getLogger(__name__)


class TaskCancelled(Exception):
    """Raised when a task detects shutdown and should exit cleanly."""

    pass


def check_cancellation():
    """Raise TaskCancelled if shutdown has been requested.

    Call at safe checkpoints in long-running tasks to allow early exit
    without losing completed work.
    """
    from backend.database.lifecycle import is_shutdown_requested

    if is_shutdown_requested():
        raise TaskCancelled("Shutdown requested")


@huey.task(unique=True)
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
        clear_active_episode,
        get_episode_status,
        get_inference_enabled,
        set_active_episode,
        set_episode_status,
    )
    from backend.graph.extract_nodes import extract_nodes
    from backend.inference.manager import cleanup_if_no_work, get_model, is_model_loading_blocked

    try:
        check_cancellation()  # Entry checkpoint

        current_status = get_episode_status(episode_uuid, journal)
        if current_status != "pending_nodes":
            logger.debug(
                "Episode %s already processed (status: %s), skipping",
                episode_uuid,
                current_status,
            )
            return {"already_processed": True, "status": current_status}

        # Mark this episode as actively processing (for spinner display)
        set_active_episode(episode_uuid, journal)

        logger.info("extract_nodes_task started for episode %s", episode_uuid)

        if not get_inference_enabled():
            logger.info("Inference disabled, leaving episode %s in pending_nodes", episode_uuid)
            return {"inference_disabled": True}

        check_cancellation()  # Before expensive model load

        # Wait for editing to end before loading model
        wait_logged = False
        while is_model_loading_blocked():
            check_cancellation()  # Allow graceful shutdown
            if not wait_logged:
                logger.info("Waiting for editing to end before loading model for episode %s", episode_uuid)
                wait_logged = True
            time.sleep(1)

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

        # Edges extraction task is not implemented yet. We mark episodes as DONE
        # immediately after node extraction so the UI stops showing spinners.
        # When extract_edges_task lands, switch this back to pending_edges and
        # enqueue the edges task.
        if result.extracted_count > 0:
            set_episode_status(episode_uuid, "done", journal, uuid_map=result.uuid_map)
            logger.info(
                "Marked episode %s done after node extraction (edges task TBD)",
                episode_uuid,
            )
        else:
            # No entities extracted (only self or nothing) - no edges possible
            set_episode_status(episode_uuid, "done", journal)
            logger.info(
                "No entities extracted for episode %s, marked done",
                episode_uuid,
            )

        check_cancellation()  # After persistence - work saved, safe to exit early

        return {
            "episode_uuid": result.episode_uuid,
            "extracted_count": result.extracted_count,
            "new_entities": result.new_entities,
            "resolved_count": result.resolved_count,
        }
    except TaskCancelled:
        logger.info("Task cancelled due to shutdown for episode %s", episode_uuid)
        return {"cancelled": True}
    except EpisodeDeletedError as e:
        from backend.database.redis_ops import remove_episode_from_queue

        logger.info("Episode %s deleted during extraction, cleaning up", e.episode_uuid)
        remove_episode_from_queue(episode_uuid, journal)
        return {"episode_deleted": True, "uuid": episode_uuid}
    except NodeNotFoundError:
        from backend.database.redis_ops import remove_episode_from_queue

        logger.info("Episode %s was deleted, cleaning up", episode_uuid)
        remove_episode_from_queue(episode_uuid, journal)
        return {"episode_deleted": True, "uuid": episode_uuid}
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
            clear_active_episode()
        except Exception:
            logger.warning("Failed to clear active episode", exc_info=True)
        try:
            cleanup_if_no_work()
        except Exception:
            logger.warning("cleanup_if_no_work() failed after extract_nodes_task", exc_info=True)


@huey.task()
def orchestrate_inference_work(reschedule: bool = True):
    """Maintenance loop: enqueue pending episodes and manage model lifecycle."""
    from backend.database.lifecycle import is_shutdown_requested

    if is_shutdown_requested():
        logger.info("Shutdown requested, skipping orchestrate_inference_work")
        return

    try:
        from backend.database.redis_ops import enqueue_pending_episodes
        from backend.inference.manager import cleanup_if_no_work

        enqueue_pending_episodes()
        cleanup_if_no_work()  # Now handles unload during editing
    except Exception:
        logger.exception("orchestrate_inference_work failed")
    finally:
        if reschedule and not is_shutdown_requested():
            try:
                orchestrate_inference_work.schedule(delay=ORCHESTRATOR_INTERVAL_SECONDS)
            except Exception:
                logger.exception("Failed to reschedule orchestrate_inference_work")
