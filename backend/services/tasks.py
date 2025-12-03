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
        clear_model_state,
        get_active_episode_uuid,
        get_episode_status,
        get_inference_enabled,
        increment_and_check_retry_count,
        remove_pending_episode,
        reset_retry_count,
        set_active_episode,
        set_episode_status,
        set_model_state,
    )
    from backend.settings import MAX_EXTRACTION_RETRIES
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

        # Check inference enabled BEFORE marking active to avoid state leakage
        if not get_inference_enabled():
            logger.debug("Inference disabled, skipping episode %s (queue preserved)", episode_uuid)
            return {"inference_disabled": True}

        # Mark this episode as actively processing (for spinner display)
        # Only after all early-return checks pass
        set_active_episode(episode_uuid, journal)

        # Verify state hasn't changed during async operation
        if not get_inference_enabled():
            clear_active_episode()
            logger.debug("Inference disabled after set_active_episode, aborting %s", episode_uuid)
            return {"inference_disabled": True}

        logger.info("extract_nodes_task started for episode %s", episode_uuid)

        check_cancellation()  # Before expensive model load

        # Wait for grace period/editing to end before loading model
        # Don't set model_state yet - processing pane should stay hidden during wait
        wait_logged = False
        while is_model_loading_blocked():
            check_cancellation()  # Allow graceful shutdown
            if not wait_logged:
                logger.info("Waiting for grace period/editing to end for episode %s", episode_uuid)
                wait_logged = True
            time.sleep(1)

        # Verify state hasn't changed during blocking operation
        if not get_inference_enabled():
            clear_active_episode()
            logger.debug("Inference disabled during wait, aborting %s", episode_uuid)
            return {"inference_disabled": True}

        # Verify still active episode after blocking operation
        current_active = get_active_episode_uuid()
        if current_active != episode_uuid:
            logger.debug("Episode %s no longer active after wait (current: %s), aborting",
                         episode_uuid, current_active)
            return {"no_longer_active": True}

        # NOW set loading state - we're actually about to load
        set_model_state("loading")

        lm = get_model("llm")

        # Transition to inferring state after model is loaded
        set_model_state("inferring")

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

        # Remove from pending queue (ZSET) now that extraction is complete
        remove_pending_episode(episode_uuid, journal)

        # Edges extraction task is not implemented yet. We mark episodes as DONE
        # immediately after node extraction so the UI stops showing spinners.
        # When extract_edges_task lands, switch this back to pending_edges and
        # enqueue the edges task.
        reset_retry_count(episode_uuid, journal)
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
        retry_count, should_mark_dead = increment_and_check_retry_count(
            episode_uuid, journal, MAX_EXTRACTION_RETRIES
        )
        if should_mark_dead:
            try:
                set_episode_status(episode_uuid, "dead", journal)
                remove_pending_episode(episode_uuid, journal)
            except Exception:
                logger.warning(
                    "Failed to mark episode %s as dead", episode_uuid, exc_info=True
                )
            logger.error(
                "Episode %s marked dead after %d failed attempts",
                episode_uuid, retry_count
            )
        else:
            logger.warning(
                "Episode %s failed (attempt %d/%d), will retry",
                episode_uuid, retry_count, MAX_EXTRACTION_RETRIES
            )
        raise
    finally:
        # Clear in reverse dependency order to minimize inconsistency window
        try:
            clear_model_state()
        except Exception:
            logger.warning("Failed to clear model state", exc_info=True)
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
