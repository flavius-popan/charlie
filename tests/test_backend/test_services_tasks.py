"""Tests for Huey background tasks."""

from __future__ import annotations

import asyncio
from unittest.mock import Mock, patch

import pytest

from backend import ExtractNodesResult
from backend.settings import DEFAULT_JOURNAL


def test_extract_nodes_task_already_processed(episode_uuid):
    """Task skips if episode already processed."""
    from backend.services.tasks import extract_nodes_task

    with patch("backend.database.redis_ops.get_episode_status") as mock_status:
        mock_status.return_value = "pending_edges"

        result = extract_nodes_task.call_local(episode_uuid, DEFAULT_JOURNAL)

        assert result["already_processed"] is True
        assert result["status"] == "pending_edges"


def test_extract_nodes_task_inference_disabled(episode_uuid):
    """Task exits early if inference disabled."""
    from backend.services.tasks import extract_nodes_task

    with patch("backend.database.redis_ops.get_episode_status") as mock_status:
        with patch("backend.database.redis_ops.get_inference_enabled", create=True) as mock_enabled:
            with patch("backend.inference.manager.cleanup_if_no_work") as mock_cleanup:
                mock_status.return_value = "pending_nodes"
                mock_enabled.return_value = False

                result = extract_nodes_task.call_local(episode_uuid, DEFAULT_JOURNAL)

                assert result["inference_disabled"] is True
                mock_cleanup.assert_called_once()


def test_extract_nodes_task_with_entities_extracted(episode_uuid):
    """Task removes episode from queue when entities found."""
    from backend.services.tasks import extract_nodes_task

    mock_result = ExtractNodesResult(
        episode_uuid=episode_uuid,
        extracted_count=3,
        resolved_count=2,
        new_entities=1,
        exact_matches=1,
        fuzzy_matches=1,
        entity_uuids=["uuid1", "uuid2"],
        uuid_map={"prov1": "canon1", "prov2": "canon2"},
    )

    with (
        patch("backend.database.redis_ops.get_episode_status") as mock_status,
        patch("backend.database.redis_ops.get_inference_enabled", create=True) as mock_enabled,
        patch("backend.database.redis_ops.set_episode_status") as mock_set_status,
        patch("backend.database.redis_ops.set_active_episode"),
        patch("backend.database.redis_ops.get_active_episode_uuid") as mock_get_active,
        patch("backend.database.redis_ops.clear_active_episode"),
        patch("backend.database.redis_ops.set_model_state"),
        patch("backend.database.redis_ops.clear_model_state"),
        patch("backend.database.redis_ops.remove_pending_episode"),
        patch("backend.inference.manager.get_model") as mock_get_model,
        patch("backend.inference.manager.is_model_loading_blocked") as mock_blocked,
        patch("backend.graph.extract_nodes.extract_nodes") as mock_extract,
        patch("backend.inference.manager.cleanup_if_no_work") as mock_cleanup,
    ):
        mock_status.return_value = "pending_nodes"
        mock_enabled.return_value = True
        mock_blocked.return_value = False
        mock_get_active.return_value = episode_uuid
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_extract.return_value = mock_result

        result = extract_nodes_task.call_local(episode_uuid, DEFAULT_JOURNAL)

        assert result["episode_uuid"] == episode_uuid
        assert result["extracted_count"] == 3
        assert result["new_entities"] == 1
        assert result["resolved_count"] == 2

        mock_get_model.assert_called_once_with("llm")
        mock_extract.assert_called_once_with(episode_uuid, DEFAULT_JOURNAL)
        # With entities extracted, should transition to done (edges task TBD)
        mock_set_status.assert_called_once_with(
            episode_uuid, "done", DEFAULT_JOURNAL, uuid_map=mock_result.uuid_map
        )
        mock_cleanup.assert_called_once()


def test_extract_nodes_task_no_entities_marks_done(episode_uuid):
    """Task marks episode done when no entities extracted."""
    from backend.services.tasks import extract_nodes_task

    mock_result = ExtractNodesResult(
        episode_uuid=episode_uuid,
        extracted_count=0,
        resolved_count=0,
        new_entities=0,
        exact_matches=0,
        fuzzy_matches=0,
        entity_uuids=[],
        uuid_map={},
    )

    with (
        patch("backend.database.redis_ops.get_episode_status") as mock_status,
        patch("backend.database.redis_ops.get_inference_enabled", create=True) as mock_enabled,
        patch("backend.database.redis_ops.set_episode_status") as mock_set_status,
        patch("backend.database.redis_ops.set_active_episode"),
        patch("backend.database.redis_ops.get_active_episode_uuid") as mock_get_active,
        patch("backend.database.redis_ops.clear_active_episode"),
        patch("backend.database.redis_ops.set_model_state"),
        patch("backend.database.redis_ops.clear_model_state"),
        patch("backend.database.redis_ops.remove_pending_episode"),
        patch("backend.inference.manager.get_model") as mock_get_model,
        patch("backend.inference.manager.is_model_loading_blocked") as mock_blocked,
        patch("backend.graph.extract_nodes.extract_nodes") as mock_extract,
        patch("backend.inference.manager.cleanup_if_no_work") as mock_cleanup,
    ):
        mock_status.return_value = "pending_nodes"
        mock_enabled.return_value = True
        mock_blocked.return_value = False
        mock_get_active.return_value = episode_uuid
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_extract.return_value = mock_result

        result = extract_nodes_task.call_local(episode_uuid, DEFAULT_JOURNAL)

        assert result["extracted_count"] == 0
        mock_set_status.assert_called_once_with(episode_uuid, "done", DEFAULT_JOURNAL)
        mock_cleanup.assert_called_once()


def test_extract_nodes_task_uses_dspy_context(episode_uuid):
    """Task sets dspy.context with LLM before calling extract_nodes."""
    from backend.services.tasks import extract_nodes_task
    import dspy

    mock_result = ExtractNodesResult(
        episode_uuid=episode_uuid,
        extracted_count=1,
        resolved_count=1,
        new_entities=0,
        exact_matches=1,
        fuzzy_matches=0,
        entity_uuids=["uuid1"],
        uuid_map={"p1": "c1"},
    )

    with (
        patch("backend.database.redis_ops.get_episode_status") as mock_status,
        patch("backend.database.redis_ops.get_inference_enabled", create=True) as mock_enabled,
        patch("backend.database.redis_ops.set_episode_status"),
        patch("backend.database.redis_ops.set_active_episode"),
        patch("backend.database.redis_ops.get_active_episode_uuid") as mock_get_active,
        patch("backend.database.redis_ops.clear_active_episode"),
        patch("backend.database.redis_ops.set_model_state"),
        patch("backend.database.redis_ops.clear_model_state"),
        patch("backend.database.redis_ops.remove_pending_episode"),
        patch("backend.inference.manager.get_model") as mock_get_model,
        patch("backend.inference.manager.is_model_loading_blocked") as mock_blocked,
        patch("backend.graph.extract_nodes.extract_nodes") as mock_extract,
        patch("backend.inference.manager.cleanup_if_no_work"),
    ):
        mock_status.return_value = "pending_nodes"
        mock_enabled.return_value = True
        mock_blocked.return_value = False
        mock_get_active.return_value = episode_uuid
        mock_model = Mock()
        mock_get_model.return_value = mock_model

        captured_lm = None

        def capture_lm(*args, **kwargs):
            nonlocal captured_lm
            captured_lm = dspy.settings.lm
            return mock_result

        mock_extract.side_effect = capture_lm

        extract_nodes_task.call_local(episode_uuid, DEFAULT_JOURNAL)

        assert captured_lm == mock_model


@pytest.mark.inference
def test_extract_nodes_task_integration(
    isolated_graph, episode_uuid, cleanup_test_episodes, require_llm
):
    """Integration test: extract_nodes_task with real model and database."""
    from backend.services.tasks import extract_nodes_task
    from backend.database.redis_ops import get_episode_status, set_episode_status
    from backend import add_journal_entry

    content = "I met Alice at the park. She works at Google."
    episode_uuid = asyncio.run(add_journal_entry(content))

    set_episode_status(episode_uuid, "pending_nodes", DEFAULT_JOURNAL)

    with patch("backend.database.redis_ops.get_inference_enabled", create=True) as mock_enabled:
        mock_enabled.return_value = True

        result = extract_nodes_task.call_local(episode_uuid, DEFAULT_JOURNAL)

        assert result["episode_uuid"] == episode_uuid
        assert isinstance(result["extracted_count"], int)
        assert isinstance(result["new_entities"], int)

        status = get_episode_status(episode_uuid, DEFAULT_JOURNAL)
        # Episodes are marked "done" after node extraction (edges task TBD)
        assert status == "done"


@pytest.mark.inference
def test_extract_nodes_task_integration_no_entities_marks_done(
    isolated_graph, cleanup_test_episodes, require_llm
):
    """Integration test: episodes with no extractable entities should end in done."""
    from backend.services.tasks import extract_nodes_task
    from backend.database.redis_ops import (
        get_episode_status,
        set_episode_status,
        remove_episode_from_queue,
    )
    from backend import add_journal_entry

    # Content chosen to be entity-free; skip if the model still extracts something.
    content = "It was a quiet, rainy afternoon with nothing notable to report."
    episode_uuid = asyncio.run(add_journal_entry(content))

    set_episode_status(episode_uuid, "pending_nodes", DEFAULT_JOURNAL)

    result = extract_nodes_task.call_local(episode_uuid, DEFAULT_JOURNAL)

    status = get_episode_status(episode_uuid, DEFAULT_JOURNAL)
    if result["extracted_count"] > 0:
        pytest.skip("LLM extracted entities unexpectedly; cannot verify done transition.")

    assert status == "done"
    remove_episode_from_queue(episode_uuid, DEFAULT_JOURNAL)


def test_extract_nodes_task_cleanup_always_called(episode_uuid):
    """cleanup_if_no_work is called even when task exits early."""
    from backend.services.tasks import extract_nodes_task

    with patch("backend.database.redis_ops.get_episode_status") as mock_status:
        with patch("backend.inference.manager.cleanup_if_no_work") as mock_cleanup:
            mock_status.return_value = None

            result = extract_nodes_task.call_local(episode_uuid, DEFAULT_JOURNAL)

            assert result["already_processed"] is True
            mock_cleanup.assert_called_once()


def test_extract_nodes_task_cleanup_called_on_exception(episode_uuid):
    """cleanup_if_no_work is called even when extraction raises exception."""
    from backend.services.tasks import extract_nodes_task

    with (
        patch("backend.database.redis_ops.get_episode_status") as mock_status,
        patch("backend.database.redis_ops.get_inference_enabled", create=True) as mock_enabled,
        patch("backend.database.redis_ops.set_episode_status") as mock_set_status,
        patch("backend.database.redis_ops.set_active_episode"),
        patch("backend.database.redis_ops.get_active_episode_uuid") as mock_get_active,
        patch("backend.database.redis_ops.clear_active_episode"),
        patch("backend.database.redis_ops.set_model_state"),
        patch("backend.database.redis_ops.clear_model_state"),
        patch("backend.database.redis_ops.increment_and_check_retry_count") as mock_retry,
        patch("backend.database.redis_ops.remove_pending_episode"),
        patch("backend.inference.manager.get_model") as mock_get_model,
        patch("backend.inference.manager.is_model_loading_blocked") as mock_blocked,
        patch("backend.graph.extract_nodes.extract_nodes") as mock_extract,
        patch("backend.inference.manager.cleanup_if_no_work") as mock_cleanup,
    ):
        mock_status.return_value = "pending_nodes"
        mock_enabled.return_value = True
        mock_blocked.return_value = False
        mock_get_active.return_value = episode_uuid
        mock_get_model.return_value = Mock()
        mock_extract.side_effect = RuntimeError("Extraction failed")
        mock_retry.return_value = (3, True)  # Simulate max retries reached

        with pytest.raises(RuntimeError, match="Extraction failed"):
            extract_nodes_task.call_local(episode_uuid, DEFAULT_JOURNAL)

        mock_set_status.assert_called_with(episode_uuid, "dead", DEFAULT_JOURNAL)
        mock_cleanup.assert_called_once()


def test_extract_nodes_task_removes_from_queue_on_success(episode_uuid):
    """Episode transitions to pending_edges after successful extraction with entities."""
    from backend.services.tasks import extract_nodes_task

    mock_result = ExtractNodesResult(
        episode_uuid=episode_uuid,
        extracted_count=2,
        resolved_count=1,
        new_entities=1,
        exact_matches=1,
        fuzzy_matches=0,
        entity_uuids=["uuid1"],
        uuid_map={"prov": "canon"},
    )

    with (
        patch("backend.database.redis_ops.get_episode_status") as mock_status,
        patch("backend.database.redis_ops.get_inference_enabled", create=True) as mock_enabled,
        patch("backend.database.redis_ops.set_episode_status") as mock_set_status,
        patch("backend.database.redis_ops.set_active_episode"),
        patch("backend.database.redis_ops.get_active_episode_uuid") as mock_get_active,
        patch("backend.database.redis_ops.clear_active_episode"),
        patch("backend.database.redis_ops.set_model_state"),
        patch("backend.database.redis_ops.clear_model_state"),
        patch("backend.database.redis_ops.remove_pending_episode"),
        patch("backend.inference.manager.get_model") as mock_get_model,
        patch("backend.inference.manager.is_model_loading_blocked") as mock_blocked,
        patch("backend.graph.extract_nodes.extract_nodes") as mock_extract,
        patch("backend.inference.manager.cleanup_if_no_work"),
    ):
        mock_status.return_value = "pending_nodes"
        mock_enabled.return_value = True
        mock_blocked.return_value = False
        mock_get_active.return_value = episode_uuid
        mock_get_model.return_value = Mock()
        mock_extract.return_value = mock_result

        extract_nodes_task.call_local(episode_uuid, DEFAULT_JOURNAL)

        # With entities extracted, should transition to done (edges task TBD)
        mock_set_status.assert_called_once_with(
            episode_uuid, "done", DEFAULT_JOURNAL, uuid_map={"prov": "canon"}
        )


def test_extract_nodes_task_idempotency_check(episode_uuid):
    """Task checks status before processing for idempotency."""
    from backend.services.tasks import extract_nodes_task

    with patch("backend.database.redis_ops.get_episode_status") as mock_status:
        with patch("backend.inference.manager.cleanup_if_no_work"):
            mock_status.return_value = None

            result1 = extract_nodes_task.call_local(episode_uuid, DEFAULT_JOURNAL)
            result2 = extract_nodes_task.call_local(episode_uuid, DEFAULT_JOURNAL)

            assert result1["already_processed"] is True
            assert result2["already_processed"] is True
            assert mock_status.call_count == 2


def test_extract_nodes_task_handles_missing_episode_gracefully(isolated_graph, cleanup_test_episodes):
    """Integration: missing episode should be cleaned up gracefully, not marked dead."""
    from backend.services.tasks import extract_nodes_task
    from backend.database.redis_ops import (
        set_episode_status,
        get_episode_status,
    )

    missing_uuid = "missing-episode-graceful"
    set_episode_status(missing_uuid, "pending_nodes", DEFAULT_JOURNAL)

    with patch("backend.inference.manager.is_model_loading_blocked", return_value=False):
        # Should NOT raise - graceful handling returns result dict
        result = extract_nodes_task.call_local(missing_uuid, DEFAULT_JOURNAL)

    # Episode deleted during extraction should be handled gracefully
    assert result["episode_deleted"] is True
    assert result["uuid"] == missing_uuid

    # Redis should be cleaned up (status removed)
    assert get_episode_status(missing_uuid, DEFAULT_JOURNAL) is None


def test_orchestrate_inference_work_reschedules_and_runs_once():
    """orchestrate_inference_work should enqueue and cleanup, then reschedule in 1s."""
    from backend.services.tasks import orchestrate_inference_work
    from backend.settings import ORCHESTRATOR_INTERVAL_SECONDS

    with patch("backend.database.redis_ops.enqueue_pending_episodes") as mock_enqueue, \
         patch("backend.inference.manager.cleanup_if_no_work") as mock_cleanup, \
         patch.object(orchestrate_inference_work, "schedule") as mock_schedule:

        orchestrate_inference_work.call_local()

        mock_enqueue.assert_called_once()
        mock_cleanup.assert_called_once()
        mock_schedule.assert_called_once()
        args, kwargs = mock_schedule.call_args
        assert kwargs.get("delay") == ORCHESTRATOR_INTERVAL_SECONDS


def test_orchestrate_inference_work_reschedule_disabled():
    """Reschedule should be skipped when reschedule=False."""
    from backend.services.tasks import orchestrate_inference_work

    with patch.object(orchestrate_inference_work, "schedule") as mock_schedule:
        orchestrate_inference_work.call_local(reschedule=False)
        mock_schedule.assert_not_called()


def test_check_cancellation_raises_when_shutdown():
    """check_cancellation() raises TaskCancelled when shutdown is requested."""
    from backend.services.tasks import TaskCancelled, check_cancellation
    import backend.database.lifecycle as lifecycle

    lifecycle._shutdown_requested = True
    try:
        with pytest.raises(TaskCancelled):
            check_cancellation()
    finally:
        lifecycle._shutdown_requested = False


def test_check_cancellation_does_not_raise_normally():
    """check_cancellation() does not raise when shutdown is not requested."""
    from backend.services.tasks import check_cancellation
    import backend.database.lifecycle as lifecycle

    lifecycle._shutdown_requested = False
    check_cancellation()  # Should not raise


def test_task_cancelled_not_marked_dead(episode_uuid):
    """TaskCancelled exception should NOT mark episode as dead."""
    from backend.services.tasks import extract_nodes_task, TaskCancelled
    import backend.database.lifecycle as lifecycle

    with patch("backend.database.redis_ops.get_episode_status") as mock_status:
        with patch("backend.database.redis_ops.set_episode_status") as mock_set_status:
            with patch("backend.inference.manager.cleanup_if_no_work"):
                mock_status.return_value = "pending_nodes"

                # Set shutdown flag so check_cancellation raises TaskCancelled
                lifecycle._shutdown_requested = True
                try:
                    result = extract_nodes_task.call_local(episode_uuid, DEFAULT_JOURNAL)

                    # Task should return cancelled status, not raise
                    assert result == {"cancelled": True}

                    # set_episode_status should NOT have been called with "dead"
                    for call in mock_set_status.call_args_list:
                        args, kwargs = call
                        if len(args) >= 2:
                            assert args[1] != "dead", "Episode should not be marked dead on cancellation"
                finally:
                    lifecycle._shutdown_requested = False


def test_orchestrator_stops_rescheduling_on_shutdown():
    """orchestrate_inference_work should not reschedule when shutdown is requested."""
    from backend.services.tasks import orchestrate_inference_work
    import backend.database.lifecycle as lifecycle

    lifecycle._shutdown_requested = True
    try:
        with patch("backend.database.redis_ops.enqueue_pending_episodes"), \
             patch("backend.inference.manager.cleanup_if_no_work"), \
             patch.object(orchestrate_inference_work, "schedule") as mock_schedule:

            orchestrate_inference_work.call_local(reschedule=True)

            # Should skip reschedule due to shutdown flag
            mock_schedule.assert_not_called()
    finally:
        lifecycle._shutdown_requested = False


def test_orchestrator_skips_work_when_shutdown():
    """orchestrate_inference_work should return early when shutdown is requested."""
    from backend.services.tasks import orchestrate_inference_work
    import backend.database.lifecycle as lifecycle

    lifecycle._shutdown_requested = True
    try:
        with patch("backend.database.redis_ops.enqueue_pending_episodes") as mock_enqueue, \
             patch("backend.inference.manager.cleanup_if_no_work") as mock_cleanup:

            orchestrate_inference_work.call_local(reschedule=False)

            # Should skip all work due to early return
            mock_enqueue.assert_not_called()
            mock_cleanup.assert_not_called()
    finally:
        lifecycle._shutdown_requested = False


def test_redis_ops_allows_operations_during_shutdown(isolated_graph):
    """redis_ops() should allow database operations when shutdown is requested.

    The shutdown flag is for tasks to check at safe checkpoints (via check_cancellation).
    It should NOT block database operations globally - in-flight tasks need to persist
    their completed work even after shutdown is requested.

    This tests the actual redis_ops() context manager, not mocked functions.
    """
    from backend.database.redis_ops import redis_ops
    import backend.database.lifecycle as lifecycle

    lifecycle._shutdown_requested = True
    try:
        # This should NOT raise - database is still available
        with redis_ops() as r:
            # Simple operation to verify Redis is accessible
            r.set("test:shutdown_check", "works")
            value = r.get("test:shutdown_check")
            assert value == b"works"
            r.delete("test:shutdown_check")
    finally:
        lifecycle._shutdown_requested = False


def test_extract_nodes_task_handles_node_not_found_error(episode_uuid):
    """Task handles NodeNotFoundError gracefully when episode deleted mid-extraction."""
    from backend.services.tasks import extract_nodes_task
    from graphiti_core.errors import NodeNotFoundError

    with (
        patch("backend.database.redis_ops.get_episode_status") as mock_status,
        patch("backend.database.redis_ops.get_inference_enabled", create=True) as mock_enabled,
        patch("backend.database.redis_ops.set_active_episode"),
        patch("backend.database.redis_ops.get_active_episode_uuid") as mock_get_active,
        patch("backend.database.redis_ops.clear_active_episode"),
        patch("backend.database.redis_ops.set_model_state"),
        patch("backend.database.redis_ops.clear_model_state"),
        patch("backend.inference.manager.get_model") as mock_get_model,
        patch("backend.inference.manager.is_model_loading_blocked") as mock_blocked,
        patch("backend.graph.extract_nodes.extract_nodes") as mock_extract,
        patch("backend.inference.manager.cleanup_if_no_work"),
        patch("backend.database.redis_ops.remove_episode_from_queue") as mock_remove,
    ):
        mock_status.return_value = "pending_nodes"
        mock_enabled.return_value = True
        mock_blocked.return_value = False
        mock_get_active.return_value = episode_uuid
        mock_get_model.return_value = Mock()
        mock_extract.side_effect = NodeNotFoundError(episode_uuid)

        result = extract_nodes_task.call_local(episode_uuid, DEFAULT_JOURNAL)

        assert result["episode_deleted"] is True
        assert result["uuid"] == episode_uuid
        mock_remove.assert_called_once_with(episode_uuid, DEFAULT_JOURNAL)


def test_extract_nodes_task_handles_episode_deleted_error(episode_uuid):
    """Task handles EpisodeDeletedError gracefully when episode deleted during persistence."""
    from backend.services.tasks import extract_nodes_task
    from backend.database.persistence import EpisodeDeletedError

    with (
        patch("backend.database.redis_ops.get_episode_status") as mock_status,
        patch("backend.database.redis_ops.get_inference_enabled", create=True) as mock_enabled,
        patch("backend.database.redis_ops.set_active_episode"),
        patch("backend.database.redis_ops.get_active_episode_uuid") as mock_get_active,
        patch("backend.database.redis_ops.clear_active_episode"),
        patch("backend.database.redis_ops.set_model_state"),
        patch("backend.database.redis_ops.clear_model_state"),
        patch("backend.inference.manager.get_model") as mock_get_model,
        patch("backend.inference.manager.is_model_loading_blocked") as mock_blocked,
        patch("backend.graph.extract_nodes.extract_nodes") as mock_extract,
        patch("backend.inference.manager.cleanup_if_no_work"),
        patch("backend.database.redis_ops.remove_episode_from_queue") as mock_remove,
    ):
        mock_status.return_value = "pending_nodes"
        mock_enabled.return_value = True
        mock_blocked.return_value = False
        mock_get_active.return_value = episode_uuid
        mock_get_model.return_value = Mock()
        mock_extract.side_effect = EpisodeDeletedError(episode_uuid)

        result = extract_nodes_task.call_local(episode_uuid, DEFAULT_JOURNAL)

        assert result["episode_deleted"] is True
        assert result["uuid"] == episode_uuid
        mock_remove.assert_called_once_with(episode_uuid, DEFAULT_JOURNAL)


# =============================================================================
# Retry Logic Tests
# =============================================================================


def test_extract_nodes_task_increments_retry_on_failure(episode_uuid):
    """Retry counter increments on each failure, not marking dead until max reached."""
    from backend.services.tasks import extract_nodes_task

    with (
        patch("backend.database.redis_ops.get_episode_status") as mock_status,
        patch("backend.database.redis_ops.get_inference_enabled", create=True) as mock_enabled,
        patch("backend.database.redis_ops.set_episode_status") as mock_set_status,
        patch("backend.database.redis_ops.set_active_episode"),
        patch("backend.database.redis_ops.get_active_episode_uuid") as mock_get_active,
        patch("backend.database.redis_ops.clear_active_episode"),
        patch("backend.database.redis_ops.set_model_state"),
        patch("backend.database.redis_ops.clear_model_state"),
        patch("backend.database.redis_ops.increment_and_check_retry_count") as mock_retry,
        patch("backend.database.redis_ops.remove_pending_episode"),
        patch("backend.inference.manager.get_model") as mock_get_model,
        patch("backend.inference.manager.is_model_loading_blocked") as mock_blocked,
        patch("backend.graph.extract_nodes.extract_nodes") as mock_extract,
        patch("backend.inference.manager.cleanup_if_no_work"),
    ):
        mock_status.return_value = "pending_nodes"
        mock_enabled.return_value = True
        mock_blocked.return_value = False
        mock_get_active.return_value = episode_uuid
        mock_get_model.return_value = Mock()
        mock_extract.side_effect = RuntimeError("Extraction failed")
        mock_retry.return_value = (1, False)  # First failure, not max yet

        with pytest.raises(RuntimeError, match="Extraction failed"):
            extract_nodes_task.call_local(episode_uuid, DEFAULT_JOURNAL)

        # Should NOT mark dead on first failure
        mock_set_status.assert_not_called()
        mock_retry.assert_called_once()


def test_extract_nodes_task_resets_retry_on_success(episode_uuid):
    """Retry counter is reset to 0 after successful extraction."""
    from backend.services.tasks import extract_nodes_task

    mock_result = ExtractNodesResult(
        episode_uuid=episode_uuid,
        extracted_count=2,
        resolved_count=1,
        new_entities=1,
        exact_matches=1,
        fuzzy_matches=0,
        entity_uuids=["uuid1"],
        uuid_map={},
    )

    with (
        patch("backend.database.redis_ops.get_episode_status") as mock_status,
        patch("backend.database.redis_ops.get_inference_enabled", create=True) as mock_enabled,
        patch("backend.database.redis_ops.set_episode_status"),
        patch("backend.database.redis_ops.set_active_episode"),
        patch("backend.database.redis_ops.get_active_episode_uuid") as mock_get_active,
        patch("backend.database.redis_ops.clear_active_episode"),
        patch("backend.database.redis_ops.set_model_state"),
        patch("backend.database.redis_ops.clear_model_state"),
        patch("backend.database.redis_ops.reset_retry_count") as mock_reset,
        patch("backend.database.redis_ops.remove_pending_episode"),
        patch("backend.inference.manager.get_model") as mock_get_model,
        patch("backend.inference.manager.is_model_loading_blocked") as mock_blocked,
        patch("backend.graph.extract_nodes.extract_nodes") as mock_extract,
        patch("backend.inference.manager.cleanup_if_no_work"),
    ):
        mock_status.return_value = "pending_nodes"
        mock_enabled.return_value = True
        mock_blocked.return_value = False
        mock_get_active.return_value = episode_uuid
        mock_get_model.return_value = Mock()
        mock_extract.return_value = mock_result

        extract_nodes_task.call_local(episode_uuid, DEFAULT_JOURNAL)

        mock_reset.assert_called_once_with(episode_uuid, DEFAULT_JOURNAL)


def test_extract_nodes_task_does_not_increment_on_cancellation(episode_uuid):
    """TaskCancelled does NOT increment retry counter."""
    from backend.services.tasks import extract_nodes_task
    import backend.database.lifecycle as lifecycle

    with (
        patch("backend.database.redis_ops.get_episode_status") as mock_status,
        patch("backend.database.redis_ops.increment_and_check_retry_count") as mock_retry,
        patch("backend.inference.manager.cleanup_if_no_work"),
    ):
        mock_status.return_value = "pending_nodes"

        lifecycle._shutdown_requested = True
        try:
            result = extract_nodes_task.call_local(episode_uuid, DEFAULT_JOURNAL)
            assert result == {"cancelled": True}
            mock_retry.assert_not_called()
        finally:
            lifecycle._shutdown_requested = False
