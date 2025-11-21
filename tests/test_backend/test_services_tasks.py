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
            mock_status.return_value = "pending_nodes"
            mock_enabled.return_value = False

            result = extract_nodes_task.call_local(episode_uuid, DEFAULT_JOURNAL)

            assert result["inference_disabled"] is True


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

    with patch("backend.database.redis_ops.get_episode_status") as mock_status:
        with patch("backend.database.redis_ops.get_inference_enabled", create=True) as mock_enabled:
            with patch("backend.database.redis_ops.set_episode_status") as mock_set_status:
                with patch("backend.inference.manager.get_model") as mock_get_model:
                    with patch("backend.graph.extract_nodes.extract_nodes") as mock_extract:
                        with patch("backend.inference.manager.cleanup_if_no_work") as mock_cleanup:
                            mock_status.return_value = "pending_nodes"
                            mock_enabled.return_value = True
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
                            # With entities extracted, should transition to pending_edges
                            mock_set_status.assert_called_once_with(
                                episode_uuid, "pending_edges", uuid_map=mock_result.uuid_map
                            )
                            mock_cleanup.assert_called_once()


def test_extract_nodes_task_no_entities_removes_from_queue(episode_uuid):
    """Task removes episode from queue when no entities extracted."""
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

    with patch("backend.database.redis_ops.get_episode_status") as mock_status:
        with patch("backend.database.redis_ops.get_inference_enabled", create=True) as mock_enabled:
            with patch("backend.database.redis_ops.remove_episode_from_queue") as mock_remove:
                with patch("backend.inference.manager.get_model") as mock_get_model:
                    with patch("backend.graph.extract_nodes.extract_nodes") as mock_extract:
                        with patch("backend.inference.manager.cleanup_if_no_work") as mock_cleanup:
                            mock_status.return_value = "pending_nodes"
                            mock_enabled.return_value = True
                            mock_model = Mock()
                            mock_get_model.return_value = mock_model
                            mock_extract.return_value = mock_result

                            result = extract_nodes_task.call_local(episode_uuid, DEFAULT_JOURNAL)

                            assert result["extracted_count"] == 0
                            mock_remove.assert_called_once_with(episode_uuid)
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

    with patch("backend.database.redis_ops.get_episode_status") as mock_status:
        with patch("backend.database.redis_ops.get_inference_enabled", create=True) as mock_enabled:
            with patch("backend.database.redis_ops.set_episode_status"):
                with patch("backend.inference.manager.get_model") as mock_get_model:
                    with patch("backend.graph.extract_nodes.extract_nodes") as mock_extract:
                        with patch("backend.inference.manager.cleanup_if_no_work"):
                            mock_status.return_value = "pending_nodes"
                            mock_enabled.return_value = True
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

    set_episode_status(episode_uuid, "pending_nodes", journal=DEFAULT_JOURNAL)

    with patch("backend.database.redis_ops.get_inference_enabled", create=True) as mock_enabled:
        mock_enabled.return_value = True

        result = extract_nodes_task.call_local(episode_uuid, DEFAULT_JOURNAL)

        assert result["episode_uuid"] == episode_uuid
        assert isinstance(result["extracted_count"], int)
        assert isinstance(result["new_entities"], int)

        status = get_episode_status(episode_uuid)
        assert status is None


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

    with patch("backend.database.redis_ops.get_episode_status") as mock_status:
        with patch("backend.database.redis_ops.get_inference_enabled", create=True) as mock_enabled:
            with patch("backend.inference.manager.get_model") as mock_get_model:
                with patch("backend.graph.extract_nodes.extract_nodes") as mock_extract:
                    with patch("backend.inference.manager.cleanup_if_no_work") as mock_cleanup:
                        mock_status.return_value = "pending_nodes"
                        mock_enabled.return_value = True
                        mock_get_model.return_value = Mock()
                        mock_extract.side_effect = RuntimeError("Extraction failed")

                        with pytest.raises(RuntimeError, match="Extraction failed"):
                            extract_nodes_task.call_local(episode_uuid, DEFAULT_JOURNAL)

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

    with patch("backend.database.redis_ops.get_episode_status") as mock_status:
        with patch("backend.database.redis_ops.get_inference_enabled", create=True) as mock_enabled:
            with patch("backend.database.redis_ops.set_episode_status") as mock_set_status:
                with patch("backend.inference.manager.get_model") as mock_get_model:
                    with patch("backend.graph.extract_nodes.extract_nodes") as mock_extract:
                        with patch("backend.inference.manager.cleanup_if_no_work"):
                            mock_status.return_value = "pending_nodes"
                            mock_enabled.return_value = True
                            mock_get_model.return_value = Mock()
                            mock_extract.return_value = mock_result

                            extract_nodes_task.call_local(episode_uuid, DEFAULT_JOURNAL)

                            # With entities extracted, should transition to pending_edges
                            mock_set_status.assert_called_once_with(
                                episode_uuid, "pending_edges", uuid_map={"prov": "canon"}
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
