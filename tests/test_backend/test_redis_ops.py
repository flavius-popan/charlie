"""Redis operations layer tests for backend (Redis API access via FalkorDB Lite)."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

import backend.database as db_utils
from backend import add_journal_entry
from backend.settings import DEFAULT_JOURNAL
from backend.database.redis_ops import (
    get_episode_data,
    get_episode_status,
    get_episode_uuid_map,
    get_episodes_by_status,
    remove_episode_from_queue,
    set_episode_status,
)


@pytest.fixture
def redis_client(falkordb_test_context):
    """Get the underlying Redis client from FalkorDB."""
    from backend.settings import DEFAULT_JOURNAL
    import backend.database.lifecycle as lifecycle

    graph = db_utils.get_falkordb_graph(DEFAULT_JOURNAL)
    if graph is None or lifecycle._db is None:
        pytest.skip("FalkorDB Lite is unavailable in this environment.")

    client = getattr(lifecycle._db, "client", None)
    if client is None:
        pytest.skip("Redis client is unavailable.")

    yield client

    # Clean up all Redis keys after each test
    try:
        keys = client.keys("*")
        if keys:
            client.delete(*keys)
    except Exception:
        pass


def test_redis_ops_import():
    """Ensure redis_ops can be imported from backend.database."""
    from backend.database import redis_ops
    assert redis_ops is not None


def test_redis_ops_context_manager_basic(falkordb_test_context):
    """Verify redis_ops context manager can be used without error."""
    from backend.database import redis_ops

    with redis_ops() as r:
        assert r is not None
        assert hasattr(r, "ping")


def test_basic_get_set_operations(redis_client, falkordb_test_context):
    """Test basic GET/SET/DEL operations."""
    from backend.database import redis_ops

    with redis_ops() as r:
        r.set("test_key", "test_value")
        value = r.get("test_key")
        assert value == b"test_value"

        r.delete("test_key")
        value = r.get("test_key")
        assert value is None


def test_hash_operations(redis_client, falkordb_test_context):
    """Test hash operations (HSET/HGET/HGETALL/HDEL)."""
    from backend.database import redis_ops

    with redis_ops() as r:
        r.hset("user_settings", "theme", "dark")
        r.hset("user_settings", "lang", "en")

        theme = r.hget("user_settings", "theme")
        assert theme == b"dark"

        lang = r.hget("user_settings", "lang")
        assert lang == b"en"

        all_settings = r.hgetall("user_settings")
        assert all_settings == {b"theme": b"dark", b"lang": b"en"}

        r.hdel("user_settings", "theme")
        theme = r.hget("user_settings", "theme")
        assert theme is None


def test_list_operations(redis_client, falkordb_test_context):
    """Test list operations (LPUSH/RPUSH/LRANGE/LPOP)."""
    from backend.database import redis_ops

    with redis_ops() as r:
        r.lpush("recent_items", "item3")
        r.lpush("recent_items", "item2")
        r.lpush("recent_items", "item1")

        items = r.lrange("recent_items", 0, -1)
        assert items == [b"item1", b"item2", b"item3"]

        first = r.lpop("recent_items")
        assert first == b"item1"

        items = r.lrange("recent_items", 0, -1)
        assert items == [b"item2", b"item3"]


def test_set_operations(redis_client, falkordb_test_context):
    """Test set operations (SADD/SMEMBERS/SREM/SISMEMBER)."""
    from backend.database import redis_ops

    with redis_ops() as r:
        r.sadd("tags", "python", "redis", "testing")

        members = r.smembers("tags")
        assert members == {b"python", b"redis", b"testing"}

        is_member = r.sismember("tags", "python")
        assert is_member == 1

        r.srem("tags", "testing")
        members = r.smembers("tags")
        assert members == {b"python", b"redis"}


def test_multi_key_operations(redis_client, falkordb_test_context):
    """Test multi-key operations (MGET, DEL with multiple keys)."""
    from backend.database import redis_ops

    with redis_ops() as r:
        r.set("key1", "value1")
        r.set("key2", "value2")
        r.set("key3", "value3")

        values = r.mget("key1", "key2", "key3")
        assert values == [b"value1", b"value2", b"value3"]

        deleted = r.delete("key1", "key2", "key3")
        assert deleted == 3

        values = r.mget("key1", "key2", "key3")
        assert values == [None, None, None]


def test_incr_decr_operations(redis_client, falkordb_test_context):
    """Test increment/decrement operations."""
    from backend.database import redis_ops

    with redis_ops() as r:
        r.set("counter", "0")

        r.incr("counter")
        assert r.get("counter") == b"1"

        r.incrby("counter", 5)
        assert r.get("counter") == b"6"

        r.decr("counter")
        assert r.get("counter") == b"5"


def test_expiration_operations(redis_client, falkordb_test_context):
    """Test key expiration operations (EXPIRE, TTL)."""
    from backend.database import redis_ops

    with redis_ops() as r:
        r.set("temp_key", "temp_value")

        r.expire("temp_key", 3600)
        ttl = r.ttl("temp_key")
        assert ttl > 0 and ttl <= 3600

        r.persist("temp_key")
        ttl = r.ttl("temp_key")
        assert ttl == -1


def test_pattern_operations(redis_client, falkordb_test_context):
    """Test pattern matching operations (KEYS, SCAN)."""
    from backend.database import redis_ops

    with redis_ops() as r:
        r.set("user:1:name", "Alice")
        r.set("user:2:name", "Bob")
        r.set("session:abc", "data1")

        user_keys = r.keys("user:*")
        assert len(user_keys) == 2
        assert b"user:1:name" in user_keys
        assert b"user:2:name" in user_keys

        cursor, keys = r.scan(0, match="session:*", count=10)
        assert b"session:abc" in keys


def test_shutdown_behavior(falkordb_test_context):
    """Test that redis_ops fails gracefully during shutdown."""
    from backend.database import redis_ops
    import backend.database.lifecycle as lifecycle

    lifecycle._shutdown_requested = True

    try:
        with pytest.raises(RuntimeError, match="shutdown"):
            with redis_ops() as r:
                r.set("key", "value")
    finally:
        lifecycle._shutdown_requested = False


# Episode Status Management Tests


def test_set_episode_status_creates_initial_status(
    episode_uuid, cleanup_test_episodes, falkordb_test_context
):
    """Set initial status for new episode."""
    set_episode_status(episode_uuid, "pending_nodes", journal=DEFAULT_JOURNAL)

    status = get_episode_status(episode_uuid)
    assert status == "pending_nodes"


def test_set_episode_status_updates_existing_status(
    episode_uuid, cleanup_test_episodes, falkordb_test_context
):
    """Update episode fields while keeping status as pending_nodes."""
    set_episode_status(episode_uuid, "pending_nodes", journal=DEFAULT_JOURNAL)
    uuid_map = {str(uuid4()): str(uuid4())}
    set_episode_status(episode_uuid, "pending_nodes", uuid_map=uuid_map)

    status = get_episode_status(episode_uuid)
    assert status == "pending_nodes"
    data = get_episode_data(episode_uuid)
    assert json.loads(data["uuid_map"]) == uuid_map


def test_set_episode_status_with_uuid_map(
    episode_uuid, cleanup_test_episodes, falkordb_test_context
):
    """Store uuid_map when entities are found."""
    uuid_map = {str(uuid4()): str(uuid4()), str(uuid4()): str(uuid4())}

    set_episode_status(episode_uuid, "pending_nodes", journal=DEFAULT_JOURNAL, uuid_map=uuid_map)

    data = get_episode_data(episode_uuid)
    assert data["status"] == "pending_nodes"
    assert json.loads(data["uuid_map"]) == uuid_map


def test_get_episodes_by_status_returns_matching_episodes(falkordb_test_context):
    """Scan all episodes with given status."""
    episode1 = str(uuid4())
    episode2 = str(uuid4())

    set_episode_status(episode1, "pending_nodes", journal=DEFAULT_JOURNAL)
    set_episode_status(episode2, "pending_nodes", journal=DEFAULT_JOURNAL)

    pending_nodes = get_episodes_by_status("pending_nodes")
    assert episode1 in pending_nodes
    assert episode2 in pending_nodes

    remove_episode_from_queue(episode1)

    pending_nodes_after = get_episodes_by_status("pending_nodes")
    assert episode1 not in pending_nodes_after
    assert episode2 in pending_nodes_after

    remove_episode_from_queue(episode2)


def test_get_episodes_by_status_empty_when_no_matches(falkordb_test_context):
    """Return empty list when no episodes have status."""
    episodes = get_episodes_by_status("nonexistent_status")
    assert episodes == []


def test_remove_episode_from_queue_removes_all_data(
    episode_uuid, cleanup_test_episodes, falkordb_test_context
):
    """Remove episode from all Redis structures."""
    uuid_map = {str(uuid4()): str(uuid4())}
    set_episode_status(
        episode_uuid, "pending_nodes", journal=DEFAULT_JOURNAL, uuid_map=uuid_map
    )

    remove_episode_from_queue(episode_uuid)

    status = get_episode_status(episode_uuid)
    assert status is None

    pending = get_episodes_by_status("pending_nodes")
    assert episode_uuid not in pending


def test_status_index_updates_when_status_changes(
    episode_uuid, cleanup_test_episodes, falkordb_test_context
):
    """Episode remains in pending_nodes until removed from queue."""
    set_episode_status(episode_uuid, "pending_nodes", journal=DEFAULT_JOURNAL)

    pending_nodes = get_episodes_by_status("pending_nodes")
    assert episode_uuid in pending_nodes

    remove_episode_from_queue(episode_uuid)

    pending_nodes = get_episodes_by_status("pending_nodes")
    assert episode_uuid not in pending_nodes


def test_get_episode_uuid_map_returns_parsed_dict(
    episode_uuid, cleanup_test_episodes, falkordb_test_context
):
    """get_episode_uuid_map returns parsed dict, not JSON string."""
    uuid_map = {str(uuid4()): str(uuid4()), str(uuid4()): str(uuid4())}
    set_episode_status(episode_uuid, "pending_nodes", journal=DEFAULT_JOURNAL, uuid_map=uuid_map)

    retrieved_map = get_episode_uuid_map(episode_uuid)
    assert retrieved_map == uuid_map
    assert isinstance(retrieved_map, dict)


def test_get_episode_uuid_map_returns_none_when_not_set(
    episode_uuid, cleanup_test_episodes, falkordb_test_context
):
    """get_episode_uuid_map returns None when uuid_map not set."""
    set_episode_status(episode_uuid, "pending_nodes", journal=DEFAULT_JOURNAL)

    uuid_map = get_episode_uuid_map(episode_uuid)
    assert uuid_map is None


@pytest.mark.asyncio
async def test_add_journal_entry_sets_pending_nodes_status(isolated_graph):
    """New journal entries are marked as pending_nodes for extraction."""
    episode_uuid = await add_journal_entry("Today I met Sarah at the park.")

    status = get_episode_status(episode_uuid)
    assert status == "pending_nodes"

    pending_episodes = get_episodes_by_status("pending_nodes")
    assert episode_uuid in pending_episodes

    remove_episode_from_queue(episode_uuid)


def test_episode_lifecycle_completes_properly(
    episode_uuid, cleanup_test_episodes, falkordb_test_context
):
    """Episode progresses through full lifecycle without getting stuck."""
    set_episode_status(episode_uuid, "pending_nodes", journal=DEFAULT_JOURNAL)

    pending = get_episodes_by_status("pending_nodes")
    assert episode_uuid in pending

    uuid_map = {str(uuid4()): str(uuid4())}
    set_episode_status(episode_uuid, "pending_nodes", uuid_map=uuid_map)

    pending_after = get_episodes_by_status("pending_nodes")
    assert episode_uuid in pending_after

    remove_episode_from_queue(episode_uuid)

    status = get_episode_status(episode_uuid)
    assert status is None

    pending_final = get_episodes_by_status("pending_nodes")
    assert episode_uuid not in pending_final


def test_enqueue_pending_episodes_processes_backlog(falkordb_test_context):
    """Pending episodes are enqueued when inference is re-enabled."""
    from backend.database.redis_ops import (
        enqueue_pending_episodes,
        get_inference_enabled,
        set_inference_enabled,
    )
    from unittest.mock import patch

    episode1 = str(uuid4())
    episode2 = str(uuid4())
    episode3 = str(uuid4())

    set_episode_status(episode1, "pending_nodes", journal=DEFAULT_JOURNAL)
    set_episode_status(episode2, "pending_nodes", journal=DEFAULT_JOURNAL)
    set_episode_status(episode3, "pending_nodes", journal=DEFAULT_JOURNAL)

    set_inference_enabled(False)
    assert get_inference_enabled() is False

    count = enqueue_pending_episodes()
    assert count == 0

    set_inference_enabled(True)
    assert get_inference_enabled() is True

    with patch("backend.database.redis_ops.extract_nodes_task") as mock_task:
        count = enqueue_pending_episodes()
        assert count == 3
        assert mock_task.call_count == 3

        called_uuids = {call[0][0] for call in mock_task.call_args_list}
        assert called_uuids == {episode1, episode2, episode3}

    remove_episode_from_queue(episode1)
    remove_episode_from_queue(episode2)
    remove_episode_from_queue(episode3)


def test_enqueue_pending_episodes_idempotent(falkordb_test_context):
    """Calling enqueue_pending_episodes multiple times is safe."""
    from backend.database.redis_ops import enqueue_pending_episodes, set_inference_enabled
    from unittest.mock import patch

    episode1 = str(uuid4())
    set_episode_status(episode1, "pending_nodes", journal=DEFAULT_JOURNAL)

    set_inference_enabled(True)

    with patch("backend.database.redis_ops.extract_nodes_task") as mock_task:
        count1 = enqueue_pending_episodes()
        assert count1 == 1

        count2 = enqueue_pending_episodes()
        assert count2 == 1

        assert mock_task.call_count == 2

    remove_episode_from_queue(episode1)


def test_get_episodes_by_status_scan_consistency(falkordb_test_context):
    """Scan-based lookup returns consistent results with episode hash state."""
    episode1 = str(uuid4())
    episode2 = str(uuid4())
    episode3 = str(uuid4())

    set_episode_status(episode1, "pending_nodes", journal=DEFAULT_JOURNAL)
    set_episode_status(episode2, "pending_nodes", journal=DEFAULT_JOURNAL)
    set_episode_status(episode3, "pending_nodes", journal=DEFAULT_JOURNAL)

    pending = get_episodes_by_status("pending_nodes")
    assert len(pending) == 3
    assert episode1 in pending
    assert episode2 in pending
    assert episode3 in pending

    remove_episode_from_queue(episode2)

    pending_after_remove = get_episodes_by_status("pending_nodes")
    assert len(pending_after_remove) == 2
    assert episode1 in pending_after_remove
    assert episode2 not in pending_after_remove
    assert episode3 in pending_after_remove

    remove_episode_from_queue(episode1)
    remove_episode_from_queue(episode3)


def test_cleanup_if_no_work_checks_pending_nodes_only(falkordb_test_context):
    """cleanup_if_no_work only needs to check pending_nodes status."""
    from backend.inference.manager import cleanup_if_no_work, MODELS
    from backend.database.redis_ops import get_episodes_by_status

    episode1 = str(uuid4())
    set_episode_status(episode1, "pending_nodes", journal=DEFAULT_JOURNAL)

    MODELS["llm"] = "fake_model"

    cleanup_if_no_work()
    assert MODELS["llm"] == "fake_model"

    remove_episode_from_queue(episode1)

    pending = get_episodes_by_status("pending_nodes")
    assert len(pending) == 0

    cleanup_if_no_work()
    assert MODELS["llm"] is None


def test_episode_cannot_use_invalid_status(
    episode_uuid, cleanup_test_episodes, falkordb_test_context
):
    """Only pending_nodes status is allowed to prevent unbounded growth."""
    with pytest.raises(ValueError, match="Invalid status 'pending_edges'"):
        set_episode_status(episode_uuid, "pending_edges", journal=DEFAULT_JOURNAL)

    with pytest.raises(ValueError, match="Invalid status 'completed'"):
        set_episode_status(episode_uuid, "completed", journal=DEFAULT_JOURNAL)

    with pytest.raises(ValueError, match="Invalid status 'processing'"):
        set_episode_status(episode_uuid, "processing", journal=DEFAULT_JOURNAL)
