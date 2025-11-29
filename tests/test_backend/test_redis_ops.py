"""Redis operations layer tests for backend (Redis API access via FalkorDB Lite)."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

import backend.database as db_utils
from backend import add_journal_entry
from backend.settings import DEFAULT_JOURNAL
from backend.database.redis_ops import (
    add_suppressed_entity,
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


def test_shutdown_flag_allows_operations_while_db_available(falkordb_test_context):
    """Test that redis_ops allows operations when shutdown is requested.

    The shutdown flag signals tasks to exit early at checkpoints (via check_cancellation),
    but should NOT block database operations. In-flight tasks need to persist their
    completed work even after shutdown is requested, as long as the DB is still available.
    """
    from backend.database import redis_ops
    import backend.database.lifecycle as lifecycle

    lifecycle._shutdown_requested = True

    try:
        # Should NOT raise - database is still available
        with redis_ops() as r:
            r.set("test:shutdown_check", "works")
            value = r.get("test:shutdown_check")
            assert value == b"works"
            r.delete("test:shutdown_check")
    finally:
        lifecycle._shutdown_requested = False


# Episode Status Management Tests


def test_set_episode_status_creates_initial_status(
    episode_uuid, cleanup_test_episodes, falkordb_test_context
):
    """Set initial status for new episode."""
    set_episode_status(episode_uuid, "pending_nodes", DEFAULT_JOURNAL)

    status = get_episode_status(episode_uuid, DEFAULT_JOURNAL)
    assert status == "pending_nodes"


def test_set_episode_status_updates_existing_status(
    episode_uuid, cleanup_test_episodes, falkordb_test_context
):
    """Update episode fields while keeping status as pending_nodes."""
    set_episode_status(episode_uuid, "pending_nodes", DEFAULT_JOURNAL)
    uuid_map = {str(uuid4()): str(uuid4())}
    set_episode_status(episode_uuid, "pending_nodes", DEFAULT_JOURNAL, uuid_map=uuid_map)

    status = get_episode_status(episode_uuid, DEFAULT_JOURNAL)
    assert status == "pending_nodes"
    data = get_episode_data(episode_uuid, DEFAULT_JOURNAL)
    assert json.loads(data["uuid_map"]) == uuid_map


def test_set_episode_status_with_uuid_map(
    episode_uuid, cleanup_test_episodes, falkordb_test_context
):
    """Store uuid_map when entities are found."""
    uuid_map = {str(uuid4()): str(uuid4()), str(uuid4()): str(uuid4())}

    set_episode_status(episode_uuid, "pending_nodes", DEFAULT_JOURNAL, uuid_map=uuid_map)

    data = get_episode_data(episode_uuid, DEFAULT_JOURNAL)
    assert data["status"] == "pending_nodes"
    assert json.loads(data["uuid_map"]) == uuid_map


def test_get_episodes_by_status_returns_matching_episodes(falkordb_test_context):
    """Scan all episodes with given status."""
    episode1 = str(uuid4())
    episode2 = str(uuid4())

    set_episode_status(episode1, "pending_nodes", DEFAULT_JOURNAL)
    set_episode_status(episode2, "pending_nodes", DEFAULT_JOURNAL)

    pending_nodes = get_episodes_by_status("pending_nodes")
    assert episode1 in pending_nodes
    assert episode2 in pending_nodes

    remove_episode_from_queue(episode1, DEFAULT_JOURNAL)

    pending_nodes_after = get_episodes_by_status("pending_nodes")
    assert episode1 not in pending_nodes_after
    assert episode2 in pending_nodes_after

    remove_episode_from_queue(episode2, DEFAULT_JOURNAL)


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
        episode_uuid, "pending_nodes", DEFAULT_JOURNAL, uuid_map=uuid_map
    )

    remove_episode_from_queue(episode_uuid, DEFAULT_JOURNAL)

    status = get_episode_status(episode_uuid, DEFAULT_JOURNAL)
    assert status is None

    pending = get_episodes_by_status("pending_nodes")
    assert episode_uuid not in pending


def test_status_index_updates_when_status_changes(
    episode_uuid, cleanup_test_episodes, falkordb_test_context
):
    """Episode remains in pending_nodes until removed from queue."""
    set_episode_status(episode_uuid, "pending_nodes", DEFAULT_JOURNAL)

    pending_nodes = get_episodes_by_status("pending_nodes")
    assert episode_uuid in pending_nodes

    remove_episode_from_queue(episode_uuid, DEFAULT_JOURNAL)

    pending_nodes = get_episodes_by_status("pending_nodes")
    assert episode_uuid not in pending_nodes


def test_get_episode_uuid_map_returns_parsed_dict(
    episode_uuid, cleanup_test_episodes, falkordb_test_context
):
    """get_episode_uuid_map returns parsed dict, not JSON string."""
    uuid_map = {str(uuid4()): str(uuid4()), str(uuid4()): str(uuid4())}
    set_episode_status(episode_uuid, "pending_nodes", DEFAULT_JOURNAL, uuid_map=uuid_map)

    retrieved_map = get_episode_uuid_map(episode_uuid)
    assert retrieved_map == uuid_map
    assert isinstance(retrieved_map, dict)


def test_get_episode_uuid_map_returns_none_when_not_set(
    episode_uuid, cleanup_test_episodes, falkordb_test_context
):
    """get_episode_uuid_map returns None when uuid_map not set."""
    set_episode_status(episode_uuid, "pending_nodes", DEFAULT_JOURNAL)

    uuid_map = get_episode_uuid_map(episode_uuid)
    assert uuid_map is None


@pytest.mark.asyncio
async def test_add_journal_entry_adds_to_pending_queue(isolated_graph):
    """New journal entries are added to the pending queue for extraction."""
    from backend.database.redis_ops import (
        get_pending_episodes,
        remove_pending_episode,
    )

    episode_uuid = await add_journal_entry("Today I met Sarah at the park.")

    # Episode should be in the chronologically-sorted pending queue
    pending_episodes = get_pending_episodes(DEFAULT_JOURNAL)
    assert episode_uuid in pending_episodes

    remove_pending_episode(episode_uuid, DEFAULT_JOURNAL)


def test_episode_lifecycle_completes_properly(
    episode_uuid, cleanup_test_episodes, falkordb_test_context
):
    """Episode progresses through full lifecycle without getting stuck."""
    set_episode_status(episode_uuid, "pending_nodes", DEFAULT_JOURNAL)

    pending = get_episodes_by_status("pending_nodes")
    assert episode_uuid in pending

    uuid_map = {str(uuid4()): str(uuid4())}
    set_episode_status(episode_uuid, "pending_nodes", DEFAULT_JOURNAL, uuid_map=uuid_map)

    pending_after = get_episodes_by_status("pending_nodes")
    assert episode_uuid in pending_after

    remove_episode_from_queue(episode_uuid, DEFAULT_JOURNAL)

    status = get_episode_status(episode_uuid, DEFAULT_JOURNAL)
    assert status is None

    pending_final = get_episodes_by_status("pending_nodes")
    assert episode_uuid not in pending_final


def test_enqueue_pending_episodes_processes_backlog(falkordb_test_context):
    """Pending episodes are enqueued when inference is re-enabled."""
    from datetime import datetime, timezone
    from backend.database.redis_ops import (
        add_pending_episode,
        enqueue_pending_episodes,
        get_inference_enabled,
        remove_pending_episode,
        set_inference_enabled,
    )
    from unittest.mock import patch

    episode1 = str(uuid4())
    episode2 = str(uuid4())
    episode3 = str(uuid4())

    # Add episodes to pending queue with chronological timestamps
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    add_pending_episode(episode1, DEFAULT_JOURNAL, base_time.replace(day=1))
    add_pending_episode(episode2, DEFAULT_JOURNAL, base_time.replace(day=2))
    add_pending_episode(episode3, DEFAULT_JOURNAL, base_time.replace(day=3))

    set_inference_enabled(False)
    assert get_inference_enabled() is False

    count = enqueue_pending_episodes()
    assert count == 0

    set_inference_enabled(True)
    assert get_inference_enabled() is True

    with patch("backend.services.tasks.extract_nodes_task") as mock_task:
        count = enqueue_pending_episodes()
        assert count == 3
        assert mock_task.call_count == 3

        called_uuids = {call[0][0] for call in mock_task.call_args_list}
        assert called_uuids == {episode1, episode2, episode3}

    remove_pending_episode(episode1, DEFAULT_JOURNAL)
    remove_pending_episode(episode2, DEFAULT_JOURNAL)
    remove_pending_episode(episode3, DEFAULT_JOURNAL)


def test_extract_nodes_task_has_unique_deduplication():
    """Verify extract_nodes_task uses Huey's unique=True for deduplication."""
    from backend.services.tasks import extract_nodes_task

    # Huey stores unique flag in the task's settings dict
    assert extract_nodes_task.settings.get("unique") is True


def test_get_episodes_by_status_scan_consistency(falkordb_test_context, redis_client):
    """Scan-based lookup returns consistent results with episode hash state."""
    # Clean up any leftover keys from other tests
    for key in redis_client.scan_iter(match="journal:*"):
        redis_client.delete(key)

    episode1 = str(uuid4())
    episode2 = str(uuid4())
    episode3 = str(uuid4())

    set_episode_status(episode1, "pending_nodes", DEFAULT_JOURNAL)
    set_episode_status(episode2, "pending_nodes", DEFAULT_JOURNAL)
    set_episode_status(episode3, "pending_nodes", DEFAULT_JOURNAL)

    pending = get_episodes_by_status("pending_nodes")
    assert len(pending) == 3
    assert episode1 in pending
    assert episode2 in pending
    assert episode3 in pending

    remove_episode_from_queue(episode2, DEFAULT_JOURNAL)

    pending_after_remove = get_episodes_by_status("pending_nodes")
    assert len(pending_after_remove) == 2
    assert episode1 in pending_after_remove
    assert episode2 not in pending_after_remove
    assert episode3 in pending_after_remove

    remove_episode_from_queue(episode1, DEFAULT_JOURNAL)
    remove_episode_from_queue(episode3, DEFAULT_JOURNAL)


@pytest.mark.asyncio
async def test_get_episodes_by_status_ignores_suppressed_entities_set(falkordb_test_context, redis_client):
    """Suppression sets should not break or pollute status scanning."""
    # Clean up any leftover keys from other tests
    for key in redis_client.scan_iter(match="journal:*"):
        redis_client.delete(key)

    episode = str(uuid4())
    set_episode_status(episode, "pending_nodes", DEFAULT_JOURNAL)

    # Create suppression set under same journal:* prefix
    await add_suppressed_entity(DEFAULT_JOURNAL, "bob")

    pending = get_episodes_by_status("pending_nodes")

    assert pending == [episode]

    remove_episode_from_queue(episode, DEFAULT_JOURNAL)


def test_cleanup_if_no_work_checks_pending_nodes_only(falkordb_test_context, redis_client):
    """cleanup_if_no_work only needs to check pending_nodes status."""
    from backend.inference.manager import cleanup_if_no_work, MODELS
    from backend.database.redis_ops import get_episodes_by_status

    # Clean up any leftover keys from other tests
    for key in redis_client.scan_iter(match="journal:*"):
        redis_client.delete(key)
    for key in redis_client.scan_iter(match="pending:*"):
        redis_client.delete(key)

    episode1 = str(uuid4())
    set_episode_status(episode1, "pending_nodes", DEFAULT_JOURNAL)

    MODELS["llm"] = "fake_model"

    cleanup_if_no_work()
    assert MODELS["llm"] == "fake_model"

    remove_episode_from_queue(episode1, DEFAULT_JOURNAL)

    pending = get_episodes_by_status("pending_nodes")
    assert len(pending) == 0

    cleanup_if_no_work()
    assert MODELS["llm"] is None


# Unresolved Entities Queue Tests


def test_append_unresolved_entities(falkordb_test_context):
    """Append unresolved entity UUIDs to batch dedup queue."""
    from backend.database.redis_ops import (
        append_unresolved_entities,
        pop_unresolved_entities,
        get_unresolved_entities_count,
    )

    uuid1 = str(uuid4())
    uuid2 = str(uuid4())

    append_unresolved_entities(DEFAULT_JOURNAL, [uuid1, uuid2])

    count = get_unresolved_entities_count(DEFAULT_JOURNAL)
    assert count == 2

    retrieved = pop_unresolved_entities(DEFAULT_JOURNAL, count=10)
    assert len(retrieved) == 2
    assert retrieved[0] == uuid1
    assert retrieved[1] == uuid2

    # Queue should be empty now
    count_after = get_unresolved_entities_count(DEFAULT_JOURNAL)
    assert count_after == 0


def test_append_unresolved_entities_empty_list(falkordb_test_context):
    """Appending empty list is a no-op."""
    from backend.database.redis_ops import (
        append_unresolved_entities,
        get_unresolved_entities_count,
    )

    append_unresolved_entities(DEFAULT_JOURNAL, [])
    count = get_unresolved_entities_count(DEFAULT_JOURNAL)
    assert count == 0


def test_pop_unresolved_entities_respects_count(falkordb_test_context):
    """Pop respects count parameter."""
    from backend.database.redis_ops import (
        append_unresolved_entities,
        pop_unresolved_entities,
        get_unresolved_entities_count,
    )

    uuids = [str(uuid4()) for _ in range(5)]

    append_unresolved_entities(DEFAULT_JOURNAL, uuids)
    assert get_unresolved_entities_count(DEFAULT_JOURNAL) == 5

    # Pop only 2
    retrieved = pop_unresolved_entities(DEFAULT_JOURNAL, count=2)
    assert len(retrieved) == 2
    assert get_unresolved_entities_count(DEFAULT_JOURNAL) == 3

    # Pop remaining
    remaining = pop_unresolved_entities(DEFAULT_JOURNAL, count=10)
    assert len(remaining) == 3
    assert get_unresolved_entities_count(DEFAULT_JOURNAL) == 0


def test_pop_unresolved_entities_empty_queue(falkordb_test_context):
    """Pop from empty queue returns empty list."""
    from backend.database.redis_ops import pop_unresolved_entities

    retrieved = pop_unresolved_entities(DEFAULT_JOURNAL, count=10)
    assert retrieved == []


# Pending Episodes Queue (Chronologically Ordered) Tests


def test_pending_episodes_chronological_order(falkordb_test_context):
    """Pending episodes are returned in reverse chronological order (newest first)."""
    from datetime import datetime, timezone
    from backend.database.redis_ops import (
        add_pending_episode,
        get_pending_episodes,
        remove_pending_episode,
    )

    episode_old = str(uuid4())
    episode_mid = str(uuid4())
    episode_new = str(uuid4())

    # Add in non-chronological order
    time_mid = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    time_old = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    time_new = datetime(2024, 12, 31, 12, 0, 0, tzinfo=timezone.utc)

    add_pending_episode(episode_mid, DEFAULT_JOURNAL, time_mid)
    add_pending_episode(episode_old, DEFAULT_JOURNAL, time_old)
    add_pending_episode(episode_new, DEFAULT_JOURNAL, time_new)

    # Should return newest first
    pending = get_pending_episodes(DEFAULT_JOURNAL)
    assert pending == [episode_new, episode_mid, episode_old]

    # Cleanup
    remove_pending_episode(episode_old, DEFAULT_JOURNAL)
    remove_pending_episode(episode_mid, DEFAULT_JOURNAL)
    remove_pending_episode(episode_new, DEFAULT_JOURNAL)


def test_remove_pending_episode(falkordb_test_context):
    """Remove episode from pending queue."""
    from datetime import datetime, timezone
    from backend.database.redis_ops import (
        add_pending_episode,
        get_pending_episodes,
        remove_pending_episode,
        get_pending_episodes_count,
    )

    episode1 = str(uuid4())
    episode2 = str(uuid4())

    time1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    time2 = datetime(2024, 1, 2, tzinfo=timezone.utc)

    add_pending_episode(episode1, DEFAULT_JOURNAL, time1)
    add_pending_episode(episode2, DEFAULT_JOURNAL, time2)
    assert get_pending_episodes_count(DEFAULT_JOURNAL) == 2

    remove_pending_episode(episode1, DEFAULT_JOURNAL)
    assert get_pending_episodes_count(DEFAULT_JOURNAL) == 1

    pending = get_pending_episodes(DEFAULT_JOURNAL)
    assert pending == [episode2]

    # Cleanup
    remove_pending_episode(episode2, DEFAULT_JOURNAL)


def test_get_pending_episodes_count(falkordb_test_context):
    """Get count of pending episodes."""
    from datetime import datetime, timezone
    from backend.database.redis_ops import (
        add_pending_episode,
        get_pending_episodes_count,
        remove_pending_episode,
    )

    assert get_pending_episodes_count(DEFAULT_JOURNAL) == 0

    episodes = [str(uuid4()) for _ in range(3)]
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

    for i, ep in enumerate(episodes):
        add_pending_episode(ep, DEFAULT_JOURNAL, base_time.replace(day=i + 1))

    assert get_pending_episodes_count(DEFAULT_JOURNAL) == 3

    # Cleanup
    for ep in episodes:
        remove_pending_episode(ep, DEFAULT_JOURNAL)


def test_get_journals_with_pending_episodes(falkordb_test_context):
    """Find journals that have pending work."""
    from datetime import datetime, timezone
    from backend.database.redis_ops import (
        add_pending_episode,
        get_journals_with_pending_episodes,
        remove_pending_episode,
    )

    episode1 = str(uuid4())
    episode2 = str(uuid4())
    time = datetime(2024, 1, 1, tzinfo=timezone.utc)

    # Add to default journal
    add_pending_episode(episode1, DEFAULT_JOURNAL, time)
    # Add to another journal
    add_pending_episode(episode2, "work", time)

    journals = get_journals_with_pending_episodes()
    assert DEFAULT_JOURNAL in journals
    assert "work" in journals

    # Remove from work journal - should no longer appear
    remove_pending_episode(episode2, "work")
    journals_after = get_journals_with_pending_episodes()
    assert DEFAULT_JOURNAL in journals_after
    assert "work" not in journals_after

    # Cleanup
    remove_pending_episode(episode1, DEFAULT_JOURNAL)


@pytest.mark.asyncio
async def test_add_journal_entry_uses_pending_queue(isolated_graph):
    """New journal entries are added to chronologically-sorted pending queue."""
    from datetime import datetime, timezone
    from backend.database.redis_ops import (
        get_pending_episodes,
        get_pending_episodes_count,
        remove_pending_episode,
    )

    # Add entries with different reference times (out of order)
    time_new = datetime(2024, 12, 1, tzinfo=timezone.utc)
    time_old = datetime(2024, 1, 1, tzinfo=timezone.utc)

    uuid_new = await add_journal_entry(
        "Entry from December",
        reference_time=time_new,
    )
    uuid_old = await add_journal_entry(
        "Entry from January",
        reference_time=time_old,
    )

    # Should be sorted newest first
    pending = get_pending_episodes(DEFAULT_JOURNAL)
    assert uuid_old in pending
    assert uuid_new in pending
    assert pending.index(uuid_new) < pending.index(uuid_old)

    # Cleanup
    remove_pending_episode(uuid_old, DEFAULT_JOURNAL)
    remove_pending_episode(uuid_new, DEFAULT_JOURNAL)


def test_get_active_episode_uuid_returns_none_when_no_active(redis_client):
    """get_active_episode_uuid returns None when no episode is processing."""
    from backend.database.redis_ops import get_active_episode_uuid

    # Clear any existing active episode
    redis_client.delete("task:active_episode")

    result = get_active_episode_uuid()
    assert result is None


def test_get_active_episode_uuid_returns_uuid_when_active(redis_client):
    """get_active_episode_uuid returns the UUID of the active episode."""
    from backend.database.redis_ops import get_active_episode_uuid

    test_uuid = str(uuid4())
    redis_client.hset("task:active_episode", mapping={"uuid": test_uuid, "name": "Test Episode"})

    result = get_active_episode_uuid()
    assert result == test_uuid

    # Cleanup
    redis_client.delete("task:active_episode")


def test_get_processing_status_returns_combined_data(redis_client):
    """get_processing_status returns both active UUID and pending count."""
    from datetime import datetime, timezone
    from backend.database.redis_ops import (
        add_pending_episode,
        get_processing_status,
        remove_pending_episode,
    )

    # Set up test data
    test_uuid = str(uuid4())
    pending_uuid = str(uuid4())
    redis_client.hset("task:active_episode", mapping={"uuid": test_uuid, "name": "Test"})
    add_pending_episode(pending_uuid, DEFAULT_JOURNAL, datetime.now(timezone.utc))

    result = get_processing_status(DEFAULT_JOURNAL)

    assert result["active_uuid"] == test_uuid
    assert result["pending_count"] >= 1

    # Cleanup
    redis_client.delete("task:active_episode")
    remove_pending_episode(pending_uuid, DEFAULT_JOURNAL)


def test_get_processing_status_idle_state(redis_client):
    """get_processing_status returns None and 0 when idle."""
    from backend.database.redis_ops import get_processing_status

    # Clear any existing state
    redis_client.delete("task:active_episode")
    redis_client.delete(f"pending:nodes:{DEFAULT_JOURNAL}")

    result = get_processing_status(DEFAULT_JOURNAL)

    assert result["active_uuid"] is None
    assert result["pending_count"] == 0
