"""Redis operations layer tests for backend (Redis API access via FalkorDB Lite)."""

from __future__ import annotations

import pytest

import backend.database as db_utils


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
