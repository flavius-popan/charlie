# Plan: Fast Home Screen (v3)

## Problem
Home screen is slow because `get_all_episodes()` fetches full content for every episode when only name/date are displayed.

## Solution Overview
1. **Hybrid query** - FalkorDB for metadata + Redis batch fetch for previews
2. **Textual messages for reactivity** - No polling; screens post messages when data changes
3. **Preview in existing Redis hash** - Smart preview generation, stored alongside status/nodes
4. **Auto-title setting** - Toggle between smart previews and classic date-based names

---

## Part 1: Preview Generation Utility

**File:** `backend/utils.py` (new file, shared by app and importers)

```python
def generate_preview(content: str, max_len: int = 80) -> str:
    """Generate preview: markdown header if at top, else first sentence, else truncated."""
    head = content[:200].lstrip()

    # Markdown header at very top?
    if head.startswith("# "):
        newline = head.find("\n")
        end = newline if newline != -1 else len(head)
        return head[2:end].strip()[:max_len]

    # First sentence
    text = head.replace("\n", " ").strip()
    for punct in ".!?":
        pos = text.find(punct)
        if 0 < pos < max_len:
            return text[:pos + 1]

    # Fallback: truncate
    return text[:max_len]
```

O(1) regardless of content size - only processes first 200 chars.

---

## Part 2: Store Preview in Redis Hash

**File:** `backend/database/redis_ops.py`

```python
def set_episode_preview(episode_uuid: str, preview: str, journal: str = DEFAULT_JOURNAL):
    """Store preview in episode's Redis hash (synchronous, call via asyncio.to_thread)."""
    cache_key = f"journal:{journal}:{episode_uuid}"
    with redis_ops() as r:
        r.hset(cache_key, "preview", preview)


def get_auto_title_enabled() -> bool:
    """Check if auto-title is enabled. Default: True (key absence = enabled)."""
    with redis_ops() as r:
        return not r.exists("app:no_auto_title")


def batch_get_previews(uuids: list[str], journal: str = DEFAULT_JOURNAL) -> list[str | None]:
    """Batch fetch previews from Redis hashes. Returns list parallel to input uuids."""
    if not uuids:
        return []
    with redis_ops() as r:
        pipe = r.pipeline()
        for uuid in uuids:
            cache_key = f"journal:{journal}:{uuid}"
            pipe.hget(cache_key, "preview")
        results = pipe.execute()
    return [r.decode() if r else None for r in results]
```

**File:** `backend/__init__.py` (in `add_journal_entry`, after `set_episode_status` call at line 178)

```python
preview = generate_preview(content)
await asyncio.to_thread(set_episode_preview, episode_uuid, preview, journal)
```

**File:** `backend/database/persistence.py` (in `update_episode`, after the existing `_invalidate_cache` block around line 371)

```python
if content_changed:
    preview = generate_preview(new_content)
    await asyncio.to_thread(set_episode_preview, episode_uuid, preview, journal)
```

Note: All Redis calls use `asyncio.to_thread()` to avoid blocking the UI thread.

---

## Part 3: Hybrid Query (FalkorDB + Redis)

**File:** `backend/database/queries.py`

```python
async def get_episodes_for_list(journal: str = DEFAULT_JOURNAL, limit: int = 100) -> list[dict]:
    """Lightweight query for home screen - metadata from FalkorDB, previews from Redis."""
    driver = get_driver(journal)
    records, _, _ = await driver.execute_query(
        """
        MATCH (e:Episodic)
        WHERE e.group_id = $group_id
        RETURN e.uuid AS uuid, e.name AS name, e.valid_at AS valid_at
        ORDER BY e.valid_at DESC
        LIMIT $limit
        """,
        group_id=journal,
        limit=limit,
    )
    episodes = [dict(r) for r in records]

    # Batch fetch previews from Redis (single pipeline call)
    if get_auto_title_enabled():
        previews = await asyncio.to_thread(
            batch_get_previews, [ep["uuid"] for ep in episodes], journal
        )
        for ep, preview in zip(episodes, previews):
            ep["preview"] = preview

    return episodes
```

---

## Part 4: Home Screen Display Logic

**File:** `frontend/screens/home_screen.py`

Change `load_episodes` to use `get_episodes_for_list` instead of `get_all_episodes`.

Display format respects auto-title setting:
```python
def get_display_title(episode: dict) -> str:
    """Return preview if available and auto-title enabled, else name."""
    return episode.get("preview") or episode["name"]

# In list rendering:
f"{valid_at:%Y-%m-%d} - {get_display_title(episode)}"
```

**Behavior:**
- `app:no_auto_title` key absent (default): shows smart preview, falls back to name
- `app:no_auto_title` key present: shows classic date-based name
- Old entries without cached preview: gracefully falls back to name

---

## Part 5: Textual Messages for Reactivity

**File:** `frontend/messages.py` (new or add to existing)

```python
from textual.message import Message

class EpisodeListChanged(Message):
    """Posted when episode list should refresh."""
    pass
```

**File:** `frontend/screens/edit_screen.py` (after successful save)

```python
self.app.post_message(EpisodeListChanged())
```

**File:** `frontend/screens/home_screen.py`

```python
def on_episode_list_changed(self, event: EpisodeListChanged) -> None:
    self.load_episodes()
```

---

## Part 6: Remove extract_title

**File:** `frontend/utils.py`
- Delete `extract_title` function
- Update `get_display_title` to use preview-or-name pattern (or remove if inlined)

**File:** `frontend/__init__.py`
- Remove `extract_title` from exports

**File:** `frontend/screens/edit_screen.py`
- Remove `extract_title` import and usage

**Tests:** Remove extract_title tests from `test_charlie_utils.py` and `test_charlie.py`

---

## Blocking Call Analysis

| Operation | Location | Blocking? | Mitigation |
|-----------|----------|-----------|------------|
| `generate_preview` | backend/utils | No | Pure string ops, microseconds |
| `set_episode_preview` | redis_ops | Yes (sync) | Wrapped with `asyncio.to_thread()` |
| `get_auto_title_enabled` | redis_ops | Yes (sync) | Called inside `asyncio.to_thread()` block |
| `batch_get_previews` | redis_ops | Yes (sync) | Wrapped with `asyncio.to_thread()` |
| FalkorDB query | queries | No | Native async via `driver.execute_query` |
| `EpisodeListChanged` | frontend | No | Textual message system |

All blocking operations are offloaded to thread pool, matching existing codebase patterns.

---

## Files to Modify

| File | Change |
|------|--------|
| `backend/utils.py` | New file with `generate_preview` |
| `backend/__init__.py` | Call `generate_preview` + `set_episode_preview` on save |
| `backend/database/redis_ops.py` | Add `set_episode_preview`, `get_auto_title_enabled`, `batch_get_previews` |
| `backend/database/persistence.py` | Call preview update on content change |
| `backend/database/queries.py` | Add `get_episodes_for_list` (hybrid FalkorDB + Redis) |
| `frontend/screens/home_screen.py` | Use new query, handle `EpisodeListChanged`, update display logic |
| `frontend/screens/edit_screen.py` | Post `EpisodeListChanged` after save, remove extract_title |
| `frontend/utils.py` | Remove `extract_title`, update/remove `get_display_title` |
| `frontend/messages.py` | Add `EpisodeListChanged` message class |

## Implementation Order

1. `backend/utils.py` - preview generation (no dependencies)
2. `backend/database/redis_ops.py` - add all three helpers
3. `backend/database/queries.py` - hybrid query
4. `backend/__init__.py` + `persistence.py` - wire up preview on save
5. `frontend/messages.py` - message class
6. `frontend/screens/home_screen.py` - use new query + message handler + display logic
7. `frontend/screens/edit_screen.py` - post message, remove extract_title
8. `frontend/utils.py` - cleanup
9. Tests - remove old, add new for preview generation and hybrid query
