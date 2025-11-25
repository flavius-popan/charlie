# Plan: Cache-First Home Screen (v2)

## Problems to fix
- UI stutter from on-thread title/preview parsing (`extract_title` in `frontend/utils.py:4-52`).
- Home load hits DB for full content (`get_all_episodes`) even though only name/date/preview are displayed.
- Cache is key-scanned per load; no ordering structure for fast slices.
- `_enqueue_extraction_task` does synchronous Redis check on the UI thread.

## Current journal hash schema (works with this plan)
- Stored in `journal:{journal}:{uuid}` hashes:
  - `status`, `journal`, `uuid_map` (set in `backend/database/redis_ops.py:128-148`).
  - `nodes` JSON + `mentions_edges` JSON (set in `backend/graph/extract_nodes.py:461-474`).
- No conflicts with adding `preview`, `valid_at`, and cached `name`. Consumers ignore unknown fields, so additive fields are safe.

## Decisions
- Preview length 80 chars; whitespace/newlines collapsed (`content.replace("\n", " ")[:80]`).
- Use a single sorted set `journal:{journal}:by_valid_at` (member=uuid, score=epoch seconds). No secondary zsets.
- Always prefer episode `name` (title) for display; no `"Untitled"` fallback.
- Cache writes are best-effort (log+continue); never fail saves on Redis errors.
- Startup still calls `ensure_database_ready(DEFAULT_JOURNAL)`; Falkor and Redis share the same process.
- Home screen reactivity: 3s poller worker checks zset cardinality/top score; refreshes list on change.

## Part 1: Remove title extraction
- `frontend/utils.py`
  - Delete `extract_title`.
  - Simplify `get_display_title` to: prefer `episode["name"]`, else `episode.get("preview", "")[:80]`.
- `frontend/__init__.py`
  - Drop `extract_title` import/export.
- `frontend/screens/edit_screen.py`
  - Remove all uses of `extract_title`; call `update_episode` without name overrides.
  - Keep `_enqueue_extraction_task` but add internal `get_inference_enabled()` guard (cheap) so callers can’t forget.
- Tests: remove `tests/test_frontend/test_charlie_utils.py`; drop title-extraction test in `tests/test_frontend/test_charlie.py` (around lines 1301-1340).

## Part 2: Cache write-through helpers
- `backend/database/redis_ops.py`
  - Add `cache_episode_display_data(episode_uuid, content, valid_at, name, journal)`:
    - Build preview (80 chars), ISO `valid_at`, HSET `preview`, `valid_at`, `name`.
    - ZADD `journal:{journal}:by_valid_at` with score = `valid_at.timestamp()`.
    - Wrap in try/except; log at debug, never raise.
  - Add `delete_episode_display_cache(episode_uuid, journal)`:
    - DEL hash, ZREM from zset; best-effort.
  - Add `get_episode_display_batch(journal, limit=400)`:
    - ZREVRANGE by_valid_at 0 limit-1 WITHSCORES.
    - Pipeline HMGET `name`, `preview`, `valid_at`.
    - Return list of dicts sorted by zset order; parse `valid_at` to datetime.
- `backend/__init__.py:add_journal_entry` (after `set_episode_status`, ~line 170s)
  - Fire-and-forget `asyncio.to_thread(cache_episode_display_data, episode_uuid, content, reference_time, title, journal)`.
- `backend/database/persistence.py:update_episode` (around line 299+)
  - After save: if `content_changed` or `valid_at` changed, fire-and-forget `cache_episode_display_data(..., episode.content, episode.valid_at, episode.name, journal)`.
  - Keep cache update best-effort; do not block return.
- `backend/database/persistence.py:delete_episode` (after graph delete)
  - Call `delete_episode_display_cache` best-effort.

## Part 3: Home screen cache-first load
- `backend/database/queries.py`
  - Add `get_all_episodes_recent(days=30, journal=DEFAULT_JOURNAL)` to limit DB fallback.
- `frontend/screens/home_screen.py`
  - Imports: `get_episode_display_batch`, `cache_episode_display_data`, `delete_episode_display_cache` as needed.
  - `load_episodes(cache_only: bool = False)`:
    - Try cache: `episodes = await asyncio.to_thread(get_episode_display_batch, DEFAULT_JOURNAL)`.
    - If cache empty and not `cache_only`: fetch recent via `get_all_episodes_recent(30)`, hydrate cache with `cache_episode_display_data` (to_thread), then render.
    - Render label as `f"{valid_at:%Y-%m-%d} - {episode['name']}"`; preview can be shown in secondary UI later.
    - Keep current focus/empty-state recomposition logic (lines 53-142).
  - Reactivity: start a 3s worker on `on_mount` that reads `ZCARD` and top score; if either changes since last check, call `load_episodes(cache_only=True)`. Stop worker on screen exit. Add a short comment explaining *why* (keeps list in sync with background imports/inference).

## Part 4: Tests (targeted)
- Cache hit: seeded zset+hash returns ordered list, no DB call.
- Cache miss: empty zset triggers recent-DB fetch, hydrates cache, returns results.
- Save/update: Redis error is swallowed (mock) and save still returns success.
- Poller: changing zset top score triggers a refresh (can mock worker loop once).

## Verification checklist
- Run full pytest suite.
- Manual: launch app → list populates instantly from cache; after import/inference, entries appear within 3s without restart.
- Manual: delete entry → disappears from home within 3s, no errors.

## Commenting note (per AGENTS.md)
- Add only high-value “why” comments, e.g., a one-liner above the home poller explaining it refreshes the list as background work completes. Avoid noisy or “what” comments elsewhere.
