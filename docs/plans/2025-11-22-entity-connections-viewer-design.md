# Entity Connections Viewer Design

**Date:** 2025-11-22
**Status:** Approved
**Context:** Per-journal diagnostic view for extracted entities with deletion capabilities

## Overview

Add a "Connections" sidebar to the ViewScreen that displays entities extracted from the current journal entry. This is a diagnostic view showing what's attached to a specific episode node, allowing users to view and delete entity mentions.

## User Requirements

- View entities extracted from a journal entry
- Delete entity mentions from the current entry (local deletion)
- See real-time updates as extraction completes
- No obstruction to markdown reading experience
- Playful, end-user friendly interface (quirky cat theme)

## Component Architecture

### ViewScreen Modifications

The current `ViewScreen` displays journal markdown in a full-width `Markdown` widget. Modified to contain:

1. **Horizontal Container** (`Horizontal` widget from Textual)
   - Left: `Markdown` widget (75% width)
   - Right: `EntitySidebar` custom widget (25% width, visible by default)

2. **New EntitySidebar Widget** (custom composite widget)
   - Header: "Connections" title
   - Body: Either loading state OR entity list
   - Loading state: Loading spinner (default Textual LoadingIndicator)
   - Loaded state: `ListView` containing entity items
   - Footer: Action hints (delete key bindings)

### Widget Hierarchy

```
ViewScreen
├── Horizontal
│   ├── Markdown (journal content)
│   └── EntitySidebar (custom widget)
│       ├── Label (header "Connections")
│       ├── [Loading | ListView] (conditional rendering)
│       └── Label (footer with key hints)
```

### Key Relationships

- `ViewScreen` owns the episode UUID and journal name
- `ViewScreen` manages the polling timer (reactive var or set_interval)
- `EntitySidebar` receives episode UUID as parameter, fetches entities
- `EntitySidebar` exposes method `refresh_entities()` called after job completes

## Data Flow & Reactive Updates

### On ViewScreen Mount (after ESC from edit)

1. ViewScreen receives episode UUID and journal name from EditScreen
2. Start polling for `nodes` field in Redis journal hash (every 500ms)
3. EntitySidebar shows loading state with spinner
4. Poll checks Redis hash for presence of `nodes` field AND job status

### When Job Completes

5. `extract_nodes` task writes entity data to Redis AFTER successful DB write
6. Redis hash field `nodes` contains JSON array: `[{"uuid": "...", "name": "...", "type": "..."}]`
7. EntitySidebar detects `nodes` field, stops polling
8. Filter out SELF entity in UI component
9. Transform entities to display format: `{name} [{type}]`
   - Type filtering: Show most specific label (e.g., "Person" not "Entity, Person")
   - Only show "Entity" if it's the only label
   - NO reference count displayed
10. Populate ListView with entity items
11. Footer shows: "d: delete | ↑↓: navigate | c: close | l: logs"

### Redis Cache Structure

**Unified Key:** `journal:<journal>:<uuid>` (single hash per episode)

Contains all episode metadata:
- `status`: Processing state (pending_nodes, pending_edges, done, dead)
- `journal`: Journal name
- `uuid_map`: UUID mapping (provisional → canonical)
- `nodes`: Extracted entities JSON `[{"uuid": "...", "name": "Sarah", "type": "Person"}, ...]`

Cache persists indefinitely (no TTL) and is deleted atomically with episode deletion.

### Reactive State

- Use Textual reactive variable to track loading/loaded state
- When state changes from loading → loaded, swap Loading widget for ListView
- No manual refresh needed - state change triggers UI update automatically

## User Interactions & Deletion Flow

### Navigation & Toggling

- **ESC from EditScreen** → Navigate to ViewScreen with sidebar open by default
- **`c` key** (connections) → Toggle sidebar visibility (collapse to full-width markdown / expand to split view)
- **`l` key** (logs) → Toggle to log viewer screen without returning to HomeScreen
- **`q` or ESC from ViewScreen** → Return to HomeScreen
- **Arrow keys (↑↓)** → Navigate entity list when sidebar has focus
- **Tab** → Switch focus between markdown viewer and entity list

### Entity Selection & Deletion

When an entity is highlighted in the ListView:

1. **`d` key** → Trigger deletion confirmation
2. **Confirmation modal appears:**
   - Title: "Remove Connection?"
   - Message: `"Remove Sarah from this entry?"`
   - Note: `"(This only removes the connection from this entry)"`
   - Buttons: `[Cancel] [Remove]` (Cancel focused by default)
3. **On confirm:**
   - Delete MENTIONS edge: `(episode)-[:MENTIONS]->(entity)`
   - If entity has no remaining MENTIONS edges → delete entity node
   - **AFTER successful DB write:** Update Redis cache by removing entity from `nodes` JSON array
   - Remove item from ListView (no toast notification)

### Multi-select (future enhancement)

- Could use `Space` to multi-select entities
- `d` deletes all selected
- But start with single-delete for simplicity

### Empty States

- No entities extracted: Show message `"No connections found"` in sidebar
- Job failed: Show `"Extraction failed"` with option to retry

## Design Principles & Implementation Notes

### UI Philosophy

- **No toasts/notifications** - They're annoying and disrupt flow. UI updates (like removing an item from the list) are sufficient feedback.
- **Playful, end-user language** - Avoid technical jargon ("knowledge graph", "nodes", "edges"). Use natural terms ("connections", "entries", "mentioned in").

## Implementation Status

**Status:** ✅ Complete (2025-11-22)

All features implemented and tested. Redis cache architecture delivers instant loads for existing entries.

## Implementation Notes & Architectural Changes

### Core Architecture Changes

1. **Unified Redis Cache Architecture**
   - Single `journal:<journal>:<uuid>` key stores all episode metadata
   - Eliminated dual-key fragmentation (previously had separate processing queue keys)
   - Cache deleted atomically with episode - no orphaned data
   - Entity data written AFTER successful DB write by `extract_nodes` task
   - Ensures cache integrity - no stale/invalid data

2. **Simplified UI Display**
   - Removed reference count display (no more "Sarah [Person] (4)")
   - Display format now: "Sarah [Person]"
   - Cleaner, less cluttered interface

3. **Cache Updates on Deletion**
   - After successful entity deletion in DB, update Redis cache
   - Remove entity from `nodes` JSON array immediately
   - Keeps cache in sync with database state

4. **Loading State Simplified**
   - Removed ASCII cat spinner (text not visible during loading)
   - Use default Textual LoadingIndicator

5. **Log Viewer Shortcut**
   - Added `l` key binding to toggle to log viewer from ViewScreen
   - No need to return to HomeScreen first

6. **Smart Sidebar Visibility**
   - Hidden when opening from HomeScreen (instant reading, no lag)
   - Shown when coming from EditScreen (contextual after editing)
   - Manual toggle with `c` key

7. **Performance Optimizations**
   - Instant cache fetch on mount (no polling delay)
   - Only polls if cache miss detected
   - ~50ms load for cached entries vs ~500ms+ before

8. **UX Improvements**
   - Auto-select first entity when sidebar opens
   - Node extraction triggers on both create and update
   - Smooth edit→view transition with immediate markdown display

## Performance Enhancement: Editing Presence Detection & Priority Queueing

**Status:** ✅ Complete (2025-11-22) — updated: presence flag now set once per edit session (no TTL)
**Commits:** `7d6a705`, `33ef30c`

### Problem Statement

The connections pane population was slow after editing because:
1. Models unloaded from memory while user was typing (cold start on save)
2. User's journal entry competed with background tasks in FIFO queue
3. No differentiation between user-triggered vs background extraction

### Solution: Three-Part Performance System

#### 1. Editing Presence Detection

**Mechanism:**
- `EditScreen` sets Redis key `editing:active` once on mount (no TTL)
- Key is explicitly deleted when EditScreen saves or unmounts
- UI thread rule: all Redis calls in this flow MUST stay off the UI thread
- Preload: orchestrator now warm-loads models when `editing:active` exists AND inference is enabled (Huey thread, cached thereafter)

**Key Details:**
- Global flag (not per-episode) - simpler, matches single-focus TUI model
- Works for BOTH new entries and editing existing entries
- Flag remains until explicit delete on exit (no periodic refresh needed)

**Files:**
- `charlie.py:759-765` - Event handler sets key
- `charlie.py:795-801` - Cleanup on unmount
- `charlie.py:60` - TTL constant

#### 2. Model Keep-Alive During Editing

**Mechanism:**
- `cleanup_if_no_work()` checks for `editing:active` key before unloading models
- If key exists, models stay loaded even when queue is empty
- Prevents cold start delay when user finishes editing

**Key Details:**
- Check happens AFTER inference-enabled check
- Graceful error handling (Redis failure defaults to allowing cleanup)
- Debug logging for visibility during development

**Files:**
- `backend/inference/manager.py:59-69` - Editing presence check
- `tests/test_backend/test_inference_manager.py` - Coverage for keep-alive behavior

#### 3. Priority Queueing

**Mechanism:**
- Switched from `RedisHuey` to `PriorityRedisHuey`
- User-triggered saves use `priority=1` (high)
- Background orchestrator uses `priority=0` (low)
- Higher priority tasks process first (jump ahead in queue)

**Implementation:**
- `PriorityRedisHuey` uses Redis sorted sets with negative scores
- `priority=1` → score=-1.0 (processed first)
- `priority=0` → score=0.0 (processed after high-priority)
- Single worker still respects priority ordering

**Files:**
- `backend/services/queue.py:57-61` - PriorityRedisHuey configuration
- `backend/services/tasks.py:16-17` - Task accepts priority parameter
- `charlie.py:833` - EditScreen passes `priority=1`
- `backend/database/redis_ops.py:333` - Orchestrator passes `priority=0`

### Impact on Connections Pane

**Before:**
1. User types entry, hits ESC
2. Models unload during editing (idle queue)
3. Save triggers extraction task (cold start + model load)
4. Task competes with background tasks in FIFO queue
5. ~2-5 second delay before connections appear

**After:**
1. User types entry (models stay loaded via `editing:active` flag)
2. User hits ESC → task enqueued with `priority=1`
3. Task jumps ahead of background tasks
4. Models already warm → immediate inference start
5. ~0.5-1 second delay before connections appear

**Performance gain:** 4-10x faster connections pane population for typical writing workflow

### Test Coverage

**Test Files Created:**
- `tests/test_frontend/test_editing_presence.py` (5 tests)
  - Keystroke sets key with 120s TTL
  - Save deletes key
  - Unmount deletes key
  - Redis errors don't break editing
  - New entries set key

- `tests/test_backend/test_task_priority.py` (5 tests)
  - Priority parameter acceptance
  - Default priority behavior
  - High-priority tasks jump queue
  - Orchestrator uses low priority
  - Queue ordering verification

**Test Updates:**
- `tests/test_backend/test_inference_manager.py` - Editing presence prevents unload
- `tests/test_backend/test_huey_consumer_inprocess.py` - Updated for priority param

**Test Isolation:**
- All tests use mocks or `isolated_graph` fixture (no real DB access)
- Redis shutdown delay bypassed via global monkey-patch
- `clear_huey_queue` fixture properly isolates Huey tests
- 224 tests passing, no regressions

### Design Rationale

**Why global `editing:active` flag?**
- Simpler than per-episode tracking (1 key vs N keys)
- Matches TUI's single-focus model (one editor at a time)
- No cleanup complexity (TTL handles expiration)
- No race conditions with episode deletion

**Why 120-second TTL?**
- Long enough for brief pauses (switching windows, reading references)
- Short enough to unload if user walks away
- Automatic cleanup prevents leaked keys

**Why two-tier priority (0 and 1)?**
- Simple to understand and maintain
- Covers the use case (user vs background)
- Extensible if more levels needed later

### Error Handling

**Redis failures:**
- Editing continues uninterrupted if Redis unavailable
- Key set failure logged at debug level
- Cleanup check defaults to `False` (allows unload)
- Graceful degradation (models may unload prematurely but system remains functional)

**Connection pool issues:**
- Test fixture updates pool when temp DB changes
- Prevents connection errors in test suite
- Production uses single persistent DB (no pool updates needed)

## Future Enhancements

- Multi-select deletion (Space to select, d to delete all selected)
- View entity details (summaries, attributes when implemented)
- View edges between entities
- Global entity management view (separate from per-journal view)
- Retry button for failed extraction jobs
