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

### Redis Cache Structure (updated)

Journal hash contains:
- Existing fields: `episode_uuid`, `content`, `status`, etc.
- **New field** `nodes`: JSON string `[{"uuid": "entity-uuid", "name": "Sarah", "type": "Person"}, ...]`
- No TTL - persists indefinitely until entry updated/deleted

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

1. **Redis Cache Instead of DB Polling**
   - Changed from polling database to polling Redis `nodes` field
   - Entity data written to cache AFTER successful DB write by `extract_nodes` task
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

## Future Enhancements

- Multi-select deletion (Space to select, d to delete all selected)
- View entity details (summaries, attributes when implemented)
- View edges between entities
- Global entity management view (separate from per-journal view)
- Retry button for failed extraction jobs
