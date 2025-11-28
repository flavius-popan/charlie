# Home Screen Redesign

## Overview

Redesign the home screen from a basic date-preview list into a three-pane live dashboard that shows entry connections, temporal context, and processing status simultaneously.

## Goals

- Quick journal access for daily writing (primary use case)
- Show reactivity and liveliness of background enrichment
- Enable temporal exploration ("What was I doing in November?")
- Clean, uncluttered UI that doesn't shift or feel busy
- See entry-level AND period-level context without leaving home screen

## Layout

Three-pane layout with titled borders (fieldset style):

```
┌─ Entries ─────────────┬─ Connections ─────────────┐
│                       │  Sarah · Coffee · Portland │  ← ↑↓ changes this
│    ── This Week ───   │  (selected entry's links)  │
│  ● Wed Nov 27 · Morn  │                            │
│    Tue Nov 26 · Coff  ├─ This Week ────────────────┤
│    Mon Nov 25 · Port  │  4 entries · 12 connections│  ← ←→ changes this
│                       │  Sarah · Mike · Portland   │
│    ── Last Week ────  │  Coffee · Planning         │
│    Sun Nov 24 · Quiet ├─ Processing ───────────────┤
│    Fri Nov 22 · Meet  │  ● Extracting Nov 27...    │  ← appears when active
│                       │    + Sarah + Coffee        │
│                       │  Queue: 3 remaining        │
└───────────────────────┴────────────────────────────┘
```

**Proportions:**
- Left pane: 2fr (entries are primary)
- Right pane: 1fr (context stack)

## Titled Borders

Use Textual's `border-title-align` for clean fieldset-style panels:

```css
#entries-pane {
    border: solid $accent;
    border-title-align: left;
}
```

Each section has a title inset in the border line, similar to Harlequin's UI.

## Entry List (Left Pane)

### Temporal Grouping

Entries grouped by time period with divider headers:

```
   ── This Week ──────────
   Wed Nov 27 · Morning thoughts about...
   Tue Nov 26 · Coffee with Sarah at...
   Mon Nov 25 · Portland trip planning...

   ── Last Week ──────────
   Sun Nov 24 · Quiet day at home...
   Fri Nov 22 · Meeting went well...

   ── November ───────────
   Wed Nov 13 · Travel prep...
```

**Grouping logic:**
- "This Week" / "Last Week" for recent entries (week starts Monday, local timezone)
- Month name for older entries
- For very old imports (2+ years), may show just year if months are sparse

**Entry format:** `Day Mon DD · Preview text...`
- Day of week immediately visible
- Dot separator (cleaner than hyphen)
- Preview truncates gracefully

### Processing Indicator

Left margin gutter (2-char fixed width) shows processing status:

```
   Wed Nov 27 · Morning...      ← empty (idle)
 ● Tue Nov 26 · Coffee...       ← dot (currently processing)
   Mon Nov 25 · Portland...     ← empty (complete or pending)
```

- Fixed width prevents layout shift
- Indicator only on actively processing entry
- Subtle visual feedback that the system is alive
- Indicator moves down the list (newest → oldest) as processing progresses

### Keyboard Navigation

- `↑↓` - Select entries (updates Connections pane)
- `←→` - Jump time periods (updates Temporal pane)
- `n` - New entry
- `space` or `enter` - View selected entry
- `d` - Delete entry

## Right Pane: Context Stack

Three stacked sections with adaptive sizing based on processing state.

### Connections Pane (Top)

Shows entities connected to the **selected entry**:

```
┌─ Connections ──────────────┐
│  Sarah · Coffee · Portland │  ← clickable entities
│  Morning pages             │
└────────────────────────────┘
```

- Updates when `↑↓` changes selected entry
- Entities are clickable (navigate to entity view - future)
- Shows "No connections" or "Pending..." for unprocessed entries

### Temporal Pane (Middle)

Shows aggregate entities for the **selected time period**:

```
┌─ This Week ─────────────────┐
│  Sarah · Mike · Portland    │  ← top entities for period
│  Coffee · Planning          │
└─────────────────────────────┘
```

- Updates when `←→` changes time period
- Most connected entities (most connected nodes) for that period (clickable)

### Processing Pane (Bottom)

Shows system activity when queue >= 1:

```
┌─ Processing ───────────────┐
│  ● Loading instruct...     │  ← LLM status (if available)
│                            │
│  Nov 27 extracting:        │
│    + Sarah + Coffee        │  ← live extraction feed (Phase 4)
│                            │
│  Queue: 3 remaining        │
│  ✓ 12 completed            │
└────────────────────────────┘
```

- Appears when queue >= 1 (any pending work shows the pane)
- LLM status is nice-to-have (may not be available in current Redis infrastructure)
- Live entity feed deferred to Phase 4 - initial version shows episode name only
- Queue count and completion stats

### Adaptive Sizing

**When idle (no processing):**
```
┌─ Connections ──┐  1/3 height
├─ This Week ────┤
│                │  2/3 height
│                │
└────────────────┘
```

**When processing:**
```
┌─ Connections ──┐  1/3 height
├─ This Week ────┤  1/3 height
├─ Processing ───┤  1/3 height
└────────────────┘
```

## Time Navigation

Left/right arrows navigate time periods:

```
→───────────────────────────────────────────────────←
This Week ← Last Week ← November ← October ← September...
```

**Behavior:**
- `←` goes back in time (older)
- `→` goes forward in time (newer)
- Navigation bounded by content - stops at oldest/newest entry periods

**Entry list response:**
- Scrolls to show entries from selected period
- First entry of that period gets focus
- Active period header emphasized
- If user manually scrolls after period jump, don't fight them - period selection stays but scroll position is user-controlled

**Temporal pane response:**
- Title updates to show period name
- Queries entities from that time window
- Shows stats and top entities for that period

**Connections pane:**
- Unaffected by ←→ (stays on selected entry)

### Handling Old/Imported Journals

When importing journals from years past (e.g., 2002-2004):

**Anchor to most recent content:**
- On load, detect the newest entry date
- Start view at that period (e.g., "December 2004"), not "This Week"
- Only show "This Week" / "Last Week" labels if entries actually exist there

**Bound navigation to content range:**
- ←→ only navigates periods that contain entries
- Skip empty periods entirely (no traversing 20 years of empties)
- `→` stops at newest content, `←` stops at oldest content

**Period labels remain absolute:**
- Use actual dates ("December 2004", "March 2003")
- Clear and unambiguous regardless of import age
- User always knows where they are in time

This ensures the home screen works naturally for both fresh daily journals and bulk imports of historical data.

## Processing Order

**Newest-first processing:**
- New entries and recent imports process first
- User sees immediate value in "This Week"
- Temporal pane populates right away
- Processing indicator moves down the list (newest → oldest)

This differs from oldest-first (better for dedup theory) but provides better UX. Batch LLM dedup handles any missed matches.

**Change required:** Flip `pending:nodes:{journal}` query to `ZREVRANGE` (descending by valid_at).

## Update Mechanism

### Polling-Based (Initial Implementation)

Use proven polling pattern from ViewScreen:

```python
async def _poll_processing_status(self):
    while True:
        status = await asyncio.to_thread(get_processing_status)
        self.active_episode_uuid = status.active_uuid
        self.queue_count = status.pending_count

        if status.pending_count == 0:
            break

        await asyncio.sleep(0.3)  # Fast during processing
```

- Poll every 0.3s during active processing
- Poll every 2s when idle (or stop polling)
- Reactive properties trigger UI updates automatically

### Push-Based (Future Enhancement)

If polling feels janky, investigate `app.call_from_thread()` for direct Huey → UI messages. Defer until polling proves insufficient.

## Implementation Phases

### Phase 0: Backend Prep

**0.1 Newest-first processing** [DONE]
- Flip `pending:nodes:{journal}` query from `ZRANGE` to `ZREVRANGE`
- Processing starts with most recent entries, works backward
- Enables realistic testing throughout development
- *Note: Will need to update tests that assume oldest-first order*

### Phase 1: Layout Foundation [DONE]

**1.1 Three-pane layout skeleton** [DONE]
- Horizontal split: entries (2fr) | context stack (1fr)
- Vertical stack on right: connections | temporal | processing
- Fieldset-style borders with titles
- *Proves: Layout proportions, border styling*
- *Must come first - establishes the canvas for all other work*

**1.2 Temporal grouping with dividers** [DONE]
- Add grouping logic to `get_home_screen()` query
- Render groups with divider headers in left pane
- Test ↑↓ navigation skipping dividers
- *Proves: Mixed ListView content works*

**1.3 Processing indicator gutter** [DONE]
- Create custom EntryListItem with 2-char left margin
- Toggle indicator via reactive property (mock data)
- *Proves: Fixed-width gutter, no layout shift*
- *Can parallelize with 1.2*

### Phase 2: Context Panes [DONE]

**2.1 Create missing queries** [DONE]
- `get_entry_entities(episode_uuid, journal)` - extract from EntitySidebar pattern
- `get_period_entities(start_date, end_date, journal)` - aggregates entities across time period
- `calculate_periods(episodes)` - utility function for period boundaries

**2.2 Connections pane** [DONE]
- Query entities for selected entry from Redis cache
- Update on ↑↓ navigation via ListView
- Shows up to 10 entities

**2.3 Temporal pane** [DONE]
- Query aggregate entities for time period
- Update on ←→ navigation via ListView
- Shows top 25 entities ranked by mention frequency

**2.4 Time navigation coordination** [DONE]
- Implement ←→ period jumping logic
- Sync entry list scroll position to selected period
- Update temporal pane title dynamically
- Preserve selected period when returning from entry view

### Phase 3: Processing Integration (Polling)

**3.1 Live indicator from Redis**
- Poll `task:active_episode` (infrastructure exists in redis_ops.py)
- Update `active_episode_uuid` reactive property
- Gutter indicator appears on matching entry, moves as processing progresses
- *Proves: Live processing state in entry list*

**3.2 Processing pane**
- Show/hide based on queue count (>= 1)
- Display queue stats and active episode name
- Adaptive sizing (1/3 each when visible, 1/3 + 2/3 when hidden)
- *Proves: Conditional pane visibility*

**3.3 Active episode display (simplified)**
- Show currently processing episode name in processing pane
- Defer live entity feed - requires new Redis structure
- *Provides value without architectural complexity*

### Phase 4: Polish + Advanced

**4.1 Live extraction feed (optional)**
- Design Redis structure for recent extractions
- Stream entities to processing pane as extracted
- *Only if simplified 3.3 feels insufficient*

**4.2 Edge cases and error states**
- Empty periods handling
- Focus management between panes
- Loading indicators
- Error states

**4.3 Keyboard shortcuts**
- Number keys (1-9) for quick entity access
- Tune polling frequencies based on usage

## Technical Notes

### Entry List Implementation

- Use `ListView` with mixed content (divider items + entry items)
- Dividers are non-selectable, ↑↓ skips them
- Use `Rule` widget or styled `Static` for dividers
- Fixed-width left margin for processing indicator

### Context Pane Implementation

- `Vertical` container with three child containers
- Each child has border with title
- Processing pane uses `display: none` when hidden (not removed from DOM)
- Use CSS `height` transitions for smooth resize

### Data Queries Needed

- `get_home_screen()` - already exists, may need `valid_at` for grouping
- `get_entry_entities(episode_uuid)` - for Connections pane
- `get_period_entities(start_date, end_date, journal)` - for Temporal pane
- `get_processing_status()` - active episode, queue count, recent extractions

## Future Ideas (Deferred)

These could enhance the context panes when idle:

- "On this day last year" - surface old memories
- "Not mentioned recently" - people you haven't written about
- "New connections discovered" - highlight graph insights
- Changelog/tips for new users in Processing pane when idle
- Writing streaks or activity patterns

## Related Work

- Entity exploration view (separate from home screen)
- Entity timeline view (accessed by clicking entities)
- Global graph view (future)

## File References for Implementation

Exact file paths and functions to modify or reference for each phase.

### Phase 0.1: Newest-first Processing

**MODIFY: backend/database/redis_ops.py**
- `get_pending_episodes()` (line 560) - Change `zrange` to `zrevrange` on line 571
- `enqueue_pending_episodes()` (line 271) - Update comment on line 279 from "oldest-first" to "newest-first"

**MODIFY: tests/test_backend/test_redis_ops.py**
- Multiple test functions expect oldest-first order - update assertions

### Phase 1.1: Three-pane Layout

**MODIFY: frontend/screens/home_screen.py**
- `compose()` (line 59) - Replace single ListView with Horizontal container
- Currently yields: Header, Static (empty state), ListView, Footer
- Change to: Header, Horizontal(entries_pane, right_stack), Footer

**REFERENCE: frontend/screens/view_screen.py**
- Lines 117-121: Horizontal layout pattern with proportional sizing
- Lines 44-56: CSS styling for panes with borders

**REFERENCE: frontend/widgets/entity_sidebar.py**
- Line 132: Border styling example

### Phase 1.2: Temporal Grouping

**MODIFY: backend/database/queries.py**
- `get_home_screen()` (line 113) - Ensure `valid_at` is returned for grouping

**CREATE: frontend/utils.py**
- `group_entries_by_period(episodes: list[dict]) -> list[tuple[str, list[dict]]]`
- Purpose: Group episodes into periods with labels

**MODIFY: frontend/screens/home_screen.py**
- `load_episodes()` (line 96) - Process grouped data, insert divider items
- Lines 113-120: Currently creates ListItems, add dividers between groups

**REFERENCE: frontend/widgets/entity_sidebar.py**
- Lines 26-41: Custom ListItem subclass pattern (EntityListItem)

### Phase 1.3: Processing Indicator Gutter

**CREATE: frontend/screens/home_screen.py**
- `EntryListItem(ListItem)` class - Custom list item with 2-char left margin
- Add after imports, before HomeScreen class

**MODIFY: frontend/screens/home_screen.py**
- Add reactive property: `active_episode_uuid: reactive[str | None] = reactive(None)`

**REFERENCE: frontend/widgets/entity_sidebar.py**
- Lines 29-33: ListItem CSS with padding
- Lines 159-166: Reactive property patterns

### Phase 2.1: Create Missing Queries

**CREATE: backend/database/queries.py**
- `get_entry_entities(episode_uuid: str, journal: str) -> list[dict]`
  - Pattern from frontend/widgets/entity_sidebar.py lines 328-334
  - Cache key: `journal:{journal}:{episode_uuid}`, hash field: `nodes`

- `get_period_entities(start_date: datetime, end_date: datetime, journal: str) -> dict`
  - NEW Cypher query for period aggregation
  - Returns: `{"entry_count": int, "connection_count": int, "top_entities": list[dict]}`

**CREATE: backend/database/redis_ops.py**
- `get_active_episode_uuid() -> str | None`
  - Simple wrapper to extract UUID from `task:active_episode`
  - Pattern from `is_episode_actively_processing()` (lines 449-459)

**MODIFY: backend/database/__init__.py**
- Export new query functions

### Phase 2.2: Connections Pane

**MODIFY: frontend/screens/home_screen.py**
- Add Container with id="connections-pane" in right_stack
- Add reactive: `selected_entry_uuid: reactive[str | None] = reactive(None)`
- Add reactive: `entry_entities: reactive[list[dict]] = reactive([])`
- Add watcher: `watch_selected_entry_uuid()` - calls get_entry_entities, updates entry_entities
- Modify `on_list_view_selected()` (line 90) - update selected_entry_uuid

**REFERENCE: frontend/screens/view_screen.py**
- Lines 166-180: Watch method patterns

### Phase 2.3: Temporal Pane

**MODIFY: frontend/screens/home_screen.py**
- Add Container with id="temporal-pane" in right_stack (title is dynamic)
- Add reactive: `selected_period: reactive[dict | None] = reactive(None)`
- Add reactive: `period_stats: reactive[dict | None] = reactive(None)`

**CREATE: frontend/utils.py**
- `calculate_periods(episodes: list[dict]) -> list[dict]`
  - Generate list of periods with labels and date ranges
  - Returns: `[{"label": str, "start": datetime, "end": datetime}, ...]`

**REFERENCE: backend/database/queries.py**
- Lines 50-59: Date parsing with `_parse_valid_at`

### Phase 2.4: Time Navigation

**MODIFY: frontend/screens/home_screen.py**
- Add BINDINGS (after line 45):
  - `Binding("left", "navigate_period_older", "Older", show=False)`
  - `Binding("right", "navigate_period_newer", "Newer", show=False)`
- Add action methods: `action_navigate_period_older()`, `action_navigate_period_newer()`
- Add watcher: `watch_selected_period()` - calls get_period_entities, scrolls entry list

**REFERENCE: frontend/screens/home_screen.py**
- Lines 174-190: Existing action method patterns

### Phase 3.1: Live Indicator from Redis

**MODIFY: frontend/screens/home_screen.py**
- Create polling worker: `_poll_processing_status()`
- Start in `on_mount()` after load_episodes
- Cancel in quit action

**REFERENCE: frontend/screens/view_screen.py**
- Lines 343-394: Full polling worker implementation (`_poll_until_complete`)
- Lines 123-141: Worker startup in on_mount
- Lines 260-268: Worker cancellation

**REFERENCE: backend/database/redis_ops.py**
- `get_pending_episodes_count()` (line 586) - Check if queue > 0
- `is_episode_actively_processing()` (line 440) - Check active episode

### Phase 3.2: Processing Pane

**MODIFY: frontend/screens/home_screen.py**
- Add Container with id="processing-pane" in right_stack
- Add reactive: `queue_count: reactive[int] = reactive(0)`
- Add watcher: `watch_queue_count()` - show/hide pane, adjust sizing

**REFERENCE: frontend/state/sidebar_state_machine.py**
- Lines 229-232: Property-based output flags for conditional display

### Phase 3.3: Active Episode Display

**MODIFY: frontend/screens/home_screen.py**
- Processing pane content: show active episode name + queue count

**REFERENCE: backend/database/queries.py**
- `get_episode()` (line 16) - Fetch episode name for display

**REFERENCE: frontend/widgets/entity_sidebar.py**
- Lines 240-311: `_update_content()` pattern for conditional rendering

## Textual Documentation References

Official examples and guides for implementation details.

**Base URLs:**
- Widgets: `https://github.com/Textualize/textual/tree/main/docs/examples/widgets/`
- Guides: `https://github.com/Textualize/textual/tree/main/docs/examples/guide/`

### Widget Examples

| Widget | File | Use Case |
|--------|------|----------|
| ListView | `list_view.py` | Entry list implementation |
| Static | `static.py` | Divider headers between periods |
| Rule | `horizontal_rules.py` | Alternative divider styling |
| LoadingIndicator | `loading_indicator.py` | Processing state feedback |
| ContentSwitcher | `content_switcher.py` | Adaptive pane content |
| Label | `label.py` | Text display in panes |
| Sparkline | `sparkline.py` | Future activity visualization |

### Guide Examples

| Topic | Directory | Use Case |
|-------|-----------|----------|
| Reactivity | `guide/reactivity/` | Reactive properties, watchers, data binding |
| Workers | `guide/workers/` | Polling worker implementation |
| Layout | `guide/layout/` | Three-pane Horizontal/Vertical layout |
| Screens | `guide/screens/` | Screen lifecycle, on_mount patterns |
| CSS | `guide/css/` | Border styling, adaptive sizing |
| Widgets | `guide/widgets/` | Custom widget creation |

### Key Documentation Pages

- Reactivity guide: https://textual.textualize.io/guide/reactivity/
- Workers guide: https://textual.textualize.io/guide/workers/
- Layout guide: https://textual.textualize.io/guide/layout/
- Widget gallery: https://textual.textualize.io/widget_gallery/
- CSS reference: https://textual.textualize.io/css_types/

## Summary: New Functions to Create

**Backend:**
1. `backend/database/queries.py::get_entry_entities()` - fetch entities for one episode
2. `backend/database/queries.py::get_period_entities()` - aggregate entities for time range
3. `backend/database/redis_ops.py::get_active_episode_uuid()` - get active episode UUID

**Frontend:**
4. `frontend/utils.py::group_entries_by_period()` - temporal grouping logic
5. `frontend/utils.py::calculate_periods()` - generate period boundaries from episodes
6. `frontend/screens/home_screen.py::EntryListItem` - custom ListItem with gutter
7. `frontend/screens/home_screen.py::_poll_processing_status()` - polling worker
8. Multiple watchers and action methods in HomeScreen
