# Reactive Properties Architecture Refactor Plan

**Status:** Planned
**Priority:** High (Architecture Excellence)
**Target:** Establish patterns for all future Textual widget interactions
**Created:** 2025-11-24

## Executive Summary

The sidebar implementation has a **code smell**: duplicate reactive properties maintained separately in `ViewScreen` and `EntitySidebar`. This creates tight coupling, synchronization burden, and fragility.

**Goal:** Refactor to single source of truth with proper parent-child reactive binding, establishing the definitive pattern for all future parent-child widget communication in this codebase.

**Scope:** ViewScreen ↔ EntitySidebar reactive property flow only. Other parent-child relationships will use this as the reference pattern.

**Framework baseline:** Textual 6.6.0 (latest stable as of 2025-11-24; matches `uv.lock`).

---

## Problem Statement

### Current Architecture (Anti-Pattern)

```
ViewScreen                          EntitySidebar
┌─────────────────────────┐         ┌──────────────────────┐
│ status: reactive[str]   │◄────────│ status: reactive[str]│
│ active_processing: bool │◄────────│ active_processing:b. │
│ inference_enabled: bool │◄────────│ inference_enabled: b.│
└─────────────────────────┘         └──────────────────────┘
        ▲                                   ▲
        │ set values manually               │ reads from parent
        │ in _sync_machine_output()         │ (stale values!)
        │ FRAGILE!                          │
        └───────────────────────────────────┘
                    Race Condition!
```

### Root Causes

1. **Duplicate State**
   - ViewScreen maintains `status`, `active_processing`, `inference_enabled`
   - EntitySidebar copies these as its own reactives
   - Each widget thinks it owns the state

2. **Manual Synchronization**
   - ViewScreen must explicitly call `sidebar.status = output.status`
   - Easy to forget this in future code paths
   - No compile-time verification

3. **Race Condition**
   - EntitySidebar's async watcher yields to event loop
   - But ViewScreen's reactive updates propagate asynchronously
   - Gap between assignment and propagation causes stale reads

4. **Single Async Watcher**
   - Only `watch_entities()` is async (unusual)
   - Other watchers are sync
   - Inconsistent pattern

---

## Design

1) **Single source of truth in the parent (ViewScreen).** `status`, `active_processing`, and `inference_enabled` live only on the screen.

2) **One-way binding with `data_bind()`.** In `compose()`, bind those reactives from `ViewScreen` to the matching reactives on `EntitySidebar`. Child treats them as read-only and communicates up via events/messages, never by mutating the bound values.

3) **Seed child reactives safely.** In `EntitySidebar.__init__`, use `set_reactive()` to load initial values without firing watchers before mount (Textual guidance).

4) **No manual synchronization.** Delete any direct assignments from parent to child reactives (_e.g._ in `_sync_machine_output`). Parent updates its own reactives; binding propagates.

5) **Watcher discipline.** Watchers stay synchronous. Use a tiny `_pending_render` flag plus `call_after_refresh()` to coalesce multiple triggers into one `_update_content()` per refresh cycle.

6) **Batching is optional.** Use `batch_update()` only for large cross-widget layout churn; normal reactive updates already benefit from Textual’s coalesced refreshes.

---

## Implementation

1) **Bind in `ViewScreen.compose()`.**
```python
def compose(self) -> ComposeResult:
    sidebar = EntitySidebar(
        episode_uuid=self.episode_uuid,
        journal=self.journal,
        inference_enabled=self.inference_enabled,  # keep during transition
        status=self.status,
        active_processing=self.active_processing,
        on_entity_deleted=self._on_entity_deleted,
        id="entity-sidebar",
    )
    sidebar.data_bind(
        status=ViewScreen.status,
        active_processing=ViewScreen.active_processing,
        inference_enabled=ViewScreen.inference_enabled,
    )
    yield Header(show_clock=False, icon="")
    yield Horizontal(Markdown("Loading...", id="journal-content"), sidebar)
    yield Footer()
```

2) **Drop manual sync in `_sync_machine_output()`.**
```python
def _sync_machine_output(self) -> None:
    output = self.sidebar_machine.output
    self.status = output.status
    self.active_processing = output.active_processing
    # Binding handles sidebar synchronization
```

3) **Dedupe renders in `EntitySidebar`.**
```python
class EntitySidebar(Container):
    _pending_render: bool = False

    def _request_render(self) -> None:
        if self._pending_render:
            return
        self._pending_render = True
        self.call_after_refresh(self._flush_render)

    def _flush_render(self) -> None:
        self._pending_render = False
        if self.is_mounted and not self.loading:
            self._update_content()

    def watch_status(self, status: str | None) -> None:
        if self.is_mounted:
            self._request_render()

    def watch_active_processing(self, active_processing: bool) -> None:
        if self.is_mounted:
            self._request_render()

    def watch_entities(self, entities: list[dict]) -> None:
        if self.is_mounted:
            self._request_render()
```
Keep watchers synchronous; remove the async/sleep variant. Keep `set_reactive()` seeding in `__init__`.

4) **Batching:** use `batch_update()` only for large cross-widget layout churn; normal reactive updates don’t require it.

---

## Testing Strategy

### Unit Tests
- Verify ViewScreen properties update correctly
- Verify EntitySidebar receives updates via binding
- Confirm child writes do not mutate parent (one-way discipline)
- Test with and without binding

### Integration Tests
- Test full deletion flow (the original bug scenario)
- Verify no "Awaiting processing..." appears
- Verify properties are synchronized at each step

### Property Binding Tests
```python
def test_data_binding_syncs_status():
    """Verify data_bind() synchronizes status from parent."""
    app = TestApp()

    async with app.run_test() as pilot:
        screen = app.screen
        sidebar = screen.query_one("#entity-sidebar", EntitySidebar)

        # Change parent property
        screen.status = "pending_edges"
        await pilot.pause()

        # Verify child is synchronized
        assert sidebar.status == "pending_edges"

        # Child write should not change parent
        sidebar.status = "should_not_propagate"
        await pilot.pause()
        assert screen.status == "pending_edges"
```

### Race Condition Verification
```python
def test_no_race_condition_on_deletion():
    """Verify stale values don't cause 'Awaiting processing' message."""
    app = TestApp()

    async with app.run_test() as pilot:
        # ... setup deletion scenario ...
        sidebar.entities = []  # Triggers watch_entities
        screen.status = None   # Binding automatically syncs
        await pilot.pause()

        # Verify sidebar reads fresh values
        content_container = sidebar.query_one("#entity-content", Container)
        text_nodes = [
            n.renderable.plain
            for n in content_container.children
            if isinstance(n, Label)
        ]
        assert any("No connections found" in t for t in text_nodes)
        assert all("Awaiting processing" not in t for t in text_nodes)

def test_sidebar_renders_once_per_cycle():
    """Rapid reactive changes only schedule one sidebar render."""
    app = TestApp()
    async with app.run_test() as pilot:
        screen = app.screen
        sidebar = screen.query_one("#entity-sidebar", EntitySidebar)

        sidebar._pending_render = False
        sidebar._update_count = 0

        # Monkeypatch to count updates
        original_update = sidebar._update_content
        def _counting_update():
            sidebar._update_count += 1
            original_update()
        sidebar._update_content = _counting_update

        # Trigger multiple watchers in one tick
        screen.status = "pending_nodes"
        sidebar.entities = []
        sidebar.active_processing = True
        await pilot.pause()

        assert sidebar._update_count == 1
```

---

## Code Quality Standards

### Comments (Per AGENTS.md)
- ✓ Explain **why** binding is used, not what it does
- ✓ Document parent-child relationship
- ✓ Evergreen (don't reference specific bugs)

**Example:**
```python
# Data binding ensures child properties stay synchronized with parent
# without manual synchronization code in _sync_machine_output()
sidebar.data_bind(status=ViewScreen.status)
```

**Not:**
```python
# This fixes the race condition where watch_entities fires before status updates
```

### Type Safety
- Use proper reactive property types
- Document binding relationships in type hints or docstrings

### Error Handling
- Keep `NoMatches` exception handling for unmounted widgets
- Verify binding success with assertions
- Fail fast if binding setup is wrong

---

## Rollout Plan

### Commit 1: Add Data Binding (Non-Breaking)
- Add `data_bind()` calls in `compose()`
- Keep manual sync code (redundant)
- Run all tests
- Verify binding works

### Commit 2: Remove Manual Sync
- Delete manual synchronization code
- Add assertions to verify binding
- Update comments per AGENTS.md

### Commit 3: Simplify Watchers
- Remove async/sleep from watchers
- Keep watchers clean and sync
- Update tests as needed

### Commit 4: Dedupe Watcher Renders
- Add `_pending_render` flag + `call_after_refresh` scheduler
- Keep `_update_content()` idempotent and single-shot per refresh cycle
- Add performance/behavioral tests around rapid status+entities changes

### Commit 5: Documentation & Pattern Guide
- Document the pattern for future developers
- Create examples for parent-child binding
- Add to development guidelines

---

## Pattern Guide for Future Development

### Checklist for All Parent-Child Reactive Relationships

- [ ] Parent owns the canonical state
- [ ] Child has reactive properties that match parent's
- [ ] `data_bind()` called in parent's `compose()`
- [ ] No manual synchronization code
- [ ] Watchers assume data is current (sync, not async) and use a scheduler to avoid duplicate renders
- [ ] Use `batch_update()` only when batching large cross-widget layout changes
- [ ] Comments explain the binding relationship
- [ ] Tests verify synchronization works

---

## Success Criteria

### Technical
- ✓ Single source of truth (ViewScreen owns state)
- ✓ Automatic binding (no manual sync code)
- ✓ No race conditions (data always current)
- ✓ Efficient rendering (coalesced refresh + watcher dedupe)
- ✓ All tests passing (63+ tests)
- ✓ Code quality (AGENTS.md compliance)

### Architectural
- ✓ Clean parent-child pattern
- ✓ Extensible to other widgets
- ✓ Self-documenting code
- ✓ Zero technical debt from duplicate state

### Developer Experience
- ✓ Easy to understand reactive flow
- ✓ Hard to make synchronization mistakes
- ✓ Pattern applies to all parent-child pairs
- ✓ Developers reference this for guidance

---

## Timeline

| Phase | Task | Effort | Duration |
|-------|------|--------|----------|
| 1 | Add data binding | Low | 1-2 hours |
| 2 | Verify & remove manual sync | Low | 1 hour |
| 3 | Simplify watchers | Low | 30 min |
| 4 | Dedupe renders (`_pending_render` scheduler) | Low | 30 min |
| 5 | Testing & verification | Medium | 1-2 hours |
| 6 | Documentation & review | Low | 1 hour |
| **Total** | | | **5-7 hours** |

---

## References & Learning

### Textual Documentation
- [Reactive Guide](https://textual.textualize.io/guide/reactivity/)
- [Data Binding](https://textual.textualize.io/guide/reactivity/#data-binding)
- [Widget Composition](https://textual.textualize.io/guide/widgets/)
- [App.batch_update](https://textual.textualize.io/api/app/#textual.app.App.batch_update)

### Versions
- Textual 6.6.0 (PyPI, released 2025-11-10) — matches `uv.lock`

### Related Code
- `frontend/screens/view_screen.py` - Parent widget (ViewScreen)
- `frontend/widgets/entity_sidebar.py` - Child widget (EntitySidebar)
- `frontend/state/sidebar_state_machine.py` - State machine (data source)

### Related Commits
- `710a634` - Current fix with manual sync (temporary)
- `f477b8c` - State machine status clearing (necessary)

---

## Risk Assessment

### Risks & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Binding doesn't work | Low | High | Test binding immediately after adding `data_bind()` |
| Performance regression | Low | Medium | Profile render scheduling; ensure `_pending_render` is in place |
| Watchers fire incorrectly | Low | Medium | Unit test each watcher |
| Type errors in binding | Low | Medium | Use type hints, add assertions |

### Rollback Plan
Each commit is independently reversible. If Phase 2+ fails, revert and keep Phase 1.

---

## Future Pattern Evolution

### Once Established
- Apply this pattern to all parent-child widget pairs
- Create utility helpers if patterns emerge
- Make this the "golden standard" for this codebase

