# Reactive Properties Architecture Refactor Plan

**Status:** Planned
**Priority:** High (Architecture Excellence)
**Target:** Establish patterns for all future Textual widget interactions
**Created:** 2025-11-24

## Executive Summary

The sidebar implementation has a **code smell**: duplicate reactive properties maintained separately in `ViewScreen` and `EntitySidebar`. This creates tight coupling, synchronization burden, and fragility.

**Goal:** Refactor to single source of truth with proper parent-child reactive binding, establishing the definitive pattern for all future parent-child widget communication in this codebase.

**Scope:** ViewScreen ↔ EntitySidebar reactive property flow only. Other parent-child relationships will use this as the reference pattern.

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
        │ set values manually              │ reads from parent
        │ in _sync_machine_output()        │ (stale values!)
        │ FRAGILE!                         │
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

## Solution Design

### Option A: Parent-Owned State (Recommended)

**Philosophy:** ViewScreen owns all state. EntitySidebar reads from parent.

```
ViewScreen (Single Source of Truth)
┌──────────────────────────────┐
│ status: reactive[str]        │
│ active_processing: reactive  │
│ inference_enabled: reactive  │
└──────────────────────────────┘
        │
        │ Data binding (automatic sync)
        ▼
EntitySidebar (Read-Only)
┌──────────────────────────────┐
│ Receives values from parent   │
│ No local copies, no duplication
│ Reads: self.screen.status    │
└──────────────────────────────┘
```

**Pros:**
- Single source of truth
- Automatic synchronization (no manual calls)
- Type-safe
- Textual best practice

**Cons:**
- Requires refactoring EntitySidebar watchers
- EntitySidebar becomes tightly coupled to ViewScreen type

**Complexity:** Medium

### Option B: Textual Data Binding (Best Practice)

**Philosophy:** Use Textual's built-in `data_bind()` for automatic reactive property synchronization.

```python
# ViewScreen.compose()
EntitySidebar(...).data_bind(
    status=ViewScreen.status,
    active_processing=ViewScreen.active_processing,
    inference_enabled=ViewScreen.inference_enabled,
)
```

**How it works:**
- Textual automatically creates two-way binding
- Child properties sync with parent automatically
- No manual synchronization code needed
- Textual handles the reactive system internally

**Pros:**
- Automatic, declarative synchronization
- Textual framework handles all complexity
- Zero manual synchronization code
- Extensible to other widgets

**Cons:**
- Requires understanding Textual's data binding semantics
- Limited to constructor-time binding (can't be dynamic)
- May need conditional binding logic

**Complexity:** Low (once understood)

### Option C: Derived Reactive Properties

**Philosophy:** Child widget has a derived reactive property that watches parent.

```python
# EntitySidebar
class EntitySidebar(Container):
    status: reactive[str | None] = reactive(None, init=False)

    async def watch_entities(self):
        # When entities change, read parent's status
        parent_status = self.screen.status if hasattr(self.screen, 'status') else None
        self.status = parent_status
        self._update_content()
```

**Pros:**
- Explicit parent-child relationship
- Child controls when to read parent
- Can add logic/filtering

**Cons:**
- Still has duplication (two properties)
- More complex watchers
- Extra rendering passes

**Complexity:** Medium

---

## Recommended Approach: Option B + Defensive Properties

**Hybrid Solution:**

1. **Use Textual's `data_bind()` for automatic synchronization**
   - Declarative in `compose()`
   - Framework handles all complexity
   - Type-safe and verifiable

2. **Keep EntitySidebar's reactive properties** (for widget interface)
   - But mark as "synced from parent" in comments
   - Document the binding relationship
   - Add assertions to verify binding worked

3. **Remove all manual synchronization code** from ViewScreen
   - Delete lines that manually set `sidebar.status = ...`
   - Trust the binding mechanism

4. **Simplify watchers** in EntitySidebar
   - Keep them sync (remove async sleep)
   - Data is always current due to binding
   - Cleaner, more straightforward

---

## Implementation Steps

### Phase 1: Add Textual Data Binding (Non-Breaking)

**File:** `frontend/screens/view_screen.py`

**Step 1.1: Import data_bind helper**
```python
# Already imported from textual
```

**Step 1.2: Modify compose() to bind properties**
```python
def compose(self) -> ComposeResult:
    # ... existing code ...

    sidebar = EntitySidebar(
        episode_uuid=self.episode_uuid,
        journal=self.journal,
        inference_enabled=self.inference_enabled,
        status=self.status,
        active_processing=self.active_processing,
        on_entity_deleted=self._on_entity_deleted,
        id="entity-sidebar",
    )

    # Data binding: automatic reactive synchronization
    sidebar.data_bind(
        status=ViewScreen.status,
        active_processing=ViewScreen.active_processing,
        inference_enabled=ViewScreen.inference_enabled,
    )

    yield sidebar
```

**Why non-breaking:** Binding happens in addition to manual sync (redundant but safe during transition).

### Phase 2: Verify Binding Works

**Testing:**
- Run integration tests to verify properties sync
- Add assertions in `_sync_machine_output()` to verify binding worked
- Test stale value scenarios

```python
def _sync_machine_output(self) -> None:
    output = self.sidebar_machine.output
    self.status = output.status
    self.active_processing = output.active_processing

    # Verify binding worked
    try:
        sidebar = self.query_one("#entity-sidebar", EntitySidebar)
        assert sidebar.status == self.status, \
            f"Binding failed: sidebar.status={sidebar.status}, self.status={self.status}"
        assert sidebar.active_processing == self.active_processing, \
            f"Binding failed: sidebar.active_processing={sidebar.active_processing}"
    except NoMatches:
        pass  # Not mounted yet
```

### Phase 3: Remove Manual Synchronization

**File:** `frontend/screens/view_screen.py`

Once binding is verified, remove manual sync code:

```python
# DELETE these lines from _sync_machine_output():
try:
    sidebar = self.query_one("#entity-sidebar", EntitySidebar)
    sidebar.status = output.status
    sidebar.active_processing = output.active_processing
except NoMatches:
    pass
```

Keep only the ViewScreen's own reactive updates:
```python
def _sync_machine_output(self) -> None:
    output = self.sidebar_machine.output
    self.status = output.status
    self.active_processing = output.active_processing
    # Binding handles EntitySidebar synchronization automatically
```

### Phase 4: Simplify EntitySidebar Watchers

**File:** `frontend/widgets/entity_sidebar.py`

Change async watcher back to sync (since data is now always current):

```python
# Before (with async sleep):
async def watch_entities(self, entities: list[dict]) -> None:
    if not self.is_mounted:
        return
    if not self.loading:
        await asyncio.sleep(0)
        self._update_content()

# After (simple and clean):
def watch_entities(self, entities: list[dict]) -> None:
    """Re-render when entities change."""
    if not self.is_mounted:
        return
    if not self.loading:
        self._update_content()
```

**Why this works:** With data binding, properties are synchronized automatically. No race condition.

---

## Batch Updates Strategy

### Current Issue
When ViewScreen's reactive properties change, EntitySidebar's multiple watchers fire:
1. `watch_status()` → calls `_update_content()`
2. `watch_active_processing()` → calls `_update_content()`
3. `watch_entities()` → calls `_update_content()`

This causes **3 renders for 1 update** (inefficient).

### Solution: Use `batch_update()` Context

**File:** `frontend/screens/view_screen.py`

```python
def _sync_machine_output(self) -> None:
    """Sync machine output with batch update for efficiency."""
    output = self.sidebar_machine.output

    # Batch all reactive updates to prevent multiple renders
    with self.app.batch_update():
        self.status = output.status
        self.active_processing = output.active_processing

    # All bound properties update atomically
    # EntitySidebar's watchers fire once at end of batch
```

**Effect:**
- ViewScreen updates 2 properties
- Binding propagates to EntitySidebar
- All watchers batch together
- Single render pass instead of 3

### Batch Update Best Practices

**When to use:**
- Multiple related property changes
- Expensive update operations (UI rendering)
- Preventing cascading updates

**Pattern:**
```python
with self.app.batch_update():
    # All changes here are batched
    self.prop1 = value1
    self.prop2 = value2
    self.prop3 = value3
# All watchers fire once here
```

**Not needed for:**
- Single property changes
- Independent unrelated changes
- Internal-only changes (not bound to UI)

---

## Testing Strategy

### Unit Tests
- Verify ViewScreen properties update correctly
- Verify EntitySidebar receives updates via binding
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
        content = sidebar._get_display_message()
        assert content == "No connections found"
        assert "Awaiting processing" not in content
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

### Commit 4: Optimize with Batch Updates
- Add `batch_update()` context
- Verify rendering efficiency
- Add performance tests

### Commit 5: Documentation & Pattern Guide
- Document the pattern for future developers
- Create examples for parent-child binding
- Add to development guidelines

---

## Pattern Guide for Future Development

### When You Add Parent-Child Widgets

**DON'T DO THIS:**
```python
# Anti-pattern: Duplicate reactive properties
class ParentWidget(Container):
    status: reactive = reactive("idle")

class ChildWidget(Container):
    status: reactive = reactive("idle")  # DUPLICATE!

    def on_mount(self):
        # Manual sync code everywhere - FRAGILE!
        self.status = self.screen.status
```

**DO THIS:**
```python
# Pattern: Parent owns state, use data binding
class ParentWidget(Container):
    status: reactive = reactive("idle")

    def compose(self):
        child = ChildWidget(status=self.status)
        # Automatic synchronization - ROBUST!
        child.data_bind(status=ParentWidget.status)
        yield child

class ChildWidget(Container):
    status: reactive = reactive("idle")  # Receives via binding

    def watch_status(self, status: str) -> None:
        # Watcher fires when parent changes status
        self._update_display()
```

### Checklist for All Parent-Child Reactive Relationships

- [ ] Parent owns the canonical state
- [ ] Child has reactive properties that match parent's
- [ ] `data_bind()` called in parent's `compose()`
- [ ] No manual synchronization code
- [ ] Watchers assume data is current (sync, not async)
- [ ] Use `batch_update()` for multiple property changes
- [ ] Comments explain the binding relationship
- [ ] Tests verify synchronization works

---

## Success Criteria

### Technical
- ✓ Single source of truth (ViewScreen owns state)
- ✓ Automatic binding (no manual sync code)
- ✓ No race conditions (data always current)
- ✓ Efficient rendering (batch updates)
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
| 4 | Batch updates | Low | 30 min |
| 5 | Testing & verification | Medium | 1-2 hours |
| 6 | Documentation & review | Low | 1 hour |
| **Total** | | | **5-7 hours** |

---

## References & Learning

### Textual Documentation
- [Reactive Guide](https://textual.textualize.io/guide/reactivity/)
- [Data Binding](https://textual.textualize.io/guide/actions/#data-binding)
- [Widget Composition](https://textual.textualize.io/guide/widgets/)

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
| Binding doesn't work | Low | High | Test thoroughly in Phase 1 |
| Performance regression | Low | Medium | Profile with batch updates |
| Watchers fire incorrectly | Low | Medium | Unit test each watcher |
| Type errors in binding | Low | Medium | Use type hints, add assertions |

### Rollback Plan
Each commit is independently reversible. If Phase 2+ fails, revert and keep Phase 1.

---

## Future Pattern Evolution

### Once Established
- Apply this pattern to all parent-child widget pairs
- Create utility helpers if patterns emerge
- Update onboarding documentation
- Make this the "golden standard" for this codebase

### Extensions
- Consider reactive composition patterns for complex apps
- Document state management best practices
- Create examples for new developers

---

**Author:** Code Review Process
**Last Updated:** 2025-11-24
**Status:** Ready for Implementation
