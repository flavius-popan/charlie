# Sidebar State Machine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace ad-hoc sidebar status handling with a focused state machine using `python-statemachine`, improving readability and robustness while keeping current UX and status strings intact.

**Architecture:** Introduce a pure logic `SidebarStateMachine` (no Textual dependencies) that models sidebar visibility, inference-disabled mode, processing states, cache results, and user deletion. `ViewScreen` owns timers/workers and drives the machine via events; `EntitySidebar` renders from machine outputs. Global inference toggle remains app-level and sends events into the sidebar machine. While refactoring, modularize UI code into a `frontend/` package (screens, widgets, state machines, diagrams) and keep `charlie.py` as the entry point.

**Tech Stack:** Python, Textual, Redis, Huey, `python-statemachine` (async-friendly).

---

## PHASE 1: FOUNDATION (COMPLETE)

### Task 1: Add dependency ✓ COMPLETE

**Files:**
- Modified: `pyproject.toml`

**Status:** python-statemachine==2.5.0 added successfully.

---

### Task 2: Create frontend package structure ✓ COMPLETE

**Files:**
- Created: `frontend/`, `frontend/screens/`, `frontend/widgets/`, `frontend/state/`, `frontend/diagrams/`
- Created: `frontend/__init__.py`, `frontend/screens/__init__.py`, `frontend/widgets/__init__.py`, `frontend/state/__init__.py`, `frontend/utils.py`

**Status:**
- All Textual widgets/screens extracted from charlie.py (1398 → 245 lines)
- Widgets: `EntitySidebar`, `EntityListItem`, `DeleteEntityModal` in `frontend/widgets/entity_sidebar.py`
- Screens: `ViewScreen`, `EditScreen`, `HomeScreen`, `SettingsScreen`, `LogScreen` in separate modules
- Helper functions: `extract_title()`, `get_display_title()` extracted to `frontend/utils.py`
- All imports updated to use frontend.* paths
- Convenience exports added to frontend/__init__.py

**Code Review:** ✓ PASSED - All critical and important issues resolved

---

### Task 3-4: Code Review Fixes ✓ COMPLETE

**Critical Issues Fixed:**
1. Test imports updated: All test files now import from frontend.* instead of charlie
2. SettingsScreen CSS moved to DEFAULT_CSS attribute (removed from charlie.py)
3. Helper functions properly isolated in frontend/utils.py (no duplication)

**Important Issues Fixed:**
1. Convenience exports added to frontend/__init__.py
2. Module docstrings added to all subpackages

**Verification:**
- All 15 tests in test_charlie_utils.py pass
- No circular imports detected
- No code duplication
- charlie.py works as entry point
- All imports functional

---

## PHASE 2: STATE MACHINE IMPLEMENTATION (IN PROGRESS)

### Task 3: Implement SidebarStateMachine module ✓ COMPLETE

**Files:**
- Created: `frontend/state/sidebar_state_machine.py` (415 lines)
- Created: `tests/test_frontend/test_sidebar_state_machine.py` (526 lines)

**Implementation Details:**

1. **States (6 flat states):** `hidden`, `disabled`, `processing_nodes`, `awaiting_edges`, `ready_entities`, `empty_idle` ✓
2. **Events (11 events with guard-based routing):**
   - `show` → routes to disabled/processing/awaiting/ready/empty based on guards
   - `hide` → hidden (from any visible state)
   - `episode_closed` → hidden (from any visible state)
   - `inference_disabled` → disabled (from any state)
   - `inference_enabled` → processing/awaiting/ready/empty based on status/entities guards
   - `status_pending_nodes` → processing_nodes
   - `status_pending_edges_or_done` → awaiting_edges/ready/empty based on guards
   - `cache_entities_found` → ready_entities
   - `cache_empty` → empty_idle
   - `user_deleted_entity` → ready_entities/empty_idle based on entities remaining ✓

3. **Outputs (SidebarOutput dataclass):** All 7 outputs implemented
   - `visible`, `should_poll`, `active_processing`, `loading`, `message`, `status`, `entities_present` ✓

4. **Status semantics:** Unchanged (`pending_nodes`, `pending_edges`, `done`, `None`) ✓

5. **API:** Pure logic module with zero Textual imports; `apply_event(event_name, **data) → SidebarOutput` ✓

6. **Tests:** 49 comprehensive tests covering:
   - Initialization (3 tests)
   - All event transitions (26 tests)
   - Guard conditions (14 tests)
   - Output flags (6 tests)
   - Complex workflows (4 tests)
   - API methods (3 tests)
   - Edge cases (8 new tests: invalid transitions, data persistence, missing data, rapid updates, inference toggle, entity depletion)

**Code Review:** ✓ APPROVED with Important recommendations addressed:

**Important Fixes Applied:**
1. Removed dead code path: `disabled → hidden` transition on `inference_enabled` event (simplified from 5 to 4 targets)
2. Added comprehensive documentation:
   - Class docstring explains guard/hook evaluation order (guards run before `before_*` hooks)
   - Explains why guards check both `data.get()` and `self._*` patterns
   - Includes example usage
3. Clarified `__init__` docstring:
   - When to use `visible=True` (from_edit initialization path)
   - When to use `show` event instead (prevents race conditions)

**Nice-to-Have Improvements Applied:**
1. Added explicit type hints to all `before_*` event methods
   - `before_show(status: str | None = None, inference_enabled: bool | None = None, entities_present: bool | None = None, **data)`
   - Similar signatures for all event handlers
2. Added 8 comprehensive edge case tests

**Test Results:** 49/49 passing (100%)

---

### Task 4: Integrate machine into ViewScreen lifecycle

**Files:**
- Modify: `frontend/screens/view_screen.py`
- Add tests: `tests/test_view_sidebar_state_machine.py`

**Steps:**
1. Instantiate `SidebarStateMachine` in `ViewScreen.__init__`, seeding with current `status`, `inference_enabled`, visibility (`from_edit`), and `entities_present` (initially False).
2. Route events:
   - On toggle display → `show`/`hide`
   - On status polling results → `status_pending_nodes` or `status_pending_edges_or_done`
   - On inference toggle refresh → `inference_disabled`/`inference_enabled`
   - On episode close/back → `episode_closed`
   - On cache fetch results → `cache_entities_found` / `cache_empty`
3. Replace manual reactive flag flips with machine outputs (e.g., set `active_processing`, `status`, `loading` from machine after each event).
4. Timer/worker ownership stays in `ViewScreen`: start polling when `should_poll` is True and no worker running; cancel when it flips False.
5. Ensure background worker stops on `action_back` and hides sidebar; sync `active_processing` to sidebar after event application.
6. Add focused tests (can be async Textual headless): simulate event sequences to ensure workers start/stop and flags sync to the sidebar widget without rendering assertions.

---

### Task 5: Adapt EntitySidebar rendering to machine outputs

**Files:**
- Modify: `frontend/widgets/entity_sidebar.py`

**Steps:**
1. Remove bespoke `user_override` logic; instead accept a `message` and `loading/active_processing` flags from machine outputs provided by ViewScreen.
2. Simplify `_update_content`: render spinner when `loading and active_processing and status in pending_*`; render message passed in; render list when entities present.
3. After delete action, emit `user_deleted_entity` event into machine via ViewScreen (or machine holder) and refresh entities; keep UX identical.
4. Keep cache fetch logic but report results via machine events instead of directly toggling flags.

---

### Task 6: Wire inference toggle to machine and sidebar

**Files:**
- Modify: `frontend/screens/view_screen.py`, `charlie.py` (App-level toggle refresh)

**Steps:**
1. After reading `get_inference_enabled()`, send `inference_enabled` or `inference_disabled` event to machine before syncing reactive fields.
2. Ensure `loading` is cleared when disabled; when re-enabled, re-evaluate current status to decide polling (machine handles).

---

### Task 7: Generate sidebar state diagram (optional but recommended)

**Files:**
- Add: `frontend/diagrams/sidebar_state_machine.png` (git-tracked)
- Helper: `frontend/state/sidebar_state_machine.py` (function to emit diagram guarded by ImportError)

**Steps:**
1. Optionally install diagrams extra locally: `uv add python-statemachine[diagrams]` (do not make CI fail if missing).
2. Add `generate_diagram()` helper that writes to `frontend/diagrams/sidebar_state_machine.png` when pydot/graphviz available.
3. Run helper once and commit the PNG.

---

### Task 8: Testing & verification

**Files:**
- Tests added in Tasks 2 and 3

**Steps:**
1. Run unit tests: `pytest tests/test_sidebar_state_machine.py -q`
2. Run integration slice: `pytest tests/test_view_sidebar_state_machine.py -q`
3. Run existing suite (if fast enough): `pytest -q`
4. Manual smoke (headless): start app, create entry, ensure sidebar toggles, delete entity path unaffected (no live run in CI).

---

### Task 9: Docs & handoff

**Files:**
- Update: `docs/plans/2025-11-24-sidebar-state-machine.md` (this file) if anything changes during implementation.

**Steps:**
1. Note chosen library and rationale (python-statemachine vs transitions) in implementation PR description, not in code.
2. If diagram generation is desired later, mention optional `python-statemachine[diagrams]` in PR notes (no change now).

---

## Session Handoff Notes (Updated 2025-11-24)

**Completed in this session:**
- ✅ Task 3 fully implemented, reviewed, and tested (49/49 tests passing)
- ✅ Code review feedback applied (dead code removed, documentation added, type hints added, edge cases tested)
- ✅ Test file organized in `tests/test_frontend/` following project structure
- ✅ Plan document updated with implementation details

**Next steps for continuation:**
1. **Task 4: ViewScreen Integration**
   - Key integration point: `ViewScreen.__init__` needs to instantiate `SidebarStateMachine`
   - See ViewScreen current structure in `frontend/screens/view_screen.py` (currently manages `status`, `inference_enabled`, `active_processing` reactives)
   - Machine will replace manual state management with `apply_event()` calls
   - Current helpers to understand: `_refresh_sidebar_context()`, `_poll_until_complete()`, `action_toggle_connections()`

2. **Task 5: EntitySidebar Adaptation**
   - Current `user_override` logic can be eliminated once ViewScreen feeds machine outputs
   - Key rendering logic in `_update_content()` method (lines 219-271)
   - Watch methods can be simplified to consume machine outputs

3. **Task 6: Inference Toggle Wiring**
   - Global toggle handler in `charlie.py` needs to send events to machine
   - Current pattern: `get_inference_enabled()` called in several places

4. **Research artifacts available:**
   - Current ViewScreen manages: visibility (`display`), status polling, sidebar synchronization
   - Current EntitySidebar has: cache fetch logic, deletion handling, override logic
   - Test patterns in `tests/test_frontend/test_view_screen.py` and `test_entity_sidebar.py` show async testing approach

**Library reference:**
- python-statemachine==2.5.0 (already in pyproject.toml)
- Guards evaluated before `before_*` hooks (see sidebar_state_machine.py docstring)
- All tests use `machine.send()` and `machine.apply_event()` patterns

Execution options:
1. Subagent-driven in this session (requires superpowers:subagent-driven-development).
2. Separate execution session using superpowers:executing-plans.
