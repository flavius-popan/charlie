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

## PHASE 2: STATE MACHINE IMPLEMENTATION (PENDING)

### Task 3: Implement SidebarStateMachine module

**Files:**
- Create: `frontend/state/sidebar_state_machine.py`
- Add tests: `tests/test_sidebar_state_machine.py`

**Steps:**
1. Define states (flat): `hidden`, `disabled`, `processing_nodes`, `awaiting_edges`, `ready_entities`, `empty_idle`.
2. Define events:  
   - `show`, `hide`, `episode_closed`  
   - `inference_disabled`, `inference_enabled`  
   - `status_pending_nodes`, `status_pending_edges_or_done`  
   - `cache_entities_found`, `cache_empty`  
   - `user_deleted_entity`
3. Embed outputs: `should_poll`, `active_processing`, `loading`, `visible`, `message`, `status` (data), `entities_present` flag. Implement as computed properties on the model or via a dataclass attached to the machine.
4. Keep status string semantics unchanged (`pending_nodes`, `pending_edges`, `done`, `None`).
5. Tests (pure unit): assert legal transitions, guard behavior when inference disabled/enabled, and output flags per state.
6. Keep module free of Textual imports; expose a small API: `apply_event(event, **data)` returning updated outputs.

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

Execution options after plan approval:
1. Subagent-driven in this session (requires superpowers:subagent-driven-development).
2. Separate execution session using superpowers:executing-plans.
