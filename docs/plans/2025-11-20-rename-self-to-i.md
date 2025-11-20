# Rename Self Node To "I" Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ensure every model-facing author node string is the literal word `"I"` (no prefixes/suffixes) so LLM outputs stay first-person; keep class/constant identifiers (e.g., `SelfNode`) unchanged.

**Architecture:** Change the canonical author name constant, then propagate to all DSPy-facing payloads and optimizer examples so training/queries emit `"I"`. Tests and docs follow the new canonical name. NL-facing texts already use first-person facts; only entity names shift.

**Tech Stack:** Python, DSPy, pytest, FalkorDB Lite, graphiti_core.

### Task 1: Add canonical-name test (RED)

**Files:**
- Create: `tests/test_pipeline/test_self_reference.py`

**Steps:**
1. Add a test asserting `SELF_ENTITY_NAME == "I"` and provisional self node uses that name.
2. Run `pytest tests/test_pipeline/test_self_reference.py -q` (expect FAIL while name is still "Self").

### Task 2: Set canonical author name to "I"

**Files:**
- Modify: `pipeline/self_reference.py`
- Modify: `backend/database/utils.py`
- Modify: `pipeline/falkordblite_driver.py`

**Steps:**
1. Update `SELF_ENTITY_NAME` default and related seed helpers to write `"I"`.
2. Keep alias set for pronoun detection; no backward-compat needed.

### Task 3: Update LLM-facing payloads & optimizer examples

**Files:**
- Modify: `pipeline/optimizers/extract_nodes_optimizer.py`
- Modify: `pipeline/optimizers/extract_edges_optimizer.py`
- Modify: `pipeline/optimizers/extract_attributes_optimizer.py`
- Modify: `pipeline/optimizers/generate_summaries_optimizer.py`
- Modify: `pipeline/README.md`

**Steps:**
1. Replace author entity names `"Self"` → `"I"` in train/val examples, entity JSON builders, and first-person checks.
2. Adjust `has_self`/author detection predicates to look for `"I"`.
3. Update README deterministic author description to match `"I"`.

### Task 4: Align tests and queries

**Files:**
- Modify: `tests/test_pipeline/test_extract_nodes.py`
- Modify: `tests/test_backend/test_extract_nodes.py`
- Modify: `tests/test_pipeline/test_falkordblite.py`
- Modify: `tests/test_backend/test_add_journal.py`

**Steps:**
1. Change assertions/queries expecting `"Self"` to expect `"I"`.
2. Ensure any seeded names in fixtures/entities use `"I"`.

### Task 5: Verify

**Commands:**
- `pytest tests/test_pipeline/test_self_reference.py -q`
- `pytest tests/test_pipeline/test_extract_nodes.py -q`
- `pytest tests/test_backend/test_extract_nodes.py -q`

**Expected:** New test passes; extract-node paths behave with `"I"`; no regressions in targeted suites.
