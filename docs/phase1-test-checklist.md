# Phase 1 PoC - Final Integration Test Checklist

**Date:** 2025-11-02

## Overview

This document provides a comprehensive test checklist for the Phase 1 PoC implementation. These tests should be performed manually in the Gradio UI to verify all functionality is working correctly.

**DO NOT RUN THE GRADIO UI AUTOMATICALLY** - User will test manually.

---

## Test 1: Database Reset

**Purpose:** Verify database initialization and reset functionality

**Steps:**
1. Start the Gradio UI: `python graphiti-poc.py`
2. Observe initial database stats in UI
3. Click "Reset Database" button
4. Verify stats show `{'nodes': 0, 'edges': 0}`

**Expected Result:** Database successfully resets to empty state

---

## Test 2: Full Pipeline with Example Text

**Purpose:** Verify end-to-end pipeline execution with provided example

**Steps:**
1. Click "Load Example" button
2. Verify example text appears in input box: "Alice works at Microsoft in Seattle. She reports to Bob, who manages the engineering team."
3. Wait 1-2 seconds for Stage 1 NER to complete automatically
4. Verify Stage 1 output shows entity names (Alice, Microsoft, Seattle, Bob)
5. Click "Run Facts" button (Stage 2)
6. Verify Stage 2 output shows JSON with facts about entities
7. Click "Run Relationships" button (Stage 3)
8. Verify Stage 3 output shows JSON with relationships (source, target, relation, context)
9. Click "Build Graphiti Objects" button (Stage 4)
10. Verify Stage 4 output shows JSON with nodes (containing UUIDs, names) and edges
11. Click "Write to Falkor" button (Stage 5)
12. Verify Stage 5 output shows write confirmation with node/edge counts and UUIDs
13. Verify Stage 6 graph visualization appears showing nodes and edges
14. Verify database stats update to show new node/edge counts (should be > 0)

**Expected Result:** Complete pipeline executes successfully with graph visualization

---

## Test 3: Persons Only Filter

**Purpose:** Verify NER person filter functionality

**Steps:**
1. Ensure example text is loaded
2. Check "Persons only" checkbox
3. Verify Stage 1 output updates to show only person names (Alice, Bob)
4. Uncheck "Persons only" checkbox
5. Verify Stage 1 output shows all entities again (Alice, Microsoft, Seattle, Bob)

**Expected Result:** Filter correctly limits entities to person names only

---

## Test 4: Error Handling - Missing Inputs

**Purpose:** Verify error handling for missing required inputs

**Steps:**
1. Clear input text box (or start fresh session)
2. Click "Run Facts" button without entering text
3. Verify Stage 2 output shows error JSON: `{"error": "Need text and entities"}`
4. Enter some text with entities
5. Wait for Stage 1 NER to complete
6. Click "Run Relationships" button WITHOUT running Facts first
7. Verify Stage 3 output shows error JSON: `{"error": "Need text, facts, and entities"}`

**Expected Result:** Clear error messages displayed for missing inputs

---

## Test 5: Custom Input Text

**Purpose:** Verify pipeline works with custom user input

**Steps:**
1. Clear input text (or enter new text)
2. Enter custom text with entities and relationships, e.g.:
   ```
   John Smith founded Acme Corp in New York. Sarah Johnson is the CTO and works closely with John on product strategy.
   ```
3. Wait for Stage 1 NER to complete
4. Run through Stages 2-5 sequentially
5. Verify graph visualization shows new entities and relationships

**Expected Result:** Pipeline processes custom text successfully

---

## Test 6: Database Persistence

**Purpose:** Verify data persists across pipeline runs

**Steps:**
1. Note current database stats (e.g., `{'nodes': 4, 'edges': 2}`)
2. Load example text again
3. Run through full pipeline (Stages 1-5)
4. Observe database stats after second run
5. Verify stats have incremented (e.g., `{'nodes': 8, 'edges': 4}`)
6. Click "Reset Database"
7. Verify stats return to `{'nodes': 0, 'edges': 0}`

**Expected Result:** Database accumulates data across runs and resets successfully

---

## Test 7: Graph Visualization Quality

**Purpose:** Verify graph rendering displays correct information

**Steps:**
1. Run full pipeline with example text
2. Examine Stage 6 graph visualization image
3. Verify:
   - Node labels show entity names (NOT `b'name'` - should be clean strings)
   - Edge labels show relationship types (e.g., "works at", "reports to")
   - No binary prefixes (`b'...'`) in any labels
   - Graph layout is readable
   - Different node colors for different entity types (if applicable)

**Expected Result:** Clean, readable graph with no encoding artifacts

---

## Test 8: Console Logging

**Purpose:** Verify logging output provides useful information

**Steps:**
1. Observe console output during pipeline execution
2. Verify INFO logs appear for each stage:
   - "Stage 1: Extracted X unique entities"
   - "Stage 2: Extracted X facts"
   - "Stage 3: Inferred X relationships"
   - "Stage 4: Built X nodes and X edges"
   - "Stage 5: Writing X nodes and X edges to FalkorDB"
3. Verify no tokenizers parallelism warning appears

**Expected Result:** Clear logging output with no warnings

---

## Test 9: Multiple Sequential Runs

**Purpose:** Verify UI state management across multiple runs

**Steps:**
1. Run full pipeline with example text
2. Note results in all stages
3. Click "Load Example" again (or enter different text)
4. Run full pipeline again
5. Verify all stages update with new results
6. Verify no stale data from previous run appears

**Expected Result:** Clean state transitions between pipeline runs

---

## Test 10: FalkorDB Cleanup on Exit

**Purpose:** Verify database closes cleanly

**Steps:**
1. Run full pipeline at least once
2. Close Gradio UI (Ctrl+C in terminal)
3. Observe console output
4. Verify message appears: "✓ FalkorDB closed successfully"

**Expected Result:** Clean shutdown with no warnings or errors

---

## Known Issues & Limitations (Expected Behavior)

The following are known limitations of Phase 1 PoC and should NOT be considered test failures:

1. **Single context window** - No text chunking for long documents
2. **Empty embeddings** - All embedding vectors are empty arrays `[]`
3. **No episode management** - Episodes array is always empty `[]`
4. **Serial execution** - Stages must be run sequentially (no auto-run)
5. **Basic error messages** - No user-friendly input validation messages
6. **No entity deduplication across runs** - Same entity will create duplicate nodes
7. **Graph layout** - May vary between renders (not deterministic positioning)

---

## Reporting Issues

If any test fails or produces unexpected results:

1. Note the test number and failure description
2. Check console output for error messages/tracebacks
3. Verify all dependencies are installed: `uv sync`
4. Verify models are in `.models/` directory
5. Check `data/graphiti-poc.db` file exists and is writable

---

## Success Criteria

All tests (1-10) should pass for Phase 1 PoC to be considered complete and ready for user testing.
