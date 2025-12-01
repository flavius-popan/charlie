# GEPA Node Extraction Examples Redesign

## Summary

Replace existing GEPA training examples with 7 new examples that address documented extraction failures. Each example covers 2-3 failure modes with globally diverse names.

## Changes

### 1. Rename Organization to Group

**File:** `backend/graph/entities_edges.py`

Change entity type from "Organization" to "Group" with updated description emphasizing informal groups alongside formal organizations.

**Rationale:** "Organization" biases model toward corporate entities. "Group" naturally captures friend circles, teams, clubs, and companies.

### 2. New Training Examples

**File:** `backend/optimizers/data/extract_nodes_examples.json`

7 new examples replacing existing 3. Each example:
- Covers 2-3 failure modes from node_issues.md
- Uses globally diverse names (not western-centric)
- Skews toward longer form (7+ sentences typical)
- Emphasizes Person and Activity extraction

### Failure Modes Addressed

From node_issues.md:
1. Missing Entities (Recognition Gaps)
2. Over-Literal Extraction (Verbatim Copying)
3. Normalization/Deduplication
4. Granularity Issues
5. Compound Entity Parsing
6. Entity Type Confusion (Action vs Entity)

### Example Coverage Matrix

| # | Length | Modes | Context |
|---|--------|-------|---------|
| 1 | Long | 1,2,3 | Morning routine - possessives, time modifiers |
| 2 | Medium | 4,5 | Outing - place granularity, compound parsing |
| 3 | Long | 2,6 | Work day - activities without action verbs |
| 4 | Short | 1,3 | Social note - implicit entities, normalization |
| 5 | Long | 3,4,5 | Family gathering - name variants, nested places |
| 6 | Medium | 1,6 | Hobby/exercise - activity recognition |
| 7 | Long | 2,4,5 | Travel - place hierarchy, compound lists |

## Scoring

Keep existing F1-based metric with strict type matching. Current feedback mechanism already penalizes:
- Missing entities (hurts recall)
- Over-extraction (hurts precision)

No scoring function changes needed initially. Can add pattern-specific feedback later if model struggles with particular failure modes.

## Entity Type IDs

- 0: Entity (generic fallback)
- 1: Person
- 2: Place
- 3: Group (renamed from Organization)
- 4: Activity
