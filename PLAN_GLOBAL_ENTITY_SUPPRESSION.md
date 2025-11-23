# Implementation Plan: Global Entity Suppression with Complete Cache Management

## Overview
Implement global journal-level entity suppression with proper cache state management across all Redis cache keys.

**Design**: Delete action → removes from episode + suppresses globally + updates all cache keys

---

## Phase 1: Add Global Suppression Storage

### Task 1.1: Create suppression management functions
**File**: `backend/database/redis_ops.py` (new functions)

Add helper functions:

1. **`add_suppressed_entity(journal: str, entity_name: str)`**
   - Redis key: `journal:{journal}:suppressed_entities`
   - Store as JSON array of lowercase entity names
   - Normalize to lowercase for case-insensitive matching

2. **`get_suppressed_entities(journal: str) -> set[str]`**
   - Returns set of suppressed entity names (lowercase)

3. **`remove_suppressed_entity(journal: str, entity_name: str)`** (future)
   - For un-suppression feature

---

## Phase 2: Update Entity Deletion with Complete Cache Management

### Task 2.1: Modify deletion query to return all needed data
**File**: `backend/database/queries.py` (lines 57-133)

Update Cypher query in `delete_entity_mention()` to return:
- Entity name (for suppression)
- Deleted edge UUID (for cache cleanup)
- Whether entity was deleted (existing)

**New query**:
```cypher
MATCH (ep:Episodic {uuid: $episode_uuid})-[r:MENTIONS]->(ent:Entity {uuid: $entity_uuid})
WITH ent.name as entity_name, r.uuid as edge_uuid, r, ent
DELETE r
WITH ent, entity_name, edge_uuid
OPTIONAL MATCH (ent)<-[remaining:MENTIONS]-()
WITH ent, entity_name, edge_uuid, count(remaining) as remaining_refs
WHERE remaining_refs = 0
DETACH DELETE ent
RETURN entity_name, edge_uuid, remaining_refs = 0 as was_deleted
```

### Task 2.2: Update all cache keys after deletion
**File**: `backend/database/queries.py` (lines 121-133)

Expand cache update logic to handle all cache keys:

```python
with redis_ops() as r:
    cache_key = f"journal:{journal}:{episode_uuid}"

    # 1. Update 'nodes' cache (existing logic)
    nodes_json = r.hget(cache_key, "nodes")
    if nodes_json:
        nodes = json.loads(nodes_json.decode())
        updated_nodes = [n for n in nodes if n["uuid"] != entity_uuid]
        r.hset(cache_key, "nodes", json.dumps(updated_nodes))

    # 2. Update 'mentions_edges' cache (NEW)
    mentions_json = r.hget(cache_key, "mentions_edges")
    if mentions_json and edge_uuid:
        edge_uuids = json.loads(mentions_json.decode())
        updated_edges = [uuid for uuid in edge_uuids if uuid != edge_uuid]
        r.hset(cache_key, "mentions_edges", json.dumps(updated_edges))

    # 3. Update 'uuid_map' cache (NEW)
    uuid_map_json = r.hget(cache_key, "uuid_map")
    if uuid_map_json:
        uuid_map = json.loads(uuid_map_json.decode())
        # Remove mappings where canonical UUID = deleted entity
        updated_map = {
            prov: canon for prov, canon in uuid_map.items()
            if canon != entity_uuid
        }
        if updated_map != uuid_map:
            r.hset(cache_key, "uuid_map", json.dumps(updated_map))

    # 4. TODO: Update 'entity_edges' when edge extraction implemented
    # When edge extraction is added, remove RELATES_TO edge UUIDs
    # that involve the deleted entity from the entity_edges list
```

### Task 2.3: Add global suppression
**File**: `backend/database/queries.py`

After cache updates, add entity to global suppression:

```python
# Suppress entity globally in journal
from backend.database.redis_ops import add_suppressed_entity
add_suppressed_entity(journal, entity_name)
```

---

## Phase 3: Filter Extraction Results Against Suppressed Entities

### Task 3.1: Add suppression filtering to extract_nodes()
**File**: `backend/graph/extract_nodes.py`

**Location**: After LLM extraction (around line 355), before creating provisional nodes

```python
# After LLM extraction
from backend.database.redis_ops import get_suppressed_entities

suppressed = get_suppressed_entities(journal)
if suppressed:
    original_count = len(extracted_entities)
    extracted_entities = [
        e for e in extracted_entities
        if e.name.lower() not in suppressed
    ]
    filtered_count = original_count - len(extracted_entities)
    if filtered_count > 0:
        logger.info(
            f"Filtered out {filtered_count} suppressed entities from episode {episode_uuid}"
        )

# Handle all entities suppressed
if not extracted_entities:
    logger.info(f"All entities suppressed for episode {episode_uuid}, returning empty result")
    return ExtractionResult(extracted_count=0, uuid_map={})
```

### Task 3.2: Update cache even when no entities extracted
Ensure empty extraction updates cache correctly:
- Clear `nodes` cache to empty array
- Clear `mentions_edges` to empty array
- Clear `uuid_map` to empty object
- Set status to "done"

---

## Phase 4: Testing

### Task 4.1: Create global suppression tests
**File**: `tests/test_backend/test_global_entity_suppression.py` (NEW)

**Test 1: Deletion suppresses globally across episodes**
```python
# Episode A: "Alice and Bob"
# Delete Bob from Episode A
# Episode B: "Bob and Charlie"
# Extract Episode B
# Assert: Only Charlie extracted (Bob suppressed)
# Assert: suppressed_entities = ["bob"]
```

**Test 2: Re-extraction respects suppression**
```python
# Episode: "Alice and Bob"
# Extract, delete Bob
# Edit to: "Alice and Bob and Charlie"
# Re-extract
# Assert: Only Alice and Charlie (Bob still suppressed)
```

**Test 3: Cache state after deletion**
```python
# Episode: "Alice and Bob"
# Extract (verify cache populated)
# Delete Bob
# Assert: nodes cache updated (Bob removed)
# Assert: mentions_edges cache updated (Bob's edge removed)
# Assert: uuid_map cache updated (Bob's mappings removed)
# Assert: suppressed_entities contains "bob"
```

**Test 4: Case-insensitive suppression**
```python
# Delete "bob" (lowercase)
# Extract content with "Bob" (capitalized)
# Assert: "Bob" filtered out
```

**Test 5: All entities suppressed**
```python
# Episode: "Alice and Bob"
# Extract, delete both
# Re-extract
# Assert: extraction result is empty
# Assert: nodes cache = []
# Assert: mentions_edges cache = []
# Assert: status = "done" (not stuck)
```

**Test 6: UUID map cleanup**
```python
# Extract entities (uuid_map populated)
# Delete one entity
# Assert: uuid_map no longer contains deleted entity UUID
# Assert: Other mappings preserved
```

**Test 7: Multiple edge deletion**
```python
# Episode mentions same entity twice (if possible)
# Delete entity
# Assert: All MENTIONS edges to entity removed from cache
```

### Task 4.2: Update existing tests
**File**: `tests/test_backend/test_extract_nodes_reextraction.py`

Update tests that call `delete_entity_mention()`:
- Expect global suppression to occur
- Verify cache keys updated correctly

---

## Phase 5: Edge Cases & Error Handling

### Task 5.1: Handle edge not found scenario
If MENTIONS edge doesn't exist (already deleted, or UUID mismatch):
- Query returns no results
- Don't crash, return gracefully
- Still add to suppression list if entity name known

### Task 5.2: Handle cache inconsistencies
If cache is missing or malformed:
- Don't crash on cache updates
- Log warnings
- Suppression still works (most critical)

### Task 5.3: Add TODO for entity_edges
**File**: `backend/database/queries.py`

Add clear TODO comment in cache update section:

```python
# TODO: When edge extraction is implemented, update entity_edges cache
# Need to remove RELATES_TO edge UUIDs that involve deleted entity:
# 1. Query for edges WHERE source_uuid = entity_uuid OR target_uuid = entity_uuid
# 2. Get edge UUIDs
# 3. Remove from episode.entity_edges list in cache
# 4. Consider: May need to check multiple episodes if edges are shared
```

---

## Phase 6: Integration & Verification

### Task 6.1: Manual UI testing
1. Create journal entry
2. Verify extraction creates entities
3. Delete entity via UI
4. Check Redis: verify all cache keys updated
5. Edit entry, trigger re-extraction
6. Verify entity doesn't return
7. Create new entry mentioning suppressed entity
8. Verify entity doesn't appear

### Task 6.2: Redis inspection
Verify cache state after each operation:
```bash
# View all cache keys for episode
redis-cli HGETALL "journal:my_journal:episode_uuid"

# View suppressed entities
redis-cli GET "journal:my_journal:suppressed_entities"
```

---

## Files Modified

- `backend/database/redis_ops.py` - Add suppression management (3 new functions)
- `backend/database/queries.py` - Update delete_entity_mention() with cache management
- `backend/graph/extract_nodes.py` - Add suppression filtering
- `tests/test_backend/test_global_entity_suppression.py` - NEW (7 tests)
- `tests/test_backend/test_extract_nodes_reextraction.py` - Update existing tests

## Success Criteria

- Deleting entity adds to global suppression list
- All cache keys updated correctly (`nodes`, `mentions_edges`, `uuid_map`)
- TODO comment added for `entity_edges` future work
- Re-extraction filters out suppressed entities
- Suppression works across episodes
- Case-insensitive matching
- Cache inconsistencies handled gracefully
- All tests pass
- Manual UI verification successful

---

## Future Enhancements (Out of Scope)

1. **UI to manage suppressions**:
   - View list of suppressed entities
   - Un-suppress entities
   - Bulk suppress/un-suppress

2. **Database persistence**:
   - Store suppressions in graph database (survive Redis restart)
   - Add property to journal node

3. **Per-episode suppression option**:
   - Alternative to global suppression
   - More granular control

4. **Entity disambiguation**:
   - Handle multiple entities with same name
   - More sophisticated matching (by type, attributes, etc.)
