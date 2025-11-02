# Graphiti PoC Analysis

## Context
- Phase 1 goal: custom pipeline should mirror `graphiti_core.graphiti.Graphiti.add_episode()` so downstream Graphiti tooling can operate unchanged (`phase1-poc.md:12`).
- Native flow produces Episode→Entity `MENTIONS` edges and `RELATES_TO` entity edges that all reference the originating episode (`research/01-episode-processing.md:166`, `.venv/lib/python3.13/site-packages/graphiti_core/utils/maintenance/edge_operations.py:221`).

## Findings
- **Missing episodes and MENTIONS edges**  
  - PoC only stages entity and relationship objects before writing; no `EpisodicNode` is created (`graphiti-poc.py:133`).  
  - Native `_process_episode_data()` always builds `EpisodicEdge` records linking the episode to each entity (`.venv/lib/python3.13/site-packages/graphiti_core/utils/maintenance/edge_operations.py:51`).  
  - Because the PoC never populates `episode.entity_edges`, RELATES_TO edges are disconnected from an episode, breaking provenance requirements (`research/01-episode-processing.md:170`).

- **Edge episodes field dropped and serialized incorrectly**  
  - `EntityEdge` objects are instantiated with `episodes=[]`, so even when an episode UUID exists it would not be attached (`entity_utils.py:96`).  
  - The Falkor writer JSON-serializes the empty list: `json.dumps(edge['episodes'])` means the DB stores a string instead of a list (`falkordb_utils.py:246`). Graphiti expects a list of UUIDs (`.venv/lib/python3.13/site-packages/graphiti_core/edges.py:225`), so downstream loaders would misinterpret the property.

- **Relationship naming parity gaps**  
  - DSPy relationships preserve free-form casing (e.g., `works_at`) when assigned to `EntityEdge.name` (`entity_utils.py:96`).  
  - Native edge extraction emits SCREAMING_SNAKE predicates and validates them against allowed edge types, defaulting to `RELATES_TO` when unsupported (`research/03-edge-extraction.md:112`, `.venv/lib/python3.13/site-packages/graphiti_core/utils/maintenance/edge_operations.py:220`).  
  - Lack of normalization risks mismatches with Graphiti’s edge-type indexing and search tooling.

## Validation Gaps
- No stage asserts that RELATES_TO edges carry non-empty `episodes` or that the database receives an Episode node.  
- Graphviz verification only reads back nodes/edges scoped by UUIDs, so it does not differentiate between RELATES_TO-only graphs and episode-linked graphs (`graphiti-poc.py:202`).

## Remediation Plan
1. **Introduce episode creation step**  
   - Generate an `EpisodicNode` per journal entry, assign `group_id`, `source_description`, `content`, etc.  
   - Use `build_episodic_edges()` to create MENTIONS edges referencing the episode (`.venv/lib/python3.13/site-packages/graphiti_core/utils/maintenance/edge_operations.py:51`).  
   - Update persistence layer to write Episodic node + edges alongside entity data, mirroring `_process_episode_data()` (`research/01-episode-processing.md:166`).
2. **Propagate and persist episode UUIDs on entity edges**  
   - Attach `episodes=[episode.uuid]` when constructing `EntityEdge` (`.venv/lib/python3.13/site-packages/graphiti_core/utils/maintenance/edge_operations.py:225`).  
   - Revise Falkor writer to keep list semantics (no `json.dumps`) and ensure Cypher literals emit arrays.
3. **Align relation naming with Graphiti expectations**  
   - Either constrain DSPy signature output to SCREAMING_SNAKE or map to canonical predicates before persistence.  
   - Consider enforcing fallback to `RELATES_TO` when no mapped predicate exists to stay compatible with Graphiti index definitions (`research/03-edge-extraction.md:112`).

## Verification Checklist
- Run PoC pipeline and confirm database contains:  
  - One `Episodic` node per ingestion, connected to entities via `:MENTIONS`.  
  - Each `:RELATES_TO` relationship stores `episodes` as a list containing the episode UUID.  
- Use Graphiti core helpers (e.g., `EntityEdge.get_by_uuid`) to load data and verify fields materialize without manual parsing (`.venv/lib/python3.13/site-packages/graphiti_core/edges.py:594`).  
- Ensure search or analytics that rely on episode provenance (community updates, temporal reasoning) have access to required fields once parity is restored.
