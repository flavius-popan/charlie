# Charlie V2: Personal Knowledge Graph from Journal Entries

## User Experience

Charlie is a desktop journaling application that builds a personal knowledge graph from your writing. When you finish a journal entry, you save it instantly - the text is preserved immediately. Behind the scenes, the system extracts entities (people, places, activities) and their relationships, showing results progressively as processing completes.

The core use case is temporal exploration: "What was I doing in March 2024?" You get a summary view showing people you spent time with, activities you engaged in, and key themes from that period - all derived from your own journal entries, not AI-generated interpretations. Entity cards display relationships with supporting facts, plus a timeline of every mention across your entries. This lets you see how relationships evolved without rereading hundreds of entries.

Users can curate their knowledge graph through the UI, correcting entity disambiguations or editing relationships. The graph serves as an augmented memory, letting you navigate your life through people, places, and activities rather than chronological text search alone.

## Architecture

The v2 rewrite eliminates the concept of "stages" in favor of discrete operations that work independently on episodic nodes. Each journal entry creates an `EpisodicNode` (episode UUID + content + timestamp + metadata). A simple state machine tracks which operations have completed for each episode, enabling background processing and progressive reveal in the UI.

### Backend API

The backend provides a clean API layer separating UI concerns from graph operations:

**Episode Creation**: `add_journal_entry(content, reference_time, journal, title, source_description, uuid)` - Saves episode to database immediately, returns episode UUID string. No LLM processing performed. Supports multiple isolated journals via `journal` parameter (maps to `group_id` internally). Auto-generates human-readable titles ("Tuesday Nov 18, 2024") when not provided. Accepts pre-existing UUIDs for imports from Day One, Notion, Obsidian, etc. Automatically creates author entity "I" for each journal on first entry.

**Validation**: Content validated for null bytes (rejected), max length (100k chars), and non-empty. Naive datetimes (no timezone) are normalized to UTC for import-friendly handling - markdown files and simple formats can pass file creation timestamps directly. Journal names restricted to alphanumeric, underscores, and hyphens (max 64 chars) for filesystem safety.

**Operation Triggers**: Background workers (Huey, single worker thread) dequeue `extract_nodes` then `extract_edges` tasks for each saved episode. Episode state in Redis advances from `pending_nodes` to `pending_edges` to complete. *(Not yet implemented)*

**Query API**: `get_episodes_by_timerange(start, end)`, `get_entity_timeline(entity_uuid)`, `search_entities(query)` - All query operations are read-only and work against the current graph state. *(Not yet implemented)*

This separation allows the TUI to call `add_journal_entry()` to instantly persist entries, while importers can batch-create episodes and trigger enrichment in the background.

### Operations (formerly Stages 1 & 2)

**Extract Entities**: Identifies people, places, organizations, and activities mentioned in the episode. Uses MinHash LSH for fuzzy deduplication against existing entities (`graphiti_core.utils.maintenance.dedup_helpers`). Stores both provisional nodes (with temporary UUIDs for LLM indexing) and resolved nodes (canonical UUIDs after deduplication via `node_operations.resolve_extracted_nodes`). The uuid_map linking provisional to canonical UUIDs is stored in episode attributes temporarily.

**Current Scope (as of Nov 21, 2025)**: Only entity extraction is live. Episodes transition from `pending_nodes` to `pending_edges` as a holding state, but no edge extraction worker exists yet. `cleanup_if_no_work` intentionally ignores `pending_edges` so models can unload even if that queue contains items. This will change once `extract_edges_task` is implemented.

**Extract Relationships** (planned): Identifies relationships between entities with supporting facts. LLM returns relationships as integer indices referencing the provisional entity list. After extraction, `edge_operations.resolve_edge_pointers()` remaps to canonical UUIDs using the uuid_map. Exact-match deduplication merges edges across episodes via the `episodes` list field (`edge_operations.resolve_extracted_edge`). Once edges are extracted, the uuid_map is cleaned up from episode attributes.

**Edge Tense Strategy - Hybrid Event/State Model**: Journal entries naturally document both temporal events and ongoing states. V2 uses **tense to encode semantic meaning**:

**Event-Based Edges (Past Tense - Capturing Moments):**
- **Met** - First encounter with someone (relationship milestone)
- **Visited** - Went to a place (can occur multiple times)
- **Attended** - Participated in a one-time event
- **Started** - Began a relationship, job, or practice

**State-Based Edges (Present Tense - Capturing Ongoing Patterns):**
- **Knows** - Ongoing awareness/familiarity
- **Supports** - Active support relationship
- **WorksAt** - Employment relationship
- **LivesAt** - Current residence
- **SpendsTimeWith** - Repeated social interactions
- **ConflictsWith** - Tension or friction in relationship
- **ParticipatesIn** - Repeated participation in activity

**Why This Works:**
1. **Semantic authenticity**: Journal entries ARE retrospective - "I met Sarah" (past event) vs "Sarah supports me" (current state)
2. **Natural queries**: "Show everyone I Met in 2024" and "Show everyone who Supports me" both read naturally
3. **Growth tracking**: First meetings mark relationship beginnings, support patterns show current connection quality
4. **Timeline + landscape**: Events provide temporal milestones, states describe current relationship fabric
5. **Natural graph visualization**: "Alice Knows Bob" reads as "Alice knows Bob" (complete sentence)

**LLM Classification**: Train DSPy optimizer to recognize event markers ("met", "visited", "went to", "attended") vs state markers ("is", "works at", "supports", "keeps visiting"). Temporal validity (valid_at/invalid_at) handles when states begin/end.

**Verb Form Convention**: State edges use **third-person singular** (Knows, Supports, SpendsTimeWith) so graph triples read as natural sentences: "Alice Knows Bob" = "Alice knows Bob". Edge types are relationship descriptors, not literal query verbs. Past-tense event edges already have correct form (Met, Visited, Attended). Standard convention across knowledge graphs.

Episodic edges (MENTIONS) are built via `edge_operations.build_episodic_edges()` to link episodes to entities.

### What We're Not Building

**No Summaries (Stage 4)**: Entity summaries are expensive (serial per-entity LLM calls), prone to hallucination, and accumulate drift over time. Users get more value from reading their original journal text than AI-generated summaries. Search works on entity names and episode text fulltext indexes instead.

**No Attributes (Stage 3)**: Type-specific attributes (Person.relationship_type, Activity.purpose) provide minimal architectural value beyond entity names and relationships. They're used only for LLM deduplication context. Skipping them saves 26s per entry and reduces hallucination risk.

Performance impact: 91s → 47s per entry (48% faster) by omitting these operations.

### Graphiti-Core Reuse

Maximize code reuse from graphiti-core for deterministic operations:
- Entity/edge resolution utilities (`edge_operations.resolve_edge_pointers`, `node_operations.resolve_extracted_nodes`)
- Deduplication algorithms (`dedup_helpers._build_candidate_indexes`, `dedup_helpers._resolve_with_similarity`)
- Data validation (`ontology_utils.validate_entity_types`, `helpers.validate_group_id`)
- Graph operations (`edge_operations.build_episodic_edges`)
- Data models (`nodes.EntityNode`, `nodes.EpisodicNode`, `edges.EntityEdge`)

Only implement custom DSPy modules for LLM extraction (entity extraction, edge extraction). This keeps the codebase aligned with upstream improvements while allowing local inference optimization.

## Supported Queries

**Time-Range Queries**: Fetch all episodes from a date range (e.g., March 2024). Return episodes sorted by timestamp with entity mentions and active relationships during that period.

**Entity Timeline**: Given an entity UUID, return all episodes mentioning it chronologically. Shows how relationships and context evolved over time.

**Relationship Queries**: Find all edges involving an entity, filtered by temporal range (valid_at/invalid_at). Supports "Who was I spending time with in spring 2024?" queries.

**Search**: BM25 fulltext search on entity names and episode content. Vector similarity search on entity name embeddings for semantic matching. No summary search (summaries not generated).

**Period Summaries**: On-demand LLM summarization of episode texts from a time range, generated when user views a period (not pre-computed during ingestion).

## Technology Stack

**FalkorDB-Lite**: Embedded graph database (Redis-based) for local deployment. Supports fulltext indexes, vector similarity, and Cypher queries. Scales to ~50k entities before performance degradation. Single-process architecture simplifies deployment.

**DSPy**: Framework for optimizing LLM extraction prompts. Enables MIPROv2 teleprompter optimization to improve entity/edge extraction quality over time. Separates extraction logic (pure dspy.Module) from orchestration (database I/O).

**llama.cpp**: Local LLM inference via CPU/GPU acceleration. Thread-safe for concurrent requests. Model runs entirely offline - no API calls, no cloud dependency. Supports structured output via JSON schema for entity/edge extraction.
**Huey (single worker)**: Thread-based consumer for asynchronous extraction tasks; only the instruct model is loaded, and it is unloaded immediately when no episodes are pending.

**State Tracking**: Episode processing state stored in database. Background workers poll for episodes needing processing. Progressive reveal: UI updates as operations complete asynchronously.
