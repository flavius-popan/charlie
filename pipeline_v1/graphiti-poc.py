"""Phase 1 PoC: Text → Graph in FalkorDBLite with Gradio UI."""

import logging
import threading
import time
from datetime import datetime

import gradio as gr

from falkordb_utils import fetch_recent_episodes
from graphiti_pipeline import (
    GraphitiPipeline,
    GraphitiPipelineError,
    PipelineConfig,
    get_db_stats,
    process_ner,
    render_verification_graph,
    reset_database,
    write_to_falkordb,
)
from settings import DB_PATH, MODEL_CONFIG, GROUP_ID

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info(f"Starting Graphiti PoC with MODEL_CONFIG: {MODEL_CONFIG}")
logger.info(f"Database path: {DB_PATH}")


def _parse_reference_time(value: str | None) -> datetime:
    if not value:
        return datetime.now()
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        logging.warning("Invalid reference time %s, using now()", value)
        return datetime.now()


def _context_to_json(episodes):
    entries = []
    for episode in episodes:
        entries.append(
            {
                "uuid": episode.uuid,
                "name": episode.name,
                "valid_at": getattr(episode, "valid_at", None),
                "content_preview": (episode.content[:200] + "...")
                if episode.content and len(episode.content) > 200
                else episode.content,
            }
        )
    return entries


def _build_config(
    context_window: int,
    dedupe_enabled: bool,
    attributes_enabled: bool,
    summary_enabled: bool,
    temporal_enabled: bool,
    temporal_enrichment_enabled: bool,
    llm_edges_enabled: bool,
) -> PipelineConfig:
    return PipelineConfig(
        group_id=GROUP_ID,
        context_window=int(context_window),
        dedupe_enabled=dedupe_enabled,
        attribute_extraction_enabled=attributes_enabled,
        entity_summary_enabled=summary_enabled,
        temporal_enabled=temporal_enabled,
        temporal_enrichment_enabled=temporal_enrichment_enabled,
        llm_edge_detection_enabled=llm_edges_enabled,
    )


def _embedding_stub():
    return {
        "status": "not_implemented",
        "message": "Embeddings will be generated post-deduplication in a future phase.",
    }


def _reranker_stub():
    return {
        "status": "not_implemented",
        "message": "Reranker scoring will be integrated alongside embeddings in a future phase.",
    }


PIPELINE = GraphitiPipeline()
DEFAULT_UI_CONFIG = PipelineConfig(
    group_id=GROUP_ID,
    context_window=3,
    dedupe_enabled=True,
    attribute_extraction_enabled=True,
    entity_summary_enabled=True,
    temporal_enabled=True,
    temporal_enrichment_enabled=True,
    llm_edge_detection_enabled=True,
)
INITIAL_CONTEXT = _context_to_json(
    fetch_recent_episodes(
        DEFAULT_UI_CONFIG.group_id,
        datetime.now(),
        DEFAULT_UI_CONFIG.context_window,
    )
)

_NER_PREVIEW_DELAY = 0.75
_ner_preview_lock = threading.Lock()
_last_ner_preview_time = 0.0


def _debounced_ner_preview(text: str, persons_only: bool):
    """Live NER preview with debounce to avoid recalculating on every keystroke."""
    global _last_ner_preview_time

    normalized_text = (text or "").strip()

    with _ner_preview_lock:
        current_time = time.time()
        _last_ner_preview_time = current_time

    if not normalized_text:
        return "Enter text to see detected entities.", gr.update(value=None)

    time.sleep(_NER_PREVIEW_DELAY)

    with _ner_preview_lock:
        if current_time != _last_ner_preview_time:
            return gr.update(), gr.update()

    try:
        entity_names, raw_entities, _ = process_ner(normalized_text, persons_only)
    except Exception as exc:  # noqa: BLE001
        logging.exception("Live NER preview failed")
        return f"NER error: {exc}", gr.update(value=None)

    names_text = "\n".join(entity_names) if entity_names else "(no entities detected)"
    return names_text, raw_entities


# Build Gradio interface
with gr.Blocks(title="Phase 1 PoC: Graphiti Pipeline") as app:
    gr.Markdown("# Graphiti Pipeline: Local Knowledge Graph Construction")
    gr.Markdown(f"**Database:** `{DB_PATH}` | **Model Config:** `{MODEL_CONFIG}`")

    with gr.Tabs():
        # Tab 0: Overview & Input
        with gr.Tab("Overview & Input"):
            gr.Markdown("""
## Pipeline Overview

This pipeline transforms unstructured text into a knowledge graph through five stages. Unlike graphiti-core's API-based LLM approach,
this implementation runs entirely offline using **DistilBERT** for entity recognition and **DSPy + Outlines** for structured extraction with local MLX inference.

### The Five Stages

1. **Entity Detection** - DistilBERT NER identifies people, organizations, and locations in your text
2. **Knowledge Extraction** - DSPy signatures extract facts and relationships from the text and entities
3. **Graph Assembly** - Entities and edges are deduplicated against existing graph data, with contradictions resolved
4. **Entity Enrichment** - Entities receive attributes, labels, and summaries to enhance searchability
5. **Persistence** - The graph is written to FalkorDB and visualized

### Key Architectural Differences from graphiti-core

- **Entity Extraction**: DistilBERT NER (local) vs LLM prompts (API)
- **Fact/Relationship Extraction**: DSPy signatures (structured) vs prompt-based LLM calls
- **Inference**: Local MLX execution vs OpenAI/Anthropic APIs
- **Deduplication**: Reuses graphiti-core's MinHash + Jaccard fuzzy matching
- **Temporal Handling**: Hybrid inline DSPy extraction + dateparser validation
- **Schema Compatibility**: Uses graphiti-core's data models (`EntityNode`, `EntityEdge`, etc.)

---

## Input Configuration

Enter your text below and configure the pipeline behavior. The live preview shows entities detected by DistilBERT NER as you type.
            """)

            with gr.Row(equal_height=True):
                with gr.Column(scale=3, min_width=400):
                    input_text = gr.Textbox(
                        label="Journal Entry / Text to Process",
                        placeholder="Enter text here...",
                        lines=10,
                    )
                    reference_time_input = gr.Textbox(
                        label="Reference Time (ISO8601, optional)",
                        placeholder="Defaults to current time",
                    )
                with gr.Column(scale=2, min_width=320):
                    gr.Markdown("### Live Entity Preview")
                    gr.Markdown("As you type, DistilBERT NER extracts entities in real-time. This preview shows what the pipeline will process.")
                    persons_only_filter = gr.Checkbox(
                        label="Persons only", value=False
                    )
                    entity_preview_box = gr.Textbox(
                        label="Detected entities",
                        value="Enter text to see detected entities.",
                        interactive=False,
                        lines=10,
                    )

            gr.Markdown("### Pipeline Configuration")
            gr.Markdown("Toggle features on/off to experiment with different pipeline behaviors. All toggles are enabled by default.")

            with gr.Row():
                context_window_slider = gr.Slider(
                    label="Context Window (previous episodes)",
                    minimum=0,
                    maximum=10,
                    step=1,
                    value=DEFAULT_UI_CONFIG.context_window,
                )
                dedupe_toggle = gr.Checkbox(
                    label="Enable deduplication", value=DEFAULT_UI_CONFIG.dedupe_enabled
                )
                attributes_toggle = gr.Checkbox(
                    label="Enable attribute extraction",
                    value=DEFAULT_UI_CONFIG.attribute_extraction_enabled,
                )
                summary_toggle = gr.Checkbox(
                    label="Enable entity summaries",
                    value=DEFAULT_UI_CONFIG.entity_summary_enabled,
                )
                temporal_toggle = gr.Checkbox(
                    label="Enable temporal defaults",
                    value=DEFAULT_UI_CONFIG.temporal_enabled,
                )
                temporal_enrichment_toggle = gr.Checkbox(
                    label="Enable temporal enrichment (dateparser)",
                    value=DEFAULT_UI_CONFIG.temporal_enrichment_enabled,
                )
                llm_edges_toggle = gr.Checkbox(
                    label="Enable LLM edge detection",
                    value=DEFAULT_UI_CONFIG.llm_edge_detection_enabled,
                )

            run_pipeline_btn = gr.Button("Run Pipeline (Dry Run - No Database Write)", variant="primary", size="lg")
            gr.Markdown("Click to process your text through all five stages. Results appear in the following tabs.")

        # Tab 1: Entity Detection
        with gr.Tab("Stage 1: Entity Detection"):
            gr.Markdown("""
## Stage 1: Entity Detection

This stage identifies entities in your text and retrieves context from previous episodes. Understanding what entities already exist helps the pipeline
make intelligent deduplication decisions later.

### What Happens Here

1. **Context Retrieval**: The pipeline queries FalkorDB for recent episodes within your configured context window
2. **Entity Recognition**: DistilBERT NER processes your text and extracts people, organizations, and locations
3. **Entity Deduplication**: Extracted entity names are normalized and deduplicated before passing to the next stage

### Why This Matters

Context episodes inform entity resolution in Stage 3. If "Alice" appears in a previous episode, the pipeline can match the new mention to the existing entity node rather than creating a duplicate.

---
            """)

            gr.Markdown("**Context Episodes** - Recent episodes from your graph, ordered by recency.")
            context_output = gr.JSON(
                label="Most recent episodes",
                value=INITIAL_CONTEXT,
            )

            gr.Markdown("**Detected Entities** - DistilBERT NER extracts named entities with type labels (PER/ORG/LOC).")
            ner_output = gr.Textbox(label="Entity Names", interactive=False, lines=6)

            with gr.Accordion("Raw NER Output", open=False):
                entity_preview_raw = gr.JSON(
                    label="Raw DistilBERT predictions with scores",
                    value=None,
                )

        # Tab 2: Knowledge Extraction
        with gr.Tab("Stage 2: Knowledge Extraction"):
            gr.Markdown("""
## Stage 2: Knowledge Extraction

This stage extracts structured facts and relationships from your text using DSPy signatures with local MLX inference. This replaces graphiti-core's
LLM-based extraction while maintaining structured output guarantees.

### What Happens Here

1. **Fact Extraction**: DSPy `FactExtractionSignature` extracts entity-specific facts grounded in the text
2. **Relationship Inference**: DSPy `RelationshipSignature` infers relationships between entities using the extracted facts
3. **LLM Edge Detection** (optional): `EntityEdgeDetectionSignature` detects relationships directly from text, providing an alternative extraction path
4. **Relationship Merging**: Both extraction methods are deduplicated and merged into a final relationship set

### Why This Matters

Facts provide grounding for relationship extraction. The two-stage approach (fact-based + direct edge detection) improves recall by catching relationships
that might be missed by either method alone. Temporal metadata (`valid_at`, `invalid_at`) is extracted inline during relationship inference.

---
            """)

            gr.Markdown("**Extracted Facts** - Entity-specific facts that ground relationship inference.")
            facts_output = gr.JSON(label="Facts")

            gr.Markdown("**DSPy Relationships** - Core relationship extraction using fact-based reasoning and temporal inference.")
            base_relationships_output = gr.JSON(label="DSPy Relationships")

            gr.Markdown("**LLM Edge Detection** - Optional direct relationship extraction (enabled when 'LLM edge detection' toggle is on).")
            llm_relationships_output = gr.JSON(label="LLM Edge Detection Results")

            gr.Markdown("**Merged Relationships** - Final deduplicated relationship set combining both extraction methods.")
            relationships_output = gr.JSON(label="Final Merged Relationships")

        # Tab 3: Graph Assembly & Resolution
        with gr.Tab("Stage 3: Graph Assembly & Resolution"):
            gr.Markdown("""
## Stage 3: Graph Assembly & Resolution

This stage builds graph objects and resolves them against existing data. Entities are matched using exact and fuzzy matching (MinHash + Jaccard).
Relationships are deduplicated and checked for contradictions.

### What Happens Here

1. **Episodic Node Creation**: An `EpisodicNode` is created for your text with a UUID and timestamp
2. **Entity Resolution**: Extracted entities are matched against existing graph entities using exact name matching, then fuzzy matching (MinHash similarity)
3. **Entity Edge Construction**: Relationships are converted to `EntityEdge` objects with temporal metadata
4. **Edge Pointer Resolution**: Edge source/target UUIDs are remapped based on entity deduplication decisions
5. **Temporal Enrichment**: Dateparser validates DSPy-extracted temporal metadata and applies cue-based heuristics ("since", "until", "currently", etc.)
6. **Edge Resolution**: Entity edges are deduplicated against existing edges; episode lists are merged
7. **Contradiction Detection**: `resolve_edge_contradictions()` identifies conflicting relationships and invalidates outdated ones
8. **Episodic Edges**: MENTIONS edges are created linking the episode to each referenced entity

### Why This Matters

Resolution prevents duplicate entities and edges from polluting your graph. Fuzzy matching catches variations like "Bob Smith" vs "Robert Smith".
Contradiction handling maintains temporal consistency by invalidating relationships that conflict with newer information.

---
            """)

            gr.Markdown("**Complete Graph Structure** - All nodes and edges assembled for this episode.")
            graphiti_output = gr.JSON(label="Graphiti Objects (Episode + Nodes + Edges)")

            gr.Markdown("**Entity Resolution Records** - Tracks how each entity was resolved: exact match, fuzzy match, or new creation.")
            dedupe_output = gr.JSON(label="Entity Deduplication Decisions")

            gr.Markdown("**Edge Resolution Records** - Shows whether relationships were merged with existing edges or created as new.")
            edge_resolution_output = gr.JSON(label="Relationship Resolution Decisions")

            gr.Markdown("**Invalidated Edges** - Relationships marked invalid due to contradictions with newer information.")
            invalidated_edges_output = gr.JSON(label="Contradicted/Invalidated Relationships")

            gr.Markdown("**Temporal Enrichment** - Dateparser validation results showing how temporal metadata was enriched or validated.")
            temporal_enrichment_output = gr.JSON(label="Temporal Metadata Enrichment Records")

        # Tab 4: Entity Enrichment
        with gr.Tab("Stage 4: Entity Enrichment"):
            gr.Markdown("""
## Stage 4: Entity Enrichment

This stage adds metadata to entities to improve search, retrieval, and context understanding. Attributes are derived from NER labels,
while summaries are generated using DSPy.

### What Happens Here

1. **Attribute Extraction**: NER labels (PER/ORG/LOC) are mapped to Graphiti labels (Person/Organization/Location) and attached to entities
2. **Entity Summary Generation**: DSPy `EntitySummarySignature` generates concise summaries for each entity using facts and relationships
3. **Embeddings** (stubbed): Placeholder for future local Qwen embedder integration
4. **Reranker** (stubbed): Placeholder for future cross-encoder integration

### Why This Matters

Attributes and labels enable type-specific filtering and validation. Summaries improve retrieval by providing context at a glance.
Embeddings (when integrated) will power semantic search across your knowledge graph.

---
            """)

            gr.Markdown("**Entity Attributes** - NER-derived labels and custom properties attached to each entity.")
            attributes_output = gr.JSON(label="Attributes & Labels")

            gr.Markdown("**Entity Summaries** - DSPy-generated summaries providing entity context at a glance.")
            summaries_output = gr.JSON(label="Entity Summaries")

            gr.Markdown("**Future Integration Stubs** - Embeddings and reranking will be integrated when local models are ready.")
            with gr.Row():
                embedding_stub_output = gr.JSON(label="Embedding Status")
                reranker_stub_output = gr.JSON(label="Reranker Status")

        # Tab 5: Persistence & Visualization
        with gr.Tab("Stage 5: Persistence & Visualization"):
            gr.Markdown("""
## Stage 5: Persistence & Visualization

This stage writes the assembled graph to FalkorDB and renders a visualization. The dry run above skips this step;
click "Write to FalkorDB" to persist your results.

### What Happens Here

1. **Bulk Write**: All nodes (episode + entities) and edges (entity edges + episodic edges + invalidated edges) are written to FalkorDB in a single transaction
2. **Verification**: The write result includes UUIDs for all created/updated objects
3. **Visualization**: Graphviz renders the graph showing the episode node, entity nodes, and their relationships

### Why This Matters

FalkorDB provides a persistent, queryable graph database compatible with Cypher. The visualization helps you verify that entities and
relationships were extracted and linked correctly.

---
            """)

            with gr.Row():
                write_falkor_btn = gr.Button("Write to FalkorDB", variant="primary", size="lg")
                db_stats_display = gr.Textbox(
                    label="Database Stats",
                    value=lambda: str(get_db_stats()),
                    interactive=False,
                )
                reset_btn = gr.Button("Reset Database", variant="stop")

            gr.Markdown("**Write Confirmation** - Result of the FalkorDB write operation with UUIDs.")
            write_output = gr.JSON(label="Write Result")

            gr.Markdown("**Graph Visualization** - Graphviz rendering showing your episode (blue), entities (green), and relationships.")
            graphviz_output = gr.Image(label="Graph Visualization")

    # State management
    config_state = gr.State(DEFAULT_UI_CONFIG)
    ner_raw_state = gr.State(None)
    entity_names_state = gr.State([])
    facts_state = gr.State(None)
    base_relationships_state = gr.State(None)
    llm_relationships_state = gr.State(None)
    relationships_state = gr.State(None)
    episode_state = gr.State(None)  # EpisodicNode
    entity_nodes_state = gr.State([])
    entity_edges_state = gr.State([])
    episodic_edges_state = gr.State([])  # EpisodicEdge list
    invalidated_edges_state = gr.State([])
    artifacts_state = gr.State(None)
    write_result_state = gr.State(None)

    def on_config_change(
        context_window,
        dedupe_enabled,
        attributes_enabled,
        summary_enabled,
        temporal_enabled,
        temporal_enrichment_enabled,
        llm_edges_enabled,
        reference_time_value,
    ):
        config = _build_config(
            context_window,
            dedupe_enabled,
            attributes_enabled,
            summary_enabled,
            temporal_enabled,
            temporal_enrichment_enabled,
            llm_edges_enabled,
        )
        reference_time = _parse_reference_time(reference_time_value)
        episodes = fetch_recent_episodes(
            config.group_id,
            reference_time,
            config.context_window,
        )
        return config, _context_to_json(episodes)

    config_inputs = [
        context_window_slider,
        dedupe_toggle,
        attributes_toggle,
        summary_toggle,
        temporal_toggle,
        temporal_enrichment_toggle,
        llm_edges_toggle,
        reference_time_input,
    ]

    for control in config_inputs:
        control.change(
            on_config_change,
            inputs=config_inputs,
            outputs=[config_state, context_output],
        )

    preview_inputs = [input_text, persons_only_filter]

    input_text.change(
        fn=_debounced_ner_preview,
        inputs=preview_inputs,
        outputs=[entity_preview_box, entity_preview_raw],
        trigger_mode="always_last",
        show_progress="hidden",
    )

    persons_only_filter.change(
        fn=_debounced_ner_preview,
        inputs=preview_inputs,
        outputs=[entity_preview_box, entity_preview_raw],
        show_progress="hidden",
    )

    def on_run_pipeline(
        text,
        persons_only,
        reference_time_value,
        config: PipelineConfig,
    ):
        if not text or not text.strip():
            message = "Enter episode text before running the pipeline."
            return (
                "(no entities)",
                [],
                {"error": message},
                {"error": message},
                {"error": message},
                {"error": message},
                {"error": message},
                {"records": []},
                {"records": []},
                {"summaries": []},
                {"records": []},
                {"invalidated_edges": []},
                {"records": []},
                _embedding_stub(),
                _reranker_stub(),
                None,
                [],
                None,
                None,
                None,
                None,
                None,
                [],
                [],
                [],
                [],
                None,
            )

        reference_time = _parse_reference_time(reference_time_value)

        try:
            artifacts = PIPELINE.run_episode(
                text=text,
                persons_only=persons_only,
                reference_time=reference_time,
                write=False,
                render_graph=False,
                config=config,
            )
        except GraphitiPipelineError as exc:
            message = getattr(exc, "message", str(exc))
            context = _context_to_json(
                fetch_recent_episodes(
                    config.group_id,
                    reference_time,
                    config.context_window,
                )
            )
            return (
                "(pipeline error)",
                context,
                {"error": message},
                {"error": message},
                {"error": message},
                {"error": message},
                {"error": message},
                {"records": []},
                {"records": []},
                {"summaries": []},
                {"records": []},
                {"invalidated_edges": []},
                {"records": []},
                _embedding_stub(),
                _reranker_stub(),
                None,
                [],
                None,
                None,
                None,
                None,
                None,
                [],
                [],
                [],
                [],
                None,
            )

        invalidated_edges_json = [
            edge.model_dump() for edge in artifacts.invalidated_edges
        ]

        return (
            artifacts.ner_display,
            artifacts.context_json,
            artifacts.facts_json or {"items": []},
            artifacts.base_relationships_json or {"items": []},
            artifacts.llm_relationships_json
            if artifacts.llm_relationships_json is not None
            else {"items": []},
            artifacts.relationships_json or {"items": []},
            artifacts.graphiti_json or {"error": "Graphiti data missing"},
            {"records": artifacts.dedupe_records},
            {"records": artifacts.entity_attributes_json},
            {"summaries": artifacts.entity_summaries_json},
            {"records": artifacts.edge_resolution_records},
            {"invalidated_edges": invalidated_edges_json},
            {"records": artifacts.temporal_enrichment_records},
            artifacts.embedding_stub,
            artifacts.reranker_stub,
            artifacts.ner_raw,
            artifacts.ner_entities,
            artifacts.facts,
            artifacts.base_relationships,
            artifacts.llm_relationships,
            artifacts.relationships,
            artifacts.episode,
            artifacts.entity_nodes,
            artifacts.entity_edges,
            artifacts.episodic_edges,
            artifacts.invalidated_edges,
            artifacts,
        )

    run_pipeline_btn.click(
        on_run_pipeline,
        inputs=[
            input_text,
            persons_only_filter,
            reference_time_input,
            config_state,
        ],
        outputs=[
            ner_output,
            context_output,
            facts_output,
            base_relationships_output,
            llm_relationships_output,
            relationships_output,
            graphiti_output,
            dedupe_output,
            attributes_output,
            summaries_output,
            edge_resolution_output,
            invalidated_edges_output,
            temporal_enrichment_output,
            embedding_stub_output,
            reranker_stub_output,
            ner_raw_state,
            entity_names_state,
            facts_state,
            base_relationships_state,
            llm_relationships_state,
            relationships_state,
            episode_state,
            entity_nodes_state,
            entity_edges_state,
            episodic_edges_state,
            invalidated_edges_state,
            artifacts_state,
        ],
    )

    def on_write_falkor(
        episode,
        entity_nodes,
        entity_edges,
        invalidated_edges,
        episodic_edges,
        config: PipelineConfig,
        reference_time_value,
    ):
        if not episode or not entity_nodes:
            return (
                {"error": "Run the pipeline before writing to FalkorDB."},
                None,
                str(get_db_stats()),
                None,
                _context_to_json(
                    fetch_recent_episodes(
                        config.group_id,
                        _parse_reference_time(reference_time_value),
                        config.context_window,
                    )
                ),
            )

        edges_for_write = list(entity_edges or []) + list(invalidated_edges or [])
        result = write_to_falkordb(
            episode,
            entity_nodes,
            edges_for_write,
            episodic_edges or [],
        )

        new_stats = str(get_db_stats())
        graph_img = render_verification_graph(result)
        reference_time = _parse_reference_time(reference_time_value)
        context = _context_to_json(
            fetch_recent_episodes(
                config.group_id,
                reference_time,
                config.context_window,
            )
        )
        return result, result, new_stats, graph_img, context

    write_falkor_btn.click(
        on_write_falkor,
        inputs=[
            episode_state,
            entity_nodes_state,
            entity_edges_state,
            invalidated_edges_state,
            episodic_edges_state,
            config_state,
            reference_time_input,
        ],
        outputs=[
            write_output,
            write_result_state,
            db_stats_display,
            graphviz_output,
            context_output,
        ],
    )

    def on_reset_db(config: PipelineConfig, reference_time_value):
        reset_database()
        reference_time = _parse_reference_time(reference_time_value)
        return (
            str(get_db_stats()),
            _context_to_json(
                fetch_recent_episodes(
                    config.group_id,
                    reference_time,
                    config.context_window,
                )
            ),
        )

    reset_btn.click(
        on_reset_db,
        inputs=[config_state, reference_time_input],
        outputs=[db_stats_display, context_output],
    )

if __name__ == "__main__":
    app.launch()
