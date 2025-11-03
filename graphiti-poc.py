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
    gr.Markdown("# Phase 1 PoC: Text → Graph in FalkorDBLite")
    gr.Markdown(f"**Database:** `{DB_PATH}` | **Model Config:** `{MODEL_CONFIG}`")

    gr.Markdown("## Input & Configuration")
    with gr.Row(equal_height=True):
        with gr.Column(scale=3, min_width=400):
            input_text = gr.Textbox(
                label="Journal Entry",
                placeholder="Enter text here...",
                lines=8,
            )
            reference_time_input = gr.Textbox(
                label="Reference Time (ISO8601, optional)",
                placeholder="Defaults to current time",
            )
        with gr.Column(scale=2, min_width=320):
            with gr.Group():
                gr.Markdown("### Entity Detection Preview")
                persons_only_filter = gr.Checkbox(
                    label="Persons only", value=False
                )
                entity_preview_box = gr.Textbox(
                    label="Entities to extract",
                    value="Enter text to see detected entities.",
                    interactive=False,
                    lines=8,
                )
                entity_preview_raw = gr.JSON(
                    label="Detected entities (raw)",
                    value=None,
                )

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

    run_pipeline_btn = gr.Button("Run Pipeline (no write)", variant="primary")

    gr.Markdown("### Context Episodes")
    context_output = gr.JSON(
        label="Most recent episodes",
        value=INITIAL_CONTEXT,
    )

    gr.Markdown("## Entity Recognition (DistilBERT NER)")
    ner_output = gr.Textbox(label="Entity Names", interactive=False)

    gr.Markdown("## DSPy Extraction")
    facts_output = gr.JSON(label="Extracted Facts")
    base_relationships_output = gr.JSON(label="DSPy Relationships")
    llm_relationships_output = gr.JSON(label="LLM Edge Detection")
    relationships_output = gr.JSON(label="Merged Relationships")

    gr.Markdown("## Graph Assembly & Deduplication")
    graphiti_output = gr.JSON(label="Graphiti Objects")
    dedupe_output = gr.JSON(label="Entity Resolution Records")
    edge_resolution_output = gr.JSON(label="Relationship Resolution Records")
    invalidated_edges_output = gr.JSON(label="Invalidated Relationships")

    gr.Markdown("## Temporal Enrichment")
    temporal_enrichment_output = gr.JSON(label="Temporal Metadata (dateparser validation)")

    gr.Markdown("## Attributes & Summaries")
    attributes_output = gr.JSON(label="Entity Attributes")
    summaries_output = gr.JSON(label="Entity Summaries")

    gr.Markdown("## Embeddings & Reranker (Stubs)")
    embedding_stub_output = gr.JSON(label="Embedding Status")
    reranker_stub_output = gr.JSON(label="Reranker Status")

    gr.Markdown("## Persistence & Visualization")
    with gr.Row():
        write_falkor_btn = gr.Button("Write to FalkorDB", variant="primary")
        db_stats_display = gr.Textbox(
            label="Database Stats",
            value=lambda: str(get_db_stats()),
            interactive=False,
        )
        reset_btn = gr.Button("Reset Database", variant="stop")

    write_output = gr.JSON(label="Write Confirmation")
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
