"""Phase 1 PoC: Text → Graph in FalkorDBLite with Gradio UI."""

import logging
import gradio as gr
from graphiti_pipeline import (
    build_graphiti_objects,
    extract_facts,
    get_db_stats,
    infer_relationships,
    process_ner,
    render_verification_graph,
    reset_database,
    write_to_falkordb,
)
from settings import DB_PATH, MODEL_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info(f"Starting Graphiti PoC with MODEL_CONFIG: {MODEL_CONFIG}")
logger.info(f"Database path: {DB_PATH}")


# Build Gradio interface
with gr.Blocks(title="Phase 1 PoC: Graphiti Pipeline") as app:
    gr.Markdown("# Phase 1 PoC: Text → Graph in FalkorDBLite")
    gr.Markdown(f"**Database:** `{DB_PATH}` | **Model Config:** `{MODEL_CONFIG}`")

    # Database stats display
    with gr.Row():
        db_stats_display = gr.Textbox(
            label="Database Stats", value=lambda: str(get_db_stats()), interactive=False
        )
        reset_btn = gr.Button("Reset Database", variant="stop")

    # Example text section
    gr.Markdown("### Example Text")
    example_text = gr.Textbox(
        value="Alice works at Microsoft in Seattle. She reports to Bob, who manages the engineering team.",
        interactive=False,
        show_label=False,
    )
    load_example_btn = gr.Button("Load Example", size="sm")

    # Stage 0: Input
    gr.Markdown("## Stage 0: Input Text")
    input_text = gr.Textbox(
        label="Journal Entry", placeholder="Enter text here...", lines=5
    )

    # Stage 1: NER
    gr.Markdown("## Stage 1: NER Entities")
    ner_output = gr.Textbox(label="Entity Names", interactive=False)
    persons_only_filter = gr.Checkbox(label="Persons only", value=False)

    # Stage 2: Facts
    gr.Markdown("## Stage 2: Fact Extraction")
    run_facts_btn = gr.Button("Run Facts", variant="primary")
    facts_output = gr.JSON(label="Extracted Facts (or error)")

    # Stage 3: Relationships
    gr.Markdown("## Stage 3: Relationship Inference")
    run_relationships_btn = gr.Button("Run Relationships", variant="primary")
    relationships_output = gr.JSON(label="Inferred Relationships (or error)")

    # Stage 4: Graphiti Objects
    gr.Markdown("## Stage 4: Build Graphiti Objects")
    build_graphiti_btn = gr.Button("Build Graphiti Objects", variant="primary")
    graphiti_output = gr.JSON(label="EntityNode + EntityEdge Objects (or error)")

    # Stage 5: FalkorDB Write
    gr.Markdown("## Stage 5: Write to FalkorDB")
    write_falkor_btn = gr.Button("Write to Falkor", variant="primary")
    write_output = gr.JSON(label="Write Confirmation (or error with traceback)")

    # Stage 6: Graphviz Preview
    gr.Markdown("## Stage 6: Graphviz Verification")
    graphviz_output = gr.Image(label="Graph Visualization")

    # State management
    ner_raw_state = gr.State(None)
    entity_names_state = gr.State([])
    facts_state = gr.State(None)
    relationships_state = gr.State(None)
    episode_state = gr.State(None)  # EpisodicNode
    entity_nodes_state = gr.State([])
    entity_edges_state = gr.State([])
    episodic_edges_state = gr.State([])  # EpisodicEdge list
    write_result_state = gr.State(None)

    # Event handlers

    # Stage 1: NER (automatic on text change)
    def on_text_change(text, persons_only):
        entity_names, raw_ner, display = process_ner(text, persons_only)
        return display, entity_names, raw_ner

    input_text.change(
        on_text_change,
        inputs=[input_text, persons_only_filter],
        outputs=[ner_output, entity_names_state, ner_raw_state],
        trigger_mode="always_last",
    )

    persons_only_filter.change(
        on_text_change,
        inputs=[input_text, persons_only_filter],
        outputs=[ner_output, entity_names_state, ner_raw_state],
    )

    # Stage 2: Fact Extraction
    def on_run_facts(text, entity_names):
        facts, facts_json = extract_facts(text, entity_names)
        return facts_json, facts

    run_facts_btn.click(
        on_run_facts,
        inputs=[input_text, entity_names_state],
        outputs=[facts_output, facts_state],
    )

    # Stage 3: Relationship Inference
    def on_run_relationships(text, facts, entity_names):
        relationships, rels_json = infer_relationships(text, facts, entity_names)
        return rels_json, relationships

    run_relationships_btn.click(
        on_run_relationships,
        inputs=[input_text, facts_state, entity_names_state],
        outputs=[relationships_output, relationships_state],
    )

    # Stage 4: Build Graphiti Objects
    def on_build_graphiti(text, entity_names, relationships):
        from datetime import datetime

        reference_time = datetime.now()

        episode, nodes, entity_edges, episodic_edges, json_output = (
            build_graphiti_objects(
                input_text=text,
                entity_names=entity_names,
                relationships=relationships,
                reference_time=reference_time,
            )
        )
        return json_output, episode, nodes, entity_edges, episodic_edges

    build_graphiti_btn.click(
        on_build_graphiti,
        inputs=[input_text, entity_names_state, relationships_state],
        outputs=[
            graphiti_output,
            episode_state,  # New
            entity_nodes_state,
            entity_edges_state,
            episodic_edges_state,  # New
        ],
    )

    # Stage 5: Write to FalkorDB (triggers Stage 6)
    def on_write_falkor(episode, entity_nodes, entity_edges, episodic_edges):
        result = write_to_falkordb(episode, entity_nodes, entity_edges, episodic_edges)
        # Update database stats
        new_stats = str(get_db_stats())
        # Render verification graph
        graph_img = render_verification_graph(result)
        return result, result, new_stats, graph_img

    write_falkor_btn.click(
        on_write_falkor,
        inputs=[
            episode_state,
            entity_nodes_state,
            entity_edges_state,
            episodic_edges_state,
        ],
        outputs=[write_output, write_result_state, db_stats_display, graphviz_output],
    )

    def on_reset_db():
        msg = reset_database()
        return str(get_db_stats())

    reset_btn.click(on_reset_db, outputs=[db_stats_display])

    # Load example handler
    def on_load_example():
        return example_text.value

    load_example_btn.click(on_load_example, outputs=[input_text])

if __name__ == "__main__":
    app.launch()
