"""Phase 1 PoC: Text → Graph in FalkorDBLite with Gradio UI."""
import gradio as gr
import dspy
from dspy_outlines.adapter import OutlinesAdapter
from dspy_outlines.lm import OutlinesLM
from settings import MODEL_CONFIG, DB_PATH
from falkordb_utils import get_db_stats, reset_database
from distilbert_ner import predict_entities
from entity_utils import deduplicate_entities
from signatures import FactExtractionSignature

# Configure DSPy once at module level
dspy.settings.configure(
    adapter=OutlinesAdapter(),
    lm=OutlinesLM(generation_config=MODEL_CONFIG),
)


# Stage 1: NER processing function
def process_ner(text: str, persons_only: bool):
    """
    Stage 1: Extract entities using NER.

    Returns: (entity_names_list, raw_ner_output, display_string)
    """
    if not text.strip():
        return [], None, ""

    # Run NER
    raw_entities = predict_entities(text)

    # Filter by type if requested
    if persons_only:
        filtered = [e for e in raw_entities if e["label"] == "PER"]
    else:
        filtered = raw_entities

    # Extract entity names and deduplicate
    entity_names = [e["text"] for e in filtered]
    unique_names = deduplicate_entities(entity_names)

    # Format for display
    display = "\n".join(unique_names) if unique_names else "(no entities found)"

    return unique_names, raw_entities, display


# Stage 2: Fact extraction function
def extract_facts(text: str, entity_names: list[str]):
    """
    Stage 2: Extract facts using DSPy.

    Returns: (Facts object, JSON for display)
    """
    if not text.strip() or not entity_names:
        return None, {"error": "Need text and entities"}

    try:
        fact_predictor = dspy.Predict(FactExtractionSignature)
        facts = fact_predictor(text=text, entities=entity_names).facts

        # Convert to JSON for display
        facts_json = {
            "items": [
                {"entity": f.entity, "text": f.text}
                for f in facts.items
            ]
        }

        return facts, facts_json
    except Exception as e:
        return None, {"error": str(e)}


# Build Gradio interface
with gr.Blocks(title="Phase 1 PoC: Graphiti Pipeline") as app:
    gr.Markdown("# Phase 1 PoC: Text → Graph in FalkorDBLite")
    gr.Markdown(f"**Database:** `{DB_PATH}` | **Model Config:** `{MODEL_CONFIG}`")

    # Database stats display
    with gr.Row():
        db_stats_display = gr.Textbox(
            label="Database Stats",
            value=lambda: str(get_db_stats()),
            interactive=False
        )
        reset_btn = gr.Button("Reset Database", variant="stop")

    # Stage 0: Input
    gr.Markdown("## Stage 0: Input Text")
    input_text = gr.Textbox(
        label="Journal Entry",
        placeholder="Enter text here...",
        lines=5
    )

    # Stage 1: NER
    gr.Markdown("## Stage 1: NER Entities")
    ner_output = gr.Textbox(label="Entity Names", interactive=False)
    persons_only_filter = gr.Checkbox(label="Persons only", value=False)

    # Stage 2: Facts
    gr.Markdown("## Stage 2: Fact Extraction")
    run_facts_btn = gr.Button("Run Facts", variant="primary")
    facts_output = gr.JSON(label="Extracted Facts")

    # Stage 3: Relationships
    gr.Markdown("## Stage 3: Relationship Inference")
    run_relationships_btn = gr.Button("Run Relationships", variant="primary")
    relationships_output = gr.JSON(label="Inferred Relationships")

    # Stage 4: Graphiti Objects
    gr.Markdown("## Stage 4: Build Graphiti Objects")
    build_graphiti_btn = gr.Button("Build Graphiti Objects", variant="primary")
    graphiti_output = gr.JSON(label="EntityNode + EntityEdge Objects")

    # Stage 5: FalkorDB Write
    gr.Markdown("## Stage 5: Write to FalkorDB")
    write_falkor_btn = gr.Button("Write to Falkor", variant="primary")
    write_output = gr.JSON(label="Write Confirmation")

    # Stage 6: Graphviz Preview
    gr.Markdown("## Stage 6: Graphviz Verification")
    graphviz_output = gr.Image(label="Graph Visualization")

    # State management
    ner_raw_state = gr.State(None)
    entity_names_state = gr.State([])
    facts_state = gr.State(None)
    relationships_state = gr.State(None)
    entity_nodes_state = gr.State([])
    entity_edges_state = gr.State([])
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
        trigger_mode="always_last"
    )

    persons_only_filter.change(
        on_text_change,
        inputs=[input_text, persons_only_filter],
        outputs=[ner_output, entity_names_state, ner_raw_state]
    )

    # Stage 2: Fact Extraction
    def on_run_facts(text, entity_names):
        facts, facts_json = extract_facts(text, entity_names)
        return facts_json, facts

    run_facts_btn.click(
        on_run_facts,
        inputs=[input_text, entity_names_state],
        outputs=[facts_output, facts_state]
    )

    def on_reset_db():
        msg = reset_database()
        return str(get_db_stats())

    reset_btn.click(
        on_reset_db,
        outputs=[db_stats_display]
    )

if __name__ == "__main__":
    app.launch()
