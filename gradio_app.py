"""Gradio UI for knowledge graph extraction."""

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
import dspy
import gradio as gr
from graphviz import Digraph

from distilbert_ner import format_entities, predict_entities
from dspy_outlines import KGExtractionModule, OutlinesAdapter, OutlinesLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Reduce noisy loggers - keep INFO to see HTTP requests, but hide DEBUG noise
logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("asyncio").setLevel(logging.WARNING)  # Hide selector spam

# Initialize LM and adapter
lm = OutlinesLM()
adapter = OutlinesAdapter()
dspy.configure(lm=lm, adapter=adapter)


# Create KG extraction module
kg_extractor = KGExtractionModule()
_PROMPTS_PATH = Path("prompts/kg_extraction_optimized.json")
if _PROMPTS_PATH.exists():
    try:
        kg_extractor.load_prompts(str(_PROMPTS_PATH))
        logger.info("Loaded optimized prompts from %s", _PROMPTS_PATH)
    except Exception as exc:
        logger.warning("Failed to load optimized prompts: %s", exc)


def _render_graph_image(graph) -> str:
    """Render the extracted graph with improved styling."""
    dot = Digraph(format="png")
    dot.attr(
        "graph",
        rankdir="LR",
        splines="spline",
        pad="0.35",
        nodesep="0.7",
        ranksep="1.0",
        bgcolor="transparent",
    )
    dot.attr(
        "node",
        shape="circle",
        style="filled",
        fontname="Helvetica",
        fontsize="11",
        color="transparent",
        fontcolor="#1f2937",
    )
    dot.attr(
        "edge",
        color="#60a5fa",
        penwidth="1.6",
        arrowsize="1.0",
        arrowhead="vee",
        fontname="Helvetica",
        fontcolor="#e2e8f0",
        fontsize="10",
    )

    for node in graph.nodes:
        label = node.label
        normalized = label.lower()
        is_author = normalized in {"author", "i"}
        fillcolor = "#facc15" if is_author else "#dbeafe"
        fontcolor = "#1f2937"
        dot.node(
            str(node.id),
            label,
            fillcolor=fillcolor,
            fontcolor=fontcolor,
            tooltip=label,
        )

    for edge in graph.edges:
        edge_label = edge.label.replace("_", " ")
        dot.edge(
            str(edge.source),
            str(edge.target),
            label=edge_label,
            color="#60a5fa",
            fontcolor="#e2e8f0",
            tooltip=edge_label,
        )

    fd, tmp_png_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    dot.render(tmp_png_path[:-4], format="png", cleanup=True)
    return tmp_png_path

# Global debounce state
_last_input_time = 0
_input_lock = threading.Lock()


def extract_ner_only(text: str):
    """Extract NER entities with 1 second debounce."""
    global _last_input_time
    logger.debug(f"extract_ner_only called with text length: {len(text)}")

    if not text.strip():
        logger.debug("Empty text, returning early")
        return "Please enter some text.", None

    # Record when this input arrived
    with _input_lock:
        current_time = time.time()
        _last_input_time = current_time
        logger.debug(f"Input timestamp recorded: {current_time}")

    # Wait for 1 second
    time.sleep(1.0)

    # Check if we're still the most recent input
    with _input_lock:
        if time.time() - _last_input_time < 0.95:  # Allow small timing variance
            logger.debug("Debounced - newer input received")
            return gr.update(), gr.update()

    try:
        logger.debug("Starting NER entity extraction")
        ner_entities = predict_entities(text)
        logger.debug(f"NER extracted {len(ner_entities)} entities")

        ner_display = format_entities(
            ner_entities, include_labels=True, include_confidence=True
        )
        ner_output = (
            json.dumps(ner_display, indent=2) if ner_display else "No entities detected"
        )

        return ner_output, ner_entities
    except Exception as e:
        logger.error(f"NER extraction failed: {str(e)}", exc_info=True)
        return f"Error: {str(e)}", None


def update_hints_preview(use_hints: bool, persons_only: bool, ner_entities):
    """Update the hints preview based on current checkbox settings."""
    if not use_hints or not ner_entities:
        return "None"

    try:
        # Filter to Person entities only if checkbox is enabled
        filtered_entities = ner_entities
        if persons_only:
            filtered_entities = [e for e in ner_entities if e.get("label") == "PER"]

        # Always use plain text format (no labels, no confidence)
        entity_hints = format_entities(
            filtered_entities,
            include_labels=False,
            include_confidence=False,
        )
        return f"{json.dumps(entity_hints, indent=2)}"
    except Exception:
        return "None"


def extract_and_display(
    text: str,
    use_hints: bool,
    persons_only: bool,
    ner_entities,
):
    """Extract knowledge graph with NER fusion and generate visualization."""
    if not text.strip():
        return None, None, gr.update()

    try:
        # Step 1: Prepare entity hints based on toggle settings
        entity_hints = None
        if use_hints and ner_entities:
            # Filter to Person entities only if checkbox is enabled
            filtered_entities = ner_entities
            if persons_only:
                filtered_entities = [e for e in ner_entities if e.get("label") == "PER"]

            # Always use plain text format (no labels, no confidence)
            entity_hints = format_entities(
                filtered_entities,
                include_labels=False,
                include_confidence=False,
            )

        # Step 2: Extract graph with LLM (with or without hints)
        result = kg_extractor(text=text, known_entities=entity_hints)
        graph = result.graph

        # Capture adapter information
        adapter_name_map = {
            "chat": "Chat (field-marker format)",
            "json": "JSON (unconstrained with json_repair)",
            "outlines_json": "OutlinesJSON (constrained generation)",
        }
        adapter_display = adapter_name_map.get(adapter.last_adapter_used, "Unknown")

        # Format JSON
        json_output = json.dumps(graph.model_dump(), indent=2)

        # Create Graphviz visualization with enhanced styling
        graph_image = _render_graph_image(graph)

        return (
            graph_image,
            json_output,
            gr.update(value=adapter_display, visible=True),
        )

    except Exception as e:
        return (
            None,
            f"Error: {str(e)}",
            gr.update(value="Error during extraction", visible=True),
        )


# Create Gradio interface
with gr.Blocks(title="KG Builder Demo", analytics_enabled=False) as demo:
    gr.Markdown("# Charlie - Knowledge Graph Extraction")
    gr.Markdown("**Model Fusion**: distilbert-NER x Qwen3-4B")

    # State to hold NER entities between steps
    ner_entities_state = gr.State(None)

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Input Text", placeholder="Enter text to analyze...", lines=10
            )

            gr.Markdown("### NER Hint Options")
            use_hints = gr.Checkbox(
                label="Use NER entity hints",
                value=True,
                info="ex: Microsoft",
            )
            persons_only = gr.Checkbox(
                label="Persons only",
                value=False,
                info="Filter NER entities to People only",
            )

            extract_btn = gr.Button("Extract Knowledge Graph", variant="primary")

            adapter_used = gr.Textbox(
                label="Adapter Used",
                interactive=False,
                visible=False,
            )

        with gr.Column():
            ner_output = gr.Code(label="distilbert-NER Output", language="json")
            hints_sent = gr.Code(label="Entity Hints Sent to LLM", language="json")
            graph_viz = gr.Image(label="Graph Visualization", type="filepath")
            json_output = gr.Code(label="Extracted Graph (JSON)", language="json")

    # When text changes, extract NER with 1 second debounce
    text_input.change(
        fn=extract_ner_only,
        inputs=[text_input],
        outputs=[ner_output, ner_entities_state],
        trigger_mode="always_last",
        show_progress="hidden",
    ).then(
        fn=update_hints_preview,
        inputs=[use_hints, persons_only, ner_entities_state],
        outputs=[hints_sent],
        show_progress="hidden",
    )

    # When extract button is clicked, show hints first, then extract graph
    extract_btn.click(
        fn=update_hints_preview,
        inputs=[use_hints, persons_only, ner_entities_state],
        outputs=[hints_sent],
        show_progress="hidden",
    ).then(
        fn=extract_and_display,
        inputs=[
            text_input,
            use_hints,
            persons_only,
            ner_entities_state,
        ],
        outputs=[graph_viz, json_output, adapter_used],
    )

    # Update hints preview when use_hints checkbox changes
    use_hints.change(
        fn=update_hints_preview,
        inputs=[use_hints, persons_only, ner_entities_state],
        outputs=[hints_sent],
        show_progress="hidden",
    )

    # Update hints preview when persons_only checkbox changes
    persons_only.change(
        fn=update_hints_preview,
        inputs=[use_hints, persons_only, ner_entities_state],
        outputs=[hints_sent],
        show_progress="hidden",
    )

    # Examples
    gr.Examples(
        examples=[
            "Residency schedule got scrambled again and the cohort Discord is a whole meltdown; I spent lunch break rewriting coverage spreadsheets while the therapist keeps texting reminders to drink water and step outside.",
            "Daycare called about Maya's fever right as Slack started lighting up about the Q3 deck, so I reheated last night's pad thai, dialed into Nana's telehealth, and promised Leo I'd actually sit for dinner even if the slides aren't done.",
            "Dropped Jonah at robotics, nudged Dad through his cardiologist's walking plan, then converted two more VHS tapes because apparently the class reunion committee thinks I'm the only one who remembers how.",
            "HOA thread blew up over someone's solar install, the library shift ran long when the new catalog crashed, and now I'm drafting a note to council because the bus route cut is going to leave Doris and Ken stranded after church.",
        ],
        inputs=text_input,
    )

if __name__ == "__main__":
    demo.launch()
