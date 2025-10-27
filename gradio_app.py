"""Gradio UI for knowledge graph extraction."""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gradio as gr
import dspy
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import tempfile
import os
from graphviz import Digraph
import time
import threading

from dspy_outlines import OutlinesLM, OutlinesAdapter
from distilbert_ner import predict_entities, format_entities

# Initialize LM and adapter
lm = OutlinesLM()
adapter = OutlinesAdapter()
dspy.configure(lm=lm, adapter=adapter)


# Pydantic models
class Node(BaseModel):
    id: int = Field(description="Unique identifier for the node")
    label: str = Field(
        description="Name of the entity (person, place, or concept). Include 'Author' or 'I' ONLY when the journal explicitly involves first-person experiences (direct interactions, feelings, thoughts)."
    )
    properties: dict = Field(default_factory=dict)


class Edge(BaseModel):
    source: int = Field(description="Source node ID")
    target: int = Field(description="Target node ID")
    label: str = Field(
        description="Relationship type in PRESENT TENSE (e.g., 'knows', 'works_with', 'feels_anxious_about', 'admires'). Use underscores between words. Max 3 words. Connect to Author node ONLY for direct interactions or emotional/subjective states."
    )
    properties: dict = Field(default_factory=dict)


class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(
        description="All entities mentioned. Include 'Author' or 'I' as a node ONLY when extracting direct interactions or emotional states. Observed third-party entities stand alone."
    )
    edges: List[Edge] = Field(
        description="Relationships in present tense. Author-connected edges for direct interactions (met_with, spoke_to) and subjective states (feels_about, admires, worried_about). Third-party relationships captured independently (e.g., Bob knows Alice)."
    )


# DSPy signature with optional entity hints
class ExtractKnowledgeGraph(dspy.Signature):
    """Extract a knowledge graph from a personal journal entry, capturing both the author's direct experiences and observed facts.

    Create an Author/I node ONLY when the entry describes:
    - Direct interactions (Author met_with Bob, Author spoke_to Alice)
    - Emotional or subjective states (Author feels_anxious_about work, Author admires mentor)

    For observed third-party relationships (Bob knows Alice, Sarah works_for Microsoft), create independent edges without forcing connection to the author.
    """

    text: str = dspy.InputField(
        desc="Journal entry text describing personal experiences, interactions, and observations"
    )
    known_entities: Optional[List[str]] = dspy.InputField(
        desc="Optional pre-extracted entity names to guide node creation",
        default=None,
    )
    graph: KnowledgeGraph = dspy.OutputField(
        desc="Graph with entities as nodes and relationships as edges (present tense). Balance author-centric edges (direct interactions, emotions) with independent observed relationships."
    )


# Create predictor
extractor = dspy.Predict(ExtractKnowledgeGraph)

# Global debounce state
_last_input_time = 0
_input_lock = threading.Lock()


def extract_ner_only(text: str):
    """Extract NER entities with 1 second debounce."""
    global _last_input_time

    if not text.strip():
        return "Please enter some text.", None

    # Record when this input arrived
    with _input_lock:
        current_time = time.time()
        _last_input_time = current_time

    # Wait for 1 second
    time.sleep(1.0)

    # Check if we're still the most recent input
    with _input_lock:
        if time.time() - _last_input_time < 0.95:  # Allow small timing variance
            # Another input came in, abort this execution
            return gr.update(), gr.update()

    try:
        # Extract entities using NER model
        ner_entities = predict_entities(text)

        # Format NER entities for display (always with labels and confidence)
        ner_display = format_entities(
            ner_entities, include_labels=True, include_confidence=True
        )
        ner_output = (
            json.dumps(ner_display, indent=2) if ner_display else "No entities detected"
        )

        return ner_output, ner_entities
    except Exception as e:
        return f"Error: {str(e)}", None


def update_hints_preview(use_hints: bool, persons_only: bool, ner_entities):
    """Update the hints preview based on current checkbox settings."""
    if not use_hints or not ner_entities:
        return "None"

    try:
        # Filter to Person entities only if checkbox is enabled
        filtered_entities = ner_entities
        if persons_only:
            filtered_entities = [
                e for e in ner_entities if e.get("label") == "PER"
            ]

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
                filtered_entities = [
                    e for e in ner_entities if e.get("label") == "PER"
                ]

            # Always use plain text format (no labels, no confidence)
            entity_hints = format_entities(
                filtered_entities,
                include_labels=False,
                include_confidence=False,
            )

        # Step 2: Extract graph with LLM (with or without hints)
        result = extractor(text=text, known_entities=entity_hints)
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

        # Create Graphviz visualization
        dot = Digraph()
        dot.attr(rankdir="LR")

        for node in graph.nodes:
            dot.node(str(node.id), node.label, shape="circle", width="1", height="1")

        for edge in graph.edges:
            dot.edge(str(edge.source), str(edge.target), label=edge.label)

        # Render to temp file (Gradio needs file path)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        dot.render(path.replace(".png", ""), format="png", cleanup=True)

        return (
            path,
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
with gr.Blocks(title="KG Builder Demo") as demo:
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
            "Alice met Bob at the coffee shop. Dr. Charlie Smith joined them later to discuss the research project with Professor Diana Lee.",
            "John works at Microsoft. He reports to Sarah, the VP of Engineering. Sarah previously worked with David at Google.",
            "Apple Inc. is headquartered in Cupertino, California. Tim Cook became CEO in 2011 after Steve Jobs.",
        ],
        inputs=text_input,
    )

if __name__ == "__main__":
    demo.launch()
