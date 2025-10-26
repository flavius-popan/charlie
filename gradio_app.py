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
    label: str = Field(description="Name of the entity")
    properties: dict = Field(default_factory=dict)


class Edge(BaseModel):
    source: int = Field(description="Source node ID")
    target: int = Field(description="Target node ID")
    label: str = Field(description="Type of relationship")
    properties: dict = Field(default_factory=dict)


class KnowledgeGraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]


# DSPy signature with optional entity hints
class ExtractKnowledgeGraph(dspy.Signature):
    """Extract knowledge graph of people and relationships."""

    text: str = dspy.InputField(desc="Text to extract entities and relationships from")
    entity_hints: Optional[List[str]] = dspy.InputField(
        desc="Clues for potential entities to find",
        default=None,
    )
    graph: KnowledgeGraph = dspy.OutputField(
        desc="Knowledge graph with people as nodes and relationships as edges"
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


def update_hints_preview(
    use_hints: bool, include_labels: bool, include_confidence: bool, ner_entities
):
    """Update the hints preview based on current checkbox settings."""
    if not use_hints or not ner_entities:
        return "None"

    try:
        entity_hints = format_entities(
            ner_entities,
            include_labels=include_labels,
            include_confidence=include_confidence and include_labels,
        )
        return f"{json.dumps(entity_hints, indent=2)}"
    except Exception:
        return "None"


def extract_and_display(
    text: str,
    use_hints: bool,
    include_labels: bool,
    include_confidence: bool,
    ner_entities,
):
    """Extract knowledge graph with NER fusion and generate visualization."""
    if not text.strip():
        return None, None, None

    try:
        # Step 1: Prepare entity hints based on toggle settings
        entity_hints = None
        if use_hints and ner_entities:
            entity_hints = format_entities(
                ner_entities,
                include_labels=include_labels,
                include_confidence=include_confidence and include_labels,
            )

        # Step 2: Extract graph with LLM (with or without hints)
        result = extractor(text=text, entity_hints=entity_hints)
        graph = result.graph

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
            f"{json.dumps(entity_hints, indent=2) if entity_hints else 'None'}",
        )

    except Exception as e:
        return None, f"Error: {str(e)}", None


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
            include_labels = gr.Checkbox(
                label="Include entity types in hints",
                value=False,
                info="ex: 'Microsoft (Organization)'",
            )
            include_confidence = gr.Checkbox(
                label="Include confidence scores in hints",
                value=False,
                info="ex: 'Microsoft (entity_type:Organization, conf:0.99)'",
            )

            extract_btn = gr.Button("Extract Knowledge Graph", variant="primary")

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
        inputs=[use_hints, include_labels, include_confidence, ner_entities_state],
        outputs=[hints_sent],
        show_progress="hidden",
    )

    # When extract button is clicked, use the cached NER entities
    extract_btn.click(
        fn=extract_and_display,
        inputs=[
            text_input,
            use_hints,
            include_labels,
            include_confidence,
            ner_entities_state,
        ],
        outputs=[graph_viz, json_output, hints_sent],
    )

    # Toggle dependent checkboxes based on use_hints and update preview
    def update_checkbox_state(use_hints_value):
        return gr.update(interactive=use_hints_value), gr.update(
            interactive=use_hints_value
        )

    use_hints.change(
        fn=update_checkbox_state,
        inputs=[use_hints],
        outputs=[include_labels, include_confidence],
    ).then(
        fn=update_hints_preview,
        inputs=[use_hints, include_labels, include_confidence, ner_entities_state],
        outputs=[hints_sent],
        show_progress="hidden",
    )

    # Update hints preview when labels checkbox changes
    include_labels.change(
        fn=update_hints_preview,
        inputs=[use_hints, include_labels, include_confidence, ner_entities_state],
        outputs=[hints_sent],
        show_progress="hidden",
    )

    # Update hints preview when confidence checkbox changes
    include_confidence.change(
        fn=update_hints_preview,
        inputs=[use_hints, include_labels, include_confidence, ner_entities_state],
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
