"""Gradio UI for knowledge graph extraction."""

import gradio as gr
import dspy
from pydantic import BaseModel, Field
from typing import List
import json
import tempfile
import os
from graphviz import Digraph

from dspy_outlines import OutlinesDSPyLM

# Initialize LM
lm = OutlinesDSPyLM()
dspy.configure(lm=lm)

# Pydantic models
class Node(BaseModel):
    id: str = Field(description="Unique identifier for the node")
    label: str = Field(description="Name of the entity")
    properties: dict = Field(default_factory=dict)

class Edge(BaseModel):
    source: str = Field(description="Source node ID")
    target: str = Field(description="Target node ID")
    label: str = Field(description="Type of relationship")
    properties: dict = Field(default_factory=dict)

class KnowledgeGraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

# DSPy signature
class ExtractKnowledgeGraph(dspy.Signature):
    """Extract knowledge graph of people and relationships."""
    text: str = dspy.InputField()
    graph: KnowledgeGraph = dspy.OutputField()

# Create predictor
extractor = dspy.Predict(ExtractKnowledgeGraph)

def extract_and_display(text: str):
    """Extract knowledge graph and generate visualization."""
    if not text.strip():
        return "Please enter some text.", None

    try:
        # Extract graph
        result = extractor(text=text)
        graph = result.graph

        # Format JSON
        json_output = json.dumps(graph.model_dump(), indent=2)

        # Create Graphviz visualization
        dot = Digraph()
        dot.attr(rankdir='LR')

        for node in graph.nodes:
            dot.node(str(node.id), node.label, shape='circle', width='1', height='1')

        for edge in graph.edges:
            dot.edge(str(edge.source), str(edge.target), label=edge.label)

        # Render to temp file (Gradio needs file path)
        fd, path = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        dot.render(path.replace('.png', ''), format='png', cleanup=True)

        return json_output, path

    except Exception as e:
        return f"Error: {str(e)}", None

# Create Gradio interface
with gr.Blocks(title="Knowledge Graph Extractor") as demo:
    gr.Markdown("# Knowledge Graph Extraction")
    gr.Markdown("Extract entities and relationships using DSPy + Outlines + MLX")

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Enter text to analyze...",
                lines=10
            )
            extract_btn = gr.Button("Extract Knowledge Graph", variant="primary")

        with gr.Column():
            json_output = gr.Code(label="Extracted Graph (JSON)", language="json")
            graph_viz = gr.Image(label="Graph Visualization", type="filepath")

    extract_btn.click(
        fn=extract_and_display,
        inputs=[text_input],
        outputs=[json_output, graph_viz]
    )

    # Examples
    gr.Examples(
        examples=[
            "Alice met Bob at the coffee shop. Dr. Charlie Smith joined them later to discuss the research project with Professor Diana Lee.",
            "John works at Microsoft. He reports to Sarah, the VP of Engineering. Sarah previously worked with David at Google.",
        ],
        inputs=text_input
    )

if __name__ == "__main__":
    demo.launch()
