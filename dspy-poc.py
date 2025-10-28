import dspy
from pydantic import BaseModel, Field
from typing import List
import json
import logging

from dspy_outlines import OutlinesLM, OutlinesAdapter

# Reduce noisy loggers - keep INFO to see HTTP requests, but hide DEBUG noise
logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("asyncio").setLevel(logging.WARNING)

# Configure Outlines+MLX hybrid LM
lm = OutlinesLM()
adapter = OutlinesAdapter()
dspy.configure(lm=lm, adapter=adapter)


# Pydantic models for knowledge graph
class Node(BaseModel):
    id: str = Field(description="Unique identifier for the node")
    label: str = Field(description="Name of the entity")
    properties: dict = Field(default_factory=dict, description="Additional attributes")


class Edge(BaseModel):
    source: str = Field(description="Source node ID")
    target: str = Field(description="Target node ID")
    label: str = Field(description="Type of relationship")
    properties: dict = Field(default_factory=dict, description="Additional attributes")


class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(description="List of entities (people)")
    edges: List[Edge] = Field(description="List of relationships between entities")


# DSPy signature for knowledge graph extraction
class ExtractKnowledgeGraph(dspy.Signature):
    """Extract a knowledge graph of people and their relationships from text."""

    text: str = dspy.InputField(desc="Text to extract entities and relationships from")
    graph: KnowledgeGraph = dspy.OutputField(
        desc="Knowledge graph with people as nodes and relationships as edges"
    )


# Create the predictor
extractor = dspy.Predict(ExtractKnowledgeGraph)


def extract_graph(text: str) -> KnowledgeGraph:
    """Extract knowledge graph from text."""
    result = extractor(text=text)
    return result.graph


# Interactive loop
if __name__ == "__main__":
    print("Knowledge Graph Extractor (Ctrl+D or Ctrl+C to exit)")
    print("=" * 50)
    print("\nPaste your text and press Enter twice:\n")

    while True:
        try:
            # Read multi-line input
            lines = []
            print("> ", end="", flush=True)
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)

            text = "\n".join(lines)

            if not text.strip():
                continue

            # Extract graph
            print("\nExtracting knowledge graph...")
            graph = extract_graph(text)

            # Output as formatted JSON
            print("\nExtracted Knowledge Graph:")
            print(json.dumps(graph.model_dump(), indent=2))
            print("\n" + "=" * 50 + "\n")

        except EOFError:
            print("\nExiting...")
            break
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("\n" + "=" * 50 + "\n")
