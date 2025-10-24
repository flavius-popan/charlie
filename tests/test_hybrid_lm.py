import dspy
from pydantic import BaseModel, Field
from typing import List

from dspy_outlines.hybrid_lm import OutlinesDSPyLM

def test_hybrid_lm_knowledge_graph_extraction():
    """Test full hybrid LM with knowledge graph extraction."""

    # Define Pydantic models (same as dspy-poc.py)
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

    # Define DSPy signature
    class ExtractKnowledgeGraph(dspy.Signature):
        """Extract knowledge graph of people and relationships."""
        text: str = dspy.InputField()
        graph: KnowledgeGraph = dspy.OutputField()

    # Initialize hybrid LM
    lm = OutlinesDSPyLM()
    dspy.configure(lm=lm)

    # Create predictor
    extractor = dspy.Predict(ExtractKnowledgeGraph)

    # Test extraction
    text = "Alice met Bob at the coffee shop. Charlie joined them."
    result = extractor(text=text)

    # Verify result
    assert hasattr(result, 'graph')
    assert isinstance(result.graph, KnowledgeGraph)
    assert len(result.graph.nodes) >= 3  # At least Alice, Bob, Charlie
    assert all(isinstance(n, Node) for n in result.graph.nodes)
    assert all(isinstance(e, Edge) for e in result.graph.edges)
