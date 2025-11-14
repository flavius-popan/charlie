import dspy
from pydantic import BaseModel, Field
from typing import List

from dspy_outlines import OutlinesLM, OutlinesAdapter


def test_outlines_actually_constrains(model_path):
    """Verify Outlines enforces schema constraints.

    This test ensures that constrained generation works by validating
    that outputs always match the schema structure 100% of the time.
    Without proper constraints, LLMs occasionally generate invalid structures.
    """

    class StrictCount(BaseModel):
        count: int

    class CountSig(dspy.Signature):
        text: str = dspy.InputField()
        result: StrictCount = dspy.OutputField()

    lm = OutlinesLM(model_path=model_path)
    adapter = OutlinesAdapter()
    dspy.configure(lm=lm, adapter=adapter)

    predictor = dspy.Predict(CountSig)

    for _ in range(10):
        result = predictor(text="How many people?")
        assert isinstance(result.result.count, int)


def test_knowledge_graph_extraction(model_path):
    """Test full integration with knowledge graph extraction."""

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

    class ExtractKnowledgeGraph(dspy.Signature):
        """Extract knowledge graph of people and relationships."""
        text: str = dspy.InputField()
        graph: KnowledgeGraph = dspy.OutputField()

    lm = OutlinesLM(model_path=model_path)
    adapter = OutlinesAdapter()
    dspy.configure(lm=lm, adapter=adapter)

    extractor = dspy.Predict(ExtractKnowledgeGraph)

    text = "Alice met Bob at the coffee shop. Charlie joined them."
    result = extractor(text=text)

    assert hasattr(result, 'graph')
    assert isinstance(result.graph, KnowledgeGraph)
    assert len(result.graph.nodes) >= 3
    assert all(isinstance(n, Node) for n in result.graph.nodes)
    assert all(isinstance(e, Edge) for e in result.graph.edges)
