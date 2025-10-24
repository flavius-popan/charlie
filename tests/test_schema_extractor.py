import dspy
from pydantic import BaseModel, Field
from typing import List

from dspy_outlines.schema_extractor import extract_output_schema

def test_extract_simple_schema():
    """Test extracting Pydantic model from simple signature."""

    class Answer(BaseModel):
        text: str
        confidence: float

    class QASignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: Answer = dspy.OutputField()

    schema = extract_output_schema(QASignature)

    assert schema == Answer

def test_extract_complex_schema():
    """Test extracting complex nested Pydantic model."""

    class Node(BaseModel):
        id: str
        label: str

    class Edge(BaseModel):
        source: str
        target: str

    class Graph(BaseModel):
        nodes: List[Node]
        edges: List[Edge]

    class GraphSignature(dspy.Signature):
        text: str = dspy.InputField()
        graph: Graph = dspy.OutputField()

    schema = extract_output_schema(GraphSignature)

    assert schema == Graph
