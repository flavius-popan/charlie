"""DSPy signatures for knowledge graph extraction."""
import dspy
from models import Facts, Relationships


class FactExtractionSignature(dspy.Signature):
    """Extract factual statements about entities from text."""
    text: str = dspy.InputField(desc="The input text to analyze")
    entities: list[str] = dspy.InputField(desc="NER-detected entity names")
    facts: Facts = dspy.OutputField(desc="Facts about entities")


class RelationshipSignature(dspy.Signature):
    """Infer relationships between entities based on facts."""
    text: str = dspy.InputField(desc="Original input text")
    facts: Facts = dspy.InputField(desc="Extracted facts about entities")
    entities: list[str] = dspy.InputField(desc="Entity names to constrain relationships")
    relationships: Relationships = dspy.OutputField(desc="Relationships between entities")
