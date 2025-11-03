"""DSPy signatures for knowledge graph extraction."""
import dspy

from models import EntityAttributes, EntitySummaries, Facts, Relationships


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


class EntitySummarySignature(dspy.Signature):
    """Summarize each entity based on context."""
    text: str = dspy.InputField(desc="Original episode text")
    entities: list[str] = dspy.InputField(desc="Entity names to summarize")
    facts: Facts = dspy.InputField(desc="Facts associated with the entities")
    relationships: Relationships = dspy.InputField(
        desc="Relationships between entities for additional context",
    )
    summaries: EntitySummaries = dspy.OutputField(
        desc="Entity summaries aligned with Graphiti schema expectations",
    )


class EntityAttributesSignature(dspy.Signature):
    """Extract attributes and labels for each entity."""
    text: str = dspy.InputField(desc="Episode text to analyze")
    entities: list[str] = dspy.InputField(desc="Entity names to enrich")
    attributes: EntityAttributes = dspy.OutputField(
        desc="Labels and attribute dictionaries for entities",
    )
