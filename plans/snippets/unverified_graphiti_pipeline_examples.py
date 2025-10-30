
"""Unverified reference snippets extracted from plans/graphiti-models.md.

NOTE: These are pending review and should not be treated as production-ready
implementations. They remain here solely for exploratory purposes.
"""

# Each snippet retains its original context heading for traceability.

# --- Snippet 1 (lines 243-253; context: Inherited from Node:) ---
# NOTE: Unverified and pending further review.
class EntityNode(Node):
    name_embedding: list[float] | None = None
    summary: str = ""
    attributes: dict[str, Any] = {}

    # Inherited from Node:
    uuid: str
    name: str
    group_id: str
    labels: list[str]
    created_at: datetime

# --- Snippet 2 (lines 260-275; context: Inherited from Edge:) ---
# NOTE: Unverified and pending further review.
class EntityEdge(Edge):
    name: str  # Relationship type
    fact: str  # Textual description
    fact_embedding: list[float] | None = None
    episodes: list[str] = []  # Episode UUIDs
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    expired_at: datetime | None = None
    attributes: dict[str, Any] = {}

    # Inherited from Edge:
    uuid: str
    group_id: str
    source_node_uuid: str
    target_node_uuid: str
    created_at: datetime

# --- Snippet 3 (lines 282-287; context: Inherited from Edge: > EpisodicNode (graphiti_core/nodes.py:295-299)) ---
# NOTE: Unverified and pending further review.
class EpisodicNode(Node):
    source: EpisodeType  # message, text, or json
    source_description: str
    content: str  # Raw episode data
    valid_at: datetime  # When original document was created
    entity_edges: list[str] = []  # UUIDs of entity edges

# --- Snippet 4 (lines 294-297; context: target_node_uuid: entity UUID) ---
# NOTE: Unverified and pending further review.
class EpisodicEdge(Edge):
    # MENTIONS relationship from episode to entity
    # source_node_uuid: episode UUID
    # target_node_uuid: entity UUID

# --- Snippet 5 (lines 314-317; context: target_node_uuid: entity UUID > Mappings Between kg_extraction.py and Graphiti Models) ---
# NOTE: Unverified and pending further review.
class Node(BaseModel):
    id: int
    label: str  # Entity name
    properties: dict

# --- Snippet 6 (lines 322-329; context: target_node_uuid: entity UUID > Mappings Between kg_extraction.py and Graphiti Models) ---
# NOTE: Unverified and pending further review.
EntityNode(
    uuid=str(uuid4()),  # Generate proper UUID
    name=node.label,    # Map label → name
    labels=["Entity"],  # Or extract entity type
    group_id=group_id,
    attributes=node.properties,  # Map properties → attributes
    created_at=utc_now()
)

# --- Snippet 7 (lines 334-338; context: target_node_uuid: entity UUID > Mappings Between kg_extraction.py and Graphiti Models) ---
# NOTE: Unverified and pending further review.
class Edge(BaseModel):
    source: int  # Node ID
    target: int  # Node ID
    label: str   # Relationship type
    properties: dict

# --- Snippet 8 (lines 343-352; context: target_node_uuid: entity UUID > Mappings Between kg_extraction.py and Graphiti Models) ---
# NOTE: Unverified and pending further review.
EntityEdge(
    uuid=str(uuid4()),
    source_node_uuid=uuid_map[edge.source],  # Resolve ID → UUID
    target_node_uuid=uuid_map[edge.target],
    name=edge.label,  # Map label → name
    fact=f"{source_name} {edge.label} {target_name}",  # Generate fact
    group_id=group_id,
    created_at=utc_now(),
    attributes=edge.properties
)

# --- Snippet 9 (lines 382-609; context: ============================================================) ---
# NOTE: Unverified and pending further review.
from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk
from graphiti_core.driver.falkordb_driver import FalkorDriver
from datetime import datetime
from uuid import uuid4

async def custom_ingestion_pipeline(
    episode_body: str,
    reference_time: datetime,
    group_id: str,
    driver: FalkorDriver,
    embedder: EmbedderClient,
    dspy_modules: dict  # Our DSPy modules
):
    """
    Custom knowledge graph ingestion pipeline using graphiti models + DSPy.
    """

    # ============================================================
    # STEP 1: Retrieve Previous Episodes for Context
    # ============================================================
    previous_episodes = await retrieve_previous_episodes(
        driver=driver,
        group_id=group_id,
        reference_time=reference_time,
        limit=3  # EPISODE_WINDOW_LEN
    )

    # ============================================================
    # STEP 2: Create EpisodicNode
    # ============================================================
    episode = EpisodicNode(
        uuid=str(uuid4()),
        name=f"Episode {reference_time.isoformat()}",
        group_id=group_id,
        source=EpisodeType.message,
        source_description="User conversation",
        content=episode_body,
        valid_at=reference_time,
        created_at=utc_now(),
        labels=[]
    )

    # ============================================================
    # STEP 3: Extract Entities (DSPy)
    # ============================================================
    extract_module = dspy_modules['extract_entities']
    extraction_result = extract_module(
        episode_content=episode_body,
        previous_episodes=[ep.content for ep in previous_episodes],
        entity_types=[]  # Optional schema
    )

    # Convert DSPy output to EntityNode objects
    extracted_nodes = [
        EntityNode(
            uuid=str(uuid4()),
            name=entity.name,
            group_id=group_id,
            labels=entity.labels or ["Entity"],
            created_at=utc_now(),
            summary="",
            attributes=entity.attributes or {}
        )
        for entity in extraction_result.entities
    ]

    # ============================================================
    # STEP 4: Deduplicate Entities (DSPy)
    # ============================================================
    dedupe_module = dspy_modules['dedupe_nodes']

    # For each extracted entity, search for existing matches
    resolved_nodes = []
    uuid_map = {}  # Maps extracted UUID → resolved UUID

    for node in extracted_nodes:
        # Search existing graph by embedding similarity
        candidates = await search_similar_entities(
            driver=driver,
            embedder=embedder,
            entity_name=node.name,
            group_id=group_id,
            top_k=5
        )

        if candidates:
            # Use DSPy to determine if match exists
            dedupe_result = dedupe_module(
                extracted_entity=node.name,
                candidate_entities=[c.name for c in candidates],
                context=episode_body
            )

            if dedupe_result.is_duplicate:
                # Merge with existing entity
                matched = candidates[dedupe_result.match_index]
                uuid_map[node.uuid] = matched.uuid

                # Update existing entity (merge attributes, etc.)
                merged = merge_entity_nodes(matched, node)
                resolved_nodes.append(merged)
            else:
                # New entity
                uuid_map[node.uuid] = node.uuid
                resolved_nodes.append(node)
        else:
            # No candidates, definitely new
            uuid_map[node.uuid] = node.uuid
            resolved_nodes.append(node)

    # ============================================================
    # STEP 5: Extract Relationships (DSPy)
    # ============================================================
    edges_module = dspy_modules['extract_relationships']
    edges_result = edges_module(
        episode_content=episode_body,
        nodes=[{"uuid": n.uuid, "name": n.name} for n in resolved_nodes],
        reference_time=reference_time.isoformat(),
        edge_types=[]  # Optional schema
    )

    # Convert to EntityEdge objects
    extracted_edges = [
        EntityEdge(
            uuid=str(uuid4()),
            source_node_uuid=uuid_map[edge.source_uuid],
            target_node_uuid=uuid_map[edge.target_uuid],
            name=edge.relation_type,
            fact=edge.fact,
            group_id=group_id,
            created_at=utc_now(),
            valid_at=parse_datetime(edge.valid_at) if edge.valid_at else reference_time,
            invalid_at=parse_datetime(edge.invalid_at) if edge.invalid_at else None,
            episodes=[episode.uuid]
        )
        for edge in edges_result.edges
    ]

    # ============================================================
    # STEP 6: Deduplicate Edges (DSPy)
    # ============================================================
    dedupe_edges_module = dspy_modules['dedupe_edges']

    resolved_edges = []
    invalidated_edges = []

    for edge in extracted_edges:
        # Find existing edges between same node pair
        existing = await find_edges_between_nodes(
            driver=driver,
            source_uuid=edge.source_node_uuid,
            target_uuid=edge.target_node_uuid,
            group_id=group_id
        )

        if existing:
            dedupe_result = dedupe_edges_module(
                new_fact=edge.fact,
                existing_facts=[e.fact for e in existing],
                context=episode_body
            )

            if dedupe_result.is_contradiction:
                # Mark contradicted edge as invalid
                contradicted = existing[dedupe_result.contradicts_index]
                contradicted.invalid_at = reference_time
                invalidated_edges.append(contradicted)
                resolved_edges.append(edge)
            elif dedupe_result.is_duplicate:
                # Update existing edge
                matched = existing[dedupe_result.match_index]
                matched.episodes.append(episode.uuid)
                resolved_edges.append(matched)
            else:
                # New edge
                resolved_edges.append(edge)
        else:
            # No existing edges
            resolved_edges.append(edge)

    # ============================================================
    # STEP 7: Generate Embeddings
    # ============================================================
    for node in resolved_nodes:
        if not node.name_embedding:
            await node.generate_name_embedding(embedder)

    for edge in resolved_edges + invalidated_edges:
        if not edge.fact_embedding:
            await edge.generate_embedding(embedder)

    # ============================================================
    # STEP 8: Create Episodic Edges (MENTIONS)
    # ============================================================
    episodic_edges = [
        EpisodicEdge(
            uuid=str(uuid4()),
            source_node_uuid=episode.uuid,
            target_node_uuid=node.uuid,
            group_id=group_id,
            created_at=utc_now()
        )
        for node in resolved_nodes
    ]

    # Update episode with entity edge references
    episode.entity_edges = [e.uuid for e in resolved_edges + invalidated_edges]

    # ============================================================
    # STEP 9: Bulk Save to FalkorDB
    # ============================================================
    await add_nodes_and_edges_bulk(
        driver=driver,
        episodic_nodes=[episode],
        episodic_edges=episodic_edges,
        entity_nodes=resolved_nodes,
        entity_edges=resolved_edges + invalidated_edges,
        embedder=embedder
    )

    return {
        "episode": episode,
        "nodes": resolved_nodes,
        "edges": resolved_edges,
        "invalidated_edges": invalidated_edges
    }

# --- Snippet 10 (lines 631-648; context: ... etc) ---
# NOTE: Unverified and pending further review.
# Add custom processing at any step
async def custom_ingestion_with_hooks(
    episode_body: str,
    hooks: dict = None
):
    hooks = hooks or {}

    # ... extraction ...

    if 'post_extraction' in hooks:
        extracted_nodes = await hooks['post_extraction'](extracted_nodes)

    # ... deduplication ...

    if 'post_deduplication' in hooks:
        resolved_nodes = await hooks['post_deduplication'](resolved_nodes)

    # ... etc

# --- Snippet 11 (lines 657-660; context: ... etc > How to Bypass Automated Episode Processing) ---
# NOTE: Unverified and pending further review.
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk
from graphiti_core.driver.falkordb_driver import FalkorDriver

# --- Snippet 12 (lines 679-687; context: Output graphiti model directly) ---
# NOTE: Unverified and pending further review.
from graphiti_core.nodes import EntityNode

class ExtractEntitiesSignature(dspy.Signature):
    """Extract entities from episode content."""

    episode_content: str = dspy.InputField()

    # Output graphiti model directly
    entities: list[EntityNode] = dspy.OutputField()

# --- Snippet 13 (lines 692-714; context: Convert after extraction) ---
# NOTE: Unverified and pending further review.
class ExtractedEntity(BaseModel):
    """Intermediate model for extraction."""
    name: str
    entity_type: str
    attributes: dict = {}

class ExtractEntitiesSignature(dspy.Signature):
    episode_content: str = dspy.InputField()
    entities: list[ExtractedEntity] = dspy.OutputField()

# Convert after extraction
def convert_to_graphiti(entities: list[ExtractedEntity], group_id: str) -> list[EntityNode]:
    return [
        EntityNode(
            uuid=str(uuid4()),
            name=e.name,
            labels=[e.entity_type],
            group_id=group_id,
            attributes=e.attributes,
            created_at=utc_now()
        )
        for e in entities
    ]

# --- Snippet 14 (lines 728-962; context: Try to construct expected model from DSPy output) ---
# NOTE: Unverified and pending further review.
from graphiti_core.llm_client import LLMClient, LLMConfig
from graphiti_core.prompts.models import Message
from pydantic import BaseModel
import dspy
from dspy_outlines import OutlinesLM, OutlinesAdapter
import json
from typing import Any

class DSPyLLMClient(LLMClient):
    """
    Custom LLMClient that routes graphiti prompts to DSPy modules.

    This allows graphiti code to call DSPy modules without modification.
    """

    def __init__(
        self,
        dspy_modules: dict[str, dspy.Module],
        config: LLMConfig | None = None,
        cache: bool = False
    ):
        super().__init__(config=config, cache=cache)

        # Configure DSPy
        lm = OutlinesLM(
            model_name="mlx-community/Qwen2.5-7B-Instruct-4bit",
            max_tokens=4096
        )
        adapter = OutlinesAdapter()
        dspy.configure(lm=lm, adapter=adapter)

        # Map prompt names to DSPy modules
        self.dspy_modules = dspy_modules

        # Context parsers for each prompt type
        self.context_parsers = {
            'extract_nodes.extract_message': self._parse_node_extraction_context,
            'extract_edges.edge': self._parse_edge_extraction_context,
            'dedupe_nodes.nodes': self._parse_dedupe_nodes_context,
            'dedupe_edges.edge': self._parse_dedupe_edges_context,
        }

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel],
        group_id: str | None = None,
        prompt_name: str | None = None,
        **kwargs
    ) -> dict[str, Any]:
        """
        Intercept graphiti LLM calls and route to DSPy modules.

        Args:
            messages: List of Message objects (system, user)
            response_model: Expected Pydantic response type
            prompt_name: Graphiti prompt identifier (e.g., 'extract_nodes.extract_message')

        Returns:
            Dict representation of response_model
        """

        # Check if we have a DSPy module for this prompt
        if prompt_name and prompt_name in self.dspy_modules:
            # Parse context from messages
            context = self._parse_context(messages, prompt_name)

            # Execute DSPy module
            module = self.dspy_modules[prompt_name]
            result = module(**context)

            # Convert DSPy output to expected format
            return self._format_response(result, response_model)

        else:
            # Fallback to default LLM behavior (if needed)
            raise NotImplementedError(
                f"No DSPy module registered for prompt: {prompt_name}"
            )

    def _parse_context(self, messages: list[Message], prompt_name: str) -> dict:
        """
        Extract structured context from graphiti's message format.

        Graphiti passes context in XML-like tags within messages.
        """
        parser = self.context_parsers.get(prompt_name)
        if parser:
            return parser(messages)

        # Default: extract from user message
        user_msg = next((m for m in messages if m.role == 'user'), None)
        if user_msg:
            return {"text": user_msg.content}

        return {}

    def _parse_node_extraction_context(self, messages: list[Message]) -> dict:
        """
        Parse context for entity extraction prompts.

        Expected format in messages:
        <PREVIOUS MESSAGES>
        episode 1 content...
        episode 2 content...
        </PREVIOUS MESSAGES>

        <CURRENT MESSAGE>
        current episode content
        </CURRENT MESSAGE>
        """
        user_msg = next((m for m in messages if m.role == 'user'), None)
        if not user_msg:
            return {}

        content = user_msg.content

        # Extract current episode
        current = self._extract_xml_content(content, "CURRENT MESSAGE")

        # Extract previous episodes
        previous_text = self._extract_xml_content(content, "PREVIOUS MESSAGES")
        previous_episodes = previous_text.split('\n\n') if previous_text else []

        return {
            "episode_content": current,
            "previous_episodes": previous_episodes,
            "entity_types": []  # TODO: Parse from message if provided
        }

    def _parse_edge_extraction_context(self, messages: list[Message]) -> dict:
        """Parse context for relationship extraction."""
        user_msg = next((m for m in messages if m.role == 'user'), None)
        if not user_msg:
            return {}

        content = user_msg.content

        # Extract episode content
        episode_content = self._extract_xml_content(content, "EPISODE")

        # Extract nodes (usually provided as JSON)
        nodes_text = self._extract_xml_content(content, "NODES")
        nodes = json.loads(nodes_text) if nodes_text else []

        # Extract reference time
        ref_time = self._extract_xml_content(content, "REFERENCE TIME")

        return {
            "episode_content": episode_content,
            "nodes": nodes,
            "reference_time": ref_time,
            "edge_types": []
        }

    def _parse_dedupe_nodes_context(self, messages: list[Message]) -> dict:
        """Parse context for entity deduplication."""
        user_msg = next((m for m in messages if m.role == 'user'), None)
        if not user_msg:
            return {}

        content = user_msg.content

        # Extract entities to compare
        new_entity = self._extract_xml_content(content, "NEW ENTITY")
        existing_entities = self._extract_xml_content(content, "EXISTING ENTITIES")

        return {
            "new_entity": new_entity,
            "existing_entities": json.loads(existing_entities) if existing_entities else [],
            "context": self._extract_xml_content(content, "CONTEXT")
        }

    def _parse_dedupe_edges_context(self, messages: list[Message]) -> dict:
        """Parse context for edge deduplication."""
        user_msg = next((m for m in messages if m.role == 'user'), None)
        if not user_msg:
            return {}

        content = user_msg.content

        return {
            "new_fact": self._extract_xml_content(content, "NEW FACT"),
            "existing_facts": json.loads(
                self._extract_xml_content(content, "EXISTING FACTS") or "[]"
            ),
            "context": self._extract_xml_content(content, "CONTEXT")
        }

    def _extract_xml_content(self, text: str, tag: str) -> str:
        """Extract content between XML-like tags."""
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"

        start_idx = text.find(start_tag)
        end_idx = text.find(end_tag)

        if start_idx == -1 or end_idx == -1:
            return ""

        return text[start_idx + len(start_tag):end_idx].strip()

    def _format_response(
        self,
        dspy_result: Any,
        expected_model: type[BaseModel]
    ) -> dict[str, Any]:
        """
        Convert DSPy output to graphiti's expected format.

        Args:
            dspy_result: Result from DSPy module
            expected_model: Graphiti's expected Pydantic model

        Returns:
            Dict matching expected_model schema
        """
        # If DSPy returned the expected model directly
        if isinstance(dspy_result, expected_model):
            return dspy_result.model_dump()

        # If DSPy returned a wrapper with the result
        if hasattr(dspy_result, expected_model.__name__.lower()):
            result_obj = getattr(dspy_result, expected_model.__name__.lower())
            if isinstance(result_obj, expected_model):
                return result_obj.model_dump()

        # Try to construct expected model from DSPy output
        try:
            model_instance = expected_model(**dspy_result.__dict__)
            return model_instance.model_dump()
        except Exception as e:
            raise ValueError(
                f"Could not convert DSPy result to {expected_model.__name__}: {e}"
            )

# --- Snippet 15 (lines 970-1010; context: Try to construct expected model from DSPy output > DSPy Module Specifications > 1. ExtractEntitiesModule) ---
# NOTE: Unverified and pending further review.
from pydantic import BaseModel, Field
import dspy

class ExtractedEntity(BaseModel):
    name: str = Field(description="Entity name")
    entity_type: str = Field(description="Entity classification (Person, Organization, etc.)")
    attributes: dict[str, Any] = Field(default_factory=dict)

class ExtractedEntities(BaseModel):
    entities: list[ExtractedEntity]

class ExtractEntitiesSignature(dspy.Signature):
    """Extract entities from episode content."""

    episode_content: str = dspy.InputField(
        desc="Current episode text to extract entities from"
    )
    previous_episodes: list[str] = dspy.InputField(
        desc="Previous episodes for context",
        default_factory=list
    )
    entity_types: list[str] = dspy.InputField(
        desc="Valid entity types to extract",
        default_factory=list
    )

    entities: ExtractedEntities = dspy.OutputField(
        desc="List of extracted entities with type classifications"
    )

class ExtractEntitiesModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self._predict = dspy.Predict(ExtractEntitiesSignature)

    def forward(self, episode_content, previous_episodes=None, entity_types=None):
        return self._predict(
            episode_content=episode_content,
            previous_episodes=previous_episodes or [],
            entity_types=entity_types or []
        )

# --- Snippet 16 (lines 1016-1049; context: Try to construct expected model from DSPy output > DSPy Module Specifications > 2. DedupeNodesModule) ---
# NOTE: Unverified and pending further review.
class NodeDuplicateResult(BaseModel):
    is_duplicate: bool = Field(description="Whether entities are duplicates")
    confidence: float = Field(description="Confidence score 0-1")
    match_index: int | None = Field(description="Index of matching entity")
    reasoning: str = Field(description="Explanation of decision")

class DedupeNodesSignature(dspy.Signature):
    """Determine if an entity is a duplicate of existing entities."""

    new_entity: str = dspy.InputField(
        desc="Name of newly extracted entity"
    )
    existing_entities: list[str] = dspy.InputField(
        desc="Names of existing entities to compare against"
    )
    context: str = dspy.InputField(
        desc="Episode context for disambiguation"
    )

    result: NodeDuplicateResult = dspy.OutputField(
        desc="Deduplication decision with reasoning"
    )

class DedupeNodesModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self._predict = dspy.Predict(DedupeNodesSignature)

    def forward(self, new_entity, existing_entities, context):
        return self._predict(
            new_entity=new_entity,
            existing_entities=existing_entities,
            context=context
        )

# --- Snippet 17 (lines 1055-1096; context: Try to construct expected model from DSPy output > DSPy Module Specifications > 3. ExtractRelationshipsModule) ---
# NOTE: Unverified and pending further review.
class ExtractedEdge(BaseModel):
    source_uuid: str
    target_uuid: str
    relation_type: str = Field(description="Relationship name")
    fact: str = Field(description="Textual description of relationship")
    valid_at: str | None = Field(description="ISO timestamp when fact became true")
    invalid_at: str | None = Field(description="ISO timestamp when fact stopped being true")

class ExtractedEdges(BaseModel):
    edges: list[ExtractedEdge]

class ExtractRelationshipsSignature(dspy.Signature):
    """Extract relationships between entities."""

    episode_content: str = dspy.InputField()
    nodes: list[dict] = dspy.InputField(
        desc="Extracted entities with UUIDs and names: [{'uuid': '...', 'name': '...'}, ...]"
    )
    reference_time: str = dspy.InputField(
        desc="ISO timestamp for temporal resolution"
    )
    edge_types: list[str] = dspy.InputField(
        desc="Valid relationship types",
        default_factory=list
    )

    edges: ExtractedEdges = dspy.OutputField(
        desc="List of relationships with temporal bounds"
    )

class ExtractRelationshipsModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self._predict = dspy.Predict(ExtractRelationshipsSignature)

    def forward(self, episode_content, nodes, reference_time, edge_types=None):
        return self._predict(
            episode_content=episode_content,
            nodes=nodes,
            reference_time=reference_time,
            edge_types=edge_types or []
        )

# --- Snippet 18 (lines 1102-1128; context: Try to construct expected model from DSPy output > DSPy Module Specifications > 4. DedupeEdgesModule) ---
# NOTE: Unverified and pending further review.
class EdgeDeduplicateResult(BaseModel):
    is_duplicate: bool
    is_contradiction: bool
    match_index: int | None
    contradicts_index: int | None
    reasoning: str

class DedupeEdgesSignature(dspy.Signature):
    """Determine if a fact duplicates or contradicts existing facts."""

    new_fact: str = dspy.InputField()
    existing_facts: list[str] = dspy.InputField()
    context: str = dspy.InputField()

    result: EdgeDeduplicateResult = dspy.OutputField()

class DedupeEdgesModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self._predict = dspy.Predict(DedupeEdgesSignature)

    def forward(self, new_fact, existing_facts, context):
        return self._predict(
            new_fact=new_fact,
            existing_facts=existing_facts,
            context=context
        )

# --- Snippet 19 (lines 1153-1183; context: Save optimized prompts) ---
# NOTE: Unverified and pending further review.
import dspy
from dspy.teleprompt import MIPRO

# Prepare training data
train_examples = [
    dspy.Example(
        episode_content="John met with Sarah at the office.",
        entities=ExtractedEntities(entities=[
            ExtractedEntity(name="John", entity_type="Person"),
            ExtractedEntity(name="Sarah", entity_type="Person"),
            ExtractedEntity(name="office", entity_type="Location")
        ])
    ).with_inputs("episode_content")
    # ... more examples
]

# Optimize module
teleprompter = MIPRO(
    metric=entity_extraction_metric,
    num_candidates=10,
    init_temperature=1.0
)

optimized_module = teleprompter.compile(
    ExtractEntitiesModule(),
    trainset=train_examples,
    num_trials=100
)

# Save optimized prompts
optimized_module.save("optimized_prompts/extract_entities.json")

# --- Snippet 20 (lines 1188-1214; context: F1 score) ---
# NOTE: Unverified and pending further review.
def entity_extraction_metric(example, prediction, trace=None):
    """
    Metric for entity extraction quality.

    Evaluates:
    - Recall: Did we extract all entities?
    - Precision: Are extracted entities valid?
    - Type accuracy: Correct entity classifications?
    """
    gold_entities = {e.name for e in example.entities.entities}
    pred_entities = {e.name for e in prediction.entities.entities}

    if len(gold_entities) == 0:
        return 0.0

    # Calculate recall
    recall = len(gold_entities & pred_entities) / len(gold_entities)

    # Calculate precision
    precision = len(gold_entities & pred_entities) / len(pred_entities) if pred_entities else 0.0

    # F1 score
    if recall + precision == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# --- Snippet 21 (lines 1224-1238; context: - Provider-specific query syntax) ---
# NOTE: Unverified and pending further review.
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.driver import GraphProvider

# Initialize FalkorDB driver
driver = FalkorDriver(
    uri="falkordb://localhost:6379",
    database="knowledge_graph",
    provider=GraphProvider.FALKORDB
)

# Driver handles:
# - Connection pooling
# - Query execution
# - Transaction management
# - Provider-specific query syntax

# --- Snippet 22 (lines 1251-1262; context: Execute query with FalkorDB-specific syntax) ---
# NOTE: Unverified and pending further review.
# Execute query with FalkorDB-specific syntax
records, summary, keys = await driver.execute_query(
    """
    MATCH (n:Entity {group_id: $group_id})
    WHERE n.name CONTAINS $search_term
    RETURN n.uuid, n.name, n.labels
    LIMIT $limit
    """,
    group_id=group_id,
    search_term="John",
    limit=10
)

# --- Snippet 23 (lines 1269-1286; context: ... etc) ---
# NOTE: Unverified and pending further review.
await driver.execute_query(
    """
    MERGE (n:Entity {uuid: $uuid})
    ON CREATE SET
        n.name = $name,
        n.group_id = $group_id,
        n.labels = $labels,
        n.created_at = $created_at,
        n.name_embedding = $embedding
    ON MATCH SET
        n.summary = $summary,
        n.attributes = $attributes
    RETURN n
    """,
    uuid=node.uuid,
    name=node.name,
    # ... etc
)

# --- Snippet 24 (lines 1291-1306; context: ... etc) ---
# NOTE: Unverified and pending further review.
await driver.execute_query(
    """
    MATCH (source:Entity {uuid: $source_uuid})
    MATCH (target:Entity {uuid: $target_uuid})
    MERGE (source)-[r:RELATES_TO {uuid: $edge_uuid}]->(target)
    SET
        r.name = $name,
        r.fact = $fact,
        r.valid_at = $valid_at,
        r.created_at = $created_at
    RETURN r
    """,
    source_uuid=edge.source_node_uuid,
    target_uuid=edge.target_node_uuid,
    # ... etc
)

# --- Snippet 25 (lines 1311-1339; context: Search by embedding) ---
# NOTE: Unverified and pending further review.
# Create vector index (once)
await driver.execute_query(
    """
    CALL db.idx.vector.createNodeIndex(
        'Entity',
        'name_embedding_idx',
        'name_embedding',
        1024,
        'COSINE'
    )
    """
)

# Search by embedding
await driver.execute_query(
    """
    CALL db.idx.vector.queryNodes(
        'Entity',
        'name_embedding_idx',
        $top_k,
        $query_embedding
    ) YIELD node, score
    WHERE node.group_id = $group_id
    RETURN node.uuid, node.name, score
    """,
    query_embedding=embedding,
    top_k=5,
    group_id=group_id
)

# --- Snippet 26 (lines 1346-1388; context: Vector indexes for similarity search) ---
# NOTE: Unverified and pending further review.
async def setup_indexes(driver: FalkorDriver):
    """Create indexes for efficient querying."""

    # UUID indexes (unique constraint)
    await driver.execute_query(
        "CREATE INDEX ON :Entity(uuid)"
    )
    await driver.execute_query(
        "CREATE INDEX ON :Episodic(uuid)"
    )

    # Group ID indexes (for partitioning)
    await driver.execute_query(
        "CREATE INDEX ON :Entity(group_id)"
    )
    await driver.execute_query(
        "CREATE INDEX ON :Episodic(group_id)"
    )

    # Vector indexes for similarity search
    await driver.execute_query(
        """
        CALL db.idx.vector.createNodeIndex(
            'Entity',
            'name_embedding_idx',
            'name_embedding',
            1024,
            'COSINE'
        )
        """
    )

    await driver.execute_query(
        """
        CALL db.idx.vector.createRelationshipIndex(
            'RELATES_TO',
            'fact_embedding_idx',
            'fact_embedding',
            1024,
            'COSINE'
        )
        """
    )

# --- Snippet 27 (lines 1396-1404; context: Manual transaction (multiple queries)) ---
# NOTE: Unverified and pending further review.
# Automatic transaction (single query)
await driver.execute_query(query, **params)

# Manual transaction (multiple queries)
async with driver.session() as session:
    async with session.begin_transaction() as tx:
        await tx.run(query1, **params1)
        await tx.run(query2, **params2)
        await tx.commit()

# --- Snippet 28 (lines 1445-1535; context: Average) ---
# NOTE: Unverified and pending further review.
from graphiti_core.embedder.client import EmbedderClient
import mlx.core as mx
from transformers import AutoTokenizer
import asyncio

class Qwen3EmbedderClient(EmbedderClient):
    """
    Qwen3 embedding client using MLX.

    Uses mlx-community/Qwen3-Embedding-4B-4bit-DWQ for local embeddings.
    """

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen3-Embedding-4B-4bit-DWQ"
    ):
        from mlx_lm import load

        # Load Qwen3 embedding model
        self.model, self.tokenizer = load(model_name)

        # Get embedding dimension from model config
        self.embedding_dim = self.model.config.hidden_size

    async def create(self, input_data: list[str]) -> list[list[float]]:
        """Create embeddings for input strings."""
        return await self.create_batch(input_data)

    async def create_batch(self, input_data: list[str]) -> list[list[float]]:
        """Create embeddings in batch mode."""
        # Run in executor for async compatibility
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_sync, input_data)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embedding generation."""
        # Tokenize all inputs
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )

        # Convert to MLX arrays
        input_ids = mx.array(encoded['input_ids'])
        attention_mask = mx.array(encoded['attention_mask'])

        # Get model outputs
        outputs = self.model(input_ids, attention_mask=attention_mask)

        # Extract embeddings (mean pooling over sequence)
        embeddings = self._mean_pooling(
            outputs.last_hidden_state,
            attention_mask
        )

        # Normalize embeddings
        embeddings = mx.nn.normalize(embeddings, axis=1)

        # Convert to list of lists
        return embeddings.tolist()

    def _mean_pooling(
        self,
        token_embeddings: mx.array,
        attention_mask: mx.array
    ) -> mx.array:
        """Mean pooling over sequence length."""
        # Expand attention mask to match embedding dimensions
        input_mask_expanded = mx.expand_dims(
            attention_mask,
            axis=-1
        ).astype(token_embeddings.dtype)

        # Sum embeddings, weighted by mask
        sum_embeddings = mx.sum(
            token_embeddings * input_mask_expanded,
            axis=1
        )

        # Sum mask values
        sum_mask = mx.clip(
            mx.sum(input_mask_expanded, axis=1),
            a_min=1e-9,
            a_max=None
        )

        # Average
        return sum_embeddings / sum_mask

# --- Snippet 29 (lines 1543-1565; context: Create vector index in FalkorDB) ---
# NOTE: Unverified and pending further review.
# Get embedding dimension from embedder
embedding_dim = embedder.embedding_dim  # e.g., 1024 for Qwen3-Embedding-4B

# Create vector index in FalkorDB
await driver.execute_query(f"""
    CALL db.idx.vector.create(
        'Entity',
        'name_embedding',
        'FP32',
        {embedding_dim},
        'COSINE'
    )
""")

await driver.execute_query(f"""
    CALL db.idx.vector.create(
        'RelatesToNode_',
        'fact_embedding',
        'FP32',
        {embedding_dim},
        'COSINE'
    )
""")

# --- Snippet 30 (lines 1602-1651; context: Create (passage, score) tuples and sort) ---
# NOTE: Unverified and pending further review.
from graphiti_core.cross_encoder.client import CrossEncoderClient
from sentence_transformers import CrossEncoder
import asyncio

class Qwen3RerankerClient(CrossEncoderClient):
    """
    Qwen3 reranker using sequence classification.

    Uses tomaarsen/Qwen3-Reranker-0.6B-seq-cls for local reranking.
    """

    def __init__(
        self,
        model_name: str = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
    ):
        # Use sentence-transformers CrossEncoder interface
        # If available for this model, otherwise use MLX directly
        self.model = CrossEncoder(model_name, device="mps")  # MPS for Apple Silicon

    async def rank(
        self,
        query: str,
        passages: list[str]
    ) -> list[tuple[str, float]]:
        """
        Rank passages by relevance to query.

        Args:
            query: The query string
            passages: List of passages to rank

        Returns:
            List of (passage, score) tuples sorted by score (descending)
        """
        # Create query-passage pairs
        pairs = [[query, passage] for passage in passages]

        # Run in executor for async compatibility
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            self.model.predict,
            pairs
        )

        # Create (passage, score) tuples and sort
        results = list(zip(passages, scores.tolist()))
        results.sort(key=lambda x: x[1], reverse=True)

        return results

# --- Snippet 31 (lines 1659-1675; context: Search with reranking) ---
# NOTE: Unverified and pending further review.
from graphiti_core.search.search_config import SearchConfig, EdgeSearchConfig
from graphiti_core.search.search_config import EdgeSearchMethod, EdgeReranker

# With cross-encoder reranking
config = SearchConfig(
    edge_config=EdgeSearchConfig(
        search_methods=[
            EdgeSearchMethod.bm25,
            EdgeSearchMethod.cosine_similarity
        ],
        reranker=EdgeReranker.cross_encoder,  # ← Enables Qwen3 reranker
        limit=10
    )
)

# Search with reranking
results = await graphiti.search_("user query", config=config)

# --- Snippet 32 (lines 1683-1712; context: ✓ Optimized for Apple Silicon) ---
# NOTE: Unverified and pending further review.
from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver
from your_module import (
    Qwen3EmbedderClient,
    Qwen3RerankerClient,
    DSPyLLMClient
)

# Initialize all three subsystems
embedder = Qwen3EmbedderClient()              # Qwen3-Embedding-4B-4bit-DWQ
reranker = Qwen3RerankerClient()              # Qwen3-Reranker-0.6B-seq-cls
llm_client = DSPyLLMClient()                  # DSPy + Qwen3 for prompts
driver = FalkorDriver(host="localhost", port=6379)

# Create Graphiti instance with all custom components
graphiti = Graphiti(
    uri="falkordb://localhost:6379",
    llm_client=llm_client,          # ← DSPy/Qwen3 for extraction
    embedder=embedder,              # ← Qwen3 for embeddings
    cross_encoder=reranker,         # ← Qwen3 for reranking
    graph_driver=driver
)

# All three subsystems now use Qwen3 + MLX
# - LLM operations: Qwen2.5-7B-Instruct-4bit via DSPy
# - Embeddings: Qwen3-Embedding-4B-4bit-DWQ
# - Reranking: Qwen3-Reranker-0.6B-seq-cls
# ✓ Complete local inference
# ✓ No API dependencies
# ✓ Optimized for Apple Silicon

# --- Snippet 33 (lines 1846-2135; context: 4. Process an episode) ---
# NOTE: Unverified and pending further review.
"""
Complete custom knowledge graph ingestion pipeline.

This example demonstrates:
- Using graphiti Pydantic models (EntityNode, EntityEdge, etc.)
- Custom DSPy modules for extraction
- FalkorDB persistence
- Full control over pipeline steps
"""

import asyncio
from datetime import datetime
from uuid import uuid4
from typing import Any

import dspy
from pydantic import BaseModel, Field

# Graphiti imports
from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.embedder import EmbedderClient
from graphiti_core.utils.datetime_utils import utc_now

# Our DSPy integration
from dspy_outlines import OutlinesLM, OutlinesAdapter


# ============================================================
# DSPy Module Definitions
# ============================================================

class ExtractedEntity(BaseModel):
    name: str
    entity_type: str = "Entity"
    attributes: dict[str, Any] = Field(default_factory=dict)

class ExtractedEntities(BaseModel):
    entities: list[ExtractedEntity]

class ExtractEntitiesSignature(dspy.Signature):
    """Extract entities from episode content."""

    episode_content: str = dspy.InputField()
    previous_episodes: list[str] = dspy.InputField(default_factory=list)

    result: ExtractedEntities = dspy.OutputField()

class ExtractEntitiesModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self._predict = dspy.Predict(ExtractEntitiesSignature)

    def forward(self, episode_content, previous_episodes=None):
        return self._predict(
            episode_content=episode_content,
            previous_episodes=previous_episodes or []
        )


class ExtractedRelationship(BaseModel):
    source_name: str
    target_name: str
    relation_type: str
    fact: str

class ExtractedRelationships(BaseModel):
    relationships: list[ExtractedRelationship]

class ExtractRelationshipsSignature(dspy.Signature):
    """Extract relationships between entities."""

    episode_content: str = dspy.InputField()
    entity_names: list[str] = dspy.InputField()

    result: ExtractedRelationships = dspy.OutputField()

class ExtractRelationshipsModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self._predict = dspy.Predict(ExtractRelationshipsSignature)

    def forward(self, episode_content, entity_names):
        return self._predict(
            episode_content=episode_content,
            entity_names=entity_names
        )


# ============================================================
# Custom Pipeline Implementation
# ============================================================

class KnowledgeGraphPipeline:
    """
    Custom knowledge graph ingestion pipeline.

    Uses graphiti models + DSPy for extraction + FalkorDB for storage.
    """

    def __init__(
        self,
        driver: FalkorDriver,
        embedder: EmbedderClient,
        group_id: str = "default"
    ):
        self.driver = driver
        self.embedder = embedder
        self.group_id = group_id

        # Configure DSPy
        lm = OutlinesLM(
            model_name="mlx-community/Qwen2.5-7B-Instruct-4bit",
            max_tokens=2048
        )
        dspy.configure(lm=lm, adapter=OutlinesAdapter())

        # Initialize DSPy modules
        self.extract_entities = ExtractEntitiesModule()
        self.extract_relationships = ExtractRelationshipsModule()

    async def ingest_episode(
        self,
        episode_body: str,
        reference_time: datetime,
        episode_name: str | None = None
    ) -> dict[str, Any]:
        """
        Ingest a single episode into the knowledge graph.

        Args:
            episode_body: Raw episode text
            reference_time: When the episode occurred
            episode_name: Optional episode identifier

        Returns:
            Dict with extracted nodes, edges, and episode
        """

        # STEP 1: Create EpisodicNode
        episode = EpisodicNode(
            uuid=str(uuid4()),
            name=episode_name or f"Episode {reference_time.isoformat()}",
            group_id=self.group_id,
            source=EpisodeType.message,
            source_description="User conversation",
            content=episode_body,
            valid_at=reference_time,
            created_at=utc_now(),
            labels=[]
        )

        # STEP 2: Extract Entities
        extraction_result = self.extract_entities(
            episode_content=episode_body,
            previous_episodes=[]  # TODO: Retrieve from DB
        )

        # Convert to EntityNode objects
        entity_nodes = []
        name_to_uuid = {}  # Map entity names to UUIDs

        for entity in extraction_result.result.entities:
            node = EntityNode(
                uuid=str(uuid4()),
                name=entity.name,
                group_id=self.group_id,
                labels=[entity.entity_type],
                created_at=utc_now(),
                summary="",
                attributes=entity.attributes
            )
            entity_nodes.append(node)
            name_to_uuid[entity.name] = node.uuid

        # STEP 3: Extract Relationships
        relationship_result = self.extract_relationships(
            episode_content=episode_body,
            entity_names=[e.name for e in entity_nodes]
        )

        # Convert to EntityEdge objects
        entity_edges = []

        for rel in relationship_result.result.relationships:
            # Look up UUIDs for source/target
            source_uuid = name_to_uuid.get(rel.source_name)
            target_uuid = name_to_uuid.get(rel.target_name)

            if source_uuid and target_uuid:
                edge = EntityEdge(
                    uuid=str(uuid4()),
                    source_node_uuid=source_uuid,
                    target_node_uuid=target_uuid,
                    name=rel.relation_type,
                    fact=rel.fact,
                    group_id=self.group_id,
                    created_at=utc_now(),
                    valid_at=reference_time,
                    episodes=[episode.uuid]
                )
                entity_edges.append(edge)

        # STEP 4: Generate Embeddings
        for node in entity_nodes:
            await node.generate_name_embedding(self.embedder)

        for edge in entity_edges:
            await edge.generate_embedding(self.embedder)

        # STEP 5: Create Episodic Edges (MENTIONS)
        episodic_edges = [
            EpisodicEdge(
                uuid=str(uuid4()),
                source_node_uuid=episode.uuid,
                target_node_uuid=node.uuid,
                group_id=self.group_id,
                created_at=utc_now()
            )
            for node in entity_nodes
        ]

        # Update episode with entity edge references
        episode.entity_edges = [e.uuid for e in entity_edges]

        # STEP 6: Save to FalkorDB
        await add_nodes_and_edges_bulk(
            driver=self.driver,
            episodic_nodes=[episode],
            episodic_edges=episodic_edges,
            entity_nodes=entity_nodes,
            entity_edges=entity_edges,
            embedder=self.embedder
        )

        return {
            "episode": episode,
            "nodes": entity_nodes,
            "edges": entity_edges
        }


# ============================================================
# Initialization Code
# ============================================================

async def main():
    """Initialize pipeline and process an episode."""

    # 1. Create FalkorDB driver
    driver = FalkorDriver(
        uri="falkordb://localhost:6379",
        database="knowledge_graph"
    )

    # 2. Create Qwen3 embedder (local, MLX-optimized)
    from your_module import Qwen3EmbedderClient

    embedder = Qwen3EmbedderClient(
        model_name="mlx-community/Qwen3-Embedding-4B-4bit-DWQ"
    )

    # 3. Create pipeline
    pipeline = KnowledgeGraphPipeline(
        driver=driver,
        embedder=embedder,
        group_id="user_123"
    )

    # 4. Process an episode
    result = await pipeline.ingest_episode(
        episode_body="John met with Sarah at the office to discuss the project.",
        reference_time=datetime.now(),
        episode_name="Meeting discussion"
    )

    print(f"Extracted {len(result['nodes'])} entities")
    print(f"Extracted {len(result['edges'])} relationships")

    for node in result['nodes']:
        print(f"  - {node.name} ({node.labels})")

    for edge in result['edges']:
        print(f"  - {edge.fact}")


if __name__ == "__main__":
    asyncio.run(main())

# --- Snippet 34 (lines 2141-2186; context: Results) ---
# NOTE: Unverified and pending further review.
"""
Minimal example: Process one episode from start to finish.
"""

import asyncio
from datetime import datetime

async def process_episode_example():
    # Setup
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from graphiti_core.embedder import BGEEmbedderClient

    driver = FalkorDriver(uri="falkordb://localhost:6379")
    embedder = BGEEmbedderClient()

    # Create pipeline
    pipeline = KnowledgeGraphPipeline(
        driver=driver,
        embedder=embedder,
        group_id="demo"
    )

    # Process episode
    result = await pipeline.ingest_episode(
        episode_body="""
        Alice and Bob are working on the machine learning project.
        Alice is the team lead. Bob joined the team last month.
        They are using Python and TensorFlow.
        """,
        reference_time=datetime.now(),
        episode_name="Team update"
    )

    # Results
    print("✓ Episode processed successfully")
    print(f"✓ {len(result['nodes'])} entities extracted:")
    for node in result['nodes']:
        print(f"    {node.name} ({', '.join(node.labels)})")

    print(f"✓ {len(result['edges'])} relationships extracted:")
    for edge in result['edges']:
        print(f"    {edge.fact}")

    print(f"✓ Saved to FalkorDB (episode UUID: {result['episode'].uuid})")

asyncio.run(process_episode_example())

# --- Snippet 35 (lines 2192-2303; context: 6. Summary) ---
# NOTE: Unverified and pending further review.
"""
Complete Graphiti initialization with Qwen3 for all three subsystems:
- LLM operations (DSPy + Qwen2.5-7B-Instruct-4bit)
- Embeddings (Qwen3-Embedding-4B-4bit-DWQ)
- Reranking (Qwen3-Reranker-0.6B-seq-cls)
"""

import asyncio
from datetime import datetime

from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.search.search_config import SearchConfig, EdgeSearchConfig
from graphiti_core.search.search_config import EdgeSearchMethod, EdgeReranker

# Import custom Qwen3 implementations
from your_module import (
    Qwen3EmbedderClient,
    Qwen3RerankerClient,
    DSPyLLMClient
)


async def complete_qwen3_pipeline_example():
    """Demonstrate complete local inference pipeline with Qwen3."""

    # 1. Initialize all three Qwen3 components
    print("Initializing Qwen3 subsystems...")

    # Embedder: Qwen3-Embedding-4B-4bit-DWQ
    embedder = Qwen3EmbedderClient()
    print(f"✓ Embedder loaded (dim={embedder.embedding_dim})")

    # Reranker: Qwen3-Reranker-0.6B-seq-cls
    reranker = Qwen3RerankerClient()
    print("✓ Reranker loaded")

    # LLM: DSPy + Qwen2.5-7B-Instruct-4bit
    llm_client = DSPyLLMClient()
    print("✓ DSPy LLM client configured")

    # 2. Create FalkorDB driver
    driver = FalkorDriver(
        uri="falkordb://localhost:6379",
        database="knowledge_graph"
    )
    print("✓ FalkorDB driver connected")

    # 3. Initialize Graphiti with all custom components
    graphiti = Graphiti(
        uri="falkordb://localhost:6379",
        llm_client=llm_client,          # ← DSPy/Qwen3 for extraction
        embedder=embedder,              # ← Qwen3 for embeddings
        cross_encoder=reranker,         # ← Qwen3 for reranking
        graph_driver=driver
    )
    print("✓ Graphiti initialized with Qwen3 subsystems")

    # 4. Add an episode (uses DSPy for extraction, Qwen3 for embeddings)
    print("\nProcessing episode...")
    await graphiti.add_episode(
        name="Team meeting",
        episode_body="""
        Alice, the team lead, met with Bob to discuss the machine learning project.
        Bob recently joined the team and is working on the Python implementation.
        They decided to use TensorFlow for the neural network components.
        The project deadline is set for next quarter.
        """,
        source_description="Meeting notes",
        reference_time=datetime.now()
    )
    print("✓ Episode processed and saved")

    # 5. Search with cross-encoder reranking
    print("\nSearching with reranking...")

    # Configure search with cross-encoder reranking enabled
    search_config = SearchConfig(
        edge_config=EdgeSearchConfig(
            search_methods=[
                EdgeSearchMethod.bm25,              # Keyword search
                EdgeSearchMethod.cosine_similarity  # Semantic search (uses Qwen3 embeddings)
            ],
            reranker=EdgeReranker.cross_encoder,    # Rerank with Qwen3 reranker
            limit=10
        )
    )

    results = await graphiti.search_(
        "What is Alice working on?",
        config=search_config
    )

    print(f"✓ Search complete: {len(results.edges)} results")
    for i, edge in enumerate(results.edges[:5], 1):
        print(f"  {i}. {edge.fact}")

    # 6. Summary
    print("\n" + "="*60)
    print("COMPLETE LOCAL INFERENCE PIPELINE")
    print("="*60)
    print("✓ LLM operations:  Qwen2.5-7B-Instruct-4bit (via DSPy)")
    print("✓ Embeddings:      Qwen3-Embedding-4B-4bit-DWQ")
    print("✓ Reranking:       Qwen3-Reranker-0.6B-seq-cls")
    print("✓ Graph storage:   FalkorDB")
    print("✓ Platform:        MLX (Apple Silicon optimized)")
    print("✓ API calls:       ZERO (completely local)")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(complete_qwen3_pipeline_example())

# --- Snippet 36 (lines 2345-2476; context: Check most relevant passage is first) ---
# NOTE: Unverified and pending further review.
"""
Unit tests for DSPy extraction modules.
"""

import pytest
import dspy
from dspy_outlines import OutlinesLM, OutlinesAdapter

@pytest.fixture(scope="module")
def configure_dspy():
    """Configure DSPy once for all tests."""
    lm = OutlinesLM(model_name="mlx-community/Qwen2.5-7B-Instruct-4bit")
    dspy.configure(lm=lm, adapter=OutlinesAdapter())

def test_extract_entities_basic(configure_dspy):
    """Test basic entity extraction."""
    module = ExtractEntitiesModule()

    result = module(
        episode_content="John works at Google in San Francisco."
    )

    entities = result.result.entities
    entity_names = {e.name for e in entities}

    assert "John" in entity_names
    assert "Google" in entity_names
    assert "San Francisco" in entity_names

def test_extract_entities_with_types(configure_dspy):
    """Test entity type classification."""
    module = ExtractEntitiesModule()

    result = module(
        episode_content="Alice is a software engineer at Microsoft."
    )

    entities = {e.name: e.entity_type for e in result.result.entities}

    assert entities.get("Alice") == "Person"
    assert entities.get("Microsoft") == "Organization"

def test_extract_relationships(configure_dspy):
    """Test relationship extraction."""
    module = ExtractRelationshipsModule()

    result = module(
        episode_content="Bob manages the engineering team.",
        entity_names=["Bob", "engineering team"]
    )

    relationships = result.result.relationships

    assert len(relationships) > 0
    assert any(
        r.source_name == "Bob" and r.target_name == "engineering team"
        for r in relationships
    )

def test_dedupe_nodes_duplicate(configure_dspy):
    """Test entity deduplication detects duplicates."""
    module = DedupeNodesModule()

    result = module(
        new_entity="John Smith",
        existing_entities=["John Smith", "Jane Doe"],
        context="John Smith sent an email."
    )

    assert result.result.is_duplicate is True
    assert result.result.match_index == 0

def test_dedupe_nodes_not_duplicate(configure_dspy):
    """Test entity deduplication detects new entities."""
    module = DedupeNodesModule()

    result = module(
        new_entity="Alice Johnson",
        existing_entities=["John Smith", "Jane Doe"],
        context="Alice Johnson joined the team."
    )

    assert result.result.is_duplicate is False

def test_qwen3_embedder(configure_dspy):
    """Test Qwen3 embedder."""
    from your_module import Qwen3EmbedderClient

    embedder = Qwen3EmbedderClient()

    texts = ["Alice", "Bob", "Stanford University"]
    embeddings = asyncio.run(embedder.create_batch(texts))

    # Check dimension
    assert all(len(emb) == embedder.embedding_dim for emb in embeddings)

    # Check normalization (if using normalized embeddings)
    import numpy as np
    norms = [np.linalg.norm(emb) for emb in embeddings]
    assert all(abs(norm - 1.0) < 1e-5 for norm in norms)

    # Check similarity makes sense
    # "Alice" and "Bob" should be more similar than "Alice" and "Stanford"
    alice_bob_sim = np.dot(embeddings[0], embeddings[1])
    alice_stanford_sim = np.dot(embeddings[0], embeddings[2])
    assert alice_bob_sim > alice_stanford_sim

def test_qwen3_reranker(configure_dspy):
    """Test Qwen3 reranker."""
    from your_module import Qwen3RerankerClient

    reranker = Qwen3RerankerClient()

    query = "What is Alice's position?"
    passages = [
        "Alice works as a research scientist",
        "Bob is a software engineer",
        "The weather is sunny"
    ]

    results = asyncio.run(reranker.rank(query, passages))

    # Check sorting (descending scores)
    assert results[0][1] >= results[1][1] >= results[2][1]

    # Check scores in valid range
    for passage, score in results:
        assert 0.0 <= score <= 1.0

    # Check most relevant passage is first
    assert "Alice" in results[0][0]
    assert "research scientist" in results[0][0]

# --- Snippet 37 (lines 2482-2631; context: Just verify the pipeline works without errors) ---
# NOTE: Unverified and pending further review.
"""
Integration tests for full pipeline with FalkorDB.
"""

import pytest
import asyncio
from datetime import datetime

@pytest.fixture(scope="module")
async def pipeline():
    """Create test pipeline with FalkorDB."""
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from graphiti_core.embedder import BGEEmbedderClient

    driver = FalkorDriver(uri="falkordb://localhost:6379", database="test_db")
    embedder = BGEEmbedderClient()

    pipeline = KnowledgeGraphPipeline(
        driver=driver,
        embedder=embedder,
        group_id="test_group"
    )

    yield pipeline

    # Cleanup: Delete test data
    from graphiti_core.nodes import Node
    await Node.delete_by_group_id(driver, "test_group")

@pytest.mark.asyncio
async def test_ingest_single_episode(pipeline):
    """Test ingesting a single episode."""
    result = await pipeline.ingest_episode(
        episode_body="Alice works at Google.",
        reference_time=datetime.now(),
        episode_name="Test episode"
    )

    assert len(result['nodes']) >= 2  # Alice, Google
    assert len(result['edges']) >= 1  # works_at
    assert result['episode'].uuid is not None

@pytest.mark.asyncio
async def test_ingest_multiple_episodes_deduplication(pipeline):
    """Test entity deduplication across episodes."""
    # First episode
    result1 = await pipeline.ingest_episode(
        episode_body="Alice works at Google.",
        reference_time=datetime.now()
    )

    # Second episode (mentions Alice again)
    result2 = await pipeline.ingest_episode(
        episode_body="Alice is a senior engineer.",
        reference_time=datetime.now()
    )

    # Query database to check Alice appears only once
    from graphiti_core.nodes import EntityNode

    alice_nodes = await EntityNode.get_by_name(
        pipeline.driver,
        name="Alice",
        group_id="test_group"
    )

    # Should be deduplicated to single entity
    assert len(alice_nodes) == 1

@pytest.mark.asyncio
async def test_temporal_edges(pipeline):
    """Test temporal edge extraction."""
    result = await pipeline.ingest_episode(
        episode_body="Bob joined Google in 2020. He left in 2023.",
        reference_time=datetime(2023, 6, 1)
    )

    # Find "Bob works_at Google" edge
    work_edge = next(
        (e for e in result['edges'] if e.name == "works_at"),
        None
    )

    assert work_edge is not None
    assert work_edge.valid_at is not None
    assert work_edge.invalid_at is not None

@pytest.mark.asyncio
async def test_search_with_embedder_and_reranker():
    """Test search with custom Qwen3 embedder and reranker."""
    from graphiti_core import Graphiti
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from graphiti_core.search.search_config import SearchConfig, EdgeSearchConfig
    from graphiti_core.search.search_config import EdgeSearchMethod, EdgeReranker
    from your_module import Qwen3EmbedderClient, Qwen3RerankerClient, DSPyLLMClient

    # Setup Graphiti with custom components
    embedder = Qwen3EmbedderClient()
    reranker = Qwen3RerankerClient()
    llm_client = DSPyLLMClient()
    driver = FalkorDriver(uri="falkordb://localhost:6379", database="test_db")

    graphiti = Graphiti(
        uri="falkordb://localhost:6379",
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=reranker,
        graph_driver=driver
    )

    # Add test data
    await graphiti.add_episode(
        name="Test episode",
        episode_body="Alice works on machine learning at Google.",
        source_description="Test",
        reference_time=datetime.now()
    )

    # Search without reranking
    config_no_rerank = SearchConfig(
        edge_config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.cosine_similarity],
            reranker=EdgeReranker.rrf,  # No cross-encoder
            limit=10
        )
    )
    results_no_rerank = await graphiti.search_(
        "What does Alice do?",
        config=config_no_rerank
    )

    # Search with reranking
    config_with_rerank = SearchConfig(
        edge_config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.cosine_similarity],
            reranker=EdgeReranker.cross_encoder,  # Use Qwen3 reranker
            limit=10
        )
    )
    results_with_rerank = await graphiti.search_(
        "What does Alice do?",
        config=config_with_rerank
    )

    # Both should return results
    assert len(results_no_rerank.edges) > 0
    assert len(results_with_rerank.edges) > 0

    # Reranking may change order or scores
    # Just verify the pipeline works without errors

# --- Snippet 38 (lines 2637-2699; context: Benchmark queries) ---
# NOTE: Unverified and pending further review.
"""
Performance benchmarks for pipeline.
"""

import time
import asyncio
from statistics import mean, stdev

async def benchmark_extraction_speed(pipeline, num_episodes=100):
    """Benchmark episode extraction speed."""

    episode_template = "Person{i} works at Company{i} on Project{i}."

    times = []

    for i in range(num_episodes):
        episode_body = episode_template.format(i=i)

        start = time.time()
        await pipeline.ingest_episode(
            episode_body=episode_body,
            reference_time=datetime.now()
        )
        elapsed = time.time() - start

        times.append(elapsed)

    print(f"Episodes processed: {num_episodes}")
    print(f"Average time: {mean(times):.2f}s")
    print(f"Std dev: {stdev(times):.2f}s")
    print(f"Min: {min(times):.2f}s, Max: {max(times):.2f}s")
    print(f"Throughput: {num_episodes / sum(times):.2f} episodes/sec")

async def benchmark_query_performance(pipeline, num_queries=1000):
    """Benchmark graph query performance."""

    # First, populate graph
    for i in range(100):
        await pipeline.ingest_episode(
            episode_body=f"Entity{i} relates_to Entity{i+1}.",
            reference_time=datetime.now()
        )

    # Benchmark queries
    from graphiti_core.nodes import EntityNode

    times = []

    for i in range(num_queries):
        start = time.time()

        await EntityNode.get_by_name(
            pipeline.driver,
            name=f"Entity{i % 100}",
            group_id=pipeline.group_id
        )

        elapsed = time.time() - start
        times.append(elapsed)

    print(f"Queries executed: {num_queries}")
    print(f"Average time: {mean(times)*1000:.2f}ms")
    print(f"Throughput: {num_queries / sum(times):.0f} queries/sec")

