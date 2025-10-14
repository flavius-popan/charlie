#!/usr/bin/env python3
"""
MLX-LM + Outlines → Graphiti Integration Test Tool

Interactive CLI for extracting knowledge graph triplets locally
and writing them to a Kuzu database with full graphiti compatibility.
"""

import os
# Fix tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import asyncio
import logging
import argparse
from datetime import datetime, timezone
from typing import List
from pathlib import Path

# MLX and Outlines
import mlx.core as mx
import mlx_lm
import outlines
from pydantic import BaseModel, Field

# Graphiti
from graphiti_core import Graphiti
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.embedder import EmbedderClient
from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
from graphiti_core.edges import EntityEdge
from graphiti_core.utils.datetime_utils import utc_now, ensure_utc
from graphiti_core.utils.maintenance.edge_operations import build_episodic_edges
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk

from app import settings


# Configure logging - reduce noise from libraries
logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s'
)
# Set application logger to INFO for our own logs
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================================
# Pydantic Models
# ============================================================================

class ExtractedEntity(BaseModel):
    id: int = Field(description="Sequential ID starting from 0")
    name: str = Field(description="Entity name")
    entity_type: str | None = Field(default=None, description="Optional entity type")


class ExtractedEntities(BaseModel):
    entities: List[ExtractedEntity]


class ExtractedRelationship(BaseModel):
    source_entity_id: int
    target_entity_id: int
    relation_type: str
    fact: str
    valid_at: str | None = None
    invalid_at: str | None = None


class ExtractedRelationships(BaseModel):
    relationships: List[ExtractedRelationship]


# ============================================================================
# MLX Local Embedder
# ============================================================================

class MLXEmbedder(EmbedderClient):
    """
    Local embedder using MLX for Apple Silicon.

    Uses mean pooling over the last hidden state of the MLX model
    to generate embeddings compatible with graphiti's storage.
    """

    def __init__(self, model, tokenizer):
        """
        Initialize MLX embedder with model and tokenizer.

        Args:
            model: MLX language model
            tokenizer: Tokenizer for the model
        """
        self.model = model
        self.tokenizer = tokenizer

    async def create(self, input_data: str | List[str]) -> List[float]:
        """
        Create embedding for a single text input.

        Args:
            input_data: Text string to embed

        Returns:
            List of floats representing the embedding vector
        """
        if isinstance(input_data, list):
            input_data = input_data[0]

        # Tokenize input
        tokens = self.tokenizer.encode(input_data)
        tokens = mx.array([tokens])

        # Get model output (last hidden state)
        outputs = self.model(tokens)

        # Mean pooling over sequence dimension
        # Shape: [batch, seq_len, hidden_dim] -> [batch, hidden_dim]
        embedding = mx.mean(outputs, axis=1)

        # Convert to list and return
        return embedding[0].tolist()

    async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
        """
        Create embeddings for a batch of text inputs.

        Args:
            input_data_list: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in input_data_list:
            embeddings.append(await self.create(text))
        return embeddings


# ============================================================================
# Prompt Templates
# ============================================================================

def build_entity_extraction_prompt(text: str) -> str:
    """Build prompt for entity extraction (adapted from graphiti)."""
    return f"""Extract entities from the text below. Find ALL people and organizations mentioned.

TEXT: {text}

Instructions:
- Extract EVERY person's name (PERSON)
- Extract EVERY organization/company name (ORGANIZATION)
- Extract EVERY location (LOCATION)
- Use the exact names from the text
- Start IDs at 0
- Return JSON array

Examples:
Text: "Alice works at Google"
Output: {{"entities": [{{"id": 0, "name": "Alice", "entity_type": "PERSON"}}, {{"id": 1, "name": "Google", "entity_type": "ORGANIZATION"}}]}}

Text: "Bob founded TechCo"
Output: {{"entities": [{{"id": 0, "name": "Bob", "entity_type": "PERSON"}}, {{"id": 1, "name": "TechCo", "entity_type": "ORGANIZATION"}}]}}

Now extract from the TEXT above. Return only JSON."""


def build_relationship_extraction_prompt(
    text: str,
    entities: List[ExtractedEntity],
    reference_time: str
) -> str:
    """Build prompt for relationship extraction (adapted from graphiti)."""
    entities_context = "\n".join([
        f"  {e.id}: {e.name}" + (f" ({e.entity_type})" if e.entity_type else "")
        for e in entities
    ])

    return f"""You are an expert fact extractor. Extract ONLY relationships explicitly stated in the TEXT.

<TEXT>
{text}
</TEXT>

<ENTITIES>
{entities_context}
</ENTITIES>

<REFERENCE_TIME>
{reference_time}
</REFERENCE_TIME>

Critical Rules:
1. ONLY use entity IDs from the ENTITIES list above (IDs 0 to {len(entities)-1})
2. ONLY extract relationships explicitly stated in the TEXT - do NOT invent facts
3. ONLY create relationships between two different entities (source_entity_id ≠ target_entity_id)
4. Use SCREAMING_SNAKE_CASE for relation_type (WORKS_AT, FOUNDED, JOINED, etc.)
5. Write fact as a direct paraphrase of what the TEXT states
6. If TEXT mentions time periods, use ISO 8601 format with Z suffix for valid_at/invalid_at
7. If no time mentioned, leave temporal fields null
8. If no relationships exist between the entities, return empty array

Do NOT:
- Invent relationships not in the TEXT
- Reference entity IDs that don't exist
- Create self-referential relationships
- Add personal opinions or general knowledge

Return valid JSON only."""


# ============================================================================
# Validation Functions
# ============================================================================

def validate_json_complete(json_str: str) -> bool:
    """Check if JSON string appears to be complete (not truncated)."""
    if not json_str:
        return False

    # Basic checks for truncation
    json_str = json_str.strip()

    # Should end with closing brace/bracket
    if not json_str.endswith('}') and not json_str.endswith(']'):
        return False

    # Count opening and closing braces/brackets
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')

    return open_braces == close_braces and open_brackets == close_brackets


# ============================================================================
# Conversion Functions
# ============================================================================

def to_graphiti_nodes(
    extracted_entities: List[ExtractedEntity],
    group_id: str,
    created_at: datetime
) -> List[EntityNode]:
    """Convert extracted entities to graphiti EntityNode objects."""
    nodes = []
    for entity in extracted_entities:
        labels = ['Entity']
        if entity.entity_type:
            labels.append(entity.entity_type)

        node = EntityNode(
            name=entity.name,
            group_id=group_id,
            labels=labels,
            summary='',
            created_at=created_at,
            name_embedding=None,
        )
        nodes.append(node)

    return nodes


def to_graphiti_edges(
    extracted_relationships: List[ExtractedRelationship],
    entity_nodes: List[EntityNode],
    episode_uuid: str,
    group_id: str,
    created_at: datetime
) -> List[EntityEdge]:
    """Convert extracted relationships to graphiti EntityEdge objects."""
    edges = []

    for rel in extracted_relationships:
        if not (0 <= rel.source_entity_id < len(entity_nodes)):
            logger.warning(f"Invalid source_entity_id: {rel.source_entity_id}")
            continue
        if not (0 <= rel.target_entity_id < len(entity_nodes)):
            logger.warning(f"Invalid target_entity_id: {rel.target_entity_id}")
            continue

        source_uuid = entity_nodes[rel.source_entity_id].uuid
        target_uuid = entity_nodes[rel.target_entity_id].uuid

        valid_at_dt = None
        invalid_at_dt = None

        # Handle valid_at - check for None or "null" string
        if rel.valid_at and rel.valid_at.lower() != "null":
            try:
                valid_at_dt = ensure_utc(
                    datetime.fromisoformat(rel.valid_at.replace('Z', '+00:00'))
                )
            except ValueError as e:
                logger.warning(f"Could not parse valid_at: {rel.valid_at} - {e}")

        # Handle invalid_at - check for None or "null" string
        if rel.invalid_at and rel.invalid_at.lower() != "null":
            try:
                invalid_at_dt = ensure_utc(
                    datetime.fromisoformat(rel.invalid_at.replace('Z', '+00:00'))
                )
            except ValueError as e:
                logger.warning(f"Could not parse invalid_at: {rel.invalid_at} - {e}")

        edge = EntityEdge(
            source_node_uuid=source_uuid,
            target_node_uuid=target_uuid,
            name=rel.relation_type,
            group_id=group_id,
            fact=rel.fact,
            episodes=[episode_uuid],
            created_at=created_at,
            valid_at=valid_at_dt,
            invalid_at=invalid_at_dt,
            fact_embedding=None,
        )
        edges.append(edge)

    return edges


# ============================================================================
# Main Extraction Pipeline
# ============================================================================

async def extract_and_save_to_graphiti(
    text: str,
    model,
    graphiti_instance: Graphiti,
    reference_time: datetime | None = None
):
    """Complete extraction pipeline: MLX-LM → Graphiti → Kuzu."""
    now = utc_now()
    reference_time = reference_time or now
    group_id = 'default'

    # Create episode
    episode = EpisodicNode(
        name=f"mlx_extraction_{now.isoformat()}",
        group_id=group_id,
        labels=[],
        source=EpisodeType.text,
        content=text,
        source_description="MLX-LM local extraction",
        created_at=now,
        valid_at=reference_time,
    )

    # Extract entities
    print("\n" + "=" * 80)
    print("ENTITY EXTRACTION")
    print("=" * 80)
    entity_prompt = build_entity_extraction_prompt(text)

    try:
        extracted_entities_json = model(entity_prompt, output_type=ExtractedEntities)

        # Validate JSON is complete
        if not validate_json_complete(extracted_entities_json):
            raise ValueError(f"Generated JSON appears truncated: {extracted_entities_json[:200]}...")

        extracted_entities = ExtractedEntities.model_validate_json(extracted_entities_json)
    except Exception as e:
        print(f"\nERROR: Entity extraction failed - {e}")
        raise

    if not extracted_entities.entities:
        print("\nNo entities found in text - nothing to save.")
        return {
            "episode": None,
            "entities": [],
            "relationships": []
        }

    print(f"\nExtracted {len(extracted_entities.entities)} entities:\n")
    for e in extracted_entities.entities:
        type_str = f" [{e.entity_type}]" if e.entity_type else ""
        print(f"  {e.id}: {e.name}{type_str}")

    # Convert to EntityNodes
    entity_nodes = to_graphiti_nodes(extracted_entities.entities, group_id, now)

    # Extract relationships (need at least 2 entities)
    if len(extracted_entities.entities) >= 2:
        print("\n" + "=" * 80)
        print("RELATIONSHIP EXTRACTION")
        print("=" * 80)
        relationship_prompt = build_relationship_extraction_prompt(
            text,
            extracted_entities.entities,
            reference_time.isoformat()
        )

        try:
            extracted_relationships_json = model(
                relationship_prompt,
                output_type=ExtractedRelationships
            )

            # Validate JSON is complete
            if not validate_json_complete(extracted_relationships_json):
                raise ValueError(f"Generated JSON appears truncated: {extracted_relationships_json[:200]}...")

            extracted_relationships = ExtractedRelationships.model_validate_json(
                extracted_relationships_json
            )
        except Exception as e:
            print(f"\nERROR: Relationship extraction failed - {e}")
            raise

        print(f"\nExtracted {len(extracted_relationships.relationships)} relationships:\n")
        valid_relationships = []
        for r in extracted_relationships.relationships:
            # Validate entity IDs before accessing
            if not (0 <= r.source_entity_id < len(extracted_entities.entities)):
                print(f"  WARNING: Invalid source_entity_id {r.source_entity_id} (only {len(extracted_entities.entities)} entities)")
                continue
            if not (0 <= r.target_entity_id < len(extracted_entities.entities)):
                print(f"  WARNING: Invalid target_entity_id {r.target_entity_id} (only {len(extracted_entities.entities)} entities)")
                continue

            # Check for self-referential relationships
            if r.source_entity_id == r.target_entity_id:
                print(f"  WARNING: Skipping self-referential relationship for entity {r.source_entity_id}")
                continue

            source_name = extracted_entities.entities[r.source_entity_id].name
            target_name = extracted_entities.entities[r.target_entity_id].name
            print(f"  {source_name}")
            print(f"    --[{r.relation_type}]-->")
            print(f"  {target_name}")
            print(f"    Fact: {r.fact}")
            if r.valid_at and r.valid_at.lower() != "null":
                print(f"    Valid from: {r.valid_at}")
            if r.invalid_at and r.invalid_at.lower() != "null":
                print(f"    Valid until: {r.invalid_at}")
            print()
            valid_relationships.append(r)

        # Update extracted_relationships to only include valid ones
        extracted_relationships.relationships = valid_relationships
    else:
        print("\n" + "=" * 80)
        print("RELATIONSHIP EXTRACTION")
        print("=" * 80)
        print(f"\nSkipping relationship extraction (need at least 2 entities, got {len(extracted_entities.entities)})")
        extracted_relationships = ExtractedRelationships(relationships=[])

    # Convert to EntityEdges
    entity_edges = to_graphiti_edges(
        extracted_relationships.relationships,
        entity_nodes,
        episode.uuid,
        group_id,
        now
    )

    # Create episodic edges
    episodic_edges = build_episodic_edges(entity_nodes, episode.uuid, now)

    # Update episode
    episode.entity_edges = [edge.uuid for edge in entity_edges]

    # Save to database
    print("=" * 80)
    print("SAVING TO DATABASE")
    print("=" * 80)
    await add_nodes_and_edges_bulk(
        graphiti_instance.driver,
        [episode],
        episodic_edges,
        entity_nodes,
        entity_edges,
        graphiti_instance.embedder
    )
    print("\nSaved successfully")
    print("=" * 80)

    return {
        "episode": episode,
        "entities": entity_nodes,
        "relationships": entity_edges
    }


# ============================================================================
# CLI Interface
# ============================================================================

async def main():
    """Interactive CLI loop."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="MLX-LM + Outlines → Graphiti Integration Test Tool"
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Database filename in brain/ directory (default: in-memory database for rapid testing)",
    )
    args = parser.parse_args()

    # Determine database path
    if args.db:
        # Use file-based database
        db_path = os.path.join(settings.BRAIN_DIR, args.db)
        # Ensure brain directory exists
        Path(settings.BRAIN_DIR).mkdir(parents=True, exist_ok=True)
        db_type = f"File: {db_path}"
    else:
        # Use in-memory database
        db_path = ":memory:"  # Special Kuzu in-memory database identifier
        db_type = "In-memory (no persistence)"

    print("\n" + "=" * 80)
    print("MLX-LM + Outlines → Graphiti Integration Test Tool")
    print("=" * 80)
    print(f"Database: {db_type}")
    print("=" * 80)

    # Load MLX model and tokenizer
    print("\nLoading MLX-LM model and tokenizer...")
    mlx_model, mlx_tokenizer = mlx_lm.load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    print("Model loaded")

    # Create outlines model for structured generation
    print("Initializing structured generation model...")
    model = outlines.from_mlxlm(mlx_model, mlx_tokenizer)
    print("Structured generation ready")

    # Initialize local MLX embedder
    print("Initializing local MLX embedder...")
    embedder = MLXEmbedder(mlx_model, mlx_tokenizer)
    print("Local embedder ready")

    # Initialize Graphiti with Kuzu
    print("Initializing Graphiti with Kuzu driver...")
    driver = KuzuDriver(db_path)

    graphiti = Graphiti(
        graph_driver=driver,
        embedder=embedder,
        llm_client=None,  # Not needed for storage-only
    )

    # Build indices on first run
    print("Building database indices...")
    await graphiti.build_indices_and_constraints()
    print("Database ready")

    # Interactive loop
    print("\n" + "=" * 80)
    print("Ready! Enter text to extract entities and relationships.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 80)

    while True:
        try:
            text = input("\n> ")

            if text.lower() in ['exit', 'quit', 'q']:
                print("\nExiting...")
                break

            if not text.strip():
                continue

            # Run extraction
            result = await extract_and_save_to_graphiti(
                text,
                model,
                graphiti,
                utc_now()
            )

            if result["episode"]:
                print("\nReady for next input.")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nERROR: {e}")
            print("Continuing...")

    # Cleanup
    await graphiti.close()
    print("\nDatabase connection closed. Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
