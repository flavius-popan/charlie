# MLX-LM + Outlines → Graphiti Integration

## Overview

This document specifies a complete implementation for extracting knowledge graph triplets (entities and relationships) using MLX-LM with outlines for structured generation, then seamlessly integrating with graphiti's storage infrastructure.

**Key Design Principle**: Bypass graphiti's LLM-based extraction pipeline while producing 100% compatible output that plugs directly into graphiti's storage, search, and query systems.

## Architecture

```
User Input Text
     ↓
MLX-LM + Outlines (Entity Extraction)
     ↓
List[ExtractedEntity] with IDs
     ↓
MLX-LM + Outlines (Relationship Extraction)
     ↓
List[ExtractedRelationship] using entity IDs
     ↓
Convert to Graphiti Objects
     ↓
EntityNode, EntityEdge, EpisodicNode
     ↓
add_nodes_and_edges_bulk()
     ↓
Kuzu Database (/brain/test.kuzu)
```

## Implementation Specification

### 1. Pydantic Models for Extraction

These models define the structured output that outlines will enforce:

```python
from pydantic import BaseModel, Field
from typing import List

class ExtractedEntity(BaseModel):
    """
    Entity extracted from text with unique ID for relationship references.

    CRITICAL: The id field is used by relationship extraction to reference entities.
    """
    id: int = Field(description="Sequential ID starting from 0")
    name: str = Field(description="Entity name, explicit and unambiguous")
    entity_type: str | None = Field(
        default=None,
        description="Optional entity type (PERSON, ORGANIZATION, LOCATION, CONCEPT, etc.)"
    )

class ExtractedEntities(BaseModel):
    """Container for all extracted entities."""
    entities: List[ExtractedEntity]


class ExtractedRelationship(BaseModel):
    """
    Relationship between two entities with temporal bounds.

    Follows graphiti's edge schema exactly for seamless integration.
    """
    source_entity_id: int = Field(description="ID of source entity from entities list")
    target_entity_id: int = Field(description="ID of target entity from entities list")
    relation_type: str = Field(
        description="Relationship type in SCREAMING_SNAKE_CASE (e.g., WORKS_AT, FOUNDED)"
    )
    fact: str = Field(
        description="Natural language description of the relationship, paraphrased from source"
    )
    valid_at: str | None = Field(
        default=None,
        description="ISO 8601 timestamp when relationship became true (e.g., 2024-01-15T00:00:00Z)"
    )
    invalid_at: str | None = Field(
        default=None,
        description="ISO 8601 timestamp when relationship ended (e.g., 2024-12-31T00:00:00Z)"
    )

class ExtractedRelationships(BaseModel):
    """Container for all extracted relationships."""
    relationships: List[ExtractedRelationship]
```

### 2. Prompt Templates

Adapted from graphiti's prompts (`extract_text` and `edge`) but simplified for local inference.

#### Entity Extraction Prompt

```python
def build_entity_extraction_prompt(text: str) -> str:
    """
    Build prompt for entity extraction.

    Based on graphiti's extract_text prompt but simplified:
    - No entity_type_id complexity (just optional string)
    - No previous episodes context
    - Focus on clarity for smaller models
    """
    return f"""You are an AI assistant that extracts entities from text.

<TEXT>
{text}
</TEXT>

Extract all significant entities explicitly or implicitly mentioned in the TEXT above.

Guidelines:
1. Extract people, organizations, places, concepts, and other significant entities
2. Be explicit and unambiguous in entity names (use full names, avoid abbreviations)
3. Assign each entity a sequential ID starting from 0
4. Optionally classify entities by type (PERSON, ORGANIZATION, LOCATION, CONCEPT, etc.)
5. Do NOT extract relationships or actions - only entities
6. Do NOT extract temporal information like dates or times
7. Do NOT extract pronouns (he/she/they/it)

Assign sequential IDs (0, 1, 2, ...) to each entity for later relationship extraction.

Extract entities as a JSON array."""
```

#### Relationship Extraction Prompt

```python
def build_relationship_extraction_prompt(
    text: str,
    entities: List[ExtractedEntity],
    reference_time: str
) -> str:
    """
    Build prompt for relationship extraction.

    Based on graphiti's edge prompt but adapted for outlines:
    - Takes extracted entities with IDs
    - Returns relationships using entity IDs (not names)
    - Includes temporal reasoning
    """
    entities_context = "\n".join([
        f"  {e.id}: {e.name}" + (f" ({e.entity_type})" if e.entity_type else "")
        for e in entities
    ])

    return f"""You are an expert fact extractor that extracts relationship triples from text.

<TEXT>
{text}
</TEXT>

<ENTITIES>
{entities_context}
</ENTITIES>

<REFERENCE_TIME>
{reference_time}
</REFERENCE_TIME>

Extract all factual relationships between the given ENTITIES based on the TEXT.

Only extract relationships that:
- Involve two DISTINCT entities from the ENTITIES list
- Are clearly stated or unambiguously implied in the TEXT
- Can be represented as edges in a knowledge graph

Extraction Rules:
1. Use entity IDs (source_entity_id, target_entity_id) from the ENTITIES list above
2. Use SCREAMING_SNAKE_CASE for relation_type (e.g., WORKS_AT, FOUNDED, KNOWS)
3. Write fact as a natural language paraphrase (don't quote verbatim)
4. Include temporal bounds when mentioned:
   - valid_at: when the relationship started or became true
   - invalid_at: when the relationship ended
   - Use ISO 8601 format with Z suffix (e.g., 2024-01-15T00:00:00Z)
   - Use REFERENCE_TIME to resolve relative dates ("last year", "recently")
5. Leave temporal fields null if no time information is stated
6. Do not hallucinate or duplicate relationships

Extract relationships as a JSON array."""
```

### 3. Conversion Functions

Convert extraction results to graphiti's native objects.

```python
from datetime import datetime
from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
from graphiti_core.edges import EntityEdge
from graphiti_core.utils.datetime_utils import utc_now, ensure_utc

def to_graphiti_nodes(
    extracted_entities: List[ExtractedEntity],
    group_id: str,
    created_at: datetime
) -> List[EntityNode]:
    """
    Convert extracted entities to graphiti EntityNode objects.

    Args:
        extracted_entities: List of entities from MLX-LM extraction
        group_id: Graph partition ID
        created_at: Creation timestamp

    Returns:
        List of EntityNode objects with auto-generated UUIDs
    """
    nodes = []
    for entity in extracted_entities:
        # Build labels list
        labels = ['Entity']
        if entity.entity_type:
            labels.append(entity.entity_type)

        # Create EntityNode (UUID auto-generated)
        node = EntityNode(
            name=entity.name,
            group_id=group_id,
            labels=labels,
            summary='',  # Empty initially; can be populated later
            created_at=created_at,
            name_embedding=None,  # Will be generated by add_nodes_and_edges_bulk
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
    """
    Convert extracted relationships to graphiti EntityEdge objects.

    Args:
        extracted_relationships: List of relationships from MLX-LM extraction
        entity_nodes: Corresponding EntityNode objects (must match entity IDs by index)
        episode_uuid: UUID of the source episode
        group_id: Graph partition ID
        created_at: Creation timestamp

    Returns:
        List of EntityEdge objects with auto-generated UUIDs
    """
    edges = []

    for rel in extracted_relationships:
        # Validate entity IDs
        if not (0 <= rel.source_entity_id < len(entity_nodes)):
            continue
        if not (0 <= rel.target_entity_id < len(entity_nodes)):
            continue

        # Map entity IDs to UUIDs
        source_uuid = entity_nodes[rel.source_entity_id].uuid
        target_uuid = entity_nodes[rel.target_entity_id].uuid

        # Parse temporal bounds
        valid_at_dt = None
        invalid_at_dt = None

        if rel.valid_at:
            try:
                valid_at_dt = ensure_utc(
                    datetime.fromisoformat(rel.valid_at.replace('Z', '+00:00'))
                )
            except ValueError as e:
                print(f"Warning: Could not parse valid_at: {rel.valid_at}")

        if rel.invalid_at:
            try:
                invalid_at_dt = ensure_utc(
                    datetime.fromisoformat(rel.invalid_at.replace('Z', '+00:00'))
                )
            except ValueError as e:
                print(f"Warning: Could not parse invalid_at: {rel.invalid_at}")

        # Create EntityEdge
        edge = EntityEdge(
            source_node_uuid=source_uuid,
            target_node_uuid=target_uuid,
            name=rel.relation_type,
            group_id=group_id,
            fact=rel.fact,
            episodes=[episode_uuid],  # Track source episode
            created_at=created_at,
            valid_at=valid_at_dt,
            invalid_at=invalid_at_dt,
            fact_embedding=None,  # Will be generated by add_nodes_and_edges_bulk
        )
        edges.append(edge)

    return edges
```

### 4. Main Extraction Pipeline

```python
import asyncio
from graphiti_core import Graphiti
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.embedder import OpenAIEmbedder
from graphiti_core.utils.maintenance.edge_operations import build_episodic_edges
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk

async def extract_and_save_to_graphiti(
    text: str,
    model,  # MLX-LM model wrapped with outlines
    graphiti_instance: Graphiti,
    reference_time: datetime | None = None
):
    """
    Complete extraction pipeline: MLX-LM → Graphiti → Kuzu.

    This function replaces add_episode() but produces identical results.

    Args:
        text: Input text to extract from
        model: Outlines-wrapped MLX-LM model
        graphiti_instance: Graphiti instance with Kuzu driver
        reference_time: Timestamp for the input (defaults to now)

    Returns:
        dict with episode, entities, and relationships
    """
    import logging
    logger = logging.getLogger(__name__)

    # 1. Setup
    now = utc_now()
    reference_time = reference_time or now
    group_id = 'default'

    logger.info(f"Starting extraction for text: {text[:100]}...")

    # 2. Create EpisodicNode (represents the source text)
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
    logger.debug(f"Created episode: {episode.uuid}")

    # 3. Extract entities using MLX-LM + outlines
    logger.info("Extracting entities...")
    entity_prompt = build_entity_extraction_prompt(text)
    extracted_entities_json = model(entity_prompt, output_type=ExtractedEntities)
    extracted_entities = ExtractedEntities.model_validate_json(extracted_entities_json)

    logger.info(f"Extracted {len(extracted_entities.entities)} entities:")
    for e in extracted_entities.entities:
        logger.info(f"  [{e.id}] {e.name}" + (f" ({e.entity_type})" if e.entity_type else ""))

    # 4. Convert to graphiti EntityNode objects
    entity_nodes = to_graphiti_nodes(
        extracted_entities.entities,
        group_id,
        now
    )
    logger.debug(f"Created {len(entity_nodes)} EntityNode objects")

    # 5. Extract relationships using MLX-LM + outlines
    logger.info("Extracting relationships...")
    relationship_prompt = build_relationship_extraction_prompt(
        text,
        extracted_entities.entities,
        reference_time.isoformat()
    )
    extracted_relationships_json = model(
        relationship_prompt,
        output_type=ExtractedRelationships
    )
    extracted_relationships = ExtractedRelationships.model_validate_json(
        extracted_relationships_json
    )

    logger.info(f"Extracted {len(extracted_relationships.relationships)} relationships:")
    for r in extracted_relationships.relationships:
        source_name = extracted_entities.entities[r.source_entity_id].name
        target_name = extracted_entities.entities[r.target_entity_id].name
        logger.info(f"  {source_name} --[{r.relation_type}]--> {target_name}")
        logger.info(f"    Fact: {r.fact}")

    # 6. Convert to graphiti EntityEdge objects
    entity_edges = to_graphiti_edges(
        extracted_relationships.relationships,
        entity_nodes,
        episode.uuid,
        group_id,
        now
    )
    logger.debug(f"Created {len(entity_edges)} EntityEdge objects")

    # 7. Create MENTIONS edges (Episode → Entity)
    episodic_edges = build_episodic_edges(entity_nodes, episode.uuid, now)
    logger.debug(f"Created {len(episodic_edges)} episodic MENTIONS edges")

    # 8. Update episode with entity_edges list
    episode.entity_edges = [edge.uuid for edge in entity_edges]

    # 9. Save everything using graphiti's bulk save
    logger.info("Saving to Kuzu database...")
    await add_nodes_and_edges_bulk(
        graphiti_instance.driver,
        [episode],
        episodic_edges,
        entity_nodes,
        entity_edges,
        graphiti_instance.embedder
    )
    logger.info("✓ Saved to database successfully")

    return {
        "episode": episode,
        "entities": entity_nodes,
        "relationships": entity_edges
    }
```

### 5. CLI Tool Implementation

Complete `mlx_test.py` with interactive loop:

```python
#!/usr/bin/env python3
"""
MLX-LM + Outlines → Graphiti Integration Test Tool

Interactive CLI for extracting knowledge graph triplets locally
and writing them to a Kuzu database with full graphiti compatibility.
"""

import asyncio
import logging
from datetime import datetime
from typing import List
import sys

# MLX and Outlines
import mlx_lm
import outlines
from pydantic import BaseModel, Field

# Graphiti
from graphiti_core import Graphiti
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.embedder import OpenAIEmbedder
from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
from graphiti_core.edges import EntityEdge
from graphiti_core.utils.datetime_utils import utc_now, ensure_utc
from graphiti_core.utils.maintenance.edge_operations import build_episodic_edges
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
# Prompt Templates
# ============================================================================

def build_entity_extraction_prompt(text: str) -> str:
    """Build prompt for entity extraction (adapted from graphiti)."""
    return f"""You are an AI assistant that extracts entities from text.

<TEXT>
{text}
</TEXT>

Extract all significant entities explicitly or implicitly mentioned in the TEXT above.

Guidelines:
1. Extract people, organizations, places, concepts, and other significant entities
2. Be explicit and unambiguous in entity names (use full names, avoid abbreviations)
3. Assign each entity a sequential ID starting from 0
4. Optionally classify entities by type (PERSON, ORGANIZATION, LOCATION, CONCEPT, etc.)
5. Do NOT extract relationships or actions - only entities
6. Do NOT extract temporal information like dates or times
7. Do NOT extract pronouns (he/she/they/it)

Extract entities as a JSON array."""


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

    return f"""You are an expert fact extractor that extracts relationship triples from text.

<TEXT>
{text}
</TEXT>

<ENTITIES>
{entities_context}
</ENTITIES>

<REFERENCE_TIME>
{reference_time}
</REFERENCE_TIME>

Extract all factual relationships between the given ENTITIES based on the TEXT.

Extraction Rules:
1. Use entity IDs (source_entity_id, target_entity_id) from the ENTITIES list
2. Use SCREAMING_SNAKE_CASE for relation_type (e.g., WORKS_AT, FOUNDED)
3. Write fact as a natural language paraphrase
4. Include temporal bounds when mentioned (ISO 8601 format with Z suffix)
5. Leave temporal fields null if no time information is stated

Extract relationships as a JSON array."""


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
            continue
        if not (0 <= rel.target_entity_id < len(entity_nodes)):
            continue

        source_uuid = entity_nodes[rel.source_entity_id].uuid
        target_uuid = entity_nodes[rel.target_entity_id].uuid

        valid_at_dt = None
        invalid_at_dt = None

        if rel.valid_at:
            try:
                valid_at_dt = ensure_utc(
                    datetime.fromisoformat(rel.valid_at.replace('Z', '+00:00'))
                )
            except ValueError:
                pass

        if rel.invalid_at:
            try:
                invalid_at_dt = ensure_utc(
                    datetime.fromisoformat(rel.invalid_at.replace('Z', '+00:00'))
                )
            except ValueError:
                pass

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

    logger.info(f"Starting extraction for text: {text[:100]}...")

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
    logger.debug(f"Created episode: {episode.uuid}")

    # Extract entities
    logger.info("Extracting entities...")
    entity_prompt = build_entity_extraction_prompt(text)
    extracted_entities_json = model(entity_prompt, output_type=ExtractedEntities)
    extracted_entities = ExtractedEntities.model_validate_json(extracted_entities_json)

    logger.info(f"Extracted {len(extracted_entities.entities)} entities:")
    for e in extracted_entities.entities:
        logger.info(f"  [{e.id}] {e.name}" + (f" ({e.entity_type})" if e.entity_type else ""))

    # Convert to EntityNodes
    entity_nodes = to_graphiti_nodes(extracted_entities.entities, group_id, now)

    # Extract relationships
    logger.info("Extracting relationships...")
    relationship_prompt = build_relationship_extraction_prompt(
        text,
        extracted_entities.entities,
        reference_time.isoformat()
    )
    extracted_relationships_json = model(
        relationship_prompt,
        output_type=ExtractedRelationships
    )
    extracted_relationships = ExtractedRelationships.model_validate_json(
        extracted_relationships_json
    )

    logger.info(f"Extracted {len(extracted_relationships.relationships)} relationships:")
    for r in extracted_relationships.relationships:
        source_name = extracted_entities.entities[r.source_entity_id].name
        target_name = extracted_entities.entities[r.target_entity_id].name
        logger.info(f"  {source_name} --[{r.relation_type}]--> {target_name}")
        logger.info(f"    Fact: {r.fact}")

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
    logger.info("Saving to Kuzu database...")
    await add_nodes_and_edges_bulk(
        graphiti_instance.driver,
        [episode],
        episodic_edges,
        entity_nodes,
        entity_edges,
        graphiti_instance.embedder
    )
    logger.info("✓ Saved to database successfully")

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
    print("=" * 80)
    print("MLX-LM + Outlines → Graphiti Integration Test Tool")
    print("=" * 80)

    # Load MLX model
    print("\nLoading MLX-LM model...")
    model = outlines.from_mlxlm(*mlx_lm.load("mlx-community/Llama-3.2-1B-Instruct-4bit"))
    print("✓ Model loaded")

    # Initialize Graphiti with Kuzu
    print("\nInitializing Graphiti with Kuzu driver...")
    kuzu_path = "/Users/flavius/repos/charlie/brain/test.kuzu"
    driver = KuzuDriver(kuzu_path)
    embedder = OpenAIEmbedder()

    graphiti = Graphiti(
        graph_driver=driver,
        embedder=embedder,
        llm_client=None,  # Not needed for storage-only
    )

    # Build indices on first run
    print("Building database indices...")
    await graphiti.build_indices_and_constraints()
    print("✓ Database ready")

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
            await extract_and_save_to_graphiti(
                text,
                model,
                graphiti,
                datetime.now()
            )

            print("\n✓ Complete! Ready for next input.")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting...")
            break
        except Exception as e:
            logger.error(f"Error during extraction: {e}", exc_info=True)
            print(f"\n✗ Error: {e}")
            print("Continuing...")

    # Cleanup
    await graphiti.close()
    print("\nDatabase connection closed. Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
```

## Testing Instructions

1. **Install dependencies:**
   ```bash
   pip install mlx-lm outlines-core pydantic graphiti-core
   ```

2. **Set OpenAI API key** (for embeddings):
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

3. **Run the CLI tool:**
   ```bash
   python mlx_test.py
   ```

4. **Test with sample input:**
   ```
   > Alice works at Acme Corp as a software engineer. She joined in January 2024.
   ```

5. **Verify output:**
   - Check console logs for extracted entities and relationships
   - Inspect `/brain/test.kuzu` database
   - Query using graphiti's search functions

## Integration Verification

To verify seamless integration, the output should be queryable using standard graphiti methods:

```python
# Search for entities
results = await graphiti.search("Alice", search_type="node")

# Get episode by UUID
episode = await EpisodicNode.get_by_uuid(driver, episode_uuid)

# Get entity relationships
edges = await EntityEdge.get_by_entity_uuid(driver, entity_uuid)
```

All queries should work identically to data extracted via graphiti's native `add_episode()` method.
