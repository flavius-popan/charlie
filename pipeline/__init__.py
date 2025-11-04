"""Modular Graphiti pipeline stages for local MLX inference.

Each module represents a discrete stage in the knowledge graph ingestion pipeline:
- extract_nodes: Entity extraction and resolution
- (Future) extract_edges: Relationship extraction
- (Future) extract_attributes: Entity attribute enrichment
- (Future) generate_summaries: Entity summarization
"""

from .extract_nodes import ExtractNodes, ExtractNodesOutput

__all__ = [
    "ExtractNodes",
    "ExtractNodesOutput",
]
