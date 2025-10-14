"""
Models package for Charlie.

Contains Pydantic models for graph nodes, edges, and API responses.
"""

from .graph import (
    EpisodicNode,
    EntityNode,
    CommunityNode,
    RelationshipNode,
    MentionsEdge,
    EntityMention,
    EpisodeWithEntities,
    EntityContext,
    EntityWithContexts,
    RelatedEntity,
    CommunityMember,
    CommunityDetail,
    EpisodeWithMentionCount,
    CommunityWithEpisodes,
)

__all__ = [
    "EpisodicNode",
    "EntityNode",
    "CommunityNode",
    "RelationshipNode",
    "MentionsEdge",
    "EntityMention",
    "EpisodeWithEntities",
    "EntityContext",
    "EntityWithContexts",
    "RelatedEntity",
    "CommunityMember",
    "CommunityDetail",
    "EpisodeWithMentionCount",
    "CommunityWithEpisodes",
]
