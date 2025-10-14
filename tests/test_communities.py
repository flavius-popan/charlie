"""
Community functionality tests for Charlie.

Tests community queries, models, routes, and data integrity
for the community features.
"""

import pytest
from pathlib import Path
from datetime import datetime, timezone


class TestCommunityModels:
    """Tests for community-related model definitions."""

    def test_community_node_creation(self):
        """Test CommunityNode model instantiation."""
        from app.models.graph import CommunityNode

        community = CommunityNode(
            uuid="test-comm-uuid",
            name="Test Community Name",
            summary="A test community summary",
            created_at=datetime.now(timezone.utc),
            member_count=10,
        )

        assert community.uuid == "test-comm-uuid"
        assert community.name == "Test Community Name"
        assert community.summary == "A test community summary"
        assert community.member_count == 10
        assert community.display_summary == "A test community summary"

    def test_community_node_with_summary(self):
        """Test CommunityNode with and without summary."""
        from app.models.graph import CommunityNode

        # Test with summary
        community_with_summary = CommunityNode(
            uuid="test-uuid",
            name="Community",
            summary="This is a community summary",
            created_at=datetime.now(timezone.utc),
            member_count=5,
        )
        assert community_with_summary.display_summary == "This is a community summary"

        # Test without summary
        community_no_summary = CommunityNode(
            uuid="test-uuid-2",
            name="Community 2",
            summary=None,
            created_at=datetime.now(timezone.utc),
            member_count=3,
        )
        assert community_no_summary.display_summary == "No summary available"

    def test_community_detail_model(self):
        """Test CommunityDetail composite model."""
        from app.models.graph import (
            CommunityDetail,
            CommunityNode,
            CommunityMember,
            EntityNode,
        )

        community = CommunityNode(
            uuid="comm-uuid",
            name="Test Community",
            summary="Summary",
            created_at=datetime.now(timezone.utc),
            member_count=2,
        )

        entity1 = EntityNode(
            uuid="e1",
            name="Entity One",
            summary="Summary 1",
            created_at=datetime.now(timezone.utc),
        )

        entity2 = EntityNode(
            uuid="e2",
            name="Entity Two",
            summary="Summary 2",
            created_at=datetime.now(timezone.utc),
        )

        members = [
            CommunityMember(entity=entity1, episode_count=5),
            CommunityMember(entity=entity2, episode_count=3),
        ]

        detail = CommunityDetail(
            community=community,
            members=members,
            member_count=2,
        )

        assert detail.community.uuid == "comm-uuid"
        assert len(detail.members) == 2
        assert detail.member_count == 2
        assert detail.members[0].episode_count == 5

    def test_episode_with_mention_count_model(self):
        """Test EpisodeWithMentionCount model."""
        from app.models.graph import EpisodeWithMentionCount, EpisodicNode

        episode = EpisodicNode(
            uuid="ep-uuid",
            name="test_episode",
            content="# Test\nContent",
            valid_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )

        ep_with_count = EpisodeWithMentionCount(
            episode=episode,
            entity_mention_count=7,
        )

        assert ep_with_count.episode.uuid == "ep-uuid"
        assert ep_with_count.entity_mention_count == 7

    def test_community_with_episodes_model(self):
        """Test CommunityWithEpisodes composite model."""
        from app.models.graph import (
            CommunityWithEpisodes,
            CommunityNode,
            EpisodicNode,
            EpisodeWithMentionCount,
        )

        community = CommunityNode(
            uuid="comm-uuid",
            name="Community",
            summary="Summary",
            created_at=datetime.now(timezone.utc),
            member_count=5,
        )

        episode = EpisodicNode(
            uuid="ep-uuid",
            name="episode",
            content="Content",
            valid_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )

        episodes = [EpisodeWithMentionCount(episode=episode, entity_mention_count=3)]

        comm_with_eps = CommunityWithEpisodes(
            community=community,
            episodes=episodes,
        )

        assert comm_with_eps.community.uuid == "comm-uuid"
        assert len(comm_with_eps.episodes) == 1
        assert comm_with_eps.episodes[0].entity_mention_count == 3


class TestCommunityQueries:
    """Tests for community database queries."""

    @pytest.mark.asyncio
    async def test_get_communities(self):
        """Test retrieving community list."""
        from app.services.kuzu_service import KuzuService

        db_path = Path("brain/charlie.kuzu")
        if not db_path.exists():
            pytest.skip("Database not found")

        service = KuzuService()
        communities = await service.get_communities(min_members=2, limit=10)

        assert isinstance(communities, list)

        if communities:
            community = communities[0]
            assert hasattr(community, "uuid")
            assert hasattr(community, "name")
            assert hasattr(community, "member_count")
            assert community.member_count >= 2

        service.close()

    @pytest.mark.asyncio
    async def test_get_community_detail(self):
        """Test retrieving detailed community information."""
        from app.services.kuzu_service import KuzuService

        db_path = Path("brain/charlie.kuzu")
        if not db_path.exists():
            pytest.skip("Database not found")

        service = KuzuService()
        communities = await service.get_communities(min_members=2, limit=1)

        if not communities:
            pytest.skip("No communities in database")

        community_uuid = communities[0].uuid
        detail = await service.get_community_detail(community_uuid)

        assert detail is not None
        assert detail.community.uuid == community_uuid
        assert isinstance(detail.members, list)
        assert len(detail.members) > 0
        assert detail.member_count == len(detail.members)

        member = detail.members[0]
        assert hasattr(member, "entity")
        assert hasattr(member, "episode_count")
        assert hasattr(member.entity, "uuid")
        assert hasattr(member.entity, "name")

        service.close()

    @pytest.mark.asyncio
    async def test_get_community_episodes(self):
        """Test retrieving episodes for a community."""
        from app.services.kuzu_service import KuzuService

        db_path = Path("brain/charlie.kuzu")
        if not db_path.exists():
            pytest.skip("Database not found")

        service = KuzuService()
        communities = await service.get_communities(min_members=2, limit=1)

        if not communities:
            pytest.skip("No communities in database")

        community_uuid = communities[0].uuid
        episodes = await service.get_community_episodes(community_uuid, limit=10)

        assert isinstance(episodes, list)

        if episodes:
            ep = episodes[0]
            assert hasattr(ep, "episode")
            assert hasattr(ep, "entity_mention_count")
            assert hasattr(ep.episode, "uuid")
            assert hasattr(ep.episode, "title")
            assert ep.entity_mention_count >= 0

        service.close()

    @pytest.mark.asyncio
    async def test_community_member_count_accuracy(self):
        """Test that community member counts are accurate."""
        from app.services.kuzu_service import KuzuService

        db_path = Path("brain/charlie.kuzu")
        if not db_path.exists():
            pytest.skip("Database not found")

        service = KuzuService()
        communities = await service.get_communities(min_members=2, limit=1)

        if not communities:
            pytest.skip("No communities in database")

        community = communities[0]
        detail = await service.get_community_detail(community.uuid)

        assert community.member_count == len(detail.members)
        assert detail.member_count == len(detail.members)

        service.close()

    @pytest.mark.asyncio
    async def test_communities_sorted_by_size(self):
        """Test that communities are returned sorted by member count."""
        from app.services.kuzu_service import KuzuService

        db_path = Path("brain/charlie.kuzu")
        if not db_path.exists():
            pytest.skip("Database not found")

        service = KuzuService()
        communities = await service.get_communities(min_members=2, limit=10)

        if len(communities) < 2:
            pytest.skip("Need at least 2 communities to test sorting")

        for i in range(len(communities) - 1):
            assert communities[i].member_count >= communities[i + 1].member_count

        service.close()


class TestCommunityRoutes:
    """Tests for community route configuration."""

    def test_community_router_exists(self):
        """Test that community router can be imported."""
        from app.routes.communities import router

        assert router is not None

    def test_community_routes_registered(self):
        """Test that community routes are registered in main app."""
        from app.main import app

        routes = [route.path for route in app.routes if hasattr(route, "path")]

        assert "/communities" in routes
        assert "/communities/{uuid}" in routes

    def test_communities_list_handler_exists(self):
        """Test that communities list handler function exists."""
        from app.routes.communities import communities_list

        assert callable(communities_list)

    def test_community_detail_handler_exists(self):
        """Test that community detail handler function exists."""
        from app.routes.communities import community_detail

        assert callable(community_detail)


class TestCommunityIntegration:
    """Integration tests for community functionality."""

    @pytest.mark.asyncio
    async def test_full_community_flow(self):
        """Test complete community list to detail flow."""
        from app.services.kuzu_service import KuzuService

        db_path = Path("brain/charlie.kuzu")
        if not db_path.exists():
            pytest.skip("Database not found")

        service = KuzuService()

        # Step 1: List communities
        communities = await service.get_communities(min_members=2, limit=5)

        if not communities:
            pytest.skip("No communities to test with")

        assert len(communities) > 0

        # Step 2: Get detail for first community
        community_uuid = communities[0].uuid
        detail = await service.get_community_detail(community_uuid)

        assert detail is not None
        assert detail.member_count > 0

        # Step 3: Get episodes for community
        episodes = await service.get_community_episodes(community_uuid, limit=10)

        assert isinstance(episodes, list)

        # Step 4: Verify data integrity
        for member in detail.members[:3]:
            assert member.entity.uuid
            assert member.entity.name
            assert member.episode_count >= 0

        service.close()

    @pytest.mark.asyncio
    async def test_community_templates_render_data(self):
        """Test that community templates can render with real data."""
        from app.services.kuzu_service import KuzuService

        db_path = Path("brain/charlie.kuzu")
        if not db_path.exists():
            pytest.skip("Database not found")

        service = KuzuService()
        communities = await service.get_communities(min_members=2, limit=1)

        if not communities:
            pytest.skip("No communities in database")

        community = communities[0]

        # Verify template would have necessary data
        assert community.uuid is not None
        assert community.name is not None
        assert community.display_summary is not None
        assert community.member_count > 0

        detail = await service.get_community_detail(community.uuid)

        # Verify detail template would have necessary data
        assert detail.community.uuid is not None
        assert len(detail.members) > 0
        assert detail.members[0].entity.name is not None

        service.close()

    @pytest.mark.asyncio
    async def test_no_regression_on_existing_features(self):
        """Test that community features don't break existing functionality."""
        from app.services.kuzu_service import KuzuService

        db_path = Path("brain/charlie.kuzu")
        if not db_path.exists():
            pytest.skip("Database not found")

        service = KuzuService()

        # Test episodes still work
        episodes = await service.get_episodes(limit=5)
        assert isinstance(episodes, list)

        if episodes:
            # Test episode detail still works
            episode_detail = await service.get_episode_with_entities(episodes[0].uuid)
            assert episode_detail is not None

        service.close()
