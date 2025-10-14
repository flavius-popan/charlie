"""
Infrastructure tests for Charlie.

Tests core infrastructure components including database connectivity,
model definitions, service layer, route configuration, templates, and static files.
"""

import pytest
from pathlib import Path
from datetime import datetime, timezone


class TestDatabase:
    """Tests for database connectivity and basic queries."""

    @pytest.mark.asyncio
    async def test_database_exists(self):
        """Verify database file exists."""
        db_path = Path("brain/charlie.kuzu")
        assert db_path.exists(), "Database file should exist at brain/charlie.kuzu"

    @pytest.mark.asyncio
    async def test_database_has_episodes(self):
        """Verify database contains episode data."""
        import kuzu

        db_path = Path("brain/charlie.kuzu")
        if not db_path.exists():
            pytest.skip("Database not found")

        db = kuzu.Database(str(db_path))
        conn = kuzu.Connection(db)
        result = conn.execute("MATCH (ep:Episodic) RETURN count(ep) as count")
        count = result.get_next()[0] if result.has_next() else 0

        assert count > 0, "Database should contain episodes"

    @pytest.mark.asyncio
    async def test_database_has_entities(self):
        """Verify database contains entity data."""
        import kuzu

        db_path = Path("brain/charlie.kuzu")
        if not db_path.exists():
            pytest.skip("Database not found")

        db = kuzu.Database(str(db_path))
        conn = kuzu.Connection(db)
        result = conn.execute("MATCH (e:Entity) RETURN count(e) as count")
        count = result.get_next()[0] if result.has_next() else 0

        assert count >= 0, "Database should be queryable for entities"


class TestModels:
    """Tests for Pydantic model definitions."""

    def test_episodic_node_model(self):
        """Test EpisodicNode model creation."""
        from app.models.graph import EpisodicNode

        episode = EpisodicNode(
            uuid="test-uuid",
            name="test_episode",
            content="# Test\nContent here",
            valid_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )

        assert episode.uuid == "test-uuid"
        assert episode.name == "test_episode"
        assert episode.content == "# Test\nContent here"
        assert episode.title == "Test"  # Extracted from markdown header

    def test_entity_node_model(self):
        """Test EntityNode model creation."""
        from app.models.graph import EntityNode

        entity = EntityNode(
            uuid="entity-uuid",
            name="Test Entity",
            summary="A test entity summary",
            created_at=datetime.now(timezone.utc),
        )

        assert entity.uuid == "entity-uuid"
        assert entity.name == "Test Entity"
        assert entity.summary == "A test entity summary"

    def test_community_node_model(self):
        """Test CommunityNode model creation."""
        from app.models.graph import CommunityNode

        community = CommunityNode(
            uuid="community-uuid",
            name="Test Community",
            summary="Community summary",
            created_at=datetime.now(timezone.utc),
            member_count=5,
        )

        assert community.uuid == "community-uuid"
        assert community.name == "Test Community"
        assert community.member_count == 5
        assert community.display_summary == "Community summary"

    def test_episode_with_entities_model(self):
        """Test EpisodeWithEntities composite model."""
        from app.models.graph import EpisodeWithEntities, EpisodicNode, EntityNode

        episode = EpisodicNode(
            uuid="ep-uuid",
            name="test_ep",
            content="Content",
            valid_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )

        entities = [
            EntityNode(
                uuid="e1",
                name="Entity 1",
                summary="Summary 1",
                created_at=datetime.now(timezone.utc),
            ),
            EntityNode(
                uuid="e2",
                name="Entity 2",
                summary="Summary 2",
                created_at=datetime.now(timezone.utc),
            ),
        ]

        episode_with_entities = EpisodeWithEntities(episode=episode, entities=entities)

        assert episode_with_entities.episode.uuid == "ep-uuid"
        assert len(episode_with_entities.entities) == 2


class TestKuzuService:
    """Tests for KuzuService database queries."""

    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """Test KuzuService can be initialized."""
        from app.services.kuzu_service import KuzuService

        service = KuzuService()
        assert service is not None
        service.close()

    @pytest.mark.asyncio
    async def test_get_episodes(self):
        """Test retrieving episode list."""
        from app.services.kuzu_service import KuzuService

        db_path = Path("brain/charlie.kuzu")
        if not db_path.exists():
            pytest.skip("Database not found")

        service = KuzuService()
        episodes = await service.get_episodes(limit=5)

        assert isinstance(episodes, list)
        if episodes:
            assert hasattr(episodes[0], "uuid")
            assert hasattr(episodes[0], "title")

        service.close()

    @pytest.mark.asyncio
    async def test_get_episode_with_entities(self):
        """Test retrieving episode with related entities."""
        from app.services.kuzu_service import KuzuService

        db_path = Path("brain/charlie.kuzu")
        if not db_path.exists():
            pytest.skip("Database not found")

        service = KuzuService()
        episodes = await service.get_episodes(limit=1)

        if not episodes:
            pytest.skip("No episodes in database")

        episode_detail = await service.get_episode_with_entities(episodes[0].uuid)

        assert episode_detail is not None
        assert episode_detail.episode.uuid == episodes[0].uuid
        assert isinstance(episode_detail.entities, list)

        service.close()

    @pytest.mark.asyncio
    async def test_count_episodes(self):
        """Test counting total episodes."""
        from app.services.kuzu_service import KuzuService

        db_path = Path("brain/charlie.kuzu")
        if not db_path.exists():
            pytest.skip("Database not found")

        service = KuzuService()
        count = await service.count_episodes()

        assert isinstance(count, int)
        assert count >= 0

        service.close()

    @pytest.mark.asyncio
    async def test_get_entity_by_uuid(self):
        """Test retrieving a single entity by UUID."""
        from app.services.kuzu_service import KuzuService

        db_path = Path("brain/charlie.kuzu")
        if not db_path.exists():
            pytest.skip("Database not found")

        service = KuzuService()
        episodes = await service.get_episodes(limit=1)

        if not episodes:
            pytest.skip("No episodes in database")

        episode_detail = await service.get_episode_with_entities(episodes[0].uuid)

        if not episode_detail.entities:
            pytest.skip("Episode has no entities")

        entity_uuid = episode_detail.entities[0].uuid
        entity = await service.get_entity_by_uuid(entity_uuid)

        assert entity is not None
        assert entity.uuid == entity_uuid
        assert hasattr(entity, "name")

        service.close()


class TestRoutes:
    """Tests for FastAPI route configuration."""

    def test_app_initialization(self):
        """Test FastAPI app can be imported and initialized."""
        from app.main import app

        assert app is not None
        assert hasattr(app, "routes")

    def test_required_routes_exist(self):
        """Test that all required routes are registered."""
        from app.main import app

        routes = [route.path for route in app.routes if hasattr(route, "path")]

        required_routes = [
            "/",
            "/episodes/{uuid}",
            "/entities/{uuid}",
            "/communities",
            "/communities/{uuid}",
        ]

        for required_route in required_routes:
            assert required_route in routes, f"Missing required route: {required_route}"

    def test_static_files_mounted(self):
        """Test that static files are mounted."""
        from app.main import app

        route_names = [route.name for route in app.routes if hasattr(route, "name")]
        assert "static" in route_names, "Static files should be mounted"


class TestTemplates:
    """Tests for Jinja2 template files."""

    def test_templates_directory_exists(self):
        """Test that templates directory exists."""
        templates_dir = Path("templates")
        assert templates_dir.exists(), "Templates directory should exist"
        assert templates_dir.is_dir(), "Templates should be a directory"

    def test_base_template_exists(self):
        """Test that base template exists."""
        template_path = Path("templates/base.html")
        assert template_path.exists(), "base.html should exist"

    def test_episode_templates_exist(self):
        """Test that episode templates exist."""
        assert Path("templates/episodes.html").exists(), "episodes.html should exist"
        assert Path("templates/episode_detail.html").exists(), (
            "episode_detail.html should exist"
        )

    def test_entity_template_exists(self):
        """Test that entity detail template exists."""
        assert Path("templates/entity_detail.html").exists(), (
            "entity_detail.html should exist"
        )

    def test_community_templates_exist(self):
        """Test that community templates exist."""
        assert Path("templates/communities.html").exists(), (
            "communities.html should exist"
        )
        assert Path("templates/community_detail.html").exists(), (
            "community_detail.html should exist"
        )


class TestStaticFiles:
    """Tests for static assets."""

    def test_static_directory_exists(self):
        """Test that static directory exists."""
        static_dir = Path("static")
        assert static_dir.exists(), "Static directory should exist"
        assert static_dir.is_dir(), "Static should be a directory"

    def test_css_directory_exists(self):
        """Test that CSS directory exists."""
        css_dir = Path("static/css")
        assert css_dir.exists(), "CSS directory should exist"

    def test_main_css_exists(self):
        """Test that main CSS file exists."""
        css_file = Path("static/css/main.css")
        assert css_file.exists(), "main.css should exist"

    def test_js_directory_exists(self):
        """Test that JavaScript directory exists."""
        js_dir = Path("static/js")
        assert js_dir.exists(), "JavaScript directory should exist"

    def test_navigation_js_exists(self):
        """Test that navigation JavaScript exists."""
        js_file = Path("static/js/navigation.js")
        assert js_file.exists(), "navigation.js should exist"

    def test_css_has_entity_styles(self):
        """Test that CSS includes entity mention styles."""
        css_content = Path("static/css/main.css").read_text()

        entity_styles = [
            ".entity-mention",
            "border-bottom:",
            "cursor: pointer",
        ]

        for style in entity_styles:
            assert style in css_content, f"CSS should include {style}"

    def test_css_has_community_styles(self):
        """Test that CSS includes community styles."""
        css_content = Path("static/css/main.css").read_text()

        community_styles = [
            ".communities-page",
            ".communities-grid",
            ".community-card",
            ".community-detail-page",
        ]

        for style in community_styles:
            assert style in css_content, f"CSS should include {style}"
