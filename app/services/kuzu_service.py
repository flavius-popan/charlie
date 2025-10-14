"""
KuzuService: Database query layer for Charlie's knowledge graph.

Handles all Kuzu database queries with connection pooling and
provides high-level methods for retrieving episodes, entities,
communities, and relationships.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import kuzu
from graphiti_core.driver.kuzu_driver import KuzuDriver

from app import settings
from app.models.graph import (
    EpisodicNode,
    EntityNode,
    CommunityNode,
    RelationshipNode,
    EpisodeWithEntities,
    EntityContext,
    CommunityDetail,
    CommunityMember,
    EpisodeWithMentionCount,
    CommunityWithEpisodes,
)

logger = logging.getLogger(__name__)


class KuzuService:
    """
    Service for querying Charlie's Kuzu knowledge graph.

    Provides async methods for retrieving nodes and relationships
    with proper connection management.
    """

    def __init__(self, db_path: str | None = None):
        """
        Initialize KuzuService with database path.

        Args:
            db_path: Path to the Kuzu database directory (uses settings.DB_PATH if None)
        """
        self.db_path = db_path if db_path is not None else settings.DB_PATH
        self._db: Optional[kuzu.Database] = None
        self._conn: Optional[kuzu.Connection] = None
        logger.info(f"KuzuService initialized with database: {self.db_path}")

    def connect(self) -> None:
        """Establish database connection."""
        if self._db is None:
            logger.info(f"Connecting to Kuzu database: {self.db_path}")
            self._db = kuzu.Database(self.db_path)
            self._conn = kuzu.Connection(self._db)
            logger.info("Kuzu connection established")

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn = None
        if self._db is not None:
            self._db = None
            logger.info("Kuzu connection closed")

    @asynccontextmanager
    async def get_connection(self):
        """Context manager for database connections."""
        self.connect()
        try:
            yield self._conn
        finally:
            pass

    def _ensure_connection(self) -> kuzu.Connection:
        """Ensure connection is established and return it."""
        if self._conn is None:
            self.connect()
        return self._conn

    async def get_episodes(
        self, limit: int = 100, offset: int = 0
    ) -> List[EpisodicNode]:
        """
        Get list of episodes ordered by date (newest first).

        Args:
            limit: Maximum number of episodes to return
            offset: Number of episodes to skip

        Returns:
            List of EpisodicNode objects
        """
        conn = self._ensure_connection()

        query = f"""
        MATCH (ep:Episodic)
        RETURN
            ep.uuid AS uuid,
            ep.name AS name,
            ep.content AS content,
            ep.valid_at AS valid_at,
            ep.created_at AS created_at
        ORDER BY ep.valid_at DESC
        SKIP {offset}
        LIMIT {limit}
        """

        try:
            result = conn.execute(query)
            episodes = []

            while result.has_next():
                row = result.get_next()
                episode = EpisodicNode(
                    uuid=row[0],
                    name=row[1],
                    content=row[2],
                    valid_at=self._parse_datetime(row[3]),
                    created_at=self._parse_datetime(row[4]),
                )
                episodes.append(episode)

            logger.info(f"Retrieved {len(episodes)} episodes")
            return episodes

        except Exception as e:
            logger.error(f"Error retrieving episodes: {e}")
            raise

    async def get_episode_by_uuid(self, uuid: str) -> Optional[EpisodicNode]:
        """
        Get a single episode by UUID.

        Args:
            uuid: Episode UUID

        Returns:
            EpisodicNode or None if not found
        """
        conn = self._ensure_connection()

        query = """
        MATCH (ep:Episodic)
        WHERE ep.uuid = $uuid
        RETURN
            ep.uuid AS uuid,
            ep.name AS name,
            ep.content AS content,
            ep.valid_at AS valid_at,
            ep.created_at AS created_at
        """

        try:
            result = conn.execute(query, parameters={"uuid": uuid})

            if result.has_next():
                row = result.get_next()
                return EpisodicNode(
                    uuid=row[0],
                    name=row[1],
                    content=row[2],
                    valid_at=self._parse_datetime(row[3]),
                    created_at=self._parse_datetime(row[4]),
                )

            return None

        except Exception as e:
            logger.error(f"Error retrieving episode {uuid}: {e}")
            raise

    async def get_episode_with_entities(
        self, uuid: str
    ) -> Optional[EpisodeWithEntities]:
        """
        Get episode with all its mentioned entities.

        Args:
            uuid: Episode UUID

        Returns:
            EpisodeWithEntities or None if episode not found
        """
        episode = await self.get_episode_by_uuid(uuid)
        if episode is None:
            return None

        conn = self._ensure_connection()

        query = """
        MATCH (ep:Episodic)-[m:MENTIONS]->(e:Entity)
        WHERE ep.uuid = $uuid
        RETURN
            e.uuid AS uuid,
            e.name AS name,
            e.summary AS summary,
            e.created_at AS created_at
        ORDER BY e.name
        """

        try:
            result = conn.execute(query, parameters={"uuid": uuid})
            entities = []

            while result.has_next():
                row = result.get_next()
                entity = EntityNode(
                    uuid=row[0],
                    name=row[1],
                    summary=row[2],
                    created_at=self._parse_datetime(row[3]),
                )
                entities.append(entity)

            logger.info(f"Retrieved episode {uuid} with {len(entities)} entities")
            return EpisodeWithEntities(episode=episode, entities=entities)

        except Exception as e:
            logger.error(f"Error retrieving episode with entities {uuid}: {e}")
            raise

    async def get_entity_by_uuid(self, uuid: str) -> Optional[EntityNode]:
        """
        Get a single entity by UUID.

        Args:
            uuid: Entity UUID

        Returns:
            EntityNode or None if not found
        """
        conn = self._ensure_connection()

        query = """
        MATCH (e:Entity)
        WHERE e.uuid = $uuid
        RETURN
            e.uuid AS uuid,
            e.name AS name,
            e.summary AS summary,
            e.created_at AS created_at
        """

        try:
            result = conn.execute(query, parameters={"uuid": uuid})

            if result.has_next():
                row = result.get_next()
                return EntityNode(
                    uuid=row[0],
                    name=row[1],
                    summary=row[2],
                    created_at=self._parse_datetime(row[3]),
                )

            return None

        except Exception as e:
            logger.error(f"Error retrieving entity {uuid}: {e}")
            raise

    async def get_communities(
        self, min_members: int = 2, limit: int = 100
    ) -> List[CommunityNode]:
        """
        Get list of communities with at least min_members.

        Args:
            min_members: Minimum number of members to include
            limit: Maximum number of communities to return

        Returns:
            List of CommunityNode objects with member_count attribute
        """
        conn = self._ensure_connection()

        query = """
        MATCH (c:Community)-[:HAS_MEMBER]->(e:Entity)
        WITH c, count(e) AS member_count
        WHERE member_count >= $min_members
        RETURN
            c.uuid AS uuid,
            c.name AS name,
            c.summary AS summary,
            c.created_at AS created_at,
            member_count
        ORDER BY member_count DESC
        LIMIT $limit
        """

        try:
            result = conn.execute(
                query, parameters={"min_members": min_members, "limit": limit}
            )
            communities = []

            while result.has_next():
                row = result.get_next()
                community = CommunityNode(
                    uuid=row[0],
                    name=row[1],
                    summary=row[2],
                    created_at=self._parse_datetime(row[3]),
                    member_count=row[4],
                )
                communities.append(community)

            logger.info(f"Retrieved {len(communities)} communities")
            return communities

        except Exception as e:
            logger.error(f"Error retrieving communities: {e}")
            raise

    async def count_episodes(self) -> int:
        """
        Get total count of episodes in the database.

        Returns:
            Total number of episodes
        """
        conn = self._ensure_connection()

        query = """
        MATCH (ep:Episodic)
        RETURN count(ep) AS count
        """

        try:
            result = conn.execute(query)
            if result.has_next():
                row = result.get_next()
                return row[0]
            return 0

        except Exception as e:
            logger.error(f"Error counting episodes: {e}")
            raise

    def _parse_datetime(self, dt: Any) -> datetime:
        """
        Parse datetime from Kuzu result, ensuring timezone awareness.

        Args:
            dt: Datetime value from Kuzu (may be string, datetime, or None)

        Returns:
            Timezone-aware datetime object
        """
        if dt is None:
            return datetime.now(timezone.utc)

        if isinstance(dt, str):
            parsed = datetime.fromisoformat(dt)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed

        if isinstance(dt, datetime):
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt

        return datetime.now(timezone.utc)

    async def get_community_detail(self, uuid: str) -> Optional[CommunityDetail]:
        """
        Get community with member entities and their episode counts.

        Args:
            uuid: Community UUID

        Returns:
            CommunityDetail with members or None if not found
        """
        conn = self._ensure_connection()

        query = """
        MATCH (c:Community {uuid: $uuid})
        OPTIONAL MATCH (c)-[:HAS_MEMBER]->(e:Entity)
        OPTIONAL MATCH (ep:Episodic)-[:MENTIONS]->(e)
        WITH c, e, count(DISTINCT ep) AS episode_count
        RETURN
            c.uuid AS c_uuid,
            c.name AS c_name,
            c.summary AS c_summary,
            c.created_at AS c_created_at,
            e.uuid AS e_uuid,
            e.name AS e_name,
            e.summary AS e_summary,
            e.created_at AS e_created_at,
            episode_count
        ORDER BY episode_count DESC
        """

        try:
            result = conn.execute(query, parameters={"uuid": uuid})

            if not result.has_next():
                logger.warning(f"Community not found: {uuid}")
                return None

            community_data = None
            members = []

            while result.has_next():
                row = result.get_next()

                # Initialize community on first row
                if community_data is None:
                    community_data = CommunityNode(
                        uuid=row[0],
                        name=row[1],
                        summary=row[2],
                        created_at=self._parse_datetime(row[3]),
                    )

                # Add member if entity data exists
                if row[4] is not None:
                    entity = EntityNode(
                        uuid=row[4],
                        name=row[5],
                        summary=row[6],
                        created_at=self._parse_datetime(row[7]),
                    )
                    member = CommunityMember(
                        entity=entity,
                        episode_count=row[8] if row[8] is not None else 0,
                    )
                    members.append(member)

            detail = CommunityDetail(
                community=community_data,
                members=members,
                member_count=len(members),
            )

            logger.info(f"Retrieved community {uuid} with {len(members)} members")
            return detail

        except Exception as e:
            logger.error(f"Error retrieving community detail {uuid}: {e}")
            raise

    async def get_community_episodes(
        self, uuid: str, limit: int = 50
    ) -> List[EpisodeWithMentionCount]:
        """
        Get episodes that mention entities from this community.

        Args:
            uuid: Community UUID
            limit: Maximum number of episodes to return

        Returns:
            List of EpisodeWithMentionCount objects
        """
        conn = self._ensure_connection()

        query = """
        MATCH (c:Community {uuid: $uuid})-[:HAS_MEMBER]->(e:Entity)
        MATCH (ep:Episodic)-[:MENTIONS]->(e)
        WITH ep, count(DISTINCT e) AS entity_mention_count
        RETURN
            ep.uuid AS uuid,
            ep.name AS name,
            ep.content AS content,
            ep.valid_at AS valid_at,
            ep.created_at AS created_at,
            entity_mention_count
        ORDER BY ep.valid_at DESC
        LIMIT $limit
        """

        try:
            result = conn.execute(query, parameters={"uuid": uuid, "limit": limit})
            episodes = []

            while result.has_next():
                row = result.get_next()
                episode = EpisodicNode(
                    uuid=row[0],
                    name=row[1],
                    content=row[2],
                    valid_at=self._parse_datetime(row[3]),
                    created_at=self._parse_datetime(row[4]),
                )
                episode_with_count = EpisodeWithMentionCount(
                    episode=episode,
                    entity_mention_count=row[5] if row[5] is not None else 0,
                )
                episodes.append(episode_with_count)

            logger.info(f"Retrieved {len(episodes)} episodes for community {uuid}")
            return episodes

        except Exception as e:
            logger.error(f"Error retrieving episodes for community {uuid}: {e}")
            raise

    async def get_entity_timeline(
        self, uuid: str, limit: int = 100
    ) -> List[EntityContext]:
        """
        Get timeline of episodes mentioning this entity.

        Args:
            uuid: Entity UUID
            limit: Maximum number of episodes to return

        Returns:
            List of EntityContext objects with episode info and excerpts
        """
        conn = self._ensure_connection()

        query = """
        MATCH (e:Entity {uuid: $uuid})
        MATCH (ep:Episodic)-[:MENTIONS]->(e)
        RETURN
            ep.uuid AS uuid,
            ep.name AS name,
            ep.content AS content,
            ep.valid_at AS valid_at
        ORDER BY ep.valid_at DESC
        LIMIT $limit
        """

        try:
            result = conn.execute(query, parameters={"uuid": uuid, "limit": limit})
            contexts = []

            # Get entity name for excerpt extraction
            entity = await self.get_entity_by_uuid(uuid)
            entity_name = entity.name if entity else ""

            while result.has_next():
                row = result.get_next()
                episode_uuid = row[0]
                episode_name = row[1]
                content = row[2]
                valid_at = self._parse_datetime(row[3])

                # Extract episode title
                episode = EpisodicNode(
                    uuid=episode_uuid,
                    name=episode_name,
                    content=content,
                    valid_at=valid_at,
                    created_at=valid_at,
                )

                # Create excerpt (first 200 chars or first paragraph)
                excerpt = self._extract_excerpt(content, entity_name)

                context = EntityContext(
                    episode_uuid=episode_uuid,
                    episode_title=episode.title,
                    episode_date=valid_at,
                    excerpt=excerpt,
                )
                contexts.append(context)

            logger.info(f"Retrieved {len(contexts)} timeline entries for entity {uuid}")
            return contexts

        except Exception as e:
            logger.error(f"Error retrieving timeline for entity {uuid}: {e}")
            raise

    def _extract_excerpt(self, content: str, entity_name: str, max_length: int = 300) -> str:
        """
        Extract relevant excerpt from content, preferring sections mentioning the entity.
        Strips markdown formatting to return clean plain text.

        Args:
            content: Full episode content
            entity_name: Name of entity to look for
            max_length: Maximum excerpt length

        Returns:
            Plain text excerpt string
        """
        # Remove markdown headers
        lines = [line for line in content.split("\n") if not line.strip().startswith("#")]
        text = " ".join(lines).strip()

        # Try to find paragraph containing entity mention
        if entity_name:
            paragraphs = text.split("\n\n")
            for para in paragraphs:
                if entity_name.lower() in para.lower():
                    para = self._strip_markdown(para.strip())[:max_length]
                    if len(para.strip()) > max_length:
                        para = para[:max_length] + "..."
                    return para

        # Fallback: return start of content
        text = self._strip_markdown(text)
        excerpt = text[:max_length]
        if len(text) > max_length:
            excerpt += "..."
        return excerpt

    def _strip_markdown(self, text: str) -> str:
        """
        Strip markdown formatting from text to return clean plain text.

        Args:
            text: Text with markdown formatting

        Returns:
            Plain text without markdown syntax
        """
        import re

        # Remove bold/italic: **text**, __text__, *text*, _text_
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)

        # Remove inline code: `code`
        text = re.sub(r'`(.+?)`', r'\1', text)

        # Remove links: [text](url) -> text
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)

        # Remove images: ![alt](url) -> alt
        text = re.sub(r'!\[(.+?)\]\(.+?\)', r'\1', text)

        # Remove strikethrough: ~~text~~
        text = re.sub(r'~~(.+?)~~', r'\1', text)

        # Remove list markers: - item, * item, + item, 1. item
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

        # Remove blockquote markers: > text
        text = re.sub(r'^\s*>\s+', '', text, flags=re.MULTILINE)

        # Remove horizontal rules: ---, ***, ___
        text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)

        # Clean up escaped characters (remove backslashes used for escaping)
        text = text.replace(r'\.', '.')
        text = text.replace(r'\(', '(')
        text = text.replace(r'\)', ')')
        text = text.replace(r'\[', '[')
        text = text.replace(r'\]', ']')
        text = text.replace(r'\-', '-')
        text = text.replace(r'\_', '_')
        text = text.replace(r'\*', '*')
        text = text.replace(r'\#', '#')

        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    async def get_entity_relationships(
        self, uuid: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get entities related to this entity through relationship nodes.

        Args:
            uuid: Entity UUID
            limit: Maximum number of relationships to return

        Returns:
            List of dictionaries with related entity and relationship info
        """
        conn = self._ensure_connection()

        # Query for relationships where this entity is involved
        # In graphiti, relationships are stored as hypernodes
        query = """
        MATCH (e1:Entity {uuid: $uuid})-[:RELATES_TO]->(rel)-[:RELATES_TO]->(e2:Entity)
        WHERE e1.uuid <> e2.uuid
        RETURN
            e2.uuid AS entity_uuid,
            e2.name AS entity_name,
            e2.summary AS entity_summary,
            e2.created_at AS entity_created_at,
            rel.name AS relationship_name,
            rel.fact AS relationship_fact,
            rel.uuid AS relationship_uuid,
            rel.created_at AS relationship_created_at
        ORDER BY rel.created_at DESC
        LIMIT $limit
        """

        try:
            result = conn.execute(query, parameters={"uuid": uuid, "limit": limit})
            relationships = []

            while result.has_next():
                row = result.get_next()

                related_entity = EntityNode(
                    uuid=row[0],
                    name=row[1],
                    summary=row[2],
                    created_at=self._parse_datetime(row[3]),
                )

                relationship = {
                    "entity": related_entity,
                    "relationship_name": row[4],
                    "relationship_fact": row[5],
                }
                relationships.append(relationship)

            logger.info(f"Retrieved {len(relationships)} relationships for entity {uuid}")
            return relationships

        except Exception as e:
            logger.error(f"Error retrieving relationships for entity {uuid}: {e}")
            raise

    async def get_entity_communities(self, uuid: str) -> List[CommunityNode]:
        """
        Get communities that this entity belongs to.

        Args:
            uuid: Entity UUID

        Returns:
            List of CommunityNode objects
        """
        conn = self._ensure_connection()

        query = """
        MATCH (c:Community)-[:HAS_MEMBER]->(e:Entity {uuid: $uuid})
        OPTIONAL MATCH (c)-[:HAS_MEMBER]->(member:Entity)
        WITH c, count(DISTINCT member) AS member_count
        RETURN
            c.uuid AS uuid,
            c.name AS name,
            c.summary AS summary,
            c.created_at AS created_at,
            member_count
        ORDER BY member_count DESC
        """

        try:
            result = conn.execute(query, parameters={"uuid": uuid})
            communities = []

            while result.has_next():
                row = result.get_next()
                community = CommunityNode(
                    uuid=row[0],
                    name=row[1],
                    summary=row[2],
                    created_at=self._parse_datetime(row[3]),
                    member_count=row[4],
                )
                communities.append(community)

            logger.info(f"Retrieved {len(communities)} communities for entity {uuid}")
            return communities

        except Exception as e:
            logger.error(f"Error retrieving communities for entity {uuid}: {e}")
            raise
