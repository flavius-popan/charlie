"""FalkorDB Lite driver implementation for backend operations."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider
from graphiti_core.embedder.client import EmbedderClient, EMBEDDING_DIM
from graphiti_core.utils.datetime_utils import convert_datetimes_to_strings

from backend.database.lifecycle import _ensure_graph
from backend.database.utils import STOPWORDS, _decode_value, _merge_episode_sync, to_cypher_literal
from backend.settings import DEFAULT_JOURNAL

logger = logging.getLogger(__name__)


class NullEmbedder(EmbedderClient):
    """Fallback embedder that returns empty vectors."""

    async def create(self, input_data):  # type: ignore[override]
        return [[0.0] * EMBEDDING_DIM for _ in input_data]


_embedder = NullEmbedder()


class FalkorLiteSession(GraphDriverSession):
    """Simplified FalkorDB session for backend operations."""

    provider = GraphProvider.FALKORDB

    def __init__(self, journal: str = DEFAULT_JOURNAL):
        self._journal = journal
        self._graph, self._lock = _ensure_graph(journal)

    async def run(self, query, **kwargs):
        """Execute bulk operations (episodes only for now)."""
        graph = self._graph
        if graph is None:
            raise RuntimeError("FalkorDB Lite is unavailable")

        if "episodes" in kwargs:
            episodes = kwargs["episodes"]
            for ep in episodes:
                # Use lock to serialize access to this journal's graph
                def _locked_merge():
                    with self._lock:
                        _merge_episode_sync(graph, ep)
                await asyncio.to_thread(_locked_merge)
            return None

        if "nodes" in kwargs:
            nodes = kwargs["nodes"]
            if nodes:
                raise NotImplementedError(
                    "Entity node persistence not yet implemented in backend"
                )
            return None

        if "entity_edges" in kwargs:
            edges = kwargs["entity_edges"]
            if edges:
                raise NotImplementedError(
                    "Entity edge persistence not yet implemented in backend"
                )
            return None

        if "episodic_edges" in kwargs:
            episodic_edges = kwargs["episodic_edges"]
            if episodic_edges:
                raise NotImplementedError(
                    "Episodic edge persistence not yet implemented in backend"
                )
            return None

        raise NotImplementedError(
            "FalkorLiteSession.run only supports bulk persistence operations"
        )

    async def close(self):
        pass

    async def execute_write(self, tx_func, *args, **kwargs):
        """Execute a write transaction."""
        return await tx_func(self, *args, **kwargs)

    async def __aenter__(self):
        """Async context manager enter."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        return False


class FalkorLiteDriver(GraphDriver):
    """Simplified FalkorDB driver for backend operations."""

    provider = GraphProvider.FALKORDB

    def __init__(self, journal: str = DEFAULT_JOURNAL):
        super().__init__()
        self._journal = journal
        self._graph, self._lock = _ensure_graph(journal)
        self._database = "falkordb-lite"

    async def execute_query(self, cypher_query_, **kwargs: Any):
        """Execute a Cypher query with parameter substitution."""
        graph = self._graph
        if graph is None:
            raise RuntimeError("FalkorDB Lite is unavailable")

        params = dict(kwargs)
        params.pop("routing_", None)
        params = convert_datetimes_to_strings(params)

        def _inject(query: str, substitutions: dict[str, Any]) -> str:
            for key, value in substitutions.items():
                placeholder = f"${key}"
                literal = to_cypher_literal(value)
                query = query.replace(placeholder, literal)
            return query

        formatted_query = _inject(cypher_query_, params)

        # Use lock to serialize access to this journal's graph
        def _locked_query():
            with self._lock:
                return graph.query(formatted_query, None)

        result = await asyncio.to_thread(_locked_query)

        raw = getattr(result, "_raw_response", None)
        records: list[dict[str, Any]] = []
        header: list[str] = []

        if isinstance(raw, list) and len(raw) >= 2:
            header = [_decode_value(col[1]) for col in raw[0]]
            rows = raw[1]

            for row in rows:
                record: dict[str, Any] = {}
                for idx, field_name in enumerate(header):
                    value = row[idx][1] if idx < len(row) else None
                    record[str(field_name)] = _decode_value(value)
                records.append(record)

        return records, header, None

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        """Build FalkorDB range and fulltext indices."""
        from graphiti_core.graph_queries import get_range_indices

        if delete_existing:
            await self.delete_all_indexes()

        # Range indices from graphiti-core
        index_queries = get_range_indices(self.provider)

        # Fulltext indices for BM25 search on episodes and entities
        fulltext_queries = [
            f"""CALL db.idx.fulltext.createNodeIndex(
                {{label: 'Episodic', stopwords: {STOPWORDS}}},
                'content', 'source', 'source_description', 'group_id'
            )""",
            f"""CALL db.idx.fulltext.createNodeIndex(
                {{label: 'Entity', stopwords: {STOPWORDS}}},
                'name', 'summary', 'group_id'
            )""",
            f"""CALL db.idx.fulltext.createNodeIndex(
                {{label: 'Community', stopwords: {STOPWORDS}}},
                'name', 'summary'
            )""",
        ]

        index_queries.extend(fulltext_queries)

        for query in index_queries:
            try:
                await self.execute_query(query)
            except Exception as exc:
                if "already indexed" in str(exc).lower():
                    logger.debug("Index already exists, skipping: %s", exc)
                else:
                    raise

    def session(self, database=None) -> GraphDriverSession:
        return FalkorLiteSession(journal=self._journal)

    async def close(self):
        """Close the driver (no-op for FalkorDB Lite)."""
        pass

    async def delete_all_indexes(self):
        """Delete all indexes (simplified implementation)."""
        pass

    @staticmethod
    def _sanitize_fulltext(query: str) -> str:
        """Remove special characters that interfere with fulltext search."""
        separator_map = str.maketrans(
            {
                ',': ' ', '.': ' ', '<': ' ', '>': ' ', '{': ' ', '}': ' ',
                '[': ' ', ']': ' ', '"': ' ', "'": ' ', ':': ' ', ';': ' ',
                '!': ' ', '@': ' ', '#': ' ', '$': ' ', '%': ' ', '^': ' ',
                '&': ' ', '*': ' ', '(': ' ', ')': ' ', '-': ' ', '+': ' ',
                '=': ' ', '~': ' ', '?': ' ',
            }
        )
        sanitized = query.translate(separator_map)
        return " ".join(sanitized.split())

    def build_fulltext_query(
        self, query: str, group_ids: list[str] | None = None, max_query_length: int = 128
    ) -> str:
        """Build a fulltext search query for FalkorDB using RedisSearch syntax."""
        group_filter = (
            f"(@group_id:{'|'.join(group_ids)})" if group_ids and len(group_ids) > 0 else ""
        )
        sanitized_query = self._sanitize_fulltext(query)
        filtered_words = [word for word in sanitized_query.split() if word.lower() not in STOPWORDS]
        sanitized_query = " | ".join(filtered_words)

        if len(sanitized_query.split(" ")) + len(group_filter.split(" ")) >= max_query_length:
            return ""

        if not sanitized_query and not group_filter:
            return ""

        if not sanitized_query:
            return group_filter

        if group_filter:
            return f"{group_filter} ({sanitized_query})"
        return f"({sanitized_query})"


def get_driver(journal: str = DEFAULT_JOURNAL) -> FalkorLiteDriver:
    """Create a driver instance for the specified journal.

    Args:
        journal: Journal name (defaults to DEFAULT_JOURNAL)

    Returns:
        New FalkorLiteDriver instance for the journal

    Note: This is not a singleton - each call creates a new driver instance.
    """
    return FalkorLiteDriver(journal=journal)


__all__ = [
    "FalkorLiteDriver",
    "FalkorLiteSession",
    "NullEmbedder",
    "get_driver",
]
