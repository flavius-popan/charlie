# FalkorDB Backend Integration in Graphiti-Core

This document details the FalkorDB backend implementation in graphiti-core, covering driver initialization, query execution, session management, indexing, and database-specific patterns.

**Key Files:**
- `/Users/flavius/repos/charlie/.venv/lib/python3.13/site-packages/graphiti_core/driver/falkordb_driver.py`
- `/Users/flavius/repos/charlie/.venv/lib/python3.13/site-packages/graphiti_core/driver/driver.py`
- `/Users/flavius/repos/charlie/.venv/lib/python3.13/site-packages/graphiti_core/graph_queries.py`
- `/Users/flavius/repos/charlie/.venv/lib/python3.13/site-packages/graphiti_core/models/nodes/node_db_queries.py`
- `/Users/flavius/repos/charlie/.venv/lib/python3.13/site-packages/graphiti_core/models/edges/edge_db_queries.py`

---

## 1. FalkorDriver Initialization and Configuration

**Location:** `driver/falkordb_driver.py` (lines 112-141)

```python
class FalkorDriver(GraphDriver):
    provider = GraphProvider.FALKORDB
    aoss_client: None = None

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        username: str | None = None,
        password: str | None = None,
        falkor_db: FalkorDB | None = None,
        database: str = 'default_db',
    ):
        """
        Initialize the FalkorDB driver.

        FalkorDB is a multi-tenant graph database.
        Default parameters assume a local (on-premises) FalkorDB instance.
        """
        super().__init__()

        self._database = database
        if falkor_db is not None:
            # If a FalkorDB instance is provided, use it directly
            self.client = falkor_db
        else:
            self.client = FalkorDB(host=host, port=port, username=username, password=password)

        # FalkorDB uses RedisSearch-like syntax for fulltext queries
        self.fulltext_syntax = '@'
```

**Key Characteristics:**
- **Multi-tenant:** Uses named databases/graphs (default: `default_db`)
- **Redis Protocol:** Runs on port 6379 by default (Redis-compatible)
- **Client Reuse:** Supports passing existing FalkorDB client for connection pooling
- **RedisSearch Integration:** Uses `@` prefix for fulltext field queries

**Graph Selection:** `driver/falkordb_driver.py` (lines 143-147)

```python
def _get_graph(self, graph_name: str | None) -> FalkorGraph:
    # FalkorDB requires a non-None database name for multi-tenant graphs
    if graph_name is None:
        graph_name = self._database
    return self.client.select_graph(graph_name)
```

---

## 2. Query Execution and Datetime Handling

**Location:** `driver/falkordb_driver.py` (lines 149-180)

### execute_query Method

```python
async def execute_query(self, cypher_query_, **kwargs: Any):
    graph = self._get_graph(self._database)

    # Convert datetime objects to ISO strings (FalkorDB does not support datetime objects directly)
    params = convert_datetimes_to_strings(dict(kwargs))

    try:
        result = await graph.query(cypher_query_, params)
    except Exception as e:
        if 'already indexed' in str(e):
            # check if index already exists
            logger.info(f'Index already exists: {e}')
            return None
        logger.error(f'Error executing FalkorDB query: {e}\n{cypher_query_}\n{params}')
        raise

    # Convert the result header to a list of strings
    header = [h[1] for h in result.header]

    # Convert FalkorDB's result format (list of lists) to the format expected by Graphiti (list of dicts)
    records = []
    for row in result.result_set:
        record = {}
        for i, field_name in enumerate(header):
            if i < len(row):
                record[field_name] = row[i]
            else:
                # If there are more fields in header than values in row, set to None
                record[field_name] = None
        records.append(record)

    return records, header, None
```

**Key Features:**
1. **Datetime Conversion:** FalkorDB doesn't support datetime objects natively - converts to ISO strings
2. **Index Existence Handling:** Gracefully handles "already indexed" errors
3. **Result Format Conversion:** Transforms FalkorDB's list-of-lists format to dict-based records

### Datetime Conversion Utility

**Location:** `utils/datetime_utils.py` (lines 45-55)

```python
def convert_datetimes_to_strings(obj):
    if isinstance(obj, dict):
        return {k: convert_datetimes_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_datetimes_to_strings(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_datetimes_to_strings(item) for item in obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj
```

**Recursively converts all datetime objects to ISO 8601 strings in nested structures.**

---

## 3. Session Management

**Location:** `driver/falkordb_driver.py` (lines 77-110)

### FalkorDriverSession

```python
class FalkorDriverSession(GraphDriverSession):
    provider = GraphProvider.FALKORDB

    def __init__(self, graph: FalkorGraph):
        self.graph = graph

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # No cleanup needed for Falkor, but method must exist
        pass

    async def close(self):
        # No explicit close needed for FalkorDB, but method must exist
        pass

    async def execute_write(self, func, *args, **kwargs):
        # Directly await the provided async function with `self` as the transaction/session
        return await func(self, *args, **kwargs)

    async def run(self, query: str | list, **kwargs: Any) -> Any:
        # FalkorDB does not support argument for Label Set, so it's converted into an array of queries
        if isinstance(query, list):
            for cypher, params in query:
                params = convert_datetimes_to_strings(params)
                await self.graph.query(str(cypher), params)
        else:
            params = dict(kwargs)
            params = convert_datetimes_to_strings(params)
            await self.graph.query(str(query), params)
        return None
```

**Critical Differences:**
- **No Explicit Transactions:** FalkorDB doesn't expose transaction boundaries like Neo4j
- **List Query Support:** Handles queries as lists for label set operations (see section 7)
- **No Cleanup Required:** Sessions don't maintain state requiring cleanup
- **Datetime Handling:** Applies conversion at session level too

### Session Creation

**Location:** `driver/falkordb_driver.py` (lines 182-183)

```python
def session(self, database: str | None = None) -> GraphDriverSession:
    return FalkorDriverSession(self._get_graph(database))
```

---

## 4. Index Creation and Management

### Range Indices

**Location:** `graph_queries.py` (lines 29-43)

```python
def get_range_indices(provider: GraphProvider) -> list[LiteralString]:
    if provider == GraphProvider.FALKORDB:
        return [
            # Entity node
            'CREATE INDEX FOR (n:Entity) ON (n.uuid, n.group_id, n.name, n.created_at)',
            # Episodic node
            'CREATE INDEX FOR (n:Episodic) ON (n.uuid, n.group_id, n.created_at, n.valid_at)',
            # Community node
            'CREATE INDEX FOR (n:Community) ON (n.uuid)',
            # RELATES_TO edge
            'CREATE INDEX FOR ()-[e:RELATES_TO]-() ON (e.uuid, e.group_id, e.name, e.created_at, e.expired_at, e.valid_at, e.invalid_at)',
            # MENTIONS edge
            'CREATE INDEX FOR ()-[e:MENTIONS]-() ON (e.uuid, e.group_id)',
            # HAS_MEMBER edge
            'CREATE INDEX FOR ()-[e:HAS_MEMBER]-() ON (e.uuid)',
        ]
```

**Differences from Neo4j:**
- **Composite Indices:** FalkorDB creates composite indices on multiple fields in one statement
- **No `IF NOT EXISTS`:** FalkorDB doesn't support this clause - must handle index existence errors
- **Simpler Syntax:** No need for named indices

### Fulltext Indices

**Location:** `graph_queries.py` (lines 72-108)

```python
def get_fulltext_indices(provider: GraphProvider) -> list[LiteralString]:
    if provider == GraphProvider.FALKORDB:
        from graphiti_core.driver.falkordb_driver import STOPWORDS

        stopwords_str = str(STOPWORDS)

        return [
            f"""CALL db.idx.fulltext.createNodeIndex(
                {{
                    label: 'Episodic',
                    stopwords: {stopwords_str}
                }},
                'content', 'source', 'source_description', 'group_id'
            )""",
            f"""CALL db.idx.fulltext.createNodeIndex(
                {{
                    label: 'Entity',
                    stopwords: {stopwords_str}
                }},
                'name', 'summary', 'group_id'
            )""",
            f"""CALL db.idx.fulltext.createNodeIndex(
                {{
                    label: 'Community',
                    stopwords: {stopwords_str}
                }},
                'name', 'group_id'
            )""",
            """CREATE FULLTEXT INDEX FOR ()-[e:RELATES_TO]-() ON (e.name, e.fact, e.group_id)""",
        ]
```

**FalkorDB Fulltext Features:**
- **Stopword Configuration:** Custom stopword lists for token filtering
- **RedisSearch Backend:** Uses RedisSearch under the hood
- **Node vs Edge Syntax:** Different procedures for nodes (`db.idx.fulltext.createNodeIndex`) vs edges (`CREATE FULLTEXT INDEX`)

### Stopwords Definition

**Location:** `driver/falkordb_driver.py` (lines 40-74)

```python
STOPWORDS = [
    'a', 'is', 'the', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by',
    'for', 'if', 'in', 'into', 'it', 'no', 'not', 'of', 'on', 'or', 'such',
    'that', 'their', 'then', 'there', 'these', 'they', 'this', 'to', 'was',
    'will', 'with',
]
```

### Index Deletion

**Location:** `driver/falkordb_driver.py` (lines 194-224)

```python
async def delete_all_indexes(self) -> None:
    result = await self.execute_query('CALL db.indexes()')
    if not result:
        return

    records, _, _ = result
    drop_tasks = []

    for record in records:
        label = record['label']
        entity_type = record['entitytype']

        for field_name, index_type in record['types'].items():
            if 'RANGE' in index_type:
                drop_tasks.append(self.execute_query(f'DROP INDEX ON :{label}({field_name})'))
            elif 'FULLTEXT' in index_type:
                if entity_type == 'NODE':
                    drop_tasks.append(
                        self.execute_query(
                            f'DROP FULLTEXT INDEX FOR (n:{label}) ON (n.{field_name})'
                        )
                    )
                elif entity_type == 'RELATIONSHIP':
                    drop_tasks.append(
                        self.execute_query(
                            f'DROP FULLTEXT INDEX FOR ()-[e:{label}]-() ON (e.{field_name})'
                        )
                    )

    if drop_tasks:
        await asyncio.gather(*drop_tasks)
```

**Index Management Strategy:**
1. Query existing indices with `CALL db.indexes()`
2. Categorize by type (RANGE vs FULLTEXT) and entity type (NODE vs RELATIONSHIP)
3. Build appropriate DROP statements based on index type
4. Execute deletions in parallel with `asyncio.gather()`

---

## 5. Fulltext Search Query Building (RedisSearch Syntax)

**Location:** `driver/falkordb_driver.py` (lines 235-308)

### Sanitization

```python
def sanitize(self, query: str) -> str:
    """
    Replace FalkorDB special characters with whitespace.
    Based on FalkorDB tokenization rules: ,.<>{}[]"':;!@#$%^&*()-+=~
    """
    separator_map = str.maketrans({
        ',': ' ', '.': ' ', '<': ' ', '>': ' ', '{': ' ', '}': ' ',
        '[': ' ', ']': ' ', '"': ' ', "'": ' ', ':': ' ', ';': ' ',
        '!': ' ', '@': ' ', '#': ' ', '$': ' ', '%': ' ', '^': ' ',
        '&': ' ', '*': ' ', '(': ' ', ')': ' ', '-': ' ', '+': ' ',
        '=': ' ', '~': ' ', '?': ' ',
    })
    sanitized = query.translate(separator_map)
    # Clean up multiple spaces
    sanitized = ' '.join(sanitized.split())
    return sanitized
```

### Fulltext Query Builder

```python
def build_fulltext_query(
    self, query: str, group_ids: list[str] | None = None, max_query_length: int = 128
) -> str:
    """
    Build a fulltext query string for FalkorDB using RedisSearch syntax.
    FalkorDB uses RedisSearch-like syntax where:
    - Field queries use @ prefix: @field:value
    - Multiple values for same field: (@field:value1|value2)
    - Text search doesn't need @ prefix for content fields
    - AND is implicit with space: (@group_id:value) (text)
    - OR uses pipe within parentheses: (@group_id:value1|value2)
    """
    if group_ids is None or len(group_ids) == 0:
        group_filter = ''
    else:
        group_values = '|'.join(group_ids)
        group_filter = f'(@group_id:{group_values})'

    sanitized_query = self.sanitize(query)

    # Remove stopwords from the sanitized query
    query_words = sanitized_query.split()
    filtered_words = [word for word in query_words if word.lower() not in STOPWORDS]
    sanitized_query = ' | '.join(filtered_words)

    # If the query is too long return no query
    if len(sanitized_query.split(' ')) + len(group_ids or '') >= max_query_length:
        return ''

    full_query = group_filter + ' (' + sanitized_query + ')'

    return full_query
```

**RedisSearch Syntax Patterns:**
1. **Field Filters:** `@group_id:value` - searches specific field
2. **OR within Field:** `@group_id:(value1|value2)` - multiple values
3. **Implicit AND:** Space between terms: `(@group_id:xyz) (search terms)`
4. **Token-level OR:** Words joined with `|` for OR matching
5. **Stopword Filtering:** Removes common words before building query
6. **Query Length Limits:** Prevents overly complex queries (max 128 tokens)

**Example Query Output:**
```
Input: query="John works at Google", group_ids=["session1", "session2"]
Output: "(@group_id:session1|session2) (John | works | Google)"
```

### Query Execution Helpers

**Location:** `graph_queries.py` (lines 130-162)

```python
def get_nodes_query(name: str, query: str, limit: int, provider: GraphProvider) -> str:
    if provider == GraphProvider.FALKORDB:
        label = NEO4J_TO_FALKORDB_MAPPING[name]
        return f"CALL db.idx.fulltext.queryNodes('{label}', {query})"
    # ...

def get_relationships_query(name: str, limit: int, provider: GraphProvider) -> str:
    if provider == GraphProvider.FALKORDB:
        label = NEO4J_TO_FALKORDB_MAPPING[name]
        return f"CALL db.idx.fulltext.queryRelationships('{label}', $query)"
    # ...
```

**Index Name Mapping:**
```python
NEO4J_TO_FALKORDB_MAPPING = {
    'node_name_and_summary': 'Entity',
    'community_name': 'Community',
    'episode_content': 'Episodic',
    'edge_name_and_fact': 'RELATES_TO',
}
```

---

## 6. Database-Specific Query Patterns

### Vector Similarity

**Location:** `graph_queries.py` (lines 142-150)

```python
def get_vector_cosine_func_query(vec1, vec2, provider: GraphProvider) -> str:
    if provider == GraphProvider.FALKORDB:
        # FalkorDB uses a different syntax for regular cosine similarity
        # Neo4j uses normalized cosine similarity
        return f'(2 - vec.cosineDistance({vec1}, vecf32({vec2})))/2'

    # Neo4j default:
    return f'vector.similarity.cosine({vec1}, {vec2})'
```

**Key Differences:**
- **Function Name:** `vec.cosineDistance` vs `vector.similarity.cosine`
- **Type Casting:** FalkorDB requires `vecf32()` wrapper for vector parameters
- **Normalization:** Formula converts distance to similarity: `(2 - distance) / 2`
- **Return Range:** Both return values in [0, 1] despite different internal representations

---

## 7. Node and Edge Save Query Generation

### Episode Nodes

**Location:** `models/nodes/node_db_queries.py` (lines 45-58)

```python
def get_episode_node_save_query(provider: GraphProvider) -> str:
    match provider:
        case GraphProvider.FALKORDB:
            return """
                MERGE (n:Episodic {uuid: $uuid})
                SET n = {uuid: $uuid, name: $name, group_id: $group_id,
                        source_description: $source_description, source: $source,
                        content: $content, entity_edges: $entity_edges,
                        created_at: $created_at, valid_at: $valid_at}
                RETURN n.uuid AS uuid
            """
        case _:  # Neo4j - same query
```

**FalkorDB uses same query as Neo4j for simple cases.**

### Bulk Episode Nodes

**Location:** `models/nodes/node_db_queries.py` (lines 86-93)

```python
case GraphProvider.FALKORDB:
    return """
        UNWIND $episodes AS episode
        MERGE (n:Episodic {uuid: episode.uuid})
        SET n = {uuid: episode.uuid, name: episode.name, group_id: episode.group_id,
                source_description: episode.source_description, source: episode.source,
                content: episode.content, entity_edges: episode.entity_edges,
                created_at: episode.created_at, valid_at: episode.valid_at}
        RETURN n.uuid AS uuid
    """
```

### Entity Nodes (with Dynamic Labels)

**Location:** `models/nodes/node_db_queries.py` (lines 129-138)

```python
def get_entity_node_save_query(provider: GraphProvider, labels: str, has_aoss: bool = False) -> str:
    match provider:
        case GraphProvider.FALKORDB:
            return f"""
                MERGE (n:Entity {{uuid: $entity_data.uuid}})
                SET n:{labels}
                SET n = $entity_data
                SET n.name_embedding = vecf32($entity_data.name_embedding)
                RETURN n.uuid AS uuid
            """
```

**Key Features:**
1. **Dynamic Label Assignment:** `SET n:{labels}` adds labels after MERGE
2. **Vector Type Casting:** `vecf32()` wrapper for embedding fields
3. **Map Assignment:** `SET n = $entity_data` for bulk property assignment

### Bulk Entity Nodes (Label Set Issue)

**Location:** `models/nodes/node_db_queries.py` (lines 186-205)

```python
case GraphProvider.FALKORDB:
    queries = []
    for node in nodes:
        for label in node['labels']:
            queries.append(
                (
                    f"""
                    UNWIND $nodes AS node
                    MERGE (n:Entity {{uuid: node.uuid}})
                    SET n:{label}
                    SET n = node
                    WITH n, node
                    SET n.name_embedding = vecf32(node.name_embedding)
                    RETURN n.uuid AS uuid
                    """,
                    {'nodes': [node]},
                )
            )
    return queries
```

**Critical Limitation:**
- **No Label Set Parameters:** FalkorDB doesn't support `SET n:$(labels)` syntax
- **Workaround:** Returns list of (query, params) tuples - one per label
- **Session Handling:** `FalkorDriverSession.run()` accepts query lists for this case

### Entity Edges

**Location:** `models/edges/edge_db_queries.py` (lines 63-73)

```python
case GraphProvider.FALKORDB:
    return """
        MATCH (source:Entity {uuid: $edge_data.source_uuid})
        MATCH (target:Entity {uuid: $edge_data.target_uuid})
        MERGE (source)-[e:RELATES_TO {uuid: $edge_data.uuid}]->(target)
        SET e = $edge_data
        SET e.fact_embedding = vecf32($edge_data.fact_embedding)
        RETURN e.uuid AS uuid
    """
```

### Bulk Entity Edges

**Location:** `models/edges/edge_db_queries.py` (lines 126-136)

```python
case GraphProvider.FALKORDB:
    return """
        UNWIND $entity_edges AS edge
        MATCH (source:Entity {uuid: edge.source_node_uuid})
        MATCH (target:Entity {uuid: edge.target_node_uuid})
        MERGE (source)-[r:RELATES_TO {uuid: edge.uuid}]->(target)
        SET r = {uuid: edge.uuid, name: edge.name, group_id: edge.group_id,
                fact: edge.fact, episodes: edge.episodes,
                created_at: edge.created_at, expired_at: edge.expired_at,
                valid_at: edge.valid_at, invalid_at: edge.invalid_at,
                fact_embedding: vecf32(edge.fact_embedding)}
        WITH r, edge
        RETURN edge.uuid AS uuid
    """
```

**Embedding Handling Pattern:**
- All embedding fields use `vecf32()` wrapper
- Applied in `SET` clause, not during MERGE
- Consistent across nodes and edges

### Community Edges

**Location:** `models/edges/edge_db_queries.py` (lines 225-234)

```python
case GraphProvider.FALKORDB:
    return """
        MATCH (community:Community {uuid: $community_uuid})
        MATCH (node {uuid: $entity_uuid})
        MERGE (community)-[e:HAS_MEMBER {uuid: $uuid}]->(node)
        SET e = {uuid: $uuid, group_id: $group_id, created_at: $created_at}
        RETURN e.uuid AS uuid
    """
```

**Note:** FalkorDB uses untyped node match (`MATCH (node {uuid: ...})`) instead of union types like Neo4j 5+ (`MATCH (node:Entity | Community ...)`).

---

## 8. Transaction Handling

**FalkorDB has no explicit transaction support in this integration.**

### Session execute_write

**Location:** `driver/falkordb_driver.py` (lines 94-96)

```python
async def execute_write(self, func, *args, **kwargs):
    # Directly await the provided async function with `self` as the transaction/session
    return await func(self, *args, **kwargs)
```

**Implications:**
- **No Rollback:** Failed operations cannot be rolled back
- **Immediate Writes:** Each `graph.query()` commits immediately
- **Atomicity at Query Level:** Only individual Cypher queries are atomic
- **No Multi-Statement Transactions:** Cannot group multiple queries in a transaction

**Comparison with Neo4j:**
```python
# Neo4j supports transactions:
async def my_write_operation(tx):
    await tx.run(query1)
    await tx.run(query2)
    # Commits or rolls back together

# FalkorDB - each runs independently:
async def my_write_operation(session):
    await session.run(query1)  # Commits immediately
    await session.run(query2)  # Commits immediately
```

---

## 9. Critical Differences from Neo4j for Custom Pipelines

### Summary Table

| Feature | Neo4j | FalkorDB | Impact |
|---------|-------|----------|--------|
| **Datetime Support** | Native | ISO strings only | Must convert all datetimes |
| **Transactions** | Full ACID | Query-level only | No multi-query rollback |
| **Vector Types** | `vector.similarity.cosine()` | `vecf32()` + `vec.cosineDistance()` | Different syntax + normalization |
| **Label Sets** | `SET n:$(labels)` | Not supported | Requires query per label |
| **Index Syntax** | `IF NOT EXISTS` supported | Not supported | Must handle "already exists" errors |
| **Fulltext Search** | Neo4j fulltext | RedisSearch | Different query syntax (`@field:value`) |
| **Result Format** | Dict-based records | List of lists | Driver converts to dicts |
| **Connection Protocol** | Bolt (7687) | Redis (6379) | Different ports |
| **Session Cleanup** | Required | Not required | Simpler session management |
| **Stopwords** | Built-in | Explicit list | Must define custom stopwords |

### Query Compatibility

**Works the Same:**
- Basic MATCH/MERGE/CREATE operations
- WHERE clauses
- RETURN projections
- ORDER BY / LIMIT
- UNWIND operations
- Property assignments with `=`

**Requires Adaptation:**
- Embedding operations (vector functions)
- Dynamic label assignment (bulk operations)
- Fulltext search queries
- Datetime parameters
- Index creation statements

### Custom Pipeline Considerations

1. **Datetime Handling:**
   ```python
   # Always convert before passing to FalkorDB
   from graphiti_core.utils.datetime_utils import convert_datetimes_to_strings

   params = convert_datetimes_to_strings({
       'created_at': datetime.now(),
       'valid_at': some_datetime
   })
   await driver.execute_query(query, **params)
   ```

2. **Vector Embeddings:**
   ```cypher
   -- FalkorDB pattern
   SET n.embedding = vecf32($embedding_param)

   -- For similarity search
   WITH (2 - vec.cosineDistance(n.embedding, vecf32($search_vector)))/2 AS similarity
   ```

3. **Dynamic Labels (Bulk Operations):**
   ```python
   # For FalkorDB, generate query per label:
   queries = []
   for node in nodes:
       for label in node['labels']:
           queries.append((
               f"MERGE (n:Entity {{uuid: $uuid}}) SET n:{label} SET n = $data",
               {'uuid': node['uuid'], 'data': node}
           ))

   # Execute via session.run() which accepts lists
   async with driver.session() as session:
       await session.run(queries)
   ```

4. **Error Handling:**
   ```python
   try:
       await driver.execute_query(index_creation_query)
   except Exception as e:
       if 'already indexed' in str(e):
           # FalkorDB index already exists - safe to continue
           pass
       else:
           raise
   ```

5. **Fulltext Search:**
   ```python
   # Build RedisSearch query
   query = driver.build_fulltext_query(
       query="search terms here",
       group_ids=["session1", "session2"]
   )
   # Result: "(@group_id:session1|session2) (search | terms | here)"

   # Execute
   cypher = f"CALL db.idx.fulltext.queryNodes('Entity', '{query}')"
   results = await driver.execute_query(cypher)
   ```

---

## 10. Best Practices for FalkorDB with Graphiti

### 1. Initialize with Proper Configuration

```python
from graphiti_core.driver.falkordb_driver import FalkorDriver

# For local development
driver = FalkorDriver(
    host='localhost',
    port=6379,
    database='my_knowledge_graph'
)

# For production with authentication
driver = FalkorDriver(
    host='falkordb.example.com',
    port=6379,
    username='app_user',
    password='secure_password',
    database='production_graph'
)
```

### 2. Always Convert Datetimes

```python
from graphiti_core.utils.datetime_utils import convert_datetimes_to_strings
from datetime import datetime

# Bad
await driver.execute_query(
    "CREATE (n:Event {created_at: $created_at})",
    created_at=datetime.now()  # Will fail!
)

# Good
params = convert_datetimes_to_strings({
    'created_at': datetime.now()
})
await driver.execute_query(
    "CREATE (n:Event {created_at: $created_at})",
    **params
)
```

### 3. Handle Index Creation Idempotently

```python
async def ensure_indices(driver):
    from graphiti_core.utils.maintenance.graph_data_operations import build_indices_and_constraints

    # This handles "already indexed" errors automatically
    await build_indices_and_constraints(driver, delete_existing=False)
```

### 4. Use Bulk Operations Efficiently

```python
# For entity nodes with multiple labels
bulk_nodes = [
    {'uuid': 'uuid1', 'name': 'Alice', 'labels': ['Person', 'Employee']},
    {'uuid': 'uuid2', 'name': 'Bob', 'labels': ['Person', 'Manager']},
]

# Use graphiti's built-in bulk query generator
from graphiti_core.models.nodes.node_db_queries import get_entity_node_save_bulk_query

queries = get_entity_node_save_bulk_query(
    provider=GraphProvider.FALKORDB,
    nodes=bulk_nodes
)

# Execute via session
async with driver.session() as session:
    await session.run(queries)
```

### 5. Optimize Fulltext Searches

```python
# Build query with stopword filtering
search_query = driver.build_fulltext_query(
    query="the important search terms",  # "the" will be filtered
    group_ids=group_ids,
    max_query_length=128  # Prevent overly complex queries
)

if not search_query:
    # Query too complex - handle gracefully
    results = []
else:
    results = await driver.execute_query(
        f"CALL db.idx.fulltext.queryNodes('Entity', '{search_query}')"
    )
```

### 6. Properly Handle Vector Embeddings

```python
# When saving nodes with embeddings
embedding = [0.1, 0.2, 0.3, ...]  # Your embedding vector

await driver.execute_query(
    """
    MERGE (n:Entity {uuid: $uuid})
    SET n.name = $name
    SET n.name_embedding = vecf32($embedding)
    """,
    uuid=node_uuid,
    name=node_name,
    embedding=embedding  # Pass as list - vecf32() converts in Cypher
)

# When searching with embeddings
results, _, _ = await driver.execute_query(
    f"""
    MATCH (n:Entity)
    WITH n, (2 - vec.cosineDistance(n.name_embedding, vecf32($search_vector)))/2 AS similarity
    WHERE similarity > $threshold
    RETURN n, similarity
    ORDER BY similarity DESC
    LIMIT $limit
    """,
    search_vector=search_embedding,
    threshold=0.7,
    limit=10
)
```

### 7. Leverage Multi-Tenancy

```python
# Create separate graphs for different users/sessions
user1_driver = driver.clone(database='user1_graph')
user2_driver = driver.clone(database='user2_graph')

# Each operates on isolated graph
await user1_driver.execute_query("CREATE (n:Data {value: 'user1'})")
await user2_driver.execute_query("CREATE (n:Data {value: 'user2'})")

# Cleanup driver but retain connection pool
await user1_driver.close()  # Doesn't close underlying FalkorDB connection
```

### 8. Handle Lack of Transactions

```python
# Design operations to be idempotent
# Bad - not atomic across queries:
async def create_related_nodes(driver):
    await driver.execute_query("CREATE (a:Node {id: 1})")
    # If this fails, first node is already created!
    await driver.execute_query("CREATE (b:Node {id: 2})")

# Good - single query for atomicity:
async def create_related_nodes(driver):
    await driver.execute_query(
        """
        CREATE (a:Node {id: 1})
        CREATE (b:Node {id: 2})
        CREATE (a)-[:RELATED]->(b)
        """
    )
```

### 9. Monitor Query Performance

```python
import time
import logging

logger = logging.getLogger(__name__)

async def execute_with_timing(driver, query, **params):
    start = time.time()
    try:
        result = await driver.execute_query(query, **params)
        elapsed = time.time() - start
        logger.info(f"Query completed in {elapsed:.3f}s")
        return result
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"Query failed after {elapsed:.3f}s: {e}")
        raise
```

### 10. Clean Shutdown

```python
async def cleanup(driver):
    # Close driver connection
    await driver.close()

    # For custom FalkorDB client, close it explicitly
    if hasattr(driver.client, 'aclose'):
        await driver.client.aclose()
```

---

## Additional Resources

**FalkorDB Documentation:**
- RedisSearch Query Syntax: https://redis.io/docs/latest/develop/ai/search-and-query/query/full-text/
- FalkorDB Procedures: https://docs.falkordb.com/

**Graphiti Integration:**
- Base driver interface: `graphiti_core/driver/driver.py`
- Query utilities: `graphiti_core/graph_queries.py`
- Example usage: Search `tests/` directory for FalkorDB-specific tests

**Related Research:**
- Neo4j comparison: See Neo4j driver implementation at `driver/neo4j_driver.py`
- Index management: `utils/maintenance/graph_data_operations.py`
