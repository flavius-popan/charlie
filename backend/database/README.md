# Database Module Organization

## What Goes Where

**utils.py** - Constants, validation, type protocols

**lifecycle.py** - Process management (init, shutdown, TCP server, graph cache)

**driver.py** - Query execution (FalkorLiteDriver, index building, fulltext search)

**persistence.py** - Write operations (create, update, delete episodes; author entity "I"; initialization)

**queries.py** - Read operations (get episode by UUID; future: time-range queries, entity timelines, search)

**redis_ops.py** - Global Redis operations for app metadata, stats, and housekeeping (non-graph data)

## Adding New Code

- Read operations → `queries.py`
- Write/update/delete operations → `persistence.py`
- Type conversion (FalkorDB ↔ Python) → `utils.py`
- DB reset/maintenance → `persistence.py`
- Connection pooling → `lifecycle.py`
- Global metadata/stats → `redis_ops.py`

## Notes

- Uses FalkorDB native arrays for list fields (entity_edges, labels)
- CRUD operations leverage graphiti-core model methods (EpisodicNode) where possible
