# Database Module Organization

## What Goes Where

**utils.py** - Constants, validation, type protocols

**lifecycle.py** - Process management (init, shutdown, TCP server, graph cache)

**driver.py** - Query execution (FalkorLiteDriver, index building, fulltext search)

**persistence.py** - Write operations (episodes, SELF entity, initialization)

**queries.py** - Read operations (time-range queries, entity timelines, search)

## Adding New Code

- Query operations → `queries.py`
- DB reset/maintenance → `persistence.py`
- New utilities → `utils.py`
- Connection pooling → `lifecycle.py`
