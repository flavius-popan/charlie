# Charlie - The Journaling Mind Mapper

Charlie is a journaling tool that ingests unstructured journal entries and uses [Graphiti](https://github.com/getzep/graphiti) to build a knowledge graph stored in a local Kuzu database. It provides a web interface for exploring your journal entries, entities, and relationships through an interactive graph explorer.

## Quick Start

### 1. Install Dependencies

```bash
uv pip install -e .
```

### 2. Load Your Journal Entries

```bash
python load_journals.py load --skip-verification
```

This loads entries from `raw_data/Journal.json` into the knowledge graph.

### 3. Build Communities (Optional)

```bash
python load_journals.py build-communities
```

This groups related entities into thematic clusters using the Leiden algorithm.

### 4. Start the Web Interface

```bash
python run.py
```

Open your browser to: http://localhost:8080

### 5. Run Tests

```bash
pytest tests/
```

## Project Structure

```
charlie/
├── app/                    # FastAPI web application
│   ├── main.py            # Application entry point
│   ├── routes/            # HTTP route handlers
│   ├── services/          # Business logic & database queries
│   └── models/            # Pydantic data models
├── templates/             # Jinja2 HTML templates
├── static/                # CSS and JavaScript
├── brain/                 # Kuzu database storage
├── raw_data/              # Source journal files
├── notes/                 # Architecture documentation
├── tests/                 # Test suite
├── load_journals.py       # Data import script
└── run.py                 # Development server launcher
```

## Documentation

- **Agent Guidelines**: [CLAUDE.md](CLAUDE.md)

## Technology Stack

- **Backend**: FastAPI, Python 3.13+
- **Database**: Kuzu (embedded graph database)
- **Knowledge Graph**: Graphiti Core
- **Templates**: Jinja2
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Package Manager**: uv

## Development

### Running the Server

```bash
# Development mode with auto-reload
python run.py

# Or directly with uvicorn
uvicorn app.main:app --reload
```

### Keyboard Shortcuts

- `h` - Return to home page (from anywhere)
