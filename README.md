# Charlie

The most over-engineered journaling app in the world.

## Quick Start

**System Requirements:**
- Linux (x86_64)
- macOS 15+ (Sequoia) on Apple Silicon (M-series)

```bash
./setup.sh
```

The setup script handles everything:
- Installs `uv` package manager (if needed)
- Installs Python 3.12+ (if needed)
- Creates virtual environment and installs dependencies
- Downloads the language model (~4.5GB, cached)
- Initializes the database
- Offers to import existing journals

## Importing Journals

Setup prompts you to import, or run importers manually:

```bash
# Day One export
python importers/dayone.py ~/Downloads/DayOneExport.zip

# Text files directory
python importers/files.py ~/journals/ --recursive

# Dry run (preview without importing)
python importers/files.py ~/journals/ --dry-run
```

See [importers/IMPORTING.md](importers/IMPORTING.md) for all options.

## Running

```bash
./charlie.sh
```

Or manually activate the virtual environment:

```bash
source .venv/bin/activate
python charlie.py
```
