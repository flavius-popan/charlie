# Charlie TUI

A minimal terminal user interface for managing journal entries.

## Installation

Requires Python 3.13+ and dependencies from `pyproject.toml`.

```bash
uv sync
```

## Usage

```bash
python charlie.py
```

## Keyboard Shortcuts

**Home Screen (List View):**
- `j/k` or `↑/↓` - Navigate entries
- `Enter` - View selected entry
- `n` - Create new entry
- `e` - Edit selected entry
- `d` - Delete selected entry
- `q` - Quit

**View Screen:**
- `e` - Edit entry
- `q` or `Esc` - Back to list

**Edit Screen:**
- `Esc` or `q` - Save and return

## Features

- Create, read, update, delete journal entries
- Markdown support with live rendering
- Auto-save on exit from editor
- Title extraction from `# Header` in markdown
- Chronological ordering (newest first)
- Cute ASCII cat for empty state

## Logging

Logs are written to `logs/charlie.log` for debugging.
