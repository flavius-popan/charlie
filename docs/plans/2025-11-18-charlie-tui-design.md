# Charlie TUI Design

**Date:** 2025-11-18
**Status:** Approved

## Overview

A minimal Textual-based TUI application for managing journal entries stored in FalkorDB Lite. The application provides CRUD operations on journal entries with markdown support.

## Architecture

### File Structure
```
charlie.py                  # Main TUI application
logs/
  .gitkeep                 # Ensure logs directory exists
  *.log                    # Excluded in .gitignore
```

### Application Lifecycle
- `on_mount()` - Initialize database with `await ensure_database_ready()`, load initial episode list
- `on_unmount()` - No explicit cleanup needed (database handles shutdown via atexit)

### Three Screen Classes
1. **HomeScreen** - List of journal entries with ASCII cat empty state
2. **ViewScreen** - Display entry using Textual's Markdown widget
3. **EditScreen** - TextArea for editing markdown content

### Navigation Flow
```
HomeScreen (list)
  ├─> [n] → EditScreen (new entry) → auto-saves → back to HomeScreen
  ├─> [Enter/click] → ViewScreen (read-only)
  │     └─> [e] → EditScreen (edit mode) → auto-saves → back to HomeScreen
  │     └─> [q/Esc] → back to HomeScreen
  └─> [d] → delete entry → refresh HomeScreen
```

## Screen Designs

### HomeScreen (List View)

**Layout:**
```
┌─ Charlie ─────────────────────────────────┐
│                                           │
│ > 2024-01-15 - First markdown entry       │
│   2024-01-14 - Meeting notes              │
│   2024-01-13 - Daily reflection           │
│                                           │
├───────────────────────────────────────────┤
│ [n] new  [e] edit  [d] delete  [q] quit   │
└───────────────────────────────────────────┘
```

**Empty State:**
```
┌─ Charlie ─────────────────────────────────┐
│                                           │
│         /\_/\                             │
│        ( o.o )                            │
│         > ^ <                             │
│                                           │
│    No entries yet!                        │
│    Press 'n' to create your first entry   │
│                                           │
├───────────────────────────────────────────┤
│ [n] new  [q] quit                         │
└───────────────────────────────────────────┘
```

**Features:**
- Display entries in reverse chronological order (newest first)
- Show date (from `valid_at`) and title
- Title extracted from `# Header` in markdown or content preview (~50 chars)
- Muted footer with keyboard shortcuts

**Keyboard Shortcuts:**
- `j`/`k` or `↓`/`↑` - Navigate list
- `Enter` - View selected entry
- `e` - Edit selected entry
- `d` - Delete selected entry immediately
- `n` - Create new entry
- `q`/`Esc` - Quit application

**Mouse Support:**
- Click to select entry
- Double-click to view
- Footer buttons for actions

### ViewScreen (Read-Only Display)

**Layout:**
```
┌─ Charlie ─────────────────────────────────┐
│                                           │
│  # My First Entry                         │
│                                           │
│  This is some **markdown** content with   │
│  formatting rendered properly.            │
│                                           │
├───────────────────────────────────────────┤
│ [e] edit  [q] back                        │
└───────────────────────────────────────────┘
```

**Features:**
- Scrollable Markdown widget (Textual built-in)
- Read-only display of episode content
- Muted footer with shortcuts

**Keyboard Shortcuts:**
- `e` - Switch to EditScreen
- `q`/`Esc` - Back to HomeScreen
- `↓`/`↑` or scroll - Scroll content

### EditScreen (Create/Edit Mode)

**Layout:**
```
┌─ Charlie ─────────────────────────────────┐
│                                           │
│  # My Entry Title                         │
│                                           │
│  Write your markdown content here...      │
│  █                                        │
│                                           │
├───────────────────────────────────────────┤
│ [Esc] save & return                       │
└───────────────────────────────────────────┘
```

**Features:**
- Full-height TextArea for markdown editing
- Auto-save on exit
- Async save to database when returning to previous screen
- Muted footer with clear help text

**Keyboard Shortcuts:**
- `Esc`/`q` - Save and return to HomeScreen
- Normal text editing (cursor, selection, etc.)

**Save Behavior:**
- On exit, fire async save operation
- Extract `# Title` from content if present
- Update episode `name` metadata in database if title found
- For new entries: create with `add_journal_entry()`, then update title if found
- For existing entries: update content and title with `update_episode()`

## Data Management

### Title Extraction Logic

```python
def extract_title(content: str) -> str | None:
    """Extract first # header from markdown content."""
    lines = content.split('\n')
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('# '):
            return stripped[2:].strip()
    return None

def get_display_title(episode: dict) -> str:
    """Get title for display in list view."""
    content = episode['content']

    # Try markdown header first
    title = extract_title(content)
    if title:
        return title[:50]  # Truncate to 50 chars

    # Fall back to content preview
    preview = content.replace('\n', ' ').strip()
    return preview[:50] if preview else episode['name']
```

### Database Operations

**List entries:**
```python
episodes = await get_all_episodes()  # Returns list[dict], newest first
```

**Get single entry:**
```python
episode = await get_episode(uuid)  # Returns dict or None
```

**Create entry:**
```python
uuid = await add_journal_entry(content=content)
if title:
    await update_episode(uuid, name=title)
```

**Update entry:**
```python
await update_episode(uuid, content=content, name=title)  # name optional
```

**Delete entry:**
```python
await delete_episode(uuid)
```

## Error Handling & Logging

### Logging Setup
- Log to `logs/charlie.log`
- Exclude `logs/*.log` in `.gitignore`
- Include `logs/.gitkeep` to ensure directory exists
- Format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

### Error Strategy
- Database initialization failure → exit with error (fatal)
- Save operation failure → show error, don't exit edit screen, re-raise
- Load operation failure → log error, show empty state, continue
- Delete operation failure → log error, show error message, keep entry
- Fail fast where appropriate, log all errors with stack traces

### User Feedback
- Show error messages in UI (Textual notifications)
- No success toasts - keep it snappy
- Immediate actions, no confirmation dialogs

## Design Constraints

- Single journal only (default journal)
- No journal parameter passing (use function defaults)
- Multi-journal support is future work
- No undo functionality for now
- Minimal and succinct implementation
- Follow Textual best practices
- Keep it snappy - no unnecessary UI elements

## Implementation Approach

1. Set up basic app skeleton with three screens
2. Implement HomeScreen with hardcoded data first
3. Wire up database calls
4. Add ViewScreen with Markdown rendering
5. Add EditScreen with save logic
6. Implement title extraction
7. Add error handling and logging
8. Polish keyboard/mouse interactions

## Dependencies

- `textual` - TUI framework (already installed)
- `backend.database` - Existing database module
- Python standard library: `logging`, `pathlib`
