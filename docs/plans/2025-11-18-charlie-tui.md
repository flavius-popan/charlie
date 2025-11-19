# Charlie TUI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a minimal Textual TUI application for CRUD operations on journal entries with markdown support.

**Architecture:** Three-screen architecture (HomeScreen for list, ViewScreen for read-only display, EditScreen for creating/editing). Auto-save on exit, title extraction from markdown headers, immediate actions with no confirmations.

**Tech Stack:** Textual (TUI framework), backend.database (FalkorDB async API), Python logging

---

## Task 1: Set Up Logging Infrastructure

**Files:**
- Create: `charlie.py` (initial setup)
- Verify: `logs/.gitkeep` exists
- Verify: `.gitignore` includes `logs/*.log`

**Step 1: Write basic app skeleton with logging**

Create `charlie.py`:

```python
import logging
from pathlib import Path
from textual.app import App

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'charlie.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('charlie')


class CharlieApp(App):
    """A minimal journal TUI application."""

    def on_mount(self):
        logger.info("Charlie TUI started")
        self.exit()


if __name__ == "__main__":
    app = CharlieApp()
    app.run()
```

**Step 2: Verify logging works**

Run: `python charlie.py`

Expected:
- App starts and exits immediately
- `logs/charlie.log` created with startup message
- Console shows log output

**Step 3: Commit**

```bash
git add charlie.py
git commit -m "feat: add logging infrastructure"
```

---

## Task 2: Create Title Extraction Utilities

**Files:**
- Modify: `charlie.py` (add helper functions)

**Step 1: Write tests for title extraction**

Add to `charlie.py` (after imports, before classes):

```python
def extract_title(content: str) -> str | None:
    """Extract first # header from markdown content.

    Args:
        content: Markdown text

    Returns:
        Title text without # prefix, or None if no header found

    Examples:
        >>> extract_title("# Hello World\\nContent")
        'Hello World'
        >>> extract_title("No header here")

        >>> extract_title("  # Trimmed  \\n")
        'Trimmed'
    """
    lines = content.split('\n')
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('# '):
            return stripped[2:].strip()
    return None


def get_display_title(episode: dict, max_chars: int = 50) -> str:
    """Get title for display in list view.

    Args:
        episode: Episode dict with 'content' and 'name' fields
        max_chars: Maximum characters to display

    Returns:
        Title string, truncated to max_chars

    Examples:
        >>> get_display_title({'content': '# My Title\\nBody', 'name': 'default'})
        'My Title'
        >>> get_display_title({'content': 'Plain text', 'name': 'default'})
        'Plain text'
    """
    content = episode.get('content', '')

    title = extract_title(content)
    if title:
        return title[:max_chars]

    preview = content.replace('\n', ' ').strip()
    return preview[:max_chars] if preview else episode.get('name', 'Untitled')
```

**Step 2: Test manually with doctests**

Run: `python -m doctest charlie.py -v`

Expected: All doctests pass

**Step 3: Commit**

```bash
git add charlie.py
git commit -m "feat: add title extraction utilities"
```

---

## Task 3: Create HomeScreen with Empty State

**Files:**
- Modify: `charlie.py` (add HomeScreen class)

**Step 1: Import Textual widgets**

Add to imports in `charlie.py`:

```python
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, ListView, ListItem, Label
from textual.binding import Binding
```

**Step 2: Create HomeScreen with ASCII cat**

Add before `CharlieApp` class:

```python
EMPTY_STATE_CAT = r"""
         /\_/\
        ( o.o )
         > ^ <

    No entries yet!
    Press 'n' to create your first entry
"""


class HomeScreen(Screen):
    """Main screen showing list of journal entries."""

    BINDINGS = [
        Binding("n", "new_entry", "New", show=True),
        Binding("e", "edit_entry", "Edit", show=True),
        Binding("d", "delete_entry", "Delete", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    def __init__(self):
        super().__init__()
        self.episodes = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        if not self.episodes:
            yield Static(EMPTY_STATE_CAT, id="empty-state")
        else:
            yield ListView(id="episodes-list")
        yield Footer()

    def action_new_entry(self):
        logger.info("New entry action triggered")
        self.app.bell()

    def action_edit_entry(self):
        logger.info("Edit entry action triggered")
        self.app.bell()

    def action_delete_entry(self):
        logger.info("Delete entry action triggered")
        self.app.bell()

    def action_quit(self):
        self.app.exit()
```

**Step 3: Update CharlieApp to use HomeScreen**

Replace `CharlieApp` class:

```python
class CharlieApp(App):
    """A minimal journal TUI application."""

    TITLE = "Charlie"

    def on_mount(self):
        logger.info("Charlie TUI started")
        self.push_screen(HomeScreen())


if __name__ == "__main__":
    app = CharlieApp()
    app.run()
```

**Step 4: Add basic CSS styling**

Add after `CharlieApp` class definition (before `if __name__`):

```python
CharlieApp.CSS = """
Screen {
    background: $surface;
}

#empty-state {
    width: 100%;
    height: 100%;
    content-align: center middle;
    color: $text-muted;
}

Footer {
    background: $panel;
}

Footer .footer--key {
    color: $text-muted;
}
"""
```

**Step 5: Test the UI**

Run: `python charlie.py`

Expected:
- TUI shows with "Charlie" header
- ASCII cat displayed in center
- Footer shows: n, e, d, q keys (muted colors)
- Pressing 'q' exits
- Pressing 'n', 'e', 'd' triggers bell sound and logs action

**Step 6: Commit**

```bash
git add charlie.py
git commit -m "feat: add HomeScreen with empty state"
```

---

## Task 4: Add Database Integration to HomeScreen

**Files:**
- Modify: `charlie.py` (add database loading)

**Step 1: Import database functions**

Add to imports:

```python
from backend.database import ensure_database_ready, get_all_episodes
```

**Step 2: Update HomeScreen to load episodes**

Replace `HomeScreen` class:

```python
class HomeScreen(Screen):
    """Main screen showing list of journal entries."""

    BINDINGS = [
        Binding("n", "new_entry", "New", show=True),
        Binding("e", "edit_entry", "Edit", show=True),
        Binding("d", "delete_entry", "Delete", show=True),
        Binding("q", "quit", "Quit", show=True),
        Binding("enter", "view_entry", "View", show=False),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
    ]

    def __init__(self):
        super().__init__()
        self.episodes = []
        self.selected_index = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        if not self.episodes:
            yield Static(EMPTY_STATE_CAT, id="empty-state")
        else:
            list_view = ListView(id="episodes-list")
            for episode in self.episodes:
                date_str = episode['valid_at'].strftime('%Y-%m-%d')
                title = get_display_title(episode)
                list_view.append(ListItem(Label(f"{date_str} - {title}")))
            yield list_view
        yield Footer()

    async def on_mount(self):
        await self.load_episodes()

    async def load_episodes(self):
        try:
            logger.info("Loading episodes from database")
            self.episodes = await get_all_episodes()
            logger.info(f"Loaded {len(self.episodes)} episodes")
            await self.recompose()
        except Exception as e:
            logger.error(f"Failed to load episodes: {e}", exc_info=True)
            self.notify("Error loading episodes", severity="error")

    def action_new_entry(self):
        logger.info("New entry action triggered")
        self.app.bell()

    def action_edit_entry(self):
        list_view = self.query_one("#episodes-list", ListView)
        if list_view.index is not None and self.episodes:
            episode = self.episodes[list_view.index]
            logger.info(f"Edit entry: {episode['uuid']}")
            self.app.bell()

    def action_view_entry(self):
        list_view = self.query_one("#episodes-list", ListView)
        if list_view.index is not None and self.episodes:
            episode = self.episodes[list_view.index]
            logger.info(f"View entry: {episode['uuid']}")
            self.app.bell()

    def action_delete_entry(self):
        list_view = self.query_one("#episodes-list", ListView)
        if list_view.index is not None and self.episodes:
            episode = self.episodes[list_view.index]
            logger.info(f"Delete entry: {episode['uuid']}")
            self.app.bell()

    def action_quit(self):
        self.app.exit()

    def action_cursor_down(self):
        try:
            list_view = self.query_one("#episodes-list", ListView)
            list_view.action_cursor_down()
        except:
            pass

    def action_cursor_up(self):
        try:
            list_view = self.query_one("#episodes-list", ListView)
            list_view.action_cursor_up()
        except:
            pass
```

**Step 3: Update CharlieApp to initialize database**

Replace `CharlieApp` class:

```python
class CharlieApp(App):
    """A minimal journal TUI application."""

    TITLE = "Charlie"

    async def on_mount(self):
        try:
            logger.info("Initializing database")
            await ensure_database_ready()
            logger.info("Database ready")
            self.push_screen(HomeScreen())
        except Exception as e:
            logger.error(f"Fatal: Database initialization failed: {e}", exc_info=True)
            self.notify("Failed to initialize database", severity="error")
            self.exit(1)
```

**Step 4: Test with actual database**

Run: `python charlie.py`

Expected:
- App loads episodes from database
- If database has entries, they appear in list with dates and titles
- If empty, shows ASCII cat
- j/k or arrow keys navigate list
- Press 'q' to exit

**Step 5: Commit**

```bash
git add charlie.py
git commit -m "feat: integrate database loading in HomeScreen"
```

---

## Task 5: Create ViewScreen

**Files:**
- Modify: `charlie.py` (add ViewScreen class)

**Step 1: Import Markdown widget**

Add to imports:

```python
from textual.widgets import Header, Footer, Static, ListView, ListItem, Label, Markdown
```

**Step 2: Import get_episode function**

Update database imports:

```python
from backend.database import ensure_database_ready, get_all_episodes, get_episode
```

**Step 3: Create ViewScreen class**

Add before `CharlieApp` class:

```python
class ViewScreen(Screen):
    """Screen for viewing a journal entry in read-only mode."""

    BINDINGS = [
        Binding("e", "edit_entry", "Edit", show=True),
        Binding("q", "back", "Back", show=True),
        Binding("escape", "back", "Back", show=False),
    ]

    def __init__(self, episode_uuid: str):
        super().__init__()
        self.episode_uuid = episode_uuid
        self.episode = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Markdown("Loading...", id="content")
        yield Footer()

    async def on_mount(self):
        await self.load_episode()

    async def load_episode(self):
        try:
            logger.info(f"Loading episode: {self.episode_uuid}")
            self.episode = await get_episode(self.episode_uuid)

            if self.episode:
                markdown = self.query_one("#content", Markdown)
                await markdown.update(self.episode['content'])
            else:
                logger.error(f"Episode not found: {self.episode_uuid}")
                self.notify("Entry not found", severity="error")
                self.app.pop_screen()
        except Exception as e:
            logger.error(f"Failed to load episode: {e}", exc_info=True)
            self.notify("Error loading entry", severity="error")
            self.app.pop_screen()

    def action_edit_entry(self):
        logger.info(f"Edit entry: {self.episode_uuid}")
        self.app.bell()

    def action_back(self):
        self.app.pop_screen()
```

**Step 4: Wire up ViewScreen from HomeScreen**

Update `action_view_entry` in `HomeScreen`:

```python
def action_view_entry(self):
    try:
        list_view = self.query_one("#episodes-list", ListView)
        if list_view.index is not None and self.episodes:
            episode = self.episodes[list_view.index]
            logger.info(f"View entry: {episode['uuid']}")
            self.app.push_screen(ViewScreen(episode['uuid']))
    except Exception as e:
        logger.error(f"Failed to open view screen: {e}", exc_info=True)
```

**Step 5: Update CSS for markdown**

Update `CharlieApp.CSS`:

```python
CharlieApp.CSS = """
Screen {
    background: $surface;
}

#empty-state {
    width: 100%;
    height: 100%;
    content-align: center middle;
    color: $text-muted;
}

#content {
    padding: 1 2;
    height: 100%;
}

Footer {
    background: $panel;
}

Footer .footer--key {
    color: $text-muted;
}
"""
```

**Step 6: Test viewing entries**

Run: `python charlie.py`

Expected:
- Navigate to entry with j/k
- Press Enter to view entry
- Markdown content renders properly
- Press 'e' triggers bell (not implemented yet)
- Press 'q' or Esc returns to list

**Step 7: Commit**

```bash
git add charlie.py
git commit -m "feat: add ViewScreen for read-only display"
```

---

## Task 6: Create EditScreen with Save Logic

**Files:**
- Modify: `charlie.py` (add EditScreen class)

**Step 1: Import TextArea widget**

Add to imports:

```python
from textual.widgets import Header, Footer, Static, ListView, ListItem, Label, Markdown, TextArea
```

**Step 2: Import database write functions**

Update database imports:

```python
from backend.database import (
    ensure_database_ready,
    get_all_episodes,
    get_episode,
    update_episode,
)
from backend import add_journal_entry
```

**Step 3: Create EditScreen class**

Add before `CharlieApp` class:

```python
class EditScreen(Screen):
    """Screen for creating or editing a journal entry."""

    BINDINGS = [
        Binding("escape", "save_and_return", "Save & Return", show=True),
        Binding("q", "save_and_return", "Save & Return", show=False),
    ]

    def __init__(self, episode_uuid: str | None = None):
        super().__init__()
        self.episode_uuid = episode_uuid
        self.is_new_entry = episode_uuid is None
        self.episode = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield TextArea("", id="editor")
        yield Footer()

    async def on_mount(self):
        if not self.is_new_entry:
            await self.load_episode()

        editor = self.query_one("#editor", TextArea)
        editor.focus()

    async def load_episode(self):
        try:
            logger.info(f"Loading episode for editing: {self.episode_uuid}")
            self.episode = await get_episode(self.episode_uuid)

            if self.episode:
                editor = self.query_one("#editor", TextArea)
                editor.text = self.episode['content']
            else:
                logger.error(f"Episode not found: {self.episode_uuid}")
                self.notify("Entry not found", severity="error")
                self.app.pop_screen()
        except Exception as e:
            logger.error(f"Failed to load episode for editing: {e}", exc_info=True)
            self.notify("Error loading entry", severity="error")
            self.app.pop_screen()

    async def action_save_and_return(self):
        await self.save_entry()

    async def save_entry(self):
        try:
            editor = self.query_one("#editor", TextArea)
            content = editor.text

            if not content.strip():
                logger.warning("Attempted to save empty entry")
                self.notify("Entry is empty", severity="warning")
                return

            title = extract_title(content)

            if self.is_new_entry:
                logger.info("Creating new entry")
                uuid = await add_journal_entry(content=content)

                if title:
                    logger.info(f"Updating title: {title}")
                    await update_episode(uuid, name=title)

                logger.info(f"Created entry: {uuid}")
            else:
                logger.info(f"Updating entry: {self.episode_uuid}")

                if title:
                    await update_episode(self.episode_uuid, content=content, name=title)
                else:
                    await update_episode(self.episode_uuid, content=content)

                logger.info("Entry updated")

            self.app.pop_screen()

        except Exception as e:
            logger.error(f"Failed to save entry: {e}", exc_info=True)
            self.notify("Failed to save entry", severity="error")
            raise
```

**Step 4: Wire up EditScreen from HomeScreen and ViewScreen**

Update `action_new_entry` in `HomeScreen`:

```python
def action_new_entry(self):
    logger.info("Creating new entry")
    self.app.push_screen(EditScreen())
```

Update `action_edit_entry` in `HomeScreen`:

```python
def action_edit_entry(self):
    try:
        list_view = self.query_one("#episodes-list", ListView)
        if list_view.index is not None and self.episodes:
            episode = self.episodes[list_view.index]
            logger.info(f"Edit entry: {episode['uuid']}")
            self.app.push_screen(EditScreen(episode['uuid']))
    except Exception as e:
        logger.error(f"Failed to open edit screen: {e}", exc_info=True)
```

Update `action_edit_entry` in `ViewScreen`:

```python
def action_edit_entry(self):
    logger.info(f"Edit entry: {self.episode_uuid}")
    self.app.push_screen(EditScreen(self.episode_uuid))
```

**Step 5: Update CSS for TextArea**

Update `CharlieApp.CSS`:

```python
CharlieApp.CSS = """
Screen {
    background: $surface;
}

#empty-state {
    width: 100%;
    height: 100%;
    content-align: center middle;
    color: $text-muted;
}

#content {
    padding: 1 2;
    height: 100%;
}

#editor {
    height: 100%;
}

Footer {
    background: $panel;
}

Footer .footer--key {
    color: $text-muted;
}
"""
```

**Step 6: Test creating and editing entries**

Run: `python charlie.py`

Test sequence:
1. Press 'n' to create new entry
2. Type some markdown content with `# Title`
3. Press Esc to save
4. Verify entry appears in list with title
5. Select entry and press 'e' to edit
6. Modify content
7. Press Esc to save
8. View entry to verify changes

Expected: All operations work smoothly, auto-save on exit

**Step 7: Commit**

```bash
git add charlie.py
git commit -m "feat: add EditScreen with auto-save"
```

---

## Task 7: Implement Delete Functionality

**Files:**
- Modify: `charlie.py` (add delete operation)

**Step 1: Import delete_episode function**

Update database imports:

```python
from backend.database import (
    ensure_database_ready,
    get_all_episodes,
    get_episode,
    update_episode,
    delete_episode,
)
```

**Step 2: Update delete action in HomeScreen**

Replace `action_delete_entry` in `HomeScreen`:

```python
async def action_delete_entry(self):
    try:
        list_view = self.query_one("#episodes-list", ListView)
        if list_view.index is not None and self.episodes:
            episode = self.episodes[list_view.index]
            logger.info(f"Deleting entry: {episode['uuid']}")

            await delete_episode(episode['uuid'])
            logger.info("Entry deleted")

            await self.load_episodes()
    except Exception as e:
        logger.error(f"Failed to delete entry: {e}", exc_info=True)
        self.notify("Failed to delete entry", severity="error")
```

**Step 3: Test delete functionality**

Run: `python charlie.py`

Test sequence:
1. Create a test entry with 'n'
2. Save it
3. Select it in list
4. Press 'd' to delete
5. Verify entry is removed from list

Expected: Entry deleted immediately, list refreshes

**Step 4: Commit**

```bash
git add charlie.py
git commit -m "feat: add delete functionality"
```

---

## Task 8: Refresh List After Edit/Create

**Files:**
- Modify: `charlie.py` (add screen refresh on return)

**Step 1: Add refresh method to HomeScreen**

Add method to `HomeScreen`:

```python
async def on_screen_resume(self):
    """Called when returning to this screen."""
    await self.load_episodes()
```

**Step 2: Test the refresh behavior**

Run: `python charlie.py`

Test sequence:
1. Create new entry with 'n', save
2. Verify new entry appears in list
3. Edit existing entry with 'e', modify, save
4. Verify changes reflected in list title

Expected: List automatically refreshes when returning from edit

**Step 3: Commit**

```bash
git add charlie.py
git commit -m "feat: auto-refresh list on screen resume"
```

---

## Task 9: Polish and Final Testing

**Files:**
- Modify: `charlie.py` (final polish)

**Step 1: Add comprehensive error handling for empty list operations**

Update HomeScreen methods to handle empty list safely:

```python
def action_edit_entry(self):
    try:
        if not self.episodes:
            return
        list_view = self.query_one("#episodes-list", ListView)
        if list_view.index is not None and self.episodes:
            episode = self.episodes[list_view.index]
            logger.info(f"Edit entry: {episode['uuid']}")
            self.app.push_screen(EditScreen(episode['uuid']))
    except Exception as e:
        logger.error(f"Failed to open edit screen: {e}", exc_info=True)

def action_view_entry(self):
    try:
        if not self.episodes:
            return
        list_view = self.query_one("#episodes-list", ListView)
        if list_view.index is not None and self.episodes:
            episode = self.episodes[list_view.index]
            logger.info(f"View entry: {episode['uuid']}")
            self.app.push_screen(ViewScreen(episode['uuid']))
    except Exception as e:
        logger.error(f"Failed to open view screen: {e}", exc_info=True)

async def action_delete_entry(self):
    try:
        if not self.episodes:
            return
        list_view = self.query_one("#episodes-list", ListView)
        if list_view.index is not None and self.episodes:
            episode = self.episodes[list_view.index]
            logger.info(f"Deleting entry: {episode['uuid']}")

            await delete_episode(episode['uuid'])
            logger.info("Entry deleted")

            await self.load_episodes()
    except Exception as e:
        logger.error(f"Failed to delete entry: {e}", exc_info=True)
        self.notify("Failed to delete entry", severity="error")
```

**Step 2: Full integration test**

Run: `python charlie.py`

Test comprehensive workflow:
1. Start with empty database → see ASCII cat
2. Press 'n' to create first entry
3. Write markdown with `# My First Entry` and content
4. Press Esc to save
5. Verify entry appears in list
6. Press Enter to view → verify markdown renders
7. Press 'e' to edit → verify content loads
8. Modify content
9. Press Esc to save
10. Press 'q' to return to list
11. Verify changes in list
12. Press 'd' to delete
13. Verify back to empty state with cat
14. Create multiple entries with dates
15. Test j/k navigation
16. Test mouse clicking
17. Press 'q' to quit

Expected: All operations smooth, no errors, log file updated

**Step 3: Verify log file structure**

Run: `cat logs/charlie.log`

Expected: Clean log entries showing all operations

**Step 4: Final commit**

```bash
git add charlie.py
git commit -m "feat: polish error handling and complete charlie TUI"
```

---

## Task 10: Documentation

**Files:**
- Create: `README-CHARLIE.md`

**Step 1: Write user documentation**

Create `README-CHARLIE.md`:

```markdown
# Charlie TUI

A minimal terminal user interface for managing journal entries.

## Installation

Requires Python 3.13+ and dependencies from `pyproject.toml`.

```bash
# Install dependencies if needed
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
```

**Step 2: Commit documentation**

```bash
git add README-CHARLIE.md
git commit -m "docs: add charlie TUI user guide"
```

---

## Verification Checklist

After completing all tasks, verify:

- [ ] `python charlie.py` runs without errors
- [ ] Empty state shows ASCII cat
- [ ] Can create new entry with 'n'
- [ ] Can edit entry with 'e'
- [ ] Can view entry with Enter
- [ ] Can delete entry with 'd'
- [ ] Markdown `# Header` becomes title in list
- [ ] List refreshes after operations
- [ ] j/k navigation works
- [ ] Logs written to `logs/charlie.log`
- [ ] `logs/*.log` excluded from git
- [ ] All commits follow conventional format
- [ ] No crashes or unhandled exceptions
- [ ] Footer shows muted key hints
- [ ] Vim-style and mouse navigation both work

## Notes

- Keep implementation minimal and focused
- Follow TDD principles where applicable
- Commit frequently with clear messages
- Test each feature before moving to next task
- Use existing backend database module (already tested)
- No need for extensive unit tests (backend is tested)
- Focus on integration and UX testing
