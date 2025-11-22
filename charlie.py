import asyncio
import atexit
import logging
import os
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    Log as LogWidget,
    Markdown,
    Static,
    Switch,
    TextArea,
)
from textual.logging import TextualHandler

from backend import add_journal_entry
from backend.database import (
    delete_episode,
    ensure_database_ready,
    get_all_episodes,
    get_episode,
    get_tcp_server_endpoint,
    shutdown_database,
    update_episode,
)
from backend.database.redis_ops import get_inference_enabled, set_inference_enabled
from backend.settings import DEFAULT_JOURNAL
from backend.services.queue import (
    is_huey_consumer_running,
    start_huey_consumer,
    stop_huey_consumer,
)

LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

os.environ.setdefault("TEXTUAL_LOG", str(LOGS_DIR / "charlie.log"))


class _NoActiveAppFilter(logging.Filter):
    """Allow records through only when no Textual app context is active."""

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        from textual._context import active_app

        try:
            active_app.get()
        except LookupError:
            return True
        return False


def _configure_logging() -> None:
    """Route all logging through Textual, with a file fallback for background threads."""
    log_path = Path(os.environ["TEXTUAL_LOG"])

    textual_handler = TextualHandler(stderr=False, stdout=False)
    textual_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    logging.basicConfig(
        level=logging.INFO,
        handlers=[textual_handler, file_handler],
        force=True,
    )

    # Quiet Huey per-task INFO logs; keep warnings/errors.
    for name in ("huey", "huey.consumer", "huey.api", "huey.signals", "huey.queue"):
        logging.getLogger(name).setLevel(logging.WARNING)


_configure_logging()

logger = logging.getLogger("charlie")


class EntityListItem(ListItem):
    """A list item for displaying entity information."""

    def __init__(self, label_text: str, **kwargs):
        super().__init__(**kwargs)
        self.label_text = label_text

    def compose(self) -> ComposeResult:
        yield Label(self.label_text)


class EntitySidebar(Container):
    """Sidebar showing entities connected to current episode."""

    DEFAULT_CSS = """
    EntitySidebar {
        width: 1fr;
        border-left: solid $accent;
        padding: 1;
    }

    EntitySidebar .sidebar-header {
        text-style: bold;
        margin-bottom: 1;
    }

    EntitySidebar .sidebar-footer {
        color: $text-muted;
        margin-top: 1;
    }
    """

    CAT_SPINNER_FRAMES = [">^.^<", "^.^", ">^.^<", "^.^"]

    episode_uuid: reactive[str] = reactive("")
    journal: reactive[str] = reactive("")
    loading: reactive[bool] = reactive(True)
    entities: reactive[list[dict]] = reactive([])

    def __init__(self, episode_uuid: str, journal: str, **kwargs):
        super().__init__(**kwargs)
        self.episode_uuid = episode_uuid
        self.journal = journal
        self._spinner_index = 0
        self._spinner_timer = None

    def compose(self) -> ComposeResult:
        yield Label("Connections", classes="sidebar-header")
        yield Container(id="entity-content")
        yield Label("d: delete | ↑↓: navigate | c: close", classes="sidebar-footer")

    def on_mount(self) -> None:
        """Start spinner animation when mounted."""
        if self.loading:
            self._spinner_timer = self.set_interval(0.3, self._update_spinner)
        self._render_content()

    def watch_loading(self, loading: bool) -> None:
        """Reactive: swap between loading indicator and entity list."""
        if loading and self._spinner_timer is None:
            self._spinner_timer = self.set_interval(0.3, self._update_spinner)
        elif not loading and self._spinner_timer:
            self._spinner_timer.stop()
            self._spinner_timer = None

        self._render_content()

    def watch_entities(self, entities: list[dict]) -> None:
        """Reactive: re-render when entities change."""
        if not self.loading:
            self._render_content()

    def _update_spinner(self) -> None:
        """Update spinner animation frame."""
        self._spinner_index = (self._spinner_index + 1) % len(self.CAT_SPINNER_FRAMES)
        if self.loading:
            self._render_content()

    def _render_content(self) -> None:
        """Render either loading state or entity list."""
        if not self.is_mounted:
            return

        content_container = self.query_one("#entity-content", Container)
        content_container.remove_children()

        if self.loading:
            cat_frame = self.CAT_SPINNER_FRAMES[self._spinner_index]
            content_container.mount(Label(f"{cat_frame} Slinging yarn..."))
        elif not self.entities:
            content_container.mount(Label("No connections found"))
        else:
            list_view = ListView()
            content_container.mount(list_view)
            for entity in self.entities:
                formatted_label = self._format_entity_label(entity)
                list_view.append(EntityListItem(formatted_label))

    def _format_entity_label(self, entity: dict) -> str:
        """Format entity as 'Name [Type] (RefCount)'."""
        name = entity["name"]
        labels = entity["labels"]
        ref_count = entity["ref_count"]

        # Filter out "Entity" if there's a more specific type
        specific_labels = [l for l in labels if l != "Entity"]
        entity_type = specific_labels[0] if specific_labels else "Entity"

        return f"{name} [{entity_type}] ({ref_count})"


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
    lines = content.split("\n")
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# "):
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
    content = episode.get("content", "")

    title = extract_title(content)
    if title:
        return title[:max_chars]

    preview = content.replace("\n", " ").strip()
    return preview[:max_chars] if preview else episode.get("name", "Untitled")


EMPTY_STATE_CAT = r"""
         /\_/\
        ( o.o )
         > ^ <

    No entries yet!
    Press 'n' to create your first entry
"""


class SettingsScreen(ModalScreen):
    """Modal screen for application settings."""

    def __init__(self):
        super().__init__()
        # Default to enabled; actual state loaded on mount from Redis
        try:
            self.inference_enabled = get_inference_enabled()
        except Exception:
            self.inference_enabled = True
        self._loading_toggle_state = False
        self._toggle_worker = None

    BINDINGS = [
        Binding("s", "dismiss_modal", "Close", show=True),
        Binding("escape", "dismiss_modal", "Close", show=False),
        Binding("j,down", "app.focus_next", "Next", show=False),
        Binding("k,up", "app.focus_previous", "Previous", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("Settings", id="settings-title"),
            Label(""),
            Label("Enable background inference"),
            Switch(id="inference-toggle", value=self.inference_enabled),
            id="settings-dialog",
        )
        yield Footer()

    def action_dismiss_modal(self) -> None:
        self.dismiss()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id != "inference-toggle":
            return

        if self._loading_toggle_state:
            event.switch.value = self.inference_enabled
            return

        desired = event.value
        previous = self.inference_enabled

        event.switch.disabled = True
        self._loading_toggle_state = True

        # Run persistence off the UI thread to avoid freezes during model load/enqueue
        self._toggle_worker = self.run_worker(
            self._persist_inference_toggle(event.switch, desired, previous),
            exclusive=True,
            name="inference-toggle",
        )

    async def _persist_inference_toggle(self, switch: Switch, desired: bool, previous: bool):
        try:
            # Persist toggle and enqueue in a thread to keep UI responsive
            await asyncio.to_thread(set_inference_enabled, desired)
            self.inference_enabled = desired
            self._on_toggle_success(switch, desired)
        except Exception as exc:
            logger.error("Failed to persist inference toggle: %s", exc, exc_info=True)
            self._on_toggle_failure(switch, previous)
        finally:
            self._clear_toggle_loading(switch)

    def _on_toggle_success(self, switch: Switch, value: bool) -> None:
        switch.value = value
        switch.disabled = False

    def _on_toggle_failure(self, switch: Switch, previous: bool) -> None:
        switch.value = previous
        self.inference_enabled = previous
        switch.disabled = False
        self.notify("ai ded 💀 check logz", severity="error")

    def _clear_toggle_loading(self, switch: Switch) -> None:
        self._loading_toggle_state = False
        switch.disabled = False
        switch.focus()


class HomeScreen(Screen):
    """Main screen showing list of journal entries."""

    BINDINGS = [
        Binding("n", "new_entry", "New", show=True),
        Binding("space", "view_entry", "View", show=True),
        Binding("d", "delete_entry", "Delete", show=True),
        Binding("s", "open_settings", "Settings", show=True),
        Binding("l", "open_logs", "Logs", show=True),
        Binding("q", "quit", "Quit", show=True),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
    ]

    def __init__(self):
        super().__init__()
        self.episodes = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False, icon="")
        if not self.episodes:
            empty = Static(EMPTY_STATE_CAT, id="empty-state")
            empty.can_focus = True
            yield empty
        else:
            yield ListView(
                *[
                    ListItem(
                        Label(
                            f"{episode['valid_at'].strftime('%Y-%m-%d')} - {get_display_title(episode)}"
                        )
                    )
                    for episode in self.episodes
                ],
                id="episodes-list",
            )
        yield Footer()

    async def on_mount(self):
        self.run_worker(self._init_and_load(), exclusive=True)

    async def _init_and_load(self):
        try:
            await ensure_database_ready(DEFAULT_JOURNAL)
            # Start Huey worker after database is ready to avoid startup races
            if hasattr(self.app, "_ensure_huey_worker_running"):
                self.app._ensure_huey_worker_running()
            await self.load_episodes()
        except Exception as e:
            logger.error(f"Database initialization failed: {e}", exc_info=True)
            self.notify("Failed to initialize database. Exiting...", severity="error")
            await asyncio.sleep(2)
            self.app.exit(1)

    async def on_screen_resume(self):
        """Called when returning to this screen."""
        await self.load_episodes()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle ListView selection (Enter key)."""
        if self.episodes and event.list_view.index is not None:
            episode = self.episodes[event.list_view.index]
            self.app.push_screen(ViewScreen(episode["uuid"]))

    async def load_episodes(self):
        try:
            new_episodes = await get_all_episodes()

            if not new_episodes:
                self.episodes = []
                # Safe to recompose: switching between Static (empty state) and ListView
                # Neither widget has state that needs to be preserved
                await self.recompose()
                empty_state = self.query_one("#empty-state", Static)
                empty_state.focus()
            else:
                try:
                    list_view = self.query_one("#episodes-list", ListView)
                    old_index = list_view.index if list_view.index is not None else 0

                    await list_view.clear()

                    items = [
                        ListItem(
                            Label(
                                f"{episode['valid_at'].strftime('%Y-%m-%d')} - {get_display_title(episode)}"
                            )
                        )
                        for episode in new_episodes
                    ]
                    if items:
                        await list_view.extend(items)

                    self.episodes = new_episodes

                    if len(self.episodes) > 0:
                        new_index = min(old_index, len(self.episodes) - 1)
                        list_view.index = new_index
                        list_view.focus()
                except Exception:
                    self.episodes = new_episodes
                    # Safe to recompose: fallback when ListView doesn't exist yet
                    # (happens when transitioning from empty to populated state)
                    await self.recompose()
        except Exception as e:
            logger.error(f"Failed to load episodes: {e}", exc_info=True)
            self.notify("Error loading episodes", severity="error")

    def action_new_entry(self):
        self.app.push_screen(EditScreen())

    def action_view_entry(self):
        try:
            if not self.episodes:
                return
            list_view = self.query_one("#episodes-list", ListView)
            if list_view.index is not None:
                episode = self.episodes[list_view.index]
                self.app.push_screen(ViewScreen(episode["uuid"]))
        except Exception as e:
            logger.error(f"Failed to open view screen: {e}", exc_info=True)

    async def action_delete_entry(self):
        try:
            if not self.episodes:
                return
            list_view = self.query_one("#episodes-list", ListView)
            if list_view.index is not None and self.episodes:
                episode = self.episodes[list_view.index]
                await delete_episode(episode["uuid"])
                await self.load_episodes()
        except Exception as e:
            logger.error(f"Failed to delete entry: {e}", exc_info=True)
            self.notify("Failed to delete entry", severity="error")

    def _graceful_shutdown(self):
        """Ensure worker stops before database teardown (idempotent)."""
        try:
            if hasattr(self.app, "stop_huey_worker"):
                self.app.stop_huey_worker()
        finally:
            shutdown_database()

    def action_quit(self):
        # Stop background worker before tearing down the database to avoid
        # connection errors/backoff during shutdown.
        self._graceful_shutdown()
        self.app.exit()

    def action_cursor_down(self):
        if not self.episodes:
            return
        try:
            list_view = self.query_one("#episodes-list", ListView)
            list_view.action_cursor_down()
        except Exception as e:
            logger.debug(f"cursor_down failed: {e}")

    def action_cursor_up(self):
        if not self.episodes:
            return
        try:
            list_view = self.query_one("#episodes-list", ListView)
            list_view.action_cursor_up()
        except Exception as e:
            logger.debug(f"cursor_up failed: {e}")

    def action_open_settings(self):
        self.app.push_screen(SettingsScreen())

    def action_open_logs(self):
        self.app.push_screen(LogScreen())


class ViewScreen(Screen):
    """Screen for viewing a journal entry in read-only mode.

    WARNING: Do NOT use recompose() in this screen - Markdown widget
    has internal state (scroll position) that would be lost.
    """

    BINDINGS = [
        Binding("e", "edit_entry", "Edit", show=True),
        Binding("q", "back", "Back", show=True),
        Binding("escape", "back", "Back", show=False),
        Binding("space", "back", "Back", show=False),
        Binding("enter", "back", "Back", show=False),
    ]

    def __init__(self, episode_uuid: str):
        super().__init__()
        self.episode_uuid = episode_uuid
        self.episode = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False, icon="")
        yield Markdown("Loading...", id="content")
        yield Footer()

    async def on_mount(self):
        await self.load_episode()

    async def on_screen_resume(self):
        """Called when returning to this screen."""
        await self.load_episode()

    async def load_episode(self):
        try:
            self.episode = await get_episode(self.episode_uuid)

            if self.episode:
                markdown = self.query_one("#content", Markdown)
                await markdown.update(self.episode["content"])
            else:
                logger.error(f"Episode not found: {self.episode_uuid}")
                self.notify("Entry not found", severity="error")
                self.app.pop_screen()
        except Exception as e:
            logger.error(f"Failed to load episode: {e}", exc_info=True)
            self.notify("Error loading entry", severity="error")
            self.app.pop_screen()

    def action_edit_entry(self):
        self.app.push_screen(EditScreen(self.episode_uuid))

    def action_back(self):
        self.app.pop_screen()


class EditScreen(Screen):
    """Screen for creating or editing a journal entry.

    WARNING: Do NOT use recompose() in this screen - TextArea widget
    has internal state (cursor position, undo history) that would be lost.
    """

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
        yield Header(show_clock=False, icon="")
        yield TextArea("", id="editor")
        yield Footer()

    async def on_mount(self):
        if not self.is_new_entry:
            await self.load_episode()

        editor = self.query_one("#editor", TextArea)
        editor.focus()

    async def load_episode(self):
        try:
            self.episode = await get_episode(self.episode_uuid)

            if self.episode:
                editor = self.query_one("#editor", TextArea)
                editor.text = self.episode["content"]
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
                self.app.pop_screen()
                return

            title = extract_title(content)

            if self.is_new_entry:
                uuid = await add_journal_entry(content=content)
                if title:
                    await update_episode(uuid, name=title)
            else:
                if title:
                    await update_episode(self.episode_uuid, content=content, name=title)
                else:
                    await update_episode(self.episode_uuid, content=content)

            self.app.pop_screen()

        except Exception as e:
            logger.error(f"Failed to save entry: {e}", exc_info=True)
            self.notify("Failed to save entry", severity="error")
            raise


class LogScreen(Screen):
    """Screen for quick log inspection using Textual's Log widget."""

    BINDINGS = [
        Binding("r", "reload", "Reload", show=True),
        Binding("escape", "close", "Close", show=True),
        Binding("q", "close", "Close", show=False),
    ]

    def __init__(self, log_path: Path | None = None):
        super().__init__()
        self.log_path = log_path or Path(os.environ.get("TEXTUAL_LOG", LOGS_DIR / "charlie.log"))

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False, icon="")
        yield LogWidget(id="log-view", auto_scroll=True, max_lines=2000)
        yield Footer()

    async def on_mount(self):
        await self._load_log()

    async def on_screen_resume(self):
        await self._load_log()

    async def action_reload(self):
        await self._load_log()

    def action_close(self):
        self.app.pop_screen()

    async def _load_log(self):
        log_widget = self.query_one("#log-view", LogWidget)
        log_widget.clear()

        if not self.log_path.exists():
            log_widget.write_line("Log file has not been created yet.")
            return

        try:
            lines = await asyncio.to_thread(self._tail_file, self.log_path, 2000)
            log_widget.write_lines(lines)
            log_widget.scroll_end(animate=False)
        except Exception as exc:
            log_widget.write_line(f"Failed to load log: {exc}")
            logger.error("Failed to load log view: %s", exc, exc_info=True)

    @staticmethod
    def _tail_file(path: Path, max_lines: int) -> list[str]:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return [line.rstrip("\n") for line in f.readlines()[-max_lines:]]


class CharlieApp(App):
    """A minimal journal TUI application."""

    TITLE = "Charlie"

    def __init__(self):
        super().__init__()

    async def on_mount(self):
        self.theme = "catppuccin-mocha"
        self.title = "Charlie"
        self.push_screen(HomeScreen())
        # Worker is started after DB readiness in HomeScreen._init_and_load.

    def _ensure_huey_worker_running(self):
        """Start Huey consumer in-process if not already running."""
        if is_huey_consumer_running():
            return

        try:
            # Ensure task registry is populated before starting the consumer.
            # Importing registers @huey.task functions with the shared Huey instance.
            import backend.services.tasks  # noqa: F401

            start_huey_consumer()
            endpoint = get_tcp_server_endpoint()
            if endpoint:
                host, port = endpoint
                logger.info(
                    "Huey consumer running in-process using embedded Redis at %s:%d",
                    host,
                    port,
                )
            else:
                logger.info(
                    "Huey consumer running in-process using embedded Redis (unix socket)"
                )
            atexit.register(self._shutdown_huey)

        except Exception as exc:
            logger.error("Failed to start Huey worker: %s", exc, exc_info=True)
            self.notify("huey ded 💀 check logz", severity="error", timeout=5)

    def _shutdown_huey(self):
        """Terminate Huey worker gracefully (finish current task first)."""
        try:
            stop_huey_consumer()
        except Exception as exc:
            logger.warning("Error while stopping Huey worker: %s", exc, exc_info=True)

    def stop_huey_worker(self):
        """Public wrapper to stop worker (used by UI handlers)."""
        self._shutdown_huey()

    def _graceful_shutdown(self):
        """Stop background worker before tearing down the database."""
        try:
            self.stop_huey_worker()
        finally:
            try:
                shutdown_database()
            except Exception as exc:
                logger.warning(
                    "Database shutdown encountered an error: %s", exc, exc_info=True
                )

    def on_unmount(self):
        # Covers exits triggered outside the key binding (e.g., cmd+q/ctrl+c)
        self._graceful_shutdown()


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

#settings-dialog {
    width: 60;
    height: auto;
    padding: 2 4;
    background: $panel;
    border: thick $primary;
}

#settings-title {
    text-align: center;
    text-style: bold;
    color: $text;
    padding-bottom: 1;
}

SettingsScreen {
    align: center middle;
}

#log-view {
    height: 100%;
}
"""


if __name__ == "__main__":
    app = CharlieApp()
    app.run()
