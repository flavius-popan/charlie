import asyncio
import logging

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Footer, Header, ListView, Markdown

from backend.database import get_episode
from backend.database.redis_ops import (
    get_episode_status,
    get_inference_enabled,
)
from backend.settings import DEFAULT_JOURNAL
from textual.worker import WorkerCancelled
from frontend.widgets.entity_sidebar import EntitySidebar

logger = logging.getLogger("charlie")


class ViewScreen(Screen):
    """Screen for viewing a journal entry in read-only mode with entity sidebar.

    WARNING: Do NOT use recompose() in this screen - Markdown widget
    has internal state (scroll position) that would be lost.
    """

    BINDINGS = [
        Binding("e", "edit_entry", "Edit", show=True),
        Binding("c", "toggle_connections", "Connections", show=True),
        Binding("l", "show_logs", "Logs", show=True),
        Binding("q", "back", "Back", show=True),
        Binding("escape", "back", "Back", show=False),
        Binding("space", "back", "Back", show=False),
        Binding("enter", "back", "Back", show=False),
    ]

    DEFAULT_CSS = """
    ViewScreen Horizontal {
        height: 100%;
    }

    ViewScreen #journal-content {
        width: 3fr;
        padding: 1 2;
        overflow-y: auto;
        height: 100%;
        scrollbar-size-vertical: 0;
        scrollbar-size-horizontal: 0;
    }
    """

    status: reactive[str | None] = reactive(None)
    inference_enabled: reactive[bool] = reactive(True)
    active_processing: reactive[bool] = reactive(False)

    def __init__(
        self,
        episode_uuid: str,
        journal: str = DEFAULT_JOURNAL,
        from_edit: bool = False,
        inference_enabled: bool | None = None,
        status: str | None = None,
        active_processing: bool = False,
    ):
        super().__init__()
        self.episode_uuid = episode_uuid
        self.journal = journal
        self.episode = None
        self.from_edit = from_edit
        inferred_enabled = (
            inference_enabled if inference_enabled is not None else get_inference_enabled()
        )
        self.set_reactive(ViewScreen.inference_enabled, inferred_enabled)
        self.set_reactive(ViewScreen.status, status)
        self.set_reactive(ViewScreen.active_processing, active_processing)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False, icon="")
        yield Horizontal(
            Markdown("Loading...", id="journal-content"),
            EntitySidebar(
                episode_uuid=self.episode_uuid,
                journal=self.journal,
                inference_enabled=self.inference_enabled,
                status=self.status,
                active_processing=self.active_processing,
                id="entity-sidebar",
            ),
        )
        yield Footer()

    async def on_mount(self):
        await self.load_episode()

        # Sync inference/status context to sidebar
        self._refresh_sidebar_context()

        # Show sidebar only if coming from EditScreen
        sidebar = self.query_one("#entity-sidebar", EntitySidebar)
        if not self.from_edit:
            sidebar.display = False
            return  # Skip polling and cache check when sidebar hidden

        # Give EntitySidebar a moment to check cache on mount
        await asyncio.sleep(0.05)

        # Only start polling if sidebar is still loading (cache miss)
        if sidebar.loading:
            # Cache doesn't have data - start polling for extraction job
            sidebar.active_processing = True
            self.run_worker(self._poll_until_complete(), exclusive=True, name="status-poll")

    async def on_screen_resume(self):
        """Called when returning to this screen."""
        self._refresh_sidebar_context()
        await self.load_episode()

    def watch_status(self, status: str | None) -> None:
        """When ViewScreen.status changes, sync to sidebar."""
        if not self.is_mounted:
            return
        try:
            sidebar = self.query_one("#entity-sidebar", EntitySidebar)
            sidebar.status = status
        except NoMatches:
            pass

    def watch_inference_enabled(self, enabled: bool) -> None:
        """When inference toggle changes, sync to sidebar."""
        if not self.is_mounted:
            return
        try:
            sidebar = self.query_one("#entity-sidebar", EntitySidebar)
            sidebar.inference_enabled = enabled
        except NoMatches:
            pass

    def watch_active_processing(self, active: bool) -> None:
        """When processing state changes, sync to sidebar."""
        if not self.is_mounted:
            return
        try:
            sidebar = self.query_one("#entity-sidebar", EntitySidebar)
            sidebar.active_processing = active
        except NoMatches:
            pass

    async def load_episode(self):
        try:
            self.episode = await get_episode(self.episode_uuid)

            if self.episode:
                # Refresh sidebar context in case status changed externally
                self._refresh_sidebar_context()
                markdown = self.query_one("#journal-content", Markdown)
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
        from frontend.screens.edit_screen import EditScreen
        self.app.push_screen(EditScreen(self.episode_uuid))

    def action_back(self):
        # Cancel any pending status polling worker before leaving
        self.workers.cancel_group(self, "status-poll")
        self.app.pop_screen()

    def action_toggle_connections(self) -> None:
        """Toggle sidebar visibility and focus if opened."""
        sidebar = self.query_one("#entity-sidebar", EntitySidebar)
        sidebar.display = not sidebar.display

        if sidebar.display:
            # Refresh context before showing
            self._refresh_sidebar_context()
            if (
                sidebar.status in ("pending_nodes", "pending_edges")
                and not sidebar.entities
            ):
                sidebar.loading = True
            if sidebar.loading:
                sidebar.run_worker(sidebar.refresh_entities(), exclusive=True)
                # If still loading, start polling for job completion
                if not any(w.name == "status-poll" for w in self.workers if w.is_running):
                    self.run_worker(self._poll_until_complete(), exclusive=True, name="status-poll")
                    sidebar.active_processing = True
            sidebar._update_content()

            if sidebar.entities:
                try:
                    list_view = sidebar.query_one(ListView)
                    if list_view.index is None:
                        list_view.index = 0
                    self.set_focus(list_view)
                except Exception:
                    pass
        else:
            self.workers.cancel_group(self, "status-poll")
            sidebar.active_processing = False

    def action_show_logs(self) -> None:
        """Navigate to log viewer."""
        from frontend.screens.log_screen import LogScreen
        self.app.push_screen(LogScreen())

    async def _poll_until_complete(self) -> None:
        """Poll Redis until extraction completes (non-blocking background task)."""
        while True:
            try:
                # Run blocking I/O in thread to keep UI responsive
                status = await asyncio.to_thread(
                    get_episode_status,
                    self.episode_uuid,
                    self.journal
                )

                # Update reactive (watch_status will sync to sidebar)
                self.status = status

                # When extraction complete, refresh and stop
                if status in ("pending_edges", "done", None):
                    self.active_processing = False
                    sidebar = self.query_one("#entity-sidebar", EntitySidebar)
                    await sidebar.refresh_entities()
                    break

                # Wait before next check
                await asyncio.sleep(0.5)
            except WorkerCancelled:
                logger.debug("Status polling worker cancelled")
                self.active_processing = False
                break
            except NoMatches:
                logger.debug("Sidebar not found during poll - stopping")
                self.active_processing = False
                break
            except Exception as exc:
                logger.exception(f"Status poll error: {exc}")
                self.active_processing = False
                break

    def _refresh_sidebar_context(self):
        """Update sidebar with latest inference toggle and processing status."""
        sidebar = self.query_one("#entity-sidebar", EntitySidebar)
        try:
            self.inference_enabled = get_inference_enabled()
            self.status = get_episode_status(self.episode_uuid, self.journal)
        except Exception as exc:
            logger.debug("Failed to refresh sidebar context: %s", exc)
            # fall back to previous values
        with self.app.batch_update():
            sidebar.inference_enabled = self.inference_enabled
            sidebar.status = self.status
            sidebar.active_processing = any(w.name == "status-poll" and w.is_running for w in self.workers)
        sidebar._update_content()
