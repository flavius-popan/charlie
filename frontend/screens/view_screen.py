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
from frontend.state.sidebar_state_machine import SidebarStateMachine

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

        # Instantiate sidebar state machine
        self.sidebar_machine = SidebarStateMachine(
            initial_status=status,
            inference_enabled=inferred_enabled,
            entities_present=False,
            visible=from_edit,
        )

        # Sync machine output to reactive properties
        self._sync_machine_output()

    def compose(self) -> ComposeResult:
        sidebar = EntitySidebar(
            episode_uuid=self.episode_uuid,
            journal=self.journal,
            inference_enabled=self.inference_enabled,
            status=self.status,
            active_processing=self.active_processing,
            on_entity_deleted=self._on_entity_deleted,
            id="entity-sidebar",
        )

        # Bind sidebar reactives to parent so state stays synchronized automatically.
        sidebar.data_bind(
            status=ViewScreen.status,
            active_processing=ViewScreen.active_processing,
            inference_enabled=ViewScreen.inference_enabled,
        )

        yield Header(show_clock=False, icon="")
        yield Horizontal(
            Markdown("Loading...", id="journal-content"),
            sidebar,
        )
        yield Footer()

    async def on_mount(self):
        await self.load_episode()

        # Sync inference/status context to sidebar
        self._refresh_sidebar_context()

        # Show sidebar only if coming from EditScreen
        sidebar = self.query_one("#entity-sidebar", EntitySidebar)
        if not self.from_edit:
            # Ensure UI matches machine state (hidden when viewing from home)
            sidebar.display = False
            if self.sidebar_machine.output.visible:
                self.sidebar_machine.send("hide")
            self._sync_machine_output()
            return  # Skip polling and cache check when sidebar hidden

        # Polling decision now handled by on_entity_sidebar_cache_check_complete
        # which fires after EntitySidebar.refresh_entities() completes on mount

    def on_entity_sidebar_cache_check_complete(
        self, event: EntitySidebar.CacheCheckComplete
    ) -> None:
        """Handle cache check completion from sidebar."""
        if not self.from_edit:
            return  # Only relevant when coming from edit

        # Route cache result to machine only when entities are found.
        # When cache is empty and processing is still pending (pending_nodes),
        # don't transition - just start polling for status updates.
        # The polling worker will route cache_empty after status changes.
        if event.entities_found:
            self.sidebar_machine.send("cache_entities_found", entities_present=True)
            self._sync_machine_output()

        # Start polling if machine indicates we should (still in processing state)
        if self.sidebar_machine.output.should_poll:
            self.run_worker(self._poll_until_complete(), exclusive=True, name="status-poll")

    async def on_screen_resume(self):
        """Called when returning to this screen."""
        self._refresh_sidebar_context()
        await self.load_episode()

    def _sync_machine_output(self) -> None:
        """Sync machine output to reactive properties for sidebar consumption."""
        output = self.sidebar_machine.output
        self.status = output.status
        self.active_processing = output.active_processing

        # Inference state is handled separately in _refresh_sidebar_context
        # since it requires reading from Redis

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
        # Notify machine that episode is closed (only if sidebar visible)
        if self.sidebar_machine.output.visible:
            self.sidebar_machine.send("episode_closed")
            self._sync_machine_output()

        # Cancel any pending status polling worker before leaving
        self.workers.cancel_group(self, "status-poll")
        self.app.pop_screen()

    def action_toggle_connections(self) -> None:
        """Toggle sidebar visibility and focus if opened."""
        sidebar = self.query_one("#entity-sidebar", EntitySidebar)
        current_visible = self.sidebar_machine.output.visible

        if current_visible:
            # Hide sidebar
            self.sidebar_machine.send("hide")
            self._sync_machine_output()
            sidebar.display = False
            self.workers.cancel_group(self, "status-poll")
        else:
            # Show sidebar - refresh context and route event
            sidebar.display = True
            self._refresh_sidebar_context()
            self.sidebar_machine.send("show", status=self.sidebar_machine.output.status)
            self._sync_machine_output()
            sidebar._update_content()

            # Start polling if machine indicates we should
            if self.sidebar_machine.output.should_poll:
                if not any(w.name == "status-poll" for w in self.workers if w.is_running):
                    sidebar.run_worker(sidebar.refresh_entities(), exclusive=True)
                    self.run_worker(self._poll_until_complete(), exclusive=True, name="status-poll")
            else:
                # Machine indicates we shouldn't poll - check cache anyway
                sidebar.run_worker(sidebar.refresh_entities(), exclusive=True)

            if sidebar.entities:
                try:
                    list_view = sidebar.query_one(ListView)
                    if list_view.index is None:
                        list_view.index = 0
                    self.set_focus(list_view)
                except Exception:
                    pass

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

                # Route status change to machine
                if status == "pending_nodes":
                    self.sidebar_machine.send("status_pending_nodes", status=status)
                elif status in ("pending_edges", "done", None):
                    self.sidebar_machine.send("status_pending_edges_or_done", status=status)

                self._sync_machine_output()

                # When extraction complete, refresh and stop
                if status in ("pending_edges", "done", None):
                    try:
                        sidebar = self.query_one("#entity-sidebar", EntitySidebar)
                        await sidebar.refresh_entities()

                        # Route cache result to machine
                        if sidebar.entities:
                            self.sidebar_machine.send("cache_entities_found", entities_present=True)
                        else:
                            self.sidebar_machine.send("cache_empty", entities_present=False)
                        self._sync_machine_output()
                    except NoMatches:
                        logger.debug("Sidebar not found during poll - stopping")
                    except Exception as exc:
                        logger.exception(f"Failed to refresh entities: {exc}")
                    break

                # Wait before next check
                await asyncio.sleep(0.5)
            except WorkerCancelled:
                logger.debug("Status polling worker cancelled")
                break
            except NoMatches:
                logger.debug("Sidebar not found during poll - stopping")
                break
            except Exception as exc:
                logger.exception(f"Status poll error: {exc}")
                break

    def _on_entity_deleted(self, entities_present: bool) -> None:
        """Handle entity deletion event from sidebar.

        Route to state machine: user deleted an entity.
        """
        self.sidebar_machine.send("user_deleted_entity", entities_present=entities_present)
        self._sync_machine_output()

    def _refresh_sidebar_context(self):
        """Update sidebar with latest inference toggle and processing status."""
        try:
            sidebar = self.query_one("#entity-sidebar", EntitySidebar)
        except NoMatches:
            logger.debug("Sidebar not yet mounted; skipping context refresh")
            return

        try:
            inferred_enabled = get_inference_enabled()
            status = get_episode_status(self.episode_uuid, self.journal)
        except Exception as exc:
            logger.debug("Failed to refresh sidebar context: %s", exc)
            # Machine already has best-known state; skip update on error
            return

        # Update machine with fresh context
        if inferred_enabled != self.sidebar_machine.inference_enabled_flag:
            if inferred_enabled:
                self.sidebar_machine.send("inference_enabled", status=status)
            else:
                self.sidebar_machine.send("inference_disabled")

        # Update status
        if status != self.sidebar_machine.output.status:
            if status == "pending_nodes":
                self.sidebar_machine.send("status_pending_nodes", status=status)
            elif status in ("pending_edges", "done", None):
                self.sidebar_machine.send("status_pending_edges_or_done", status=status)

        self._sync_machine_output()
