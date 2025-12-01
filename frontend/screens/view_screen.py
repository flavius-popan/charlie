import asyncio
import logging

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Footer, Header, ListView, Markdown

from backend.database import get_episode, get_entry_entities_with_counts
from backend.database.persistence import delete_episode
from backend.database.redis_ops import redis_ops
from frontend.utils import inject_entity_links, rough_token_estimate
from backend.database.redis_ops import (
    get_episode_status,
    get_inference_enabled,
)
from backend.settings import DEFAULT_JOURNAL
from textual.worker import WorkerCancelled
from frontend.widgets.confirmation_modal import ConfirmationModal
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
        Binding("d", "delete", "Delete", show=True),
        Binding("c", "toggle_connections", "Connections", show=True),
        Binding("q", "back", "Back", show=True),
        Binding("left", "prev_entry", "Prev", show=True),
        Binding("right", "next_entry", "Next", show=True),
        Binding("escape", "back", "Back", show=False),
        Binding("space", "back", "Back", show=False),
        Binding("enter", "back", "Back", show=False),
        Binding("up", "scroll_up", "Scroll Up", show=False),
        Binding("down", "scroll_down", "Scroll Down", show=False),
        Binding("k", "scroll_up", "Scroll Up", show=False),
        Binding("j", "scroll_down", "Scroll Down", show=False),
    ]

    DEFAULT_CSS = """
    ViewScreen Horizontal {
        height: 100%;
    }

    ViewScreen #content-area {
        width: 3fr;
        height: 100%;
        align: center top;
    }

    ViewScreen #content-wrapper {
        max-width: 80;
        height: 100%;
        border: solid $foreground-muted;
        border-title-align: center;
        border-subtitle-align: center;
    }

    ViewScreen #journal-content {
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
    active_episode_uuid: reactive[str | None] = reactive(None)
    _current_idx: reactive[int | None] = reactive(None, bindings=True)

    def __init__(
        self,
        episode_uuid: str,
        journal: str = DEFAULT_JOURNAL,
        from_edit: bool = False,
        inference_enabled: bool | None = None,
        status: str | None = None,
        active_processing: bool = False,
        episodes: list[dict] | None = None,
    ):
        super().__init__()
        self.episode_uuid = episode_uuid
        self.journal = journal
        self.episode = None
        self.from_edit = from_edit
        self.episodes = episodes or []
        self._current_idx = self._find_episode_index(episode_uuid)
        inferred_enabled = (
            inference_enabled
            if inference_enabled is not None
            else get_inference_enabled()
        )

        # Instantiate sidebar state machine
        self.sidebar_machine = SidebarStateMachine(
            initial_status=status,
            inference_enabled=inferred_enabled,
            entities_present=False,
            visible=from_edit,
        )

        # Guard against reentrant toggle calls during async operations
        self._toggling = False

        # Sync machine output to reactive properties
        self._sync_machine_output()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Conditionally hide prev/next bindings based on navigation state."""
        if action == "prev_entry":
            # Hide if no episodes, no index, or at oldest
            if not self.episodes or self._current_idx is None:
                return False
            if self._current_idx >= len(self.episodes) - 1:
                return False
        elif action == "next_entry":
            # Hide if no episodes, no index, or at newest
            if not self.episodes or self._current_idx is None:
                return False
            if self._current_idx <= 0:
                return False
        return True

    def compose(self) -> ComposeResult:
        sidebar = EntitySidebar(
            episode_uuid=self.episode_uuid,
            journal=self.journal,
            inference_enabled=self.inference_enabled,
            status=self.status,
            active_processing=self.active_processing,
            active_episode_uuid=self.active_episode_uuid,
            on_entity_deleted=self._on_entity_deleted,
            id="entity-sidebar",
        )

        # Bind sidebar reactives to parent so state stays synchronized automatically.
        sidebar.data_bind(
            status=ViewScreen.status,
            active_processing=ViewScreen.active_processing,
            inference_enabled=ViewScreen.inference_enabled,
            active_episode_uuid=ViewScreen.active_episode_uuid,
        )

        content_wrapper = Container(
            Markdown("Loading...", id="journal-content", open_links=False),
            id="content-wrapper",
        )

        yield Header(show_clock=False, icon="")
        yield Horizontal(
            Container(content_wrapper, id="content-area"),
            sidebar,
        )
        yield Footer()

    async def on_mount(self):
        await self.load_episode()

        # Sync all sidebar context in one batched Redis call
        await self._refresh_all_sidebar_state()

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
            self.run_worker(
                self._poll_until_complete(), exclusive=True, name="status-poll"
            )

    async def on_screen_resume(self):
        """Called when returning to this screen."""
        await self._refresh_all_sidebar_state()
        await self.load_episode()

        # Refresh sidebar entities in case they changed while navigating
        try:
            sidebar = self.query_one("#entity-sidebar", EntitySidebar)
            sidebar.run_worker(sidebar.refresh_entities(), exclusive=True)
        except NoMatches:
            pass

    def _sync_machine_output(self) -> None:
        """Sync machine output to reactive properties for sidebar consumption.

        NOTE: Only sets active_processing=False when machine indicates processing is DONE.
        When machine says we SHOULD be processing, active_processing is controlled by
        _update_active_processing_async() to distinguish actively-processing vs queued episodes.
        """
        output = self.sidebar_machine.output
        self.status = output.status

        # When machine says processing is done, we know active_processing is False
        # (no Redis check needed). When machine says we should be processing,
        # leave it alone - _update_active_processing_async() handles that case.
        if not output.active_processing:
            self.active_processing = False

    async def _fetch_sidebar_context_batched(
        self,
    ) -> tuple[bool, str | None, str | None]:
        """Fetch all sidebar context from Redis in a single batched call.

        Returns:
            Tuple of (inference_enabled, status, active_episode_uuid)
        """

        def _fetch_all():
            # Use existing helper functions for consistency with tests
            inferred_enabled = get_inference_enabled()
            status = get_episode_status(self.episode_uuid, self.journal)

            # Fetch active episode directly for batching
            with redis_ops() as r:
                active_data = r.hgetall("task:active_episode")
            active_uuid = (
                active_data.get(b"uuid", b"").decode() if active_data else None
            )

            return (inferred_enabled, status, active_uuid)

        return await asyncio.to_thread(_fetch_all)

    async def _refresh_all_sidebar_state(self) -> None:
        """Refresh all sidebar context in one batched Redis call."""
        try:
            self.query_one("#entity-sidebar", EntitySidebar)
        except NoMatches:
            return

        try:
            (
                inferred_enabled,
                status,
                active_uuid,
            ) = await self._fetch_sidebar_context_batched()
        except Exception as exc:
            logger.debug("Failed to refresh sidebar context: %s", exc)
            return

        self._apply_sidebar_context(inferred_enabled, status)
        self.active_episode_uuid = active_uuid
        self.active_processing = bool(active_uuid and active_uuid == self.episode_uuid)

    async def _update_active_processing_async(self) -> None:
        """Check if THIS episode is actively processing (async-safe).

        Spinner should show only for the single episode Huey is working on.
        """

        def _get_active_uuid():
            with redis_ops() as r:
                data = r.hgetall("task:active_episode")
                if not data:
                    return None
                try:
                    return data.get(b"uuid", b"").decode()
                except Exception:
                    return None

        active_uuid = await asyncio.to_thread(_get_active_uuid)
        self.active_episode_uuid = active_uuid
        self.active_processing = bool(active_uuid and active_uuid == self.episode_uuid)

    async def load_episode(self):
        try:
            self.episode = await get_episode(self.episode_uuid)

            if self.episode:
                # Refresh sidebar context in case status changed externally (batched for speed)
                await self._refresh_all_sidebar_state()

                # Fetch entities and inject clickable links (only entities mentioned 2+ times)
                entities = await get_entry_entities_with_counts(
                    self.episode_uuid, self.journal
                )
                content = self.episode["content"]
                if entities:
                    content = inject_entity_links(content, entities, min_mentions=2)

                markdown = self.query_one("#journal-content", Markdown)
                await markdown.update(content)

                # Update content wrapper border with date and stats
                content_wrapper = self.query_one("#content-wrapper", Container)
                valid_at = self.episode.get("valid_at")
                if valid_at and hasattr(valid_at, "strftime"):
                    date_str = valid_at.strftime("%A, %B %-d, %Y")
                    content_wrapper.border_title = date_str

                # Compute and display stats
                word_count = len(content.split())
                char_count = len(content)
                token_count = rough_token_estimate(content)
                content_wrapper.border_subtitle = f"[dim]{word_count:,} words | {char_count:,} chars | ~{token_count:,} tokens[/dim]"
            else:
                logger.error(f"Episode not found: {self.episode_uuid}")
                self.notify("Entry not found", severity="error")
                self.app.pop_screen()
        except Exception as e:
            logger.error(f"Failed to load episode: {e}", exc_info=True)
            self.notify("Error loading entry", severity="error")
            self.app.pop_screen()

    def on_markdown_link_clicked(self, event: Markdown.LinkClicked) -> None:
        """Handle link clicks, including custom entity:// protocol."""
        href = event.href

        if href.startswith("entity://"):
            entity_uuid = href.replace("entity://", "")
            from frontend.screens.entity_browser_screen import EntityBrowserScreen

            self.app.push_screen(EntityBrowserScreen(entity_uuid, self.journal))
        elif href.startswith(("http://", "https://")):
            self.app.open_url(href)

    def action_edit_entry(self):
        from frontend.screens.edit_screen import EditScreen

        self.app.push_screen(EditScreen(self.episode_uuid))

    def _sidebar_entity_focused(self) -> bool:
        """Check if sidebar is visible with an entity focused."""
        try:
            sidebar = self.query_one("#entity-sidebar", EntitySidebar)
            if sidebar.display and sidebar.entities:
                list_view = sidebar.query_one(ListView)
                if (
                    list_view.has_focus
                    and list_view.index is not None
                    and list_view.index >= 0
                ):
                    return True
        except NoMatches:
            pass
        return False

    def action_delete(self):
        """Context-aware delete: connection if sidebar focused, otherwise entry."""
        if self._sidebar_entity_focused():
            sidebar = self.query_one("#entity-sidebar", EntitySidebar)
            sidebar.action_delete_entity()
        else:
            modal = ConfirmationModal(
                title="Delete this entry?",
                hint="This action cannot be undone.",
                confirm_label="Delete",
            )
            self.app.push_screen(modal, self._handle_entry_delete_result)

    async def _handle_entry_delete_result(self, confirmed: bool) -> None:
        """Handle entry deletion confirmation result."""
        if not confirmed:
            return

        try:
            await delete_episode(self.episode_uuid)
            self.app.pop_screen()
        except Exception as e:
            logger.error("Failed to delete entry: %s", e, exc_info=True)
            self.notify("Failed to delete entry", severity="error")

    def action_back(self):
        # Notify machine that episode is closed (only if sidebar visible)
        if self.sidebar_machine.output.visible:
            self.sidebar_machine.send("episode_closed")
            self._sync_machine_output()

        # Cancel any pending status polling worker before leaving
        self.workers.cancel_group(self, "status-poll")
        # Return current episode UUID so HomeScreen can select it
        self.dismiss(self.episode_uuid)

    async def action_toggle_connections(self) -> None:
        """Toggle sidebar visibility and focus if opened."""
        # Prevent reentrant calls during async operations
        if self._toggling:
            return
        self._toggling = True

        try:
            sidebar = self.query_one("#entity-sidebar", EntitySidebar)
            current_visible = self.sidebar_machine.output.visible

            if current_visible:
                # Hide sidebar
                self.sidebar_machine.send("hide")
                self._sync_machine_output()
                self.active_processing = False  # No spinner when hidden
                sidebar.display = False
                # IMPORTANT: Cancel ALL workers when hiding to prevent race conditions.
                # Workers started on child widgets (sidebar.run_worker) continue running
                # even when the widget is hidden. If not cancelled, they race with the
                # next "show" operation, corrupting state. Always cancel child widget
                # workers when hiding/destroying the parent context.
                self.workers.cancel_group(self, "status-poll")
                sidebar.workers.cancel_all()
            else:
                # Show sidebar - refresh context and route event
                sidebar.display = True

                # Batch all Redis reads into a single call for minimal latency
                (
                    inferred_enabled,
                    status,
                    active_uuid,
                ) = await self._fetch_sidebar_context_batched()

                self._apply_sidebar_context(inferred_enabled, status)
                self.sidebar_machine.send(
                    "show", status=self.sidebar_machine.output.status
                )
                self._sync_machine_output()

                # Update active processing state
                self.active_episode_uuid = active_uuid
                self.active_processing = bool(
                    active_uuid and active_uuid == self.episode_uuid
                )

                sidebar._update_content()

                # Start polling if machine indicates we should
                if self.sidebar_machine.output.should_poll:
                    if not any(
                        w.name == "status-poll" for w in self.workers if w.is_running
                    ):
                        sidebar.run_worker(sidebar.refresh_entities(), exclusive=True)
                        self.run_worker(
                            self._poll_until_complete(),
                            exclusive=True,
                            name="status-poll",
                        )
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
                else:
                    # No entities yet - keep focus on markdown to avoid focus limbo
                    try:
                        markdown = self.query_one("#journal-content", Markdown)
                        self.set_focus(markdown)
                    except Exception:
                        pass
        finally:
            self._toggling = False

    async def _poll_until_complete(self) -> None:
        """Poll Redis until extraction completes (non-blocking background task)."""
        while True:
            try:
                # Run blocking I/O in thread to keep UI responsive
                status = await asyncio.to_thread(
                    get_episode_status, self.episode_uuid, self.journal
                )

                # Route status change to machine
                if status == "pending_nodes":
                    self.sidebar_machine.send("status_pending_nodes", status=status)
                elif status in ("pending_edges", "done", None):
                    self.sidebar_machine.send(
                        "status_pending_edges_or_done", status=status
                    )

                self._sync_machine_output()

                # Check if this episode is actively processing (async)
                await self._update_active_processing_async()

                # When extraction complete, refresh and stop
                if status in ("pending_edges", "done", None):
                    try:
                        sidebar = self.query_one("#entity-sidebar", EntitySidebar)
                        await sidebar.refresh_entities()

                        # Route cache result to machine
                        if sidebar.entities:
                            self.sidebar_machine.send(
                                "cache_entities_found", entities_present=True
                            )
                        else:
                            self.sidebar_machine.send(
                                "cache_empty", entities_present=False
                            )
                        self._sync_machine_output()
                        await self._update_active_processing_async()
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
        self.sidebar_machine.send(
            "user_deleted_entity", entities_present=entities_present
        )
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

        self._apply_sidebar_context(inferred_enabled, status)

    async def _refresh_sidebar_context_async(self):
        """Update sidebar context without blocking the UI (async version)."""
        try:
            self.query_one("#entity-sidebar", EntitySidebar)
        except NoMatches:
            logger.debug("Sidebar not yet mounted; skipping context refresh")
            return

        try:
            # Parallelize Redis calls for faster response
            inferred_enabled, status = await asyncio.gather(
                asyncio.to_thread(get_inference_enabled),
                asyncio.to_thread(get_episode_status, self.episode_uuid, self.journal),
            )
        except Exception as exc:
            logger.debug("Failed to refresh sidebar context: %s", exc)
            return

        self._apply_sidebar_context(inferred_enabled, status)

    def _apply_sidebar_context(self, inferred_enabled: bool, status: str | None):
        """Apply inference and status context to sidebar state machine."""
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

    def _find_episode_index(self, episode_uuid: str) -> int | None:
        """Find index of episode in episodes list."""
        for i, ep in enumerate(self.episodes):
            if ep.get("uuid") == episode_uuid:
                return i
        return None

    def action_scroll_up(self) -> None:
        """Scroll markdown content up."""
        try:
            markdown = self.query_one("#journal-content", Markdown)
            markdown.scroll_up(animate=False)
        except NoMatches:
            pass

    def action_scroll_down(self) -> None:
        """Scroll markdown content down."""
        try:
            markdown = self.query_one("#journal-content", Markdown)
            markdown.scroll_down(animate=False)
        except NoMatches:
            pass

    def action_prev_entry(self) -> None:
        """Navigate to older entry (next index since list is newest-first)."""
        if not self.episodes or self._current_idx is None:
            return
        if self._current_idx >= len(self.episodes) - 1:
            return  # Already at oldest
        new_idx = self._current_idx + 1
        self._navigate_to_episode(new_idx)

    def action_next_entry(self) -> None:
        """Navigate to newer entry (prev index since list is newest-first)."""
        if not self.episodes or self._current_idx is None:
            return
        if self._current_idx <= 0:
            return  # Already at newest
        new_idx = self._current_idx - 1
        self._navigate_to_episode(new_idx)

    def _navigate_to_episode(self, new_idx: int) -> None:
        """Switch to a different episode in-place (no screen push)."""
        if new_idx < 0 or new_idx >= len(self.episodes):
            return

        new_episode = self.episodes[new_idx]
        new_uuid = new_episode.get("uuid")
        if not new_uuid:
            return

        # Update current episode tracking
        self._current_idx = new_idx
        self.episode_uuid = new_uuid

        # Reset sidebar state for new episode
        self.sidebar_machine = SidebarStateMachine(
            initial_status=None,
            inference_enabled=self.inference_enabled,
            entities_present=False,
            visible=self.sidebar_machine.output.visible,
        )
        self._sync_machine_output()

        # Cancel any pending workers
        self.workers.cancel_group(self, "status-poll")

        # Update sidebar episode reference
        try:
            sidebar = self.query_one("#entity-sidebar", EntitySidebar)
            sidebar.episode_uuid = new_uuid
            sidebar.entities = []
            sidebar.cache_loading = True
            if sidebar.display:
                sidebar.run_worker(sidebar.refresh_entities(), exclusive=True)
        except NoMatches:
            pass

        # Reload episode content
        self.run_worker(self.load_episode(), exclusive=True)
