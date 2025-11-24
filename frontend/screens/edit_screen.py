import asyncio
import logging

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Header, TextArea

from backend import add_journal_entry
from backend.database import get_episode, update_episode
from backend.database.redis_ops import (
    get_episode_status,
    get_inference_enabled,
    redis_ops,
)
from backend.settings import DEFAULT_JOURNAL
from frontend.utils import extract_title

logger = logging.getLogger("charlie")


class EditScreen(Screen):
    """Screen for creating or editing a journal entry.

    WARNING: Do NOT use recompose() in this screen - TextArea widget
    has internal state (cursor position, undo history) that would be lost.
    UI thread rule: never add blocking I/O here; offload with run_worker(thread=True)
    or asyncio.to_thread.
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

    @staticmethod
    def _set_editing_presence() -> None:
        """Mark editing as active (non-blocking via worker)."""
        try:
            with redis_ops() as r:
                r.set("editing:active", "active")
        except Exception as e:
            logger.debug("Failed to set editing presence key: %s", e)

    @staticmethod
    def _clear_editing_presence() -> None:
        """Clear editing presence flag (non-blocking via worker)."""
        try:
            with redis_ops() as r:
                r.delete("editing:active")
        except Exception as e:
            logger.debug("Failed to delete editing presence key: %s", e)

    def _schedule_set_editing_presence(self) -> None:
        try:
            self.run_worker(
                self._set_editing_presence,
                exclusive=True,
                name="editing-presence-set",
                thread=True,
            )
        except Exception as e:
            logger.debug("Failed to schedule editing presence set: %s", e)

    def _schedule_clear_editing_presence(self) -> None:
        try:
            self.run_worker(
                self._clear_editing_presence,
                exclusive=True,
                name="editing-presence-clear",
                thread=True,
            )
        except Exception as e:
            logger.debug("Failed to schedule editing presence clear: %s", e)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False, icon="")
        yield TextArea("", id="editor")
        yield Footer()

    async def on_mount(self):
        if not self.is_new_entry:
            await self.load_episode()

        # Fire-and-forget: mark editing active without blocking UI thread.
        self._schedule_set_editing_presence()

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

    def _delete_editing_key(self) -> None:
        """Delete Redis editing presence key."""
        self._schedule_clear_editing_presence()

    def on_unmount(self) -> None:
        """Delete Redis key when screen is unmounted."""
        self._delete_editing_key()

    async def action_save_and_return(self):
        await self.save_entry()

    async def save_entry(self):
        try:
            editor = self.query_one("#editor", TextArea)
            content = editor.text

            # Clear editing presence as soon as a save is initiated (non-blocking).
            self._schedule_clear_editing_presence()

            if not content.strip():
                self.app.pop_screen()
                return

            title = extract_title(content)

            if self.is_new_entry:
                from frontend.screens.view_screen import ViewScreen
                uuid = await add_journal_entry(content=content)
                if title:
                    await update_episode(uuid, name=title)
                inference_enabled = await asyncio.to_thread(get_inference_enabled)

                # Determine if connections pane should be shown
                status = await asyncio.to_thread(get_episode_status, uuid, DEFAULT_JOURNAL)
                show_connections = (
                    inference_enabled and status in ("pending_nodes", "pending_edges")
                )

                # Navigate to ViewScreen (atomic screen replacement)
                self.app.switch_screen(
                    ViewScreen(
                        uuid,
                        DEFAULT_JOURNAL,
                        from_edit=show_connections,
                        inference_enabled=inference_enabled,
                        status=status,
                        active_processing=False,
                    )
                )
                # THEN enqueue extraction task in background
                if inference_enabled:
                    self._enqueue_extraction_task(uuid, DEFAULT_JOURNAL)
            else:
                # Update episode and check if content changed
                if title:
                    content_changed = await update_episode(
                        self.episode_uuid, content=content, name=title
                    )
                else:
                    content_changed = await update_episode(
                        self.episode_uuid, content=content
                    )
                inference_enabled = await asyncio.to_thread(get_inference_enabled)

                status = await asyncio.to_thread(
                    get_episode_status, self.episode_uuid, DEFAULT_JOURNAL
                )
                show_connections = (
                    inference_enabled
                    and content_changed
                    and status in ("pending_nodes", "pending_edges")
                )
                # Remove the previous ViewScreen (if present) before showing the fresh one
                from frontend.screens.view_screen import ViewScreen
                if len(self.app.screen_stack) >= 2 and isinstance(
                    self.app.screen_stack[-2], ViewScreen
                ):
                    # pop EditScreen
                    self.app.pop_screen()
                    # pop the prior ViewScreen
                    self.app.pop_screen()
                else:
                    # pop EditScreen
                    self.app.pop_screen()

                # Push a fresh ViewScreen reflecting the updated episode state
                self.app.push_screen(
                    ViewScreen(
                        self.episode_uuid,
                        DEFAULT_JOURNAL,
                        from_edit=show_connections,
                        inference_enabled=inference_enabled,
                        status=status,
                        active_processing=False,
                    )
                )

                # THEN enqueue extraction task in background if content changed
                if content_changed and inference_enabled:
                    self._enqueue_extraction_task(self.episode_uuid, DEFAULT_JOURNAL)

        except Exception as e:
            logger.error(f"Failed to save entry: {e}", exc_info=True)
            self.notify("Failed to save entry", severity="error")
            raise

    def _enqueue_extraction_task(self, episode_uuid: str, journal: str):
        """Enqueue node extraction task in background (non-blocking)."""
        from backend.database.redis_ops import get_inference_enabled
        from frontend.widgets.entity_sidebar import EntitySidebar

        if get_inference_enabled():
            try:
                from backend.services.tasks import extract_nodes_task

                extract_nodes_task(episode_uuid, journal, priority=1)
                self.status = "pending_nodes"
                try:
                    sidebar = self.app.query_one("#entity-sidebar", EntitySidebar)
                    sidebar.status = "pending_nodes"
                except Exception:
                    pass
            except Exception as exc:
                logger.warning(
                    "Failed to enqueue extract_nodes_task for %s: %s", episode_uuid, exc
                )
