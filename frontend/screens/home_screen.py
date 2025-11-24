import asyncio
import logging

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, ListItem, ListView, Static

from backend.database import (
    delete_episode,
    ensure_database_ready,
    get_all_episodes,
    shutdown_database,
)
from backend.settings import DEFAULT_JOURNAL
from backend.services.queue import start_huey_consumer, stop_huey_consumer
from frontend.screens.settings_screen import SettingsScreen
from frontend.screens.view_screen import ViewScreen
from frontend.screens.log_screen import LogScreen
from frontend.screens.edit_screen import EditScreen
from frontend.utils import get_display_title

EMPTY_STATE_CAT = r"""
         /\_/\
        ( o.o )
         > ^ <

    No entries yet!
    Press 'n' to create your first entry
"""

logger = logging.getLogger("charlie")


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
                await asyncio.to_thread(self.app._ensure_huey_worker_running)
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
            self.app.push_screen(ViewScreen(episode["uuid"], DEFAULT_JOURNAL))

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
                self.app.push_screen(ViewScreen(episode["uuid"], DEFAULT_JOURNAL))
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
