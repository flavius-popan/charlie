import asyncio
import logging

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.css.query import NoMatches
from textual.widgets import Footer, Header, Label, ListItem, ListView, LoadingIndicator, Static

from backend.database import (
    delete_episode,
    ensure_database_ready,
    get_entry_entities,
    get_home_screen,
    get_period_entities,
    get_processing_status,
)
from backend.settings import DEFAULT_JOURNAL
from frontend.screens.settings_screen import SettingsScreen
from frontend.screens.view_screen import ViewScreen
from frontend.screens.log_screen import LogScreen
from frontend.screens.edit_screen import EditScreen
from frontend.utils import calculate_periods, get_display_title, group_entries_by_period

EMPTY_STATE_CAT = r"""
         /\_/\
        ( o.o )
         > ^ <

    No entries yet!
    Press 'n' to create your first entry
"""

logger = logging.getLogger("charlie")


class PeriodDivider(Static):
    """Non-selectable divider showing period label."""

    DEFAULT_CSS = """
    PeriodDivider {
        height: 1;
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self, label: str):
        super().__init__(f"── {label} ──")


class EntryLabel(Label):
    """Entry label with processing indicator gutter."""

    DEFAULT_CSS = """
    EntryLabel {
        padding-left: 2;
    }
    EntryLabel.processing {
        padding-left: 0;
    }
    """

    def __init__(self, text: str, episode_uuid: str):
        super().__init__(text)
        self.episode_uuid = episode_uuid
        self._text = text

    def set_processing(self, is_processing: bool) -> None:
        """Update the processing indicator."""
        if is_processing:
            self.update(f"● {self._text}")
            self.add_class("processing")
        else:
            self.update(self._text)
            self.remove_class("processing")


class HomeScreen(Screen):
    """Main screen showing list of journal entries."""

    DEFAULT_CSS = """
    HomeScreen #main-container {
        height: 100%;
    }

    HomeScreen #entries-pane {
        width: 2fr;
        border: solid $accent;
        border-title-align: left;
    }

    HomeScreen #context-stack {
        width: 1fr;
    }

    HomeScreen #connections-pane {
        height: 1fr;
        border: solid $accent;
        border-title-align: left;
    }

    HomeScreen #temporal-pane {
        height: 2fr;
        border: solid $accent;
        border-title-align: left;
    }

    HomeScreen #processing-pane {
        height: 1fr;
        border: solid $accent;
        border-title-align: left;
    }

    HomeScreen #episodes-list {
        height: 100%;
    }

    HomeScreen .pane-content {
        padding: 0 1;
        color: $text-muted;
    }

    HomeScreen .entity-list {
        height: 100%;
        scrollbar-size: 0 0;
    }

    HomeScreen .entity-list > ListItem {
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("n", "new_entry", "New", show=True),
        Binding("space", "view_entry", "View", show=True),
        Binding("d", "delete_entry", "Delete", show=True),
        Binding("s", "open_settings", "Settings", show=True),
        Binding("l", "open_logs", "Logs", show=True),
        Binding("q", "quit", "Quit", show=True),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("left", "navigate_period_older", "Older", show=False),
        Binding("right", "navigate_period_newer", "Newer", show=False),
    ]

    active_episode_uuid: reactive[str | None] = reactive(None)
    selected_entry_uuid: reactive[str | None] = reactive(None)
    entry_entities: reactive[list[dict]] = reactive([])
    selected_period_index: reactive[int] = reactive(0)
    period_stats: reactive[dict | None] = reactive(None)
    queue_count: reactive[int] = reactive(0)

    @property
    def connections_loading(self) -> bool:
        """True if selected entry is currently being processed."""
        return (
            self.selected_entry_uuid is not None
            and self.selected_entry_uuid == self.active_episode_uuid
        )

    def __init__(self):
        super().__init__()
        self.episodes = []
        self.periods: list[dict] = []
        self.list_index_to_episode: dict[int, int] = {}  # Maps list index to episode index

    def watch_active_episode_uuid(self, old_uuid: str | None, new_uuid: str | None) -> None:
        """Update processing indicator and refresh panes when processing completes."""
        # Update entry list indicators
        try:
            for entry_label in self.query(EntryLabel):
                entry_label.set_processing(entry_label.episode_uuid == new_uuid)
        except Exception as e:
            logger.debug(f"Failed to update processing indicators: {e}")

        # Refresh connections pane to update loading state (computed property changed)
        self._refresh_connections_pane()

        # If an entry just finished processing (old_uuid was set, now different)
        if old_uuid and old_uuid != new_uuid:
            # Refresh connections if the finished entry is selected
            if old_uuid == self.selected_entry_uuid:
                self.run_worker(
                    self._fetch_entry_entities(self.selected_entry_uuid),
                    exclusive=True,
                    group="entities",
                )

            # Refresh temporal pane if finished entry is in current period
            if self._is_episode_in_current_period(old_uuid):
                # Guard against concurrent period list modification
                if self.periods and 0 <= self.selected_period_index < len(self.periods):
                    period = self.periods[self.selected_period_index]
                    self.run_worker(
                        self._fetch_period_stats(period["start"], period["end"]),
                        exclusive=True,
                        group="period_stats",
                    )

    def watch_selected_entry_uuid(self, old_uuid: str | None, new_uuid: str | None) -> None:
        """Fetch entities when selected entry changes."""
        # Refresh connections pane immediately (loading state may have changed)
        self._refresh_connections_pane()

        if new_uuid:
            self.run_worker(self._fetch_entry_entities(new_uuid), exclusive=True, group="entities")
        else:
            self.entry_entities = []

    async def _fetch_entry_entities(self, episode_uuid: str) -> None:
        """Fetch entities for selected entry from cache."""
        from textual.worker import WorkerCancelled

        try:
            entities = await get_entry_entities(episode_uuid, DEFAULT_JOURNAL)
            self.entry_entities = entities
        except WorkerCancelled:
            raise  # Don't update state on cancellation
        except Exception as e:
            logger.debug(f"Failed to fetch entry entities: {e}")
            self.entry_entities = []

    def watch_entry_entities(self, old_entities: list[dict], new_entities: list[dict]) -> None:
        """Update Connections pane when entities change."""
        self._refresh_connections_pane()

    def _refresh_connections_pane(self) -> None:
        """Update Connections pane based on current state."""
        try:
            connections_pane = self.query_one("#connections-pane", Container)
            empty_msg = self.query_one("#connections-empty", Static)
            entity_list = self.query_one("#connections-list", ListView)

            # Remove all existing LoadingIndicators atomically
            for indicator in connections_pane.query(LoadingIndicator):
                indicator.remove()

            # Show loading indicator when selected entry is being processed
            if self.connections_loading and not self.entry_entities:
                empty_msg.display = False
                entity_list.display = False
                connections_pane.mount(LoadingIndicator())
                return

            if not self.entry_entities:
                empty_msg.update("No connections")
                empty_msg.display = True
                entity_list.display = False
            else:
                empty_msg.display = False
                entity_list.display = True
                entity_list.clear()
                for entity in self.entry_entities[:10]:
                    name = entity.get("name", "")
                    entity_list.append(ListItem(Label(name)))
        except Exception as e:
            logger.debug(f"Failed to update connections pane: {e}")

    def watch_selected_period_index(self, old_idx: int, new_idx: int) -> None:
        """Fetch period stats and update temporal pane when period changes."""
        if not self.periods or new_idx < 0 or new_idx >= len(self.periods):
            return

        period = self.periods[new_idx]

        # Update temporal pane title
        try:
            temporal_pane = self.query_one("#temporal-pane", Container)
            temporal_pane.border_title = period["label"]
        except Exception as e:
            logger.debug(f"Failed to update temporal pane title: {e}")

        # Fetch period stats
        self.run_worker(
            self._fetch_period_stats(period["start"], period["end"]),
            exclusive=True,
            group="period_stats",
        )

    async def _fetch_period_stats(self, start: "datetime", end: "datetime") -> None:
        """Fetch aggregate entities for the selected period."""
        from datetime import datetime  # noqa: F811
        from textual.worker import WorkerCancelled

        try:
            stats = await get_period_entities(start, end, DEFAULT_JOURNAL)
            self.period_stats = stats
        except WorkerCancelled:
            raise  # Don't update state on cancellation
        except Exception as e:
            logger.debug(f"Failed to fetch period stats: {e}")
            self.period_stats = None

    def watch_period_stats(self, old_stats: dict | None, new_stats: dict | None) -> None:
        """Update Temporal pane content when stats change."""
        try:
            empty_msg = self.query_one("#temporal-empty", Static)
            entity_list = self.query_one("#temporal-list", ListView)

            top_entities = new_stats.get("top_entities", []) if new_stats else []

            if not top_entities:
                empty_msg.update("No connections")
                empty_msg.display = True
                entity_list.display = False
            else:
                empty_msg.display = False
                entity_list.display = True
                entity_list.clear()
                for entity in top_entities[:25]:
                    name = entity.get("name", "")
                    entity_list.append(ListItem(Label(name)))
        except Exception as e:
            logger.debug(f"Failed to update temporal pane: {e}")

    def watch_queue_count(self, old_count: int, new_count: int) -> None:
        """Show/hide processing pane based on queue count."""
        try:
            processing_pane = self.query_one("#processing-pane", Container)
            temporal_pane = self.query_one("#temporal-pane", Container)

            if new_count > 0:
                processing_pane.display = True
                temporal_pane.styles.height = "1fr"
            else:
                processing_pane.display = False
                temporal_pane.styles.height = "2fr"
        except Exception as e:
            logger.debug(f"Failed to update processing pane visibility: {e}")

    def action_navigate_period_older(self) -> None:
        """Navigate to older time period (←)."""
        if not self.periods:
            return

        new_idx = self.selected_period_index + 1
        if new_idx < len(self.periods):
            self.selected_period_index = new_idx
            self._scroll_to_period(new_idx)

    def action_navigate_period_newer(self) -> None:
        """Navigate to newer time period (→)."""
        if not self.periods:
            return

        new_idx = self.selected_period_index - 1
        if new_idx >= 0:
            self.selected_period_index = new_idx
            self._scroll_to_period(new_idx)

    def _scroll_to_period(self, period_idx: int) -> None:
        """Scroll entry list to show selected period and select first entry."""
        if period_idx < 0 or period_idx >= len(self.periods):
            return

        period = self.periods[period_idx]
        first_episode_idx = period.get("first_episode_index", 0)

        # Find the list index for this episode
        for list_idx, ep_idx in self.list_index_to_episode.items():
            if ep_idx == first_episode_idx:
                try:
                    list_view = self.query_one("#episodes-list", ListView)
                    list_view.index = list_idx
                    # Update selected entry
                    if first_episode_idx < len(self.episodes):
                        self.selected_entry_uuid = self.episodes[first_episode_idx]["uuid"]
                except Exception as e:
                    logger.debug(f"Failed to scroll to period: {e}")
                break

    @staticmethod
    def _format_date(valid_at) -> str:
        """Format date as 'Day Mon DD' for list rendering."""
        try:
            return valid_at.strftime("%a %b %d")  # type: ignore[arg-type]
        except Exception:
            return str(valid_at)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False, icon="")

        empty = Static(EMPTY_STATE_CAT, id="empty-state")
        empty.can_focus = True
        empty.display = False
        yield empty

        entries_pane = Container(
            ListView(id="episodes-list"),
            id="entries-pane",
        )
        entries_pane.border_title = "Entries"

        connections_pane = Container(
            Static("Select an entry", classes="pane-content", id="connections-empty"),
            ListView(id="connections-list", classes="entity-list"),
            id="connections-pane",
        )
        connections_pane.border_title = "Connections"

        temporal_pane = Container(
            Static("", classes="pane-content", id="temporal-empty"),
            ListView(id="temporal-list", classes="entity-list"),
            id="temporal-pane",
        )
        temporal_pane.border_title = "This Week"

        processing_pane = Container(
            Static("Idle", classes="pane-content"),
            id="processing-pane",
        )
        processing_pane.border_title = "Processing"
        processing_pane.display = False

        context_stack = Vertical(
            connections_pane,
            temporal_pane,
            processing_pane,
            id="context-stack",
        )

        main_container = Horizontal(
            entries_pane,
            context_stack,
            id="main-container",
        )
        main_container.display = False
        yield main_container

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
            self._start_processing_poll()
        except Exception as e:
            logger.error(f"Database initialization failed: {e}", exc_info=True)
            self.notify("Failed to initialize database. Exiting...", severity="error")
            await asyncio.sleep(2)
            self.app.exit(1)

    async def on_screen_resume(self):
        """Called when returning to this screen."""
        await self.load_episodes()
        self._start_processing_poll()

    def on_screen_suspend(self) -> None:
        """Called when this screen is suspended (user navigates away)."""
        self.workers.cancel_group(self, "processing_poll")
        self.workers.cancel_group(self, "entities")
        self.workers.cancel_group(self, "period_stats")

    def _start_processing_poll(self) -> None:
        """Start the processing status polling worker.

        Worker lifecycle pattern:
        - Workers are started with group="processing_poll"
        - Cancel via self.workers.cancel_group(self, "processing_poll")
        - exclusive=True ensures only one worker runs at a time in the group
        - on_screen_suspend() cancels workers when navigating away
        - on_screen_resume() restarts workers when returning
        """
        self.run_worker(
            self._poll_processing_status(), exclusive=True, group="processing_poll"
        )

    async def _poll_processing_status(self) -> None:
        """Poll Redis for processing status updates."""
        from textual.worker import WorkerCancelled

        while True:
            try:
                status = await asyncio.to_thread(get_processing_status, DEFAULT_JOURNAL)
                new_uuid = status["active_uuid"]
                new_count = status["pending_count"]

                # Update reactive properties (triggers watchers)
                if new_uuid != self.active_episode_uuid:
                    self.active_episode_uuid = new_uuid
                if new_count != self.queue_count:
                    self.queue_count = new_count
                    self._update_processing_pane_content(new_uuid, new_count)

                # Poll faster during active processing, slower when idle
                poll_interval = 0.3 if new_count > 0 else 2.0
                await asyncio.sleep(poll_interval)
            except WorkerCancelled:
                logger.debug("Processing poll worker cancelled")
                break
            except Exception as e:
                logger.debug(f"Processing poll error: {e}")
                await asyncio.sleep(2.0)

    def _update_processing_pane_content(self, active_uuid: str | None, pending_count: int) -> None:
        """Update the processing pane content display."""
        try:
            pane_content = self.query_one("#processing-pane .pane-content", Static)
            if pending_count == 0:
                pane_content.update("Idle")
            elif active_uuid:
                # Find the episode name from our loaded episodes
                episode_name = None
                for ep in self.episodes:
                    if ep["uuid"] == active_uuid:
                        episode_name = get_display_title(ep)
                        break
                if episode_name:
                    pane_content.update(f"Extracting: {episode_name}\nQueue: {pending_count}")
                else:
                    pane_content.update(f"Extracting...\nQueue: {pending_count}")
            else:
                pane_content.update(f"Queue: {pending_count}")
        except Exception as e:
            logger.debug(f"Failed to update processing pane content: {e}")

    def _get_period_for_episode(self, episode_idx: int) -> int:
        """Find which period contains the given episode index."""
        period_idx = 0
        for i, period in enumerate(self.periods):
            if episode_idx >= period["first_episode_index"]:
                period_idx = i
            else:
                break
        return period_idx

    def _is_episode_in_current_period(self, episode_uuid: str) -> bool:
        """Check if an episode belongs to the currently selected period."""
        if not self.periods or self.selected_period_index >= len(self.periods):
            return False

        for idx, ep in enumerate(self.episodes):
            if ep["uuid"] == episode_uuid:
                ep_period_idx = self._get_period_for_episode(idx)
                return ep_period_idx == self.selected_period_index
        return False

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Handle ListView cursor movement (↑↓ navigation)."""
        if self.episodes and event.list_view.index is not None:
            episode_idx = self.list_index_to_episode.get(event.list_view.index)
            if episode_idx is not None:
                episode = self.episodes[episode_idx]
                self.selected_entry_uuid = episode["uuid"]

                # Update temporal pane if we crossed into a different period
                if self.periods:
                    new_period_idx = self._get_period_for_episode(episode_idx)
                    if new_period_idx != self.selected_period_index:
                        self.selected_period_index = new_period_idx

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle ListView selection (Enter key)."""
        if self.episodes and event.list_view.index is not None:
            episode_idx = self.list_index_to_episode.get(event.list_view.index)
            if episode_idx is not None:
                episode = self.episodes[episode_idx]
                self.app.push_screen(ViewScreen(episode["uuid"], DEFAULT_JOURNAL))

    async def load_episodes(self):
        try:
            new_episodes = await get_home_screen()
            list_view = self.query_one("#episodes-list", ListView)
            empty_state = self.query_one("#empty-state", Static)
            main_container = self.query_one("#main-container", Horizontal)

            with self.app.batch_update():
                if not new_episodes:
                    self.episodes = []
                    self.periods = []
                    self.list_index_to_episode = {}
                    await list_view.clear()
                    main_container.display = False
                    empty_state.display = True
                    empty_state.focus()
                else:
                    old_index = list_view.index if list_view.index is not None else 0
                    await list_view.clear()

                    grouped = group_entries_by_period(new_episodes)
                    items: list[ListItem] = []
                    self.list_index_to_episode = {}
                    episode_idx = 0

                    for period_label, period_episodes in grouped:
                        divider = ListItem(PeriodDivider(period_label), disabled=True)
                        items.append(divider)

                        for episode in period_episodes:
                            text = f"{self._format_date(episode['valid_at'])} · {get_display_title(episode)}"
                            entry_label = EntryLabel(text, episode["uuid"])
                            if episode["uuid"] == self.active_episode_uuid:
                                entry_label.set_processing(True)
                            item = ListItem(entry_label)
                            self.list_index_to_episode[len(items)] = episode_idx
                            items.append(item)
                            episode_idx += 1

                    await list_view.extend(items)
                    self.episodes = new_episodes
                    self.periods = calculate_periods(new_episodes)

                    empty_state.display = False
                    main_container.display = True

                    # Find first selectable index
                    new_index = 0
                    if self.list_index_to_episode:
                        first_selectable = min(self.list_index_to_episode.keys())
                        new_index = first_selectable
                        if old_index in self.list_index_to_episode:
                            new_index = old_index

                    list_view.index = new_index
                    list_view.focus()

                    # Set initial selected entry for Connections pane
                    episode_idx = self.list_index_to_episode.get(new_index)
                    if episode_idx is not None:
                        self.selected_entry_uuid = new_episodes[episode_idx]["uuid"]

                    # Set period based on selected entry's position
                    if self.periods and episode_idx is not None:
                        period_idx = self._get_period_for_episode(episode_idx)
                        self.selected_period_index = period_idx
                        period = self.periods[period_idx]
                        temporal_pane = self.query_one("#temporal-pane", Container)
                        temporal_pane.border_title = period["label"]
                        self.run_worker(
                            self._fetch_period_stats(period["start"], period["end"]),
                            exclusive=True,
                            group="period_stats",
                        )
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
                episode_idx = self.list_index_to_episode.get(list_view.index)
                if episode_idx is not None:
                    episode = self.episodes[episode_idx]
                    self.app.push_screen(ViewScreen(episode["uuid"], DEFAULT_JOURNAL))
        except Exception as e:
            logger.error(f"Failed to open view screen: {e}", exc_info=True)

    async def action_delete_entry(self):
        try:
            if not self.episodes:
                return
            list_view = self.query_one("#episodes-list", ListView)
            if list_view.index is not None:
                episode_idx = self.list_index_to_episode.get(list_view.index)
                if episode_idx is not None:
                    episode = self.episodes[episode_idx]
                    await delete_episode(episode["uuid"])
                    await self.load_episodes()
        except Exception as e:
            logger.error(f"Failed to delete entry: {e}", exc_info=True)
            self.notify("Failed to delete entry", severity="error")

    def action_quit(self):
        """Request graceful shutdown via unified async path (fire-and-forget)."""
        self.workers.cancel_group(self, "processing_poll")

        def handle_shutdown_done(task):
            try:
                exc = task.exception()
                if exc is not None:
                    logger.error(f"Shutdown failed: {exc}", exc_info=exc)
            except asyncio.CancelledError:
                pass

        task = asyncio.create_task(self.app._async_shutdown())
        task.add_done_callback(handle_shutdown_done)

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
