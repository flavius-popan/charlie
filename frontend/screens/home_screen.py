import asyncio
import logging

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.css.query import NoMatches
from textual.widgets import (
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    LoadingIndicator,
    Static,
)

from backend.database import (
    ensure_database_ready,
    get_entry_entities,
    get_episode_status,
    get_home_screen,
    get_home_screen_paginated,
    get_period_entities,
    get_processing_status,
)
from backend.settings import DEFAULT_JOURNAL
from frontend.screens.entity_browser_screen import EntityBrowserScreen
from frontend.screens.settings_screen import SettingsScreen
from frontend.screens.view_screen import ViewScreen
from frontend.screens.log_screen import LogScreen
from frontend.screens.edit_screen import EditScreen
from frontend.state.processing_state_machine import (
    ProcessingStateMachine,
    ProcessingOutput,
)
from frontend.utils import calculate_periods, get_display_title, group_entries_by_period
from frontend.widgets import EntityListItem, ProcessingDot

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
        self.label = label
        super().__init__(f"── {label} ──")


class EntryLabel(Horizontal):
    """Entry label with animated processing indicator."""

    DEFAULT_CSS = """
    EntryLabel {
        height: 1;
        width: 100%;
    }
    EntryLabel ProcessingDot {
        display: none;
    }
    EntryLabel.processing ProcessingDot {
        display: block;
    }
    EntryLabel > .entry-text {
        padding-left: 2;
    }
    EntryLabel.processing > .entry-text {
        padding-left: 0;
        text-style: bold;
    }
    """

    def __init__(self, text: str, episode_uuid: str):
        super().__init__()
        self.episode_uuid = episode_uuid
        self._text = text

    def compose(self) -> ComposeResult:
        yield ProcessingDot()
        yield Label(self._text, classes="entry-text")

    def set_processing(self, is_processing: bool) -> None:
        """Update the processing indicator via CSS class toggle."""
        if is_processing:
            self.add_class("processing")
        else:
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
        border-subtitle-align: left;
    }

    HomeScreen #processing-pane {
        height: 1fr;
        border: solid $accent;
        border-title-align: left;
    }

    HomeScreen #processing-status-line {
        height: 1;
        width: 100%;
        padding: 0 1;
    }

    HomeScreen #processing-dot {
        display: none;
    }

    HomeScreen #processing-pane.active #processing-dot {
        display: block;
    }

    HomeScreen #processing-status {
        height: 1;
    }

    HomeScreen #processing-entry {
        height: 1;
        padding: 0 1 0 3;
        color: $text;
    }

    HomeScreen #processing-queue {
        dock: bottom;
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }

    HomeScreen #episodes-list {
        height: 100%;
    }

    HomeScreen .pane-content {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }

    HomeScreen .entity-list {
        height: 100%;
        scrollbar-size: 0 0;
    }

    HomeScreen .entity-list > ListItem {
        height: 1;
        padding: 0 1;
    }

    HomeScreen .entity-list > ListItem > Label {
        padding: 0;
    }

    HomeScreen #episodes-list > ListItem.-highlight EntryLabel.processing .entry-text {
        color: $block-cursor-foreground;
    }

    HomeScreen #episodes-list:focus > ListItem.-highlight EntryLabel.processing .entry-text {
        color: $block-cursor-foreground;
    }

    HomeScreen #episodes-list > ListItem.-highlight ProcessingDot {
        color: $block-cursor-foreground;
    }

    HomeScreen #episodes-list:focus > ListItem.-highlight ProcessingDot {
        color: $block-cursor-foreground;
    }
    """

    BINDINGS = [
        Binding("n", "new_entry", "New", show=True),
        Binding("space", "view_entry", "View", show=True),
        Binding("s", "open_settings", "Settings", show=True),
        Binding("l", "open_logs", "Logs", show=True),
        Binding("q", "quit", "Quit", show=True),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("left", "navigate_period_older", "Older", show=False),
        Binding("right", "navigate_period_newer", "Newer", show=False),
        Binding("1", "focus_entries", "Entries", show=False),
        Binding("2", "focus_connections", "Connections", show=False),
        Binding("3", "focus_temporal", "Temporal", show=False),
    ]

    selected_entry_uuid: reactive[str | None] = reactive(None)
    selected_entry_status: reactive[str | None] = reactive(None)
    entry_entities: reactive[list[dict]] = reactive([])
    selected_period_index: reactive[int] = reactive(0)
    period_stats: reactive[dict | None] = reactive(None)
    processing_output: reactive[ProcessingOutput] = reactive(ProcessingOutput.idle())

    @property
    def connections_loading(self) -> bool:
        """True if selected entry is currently being inferred (not just loading)."""
        return (
            self.processing_output.is_inferring
            and self.selected_entry_uuid is not None
            and self.selected_entry_uuid == self.processing_output.active_episode_uuid
        )

    def __init__(self):
        super().__init__()
        self.episodes = []
        self.periods: list[dict] = []
        self.list_index_to_episode: dict[
            int, int
        ] = {}  # Maps list index to episode index
        # Track last selected index for each list view (for position memory on focus switch)
        self._last_entries_index: int | None = None
        self._last_connections_index: int | None = None
        self._last_temporal_index: int | None = None
        # UUID to select when returning from ViewScreen
        self._select_uuid_on_resume: str | None = None
        # Processing state machine
        self.processing_machine = ProcessingStateMachine()
        # Pagination state
        self._has_more: bool = True
        self._is_loading_more: bool = False
        self._current_offset: int = 0
        self._page_size: int = 100
        self._pagination_generation: int = 0  # Detect stale responses after reload
        self._last_period_label: str | None = None  # Avoid duplicate dividers at page boundary
        # O(1) UUID to episode index lookup for _is_episode_in_current_period
        self._uuid_to_episode_idx: dict[str, int] = {}

    def _update_entry_processing_dots(self, active_uuid: str | None) -> None:
        """Update processing dots on all entry labels.

        Args:
            active_uuid: The UUID of the entry being processed, or None when not inferring.
        """
        try:
            with self.app.batch_update():
                for entry_label in self.query(EntryLabel):
                    entry_label.set_processing(entry_label.episode_uuid == active_uuid)
        except Exception as e:
            logger.debug("Failed to update processing indicators: %s", e)

    def _resolve_entry_name(self, uuid: str | None) -> str:
        """Resolve episode UUID to display name."""
        if not uuid:
            return ""
        for ep in self.episodes:
            if ep["uuid"] == uuid:
                return get_display_title(ep)
        return ""  # Entry was deleted

    def watch_processing_output(
        self, old: ProcessingOutput, new: ProcessingOutput
    ) -> None:
        """Single watcher handles all processing pane updates."""
        self._update_processing_pane(new)
        self._update_entry_processing_dots(
            new.active_episode_uuid if new.is_inferring else None
        )

        # Refresh connections pane if processing state changed
        # (connections_loading depends on is_inferring and active_episode_uuid)
        if (
            old.active_episode_uuid != new.active_episode_uuid
            or old.is_inferring != new.is_inferring
        ):
            self._refresh_connections_pane()

        # Detect when an episode just finished processing
        # Track UUID change, not is_inferring transition (queue may skip idle state)
        finished_uuid = None
        if (
            old.active_episode_uuid
            and old.active_episode_uuid != new.active_episode_uuid
        ):
            finished_uuid = old.active_episode_uuid

        if finished_uuid:
            # Refresh connections if finished entry is selected
            if finished_uuid == self.selected_entry_uuid:
                self.run_worker(
                    self._fetch_entry_entities(finished_uuid),
                    exclusive=True,
                    group="entities",
                )

            # Refresh temporal pane if finished entry is in current period
            if self._is_episode_in_current_period(finished_uuid):
                if self.periods and 0 <= self.selected_period_index < len(self.periods):
                    period = self.periods[self.selected_period_index]
                    self.run_worker(
                        self._fetch_period_stats(period["start"], period["end"]),
                        exclusive=True,
                        group="period_stats",
                    )

    def watch_selected_entry_uuid(
        self, old_uuid: str | None, new_uuid: str | None
    ) -> None:
        """Fetch entities when selected entry changes."""
        # Refresh connections pane immediately (loading state may have changed)
        self._refresh_connections_pane()

        if new_uuid:
            self.run_worker(
                self._fetch_entry_entities(new_uuid), exclusive=True, group="entities"
            )
        else:
            self.selected_entry_status = None
            self.entry_entities = []

    async def _fetch_entry_entities(self, episode_uuid: str) -> None:
        """Fetch entities and processing status for selected entry."""
        from textual.worker import WorkerCancelled

        try:
            status = await asyncio.to_thread(
                get_episode_status, episode_uuid, DEFAULT_JOURNAL
            )
            entities = await get_entry_entities(episode_uuid, DEFAULT_JOURNAL)
            # Verify entry is still selected before updating (prevents race with rapid navigation)
            if self.selected_entry_uuid == episode_uuid:
                self.selected_entry_status = status
                self.entry_entities = entities
        except WorkerCancelled:
            raise  # Don't update state on cancellation
        except Exception as e:
            logger.debug("Failed to fetch entry entities: %s", e)
            self.selected_entry_status = None
            self.entry_entities = []

    def watch_entry_entities(
        self, old_entities: list[dict], new_entities: list[dict]
    ) -> None:
        """Update Connections pane when entities change."""
        self._refresh_connections_pane()

    def watch_selected_entry_status(
        self, old_status: str | None, new_status: str | None
    ) -> None:
        """Refresh connections pane when entry processing status changes."""
        self._refresh_connections_pane()

    def _refresh_connections_pane(self) -> None:
        """Update Connections pane based on current state."""
        try:
            connections_pane = self.query_one("#connections-pane", Container)
            empty_msg = self.query_one("#connections-empty", Static)
            entity_list = self.query_one("#connections-list", ListView)

            with self.app.batch_update():
                for indicator in connections_pane.query(LoadingIndicator):
                    indicator.remove()

                if self.connections_loading and not self.entry_entities:
                    empty_msg.display = False
                    entity_list.display = False
                    connections_pane.mount(LoadingIndicator())
                elif (
                    self.selected_entry_status == "pending_nodes"
                    and not self.entry_entities
                ):
                    empty_msg.update("Awaiting processing...")
                    empty_msg.display = True
                    entity_list.display = False
                elif not self.entry_entities:
                    empty_msg.update("No connections")
                    empty_msg.display = True
                    entity_list.display = False
                else:
                    empty_msg.display = False
                    entity_list.display = True
                    entity_list.clear()
                    self._last_connections_index = None
                    for entity in self.entry_entities:
                        name = entity.get("name", "")
                        uuid = entity.get("uuid", "")
                        entity_list.append(EntityListItem(name, uuid))
        except Exception as e:
            logger.debug("Failed to update connections pane: %s", e)

    def watch_selected_period_index(self, old_idx: int, new_idx: int) -> None:
        """Fetch period stats and update temporal pane when period changes."""
        if not self.periods or new_idx < 0 or new_idx >= len(self.periods):
            return

        period = self.periods[new_idx]

        # Update temporal pane title
        try:
            temporal_pane = self.query_one("#temporal-pane", Container)
            temporal_pane.border_title = f"{period['label']} [dim](3)[/dim]"
        except Exception as e:
            logger.debug("Failed to update temporal pane title: %s", e)

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
            # Verify period is still selected before updating (prevents race with rapid navigation)
            if (
                self.periods
                and 0 <= self.selected_period_index < len(self.periods)
                and self.periods[self.selected_period_index]["start"] == start
                and self.periods[self.selected_period_index]["end"] == end
            ):
                self.period_stats = stats
        except WorkerCancelled:
            raise  # Don't update state on cancellation
        except Exception as e:
            logger.debug("Failed to fetch period stats: %s", e)
            self.period_stats = None

    def watch_period_stats(
        self, old_stats: dict | None, new_stats: dict | None
    ) -> None:
        """Update Temporal pane content when stats change."""
        try:
            empty_msg = self.query_one("#temporal-empty", Static)
            entity_list = self.query_one("#temporal-list", ListView)

            top_entities = new_stats.get("top_entities", []) if new_stats else []

            with self.app.batch_update():
                if not top_entities:
                    empty_msg.update("No connections")
                    empty_msg.display = True
                    entity_list.display = False
                else:
                    empty_msg.display = False
                    entity_list.display = True
                    entity_list.clear()
                    self._last_temporal_index = None
                    for entity in top_entities[:100]:
                        name = entity.get("name", "")
                        uuid = entity.get("uuid", "")
                        entity_list.append(EntityListItem(name, uuid))
        except Exception as e:
            logger.debug("Failed to update temporal pane: %s", e)

    def _update_processing_pane(self, output: ProcessingOutput) -> None:
        """Update processing pane display based on machine output."""
        try:
            processing_pane = self.query_one("#processing-pane", Container)
            temporal_pane = self.query_one("#temporal-pane", Container)
            status_widget = self.query_one("#processing-status", Static)
            entry_widget = self.query_one("#processing-entry", Static)
            queue_widget = self.query_one("#processing-queue", Static)

            with self.app.batch_update():
                # Toggle pane visibility
                if output.pane_visible:
                    processing_pane.display = True
                    temporal_pane.styles.height = "1fr"
                else:
                    processing_pane.display = False
                    temporal_pane.styles.height = "2fr"

                # Toggle .active class for ProcessingDot visibility
                if output.show_dot:
                    processing_pane.add_class("active")
                else:
                    processing_pane.remove_class("active")

                # Status text from machine output
                status_widget.update(output.status_text)

                # Entry name resolved HERE, not in machine
                entry_name = (
                    self._resolve_entry_name(output.active_episode_uuid)
                    if output.is_inferring
                    else ""
                )
                entry_widget.update(entry_name)

                # Queue text from machine output
                queue_widget.update(output.queue_text)

        except Exception as e:
            logger.debug("Failed to update processing pane: %s", e)

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
                        self.selected_entry_uuid = self.episodes[first_episode_idx][
                            "uuid"
                        ]
                except Exception as e:
                    logger.debug("Failed to scroll to period: %s", e)
                break

    # Number of episodes from end to trigger loading more
    PAGINATION_THRESHOLD = 20

    def _check_load_more(self, episode_idx: int) -> None:
        """Check if we need to load more episodes based on cursor position."""
        if not self._has_more or self._is_loading_more:
            return

        if episode_idx >= len(self.episodes) - self.PAGINATION_THRESHOLD:
            # Set flag BEFORE spawning worker to prevent race condition
            self._is_loading_more = True
            self.run_worker(
                self._load_more_episodes(),
                exclusive=True,
                group="load_more",
            )

    async def _load_more_episodes(self) -> None:
        """Load more episodes when scrolling near end of list.

        Note: _is_loading_more is set by _check_load_more() before this worker starts.
        """
        from textual.worker import WorkerCancelled

        my_generation = self._pagination_generation

        try:
            new_episodes, has_more = await get_home_screen_paginated(
                limit=self._page_size,
                offset=self._current_offset,
            )

            # Stale check: discard if reload happened
            if my_generation != self._pagination_generation:
                return

            if not new_episodes:
                self._has_more = False
                return

            self._has_more = has_more
            self._current_offset += len(new_episodes)
            await self._append_episodes_to_list(new_episodes)

        except WorkerCancelled:
            raise
        except Exception as e:
            logger.error("Failed to load more episodes: %s", e, exc_info=True)
        finally:
            # Always clear flag to unblock future pagination
            self._is_loading_more = False

    async def _append_episodes_to_list(self, new_episodes: list[dict]) -> None:
        """Append new episodes to the existing list."""
        list_view = self.query_one("#episodes-list", ListView)
        grouped = group_entries_by_period(new_episodes)

        # Derive last period label from existing list to avoid duplicate dividers
        last_label = self._last_period_label
        if last_label is None:
            for child in reversed(list_view.children):
                if isinstance(child, ListItem) and child.children:
                    divider = child.children[0]
                    if isinstance(divider, PeriodDivider):
                        last_label = divider.label
                        break

        items: list[ListItem] = []
        current_list_idx = len(list_view.children)  # Start index for new items
        episode_offset = len(self.episodes)  # Start index in episodes list

        with self.app.batch_update():
            for period_label, period_episodes in grouped:
                # Only add divider if NEW period
                if period_label != last_label:
                    items.append(ListItem(PeriodDivider(period_label), disabled=True))
                    current_list_idx += 1
                    last_label = period_label

                for episode in period_episodes:
                    date_str = self._format_date(episode["valid_at"])
                    preview = get_display_title(episode)
                    text = f"[bold]{date_str}[/bold]  {preview}"
                    entry_label = EntryLabel(text, episode["uuid"])

                    # Check if this episode is currently being processed
                    if (
                        self.processing_output.is_inferring
                        and episode["uuid"] == self.processing_output.active_episode_uuid
                    ):
                        entry_label.set_processing(True)

                    item = ListItem(entry_label)
                    self.list_index_to_episode[current_list_idx] = episode_offset
                    self._uuid_to_episode_idx[episode["uuid"]] = episode_offset
                    items.append(item)
                    current_list_idx += 1
                    episode_offset += 1

            await list_view.extend(items)

            # Update data model
            self.episodes.extend(new_episodes)
            self.periods = calculate_periods(self.episodes)
            self._last_period_label = last_label

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
        entries_pane.border_title = "Entries [dim](1)[/dim]"

        connections_pane = Container(
            Static("Select an entry", classes="pane-content", id="connections-empty"),
            ListView(id="connections-list", classes="entity-list"),
            id="connections-pane",
        )
        connections_pane.border_title = "Connections [dim](2)[/dim]"

        temporal_pane = Container(
            Static("", classes="pane-content", id="temporal-empty"),
            ListView(id="temporal-list", classes="entity-list"),
            id="temporal-pane",
        )
        temporal_pane.border_title = "This Week [dim](3)[/dim]"
        temporal_pane.border_subtitle = "[dim]← older | newer →[/dim]"

        processing_pane = Container(
            Horizontal(
                ProcessingDot(id="processing-dot"),
                Static("", id="processing-status"),
                id="processing-status-line",
            ),
            Static("", id="processing-entry"),
            Static("", id="processing-queue"),
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
        from backend.database.redis_ops import clear_transient_state, migrate_dead_episodes

        try:
            await ensure_database_ready(DEFAULT_JOURNAL)
            # Clear any stale state from previous crashes before starting worker
            await asyncio.to_thread(clear_transient_state)
            await asyncio.to_thread(migrate_dead_episodes, DEFAULT_JOURNAL)
            # Start Huey worker after database is ready to avoid startup races
            if hasattr(self.app, "_ensure_huey_worker_running"):
                await asyncio.to_thread(self.app._ensure_huey_worker_running)
            await self.load_episodes()
            self._start_processing_poll()
        except Exception as e:
            logger.error("Database initialization failed: %s", e, exc_info=True)
            self.notify("Failed to initialize database. Exiting...", severity="error")
            await asyncio.sleep(2)
            self.app.exit(1)

    async def on_screen_resume(self):
        """Called when returning to this screen."""
        # Cancel any stale workers first to prevent duplicates on rapid navigation
        self.workers.cancel_group(self, "processing_poll")
        self.workers.cancel_group(self, "entities")
        self.workers.cancel_group(self, "period_stats")
        self.workers.cancel_group(self, "load_more")

        # Clear selected entry so watcher fires when load_episodes sets it again.
        # This ensures entities are re-fetched (the previous fetch may have been
        # cancelled by on_screen_suspend).
        self.selected_entry_uuid = None

        await self.load_episodes()
        self._start_processing_poll()

    def on_screen_suspend(self) -> None:
        """Called when this screen is suspended (user navigates away)."""
        self.workers.cancel_group(self, "processing_poll")
        self.workers.cancel_group(self, "entities")
        self.workers.cancel_group(self, "period_stats")
        self.workers.cancel_group(self, "load_more")
        # Clear saved positions since data may change while suspended
        self._last_entries_index = None
        self._last_connections_index = None
        self._last_temporal_index = None

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
                output = self.processing_machine.apply_status(status)
                self.processing_output = output
                await asyncio.sleep(output.poll_interval)
            except WorkerCancelled:
                logger.debug("Processing poll worker cancelled")
                break
            except Exception as e:
                logger.debug("Processing poll error: %s", e)
                await asyncio.sleep(2.0)

    def _save_current_list_positions(self) -> None:
        """Save current index for all list views before switching focus."""
        try:
            entries = self.query_one("#episodes-list", ListView)
            if entries.index is not None:
                self._last_entries_index = entries.index
        except NoMatches:
            pass

        try:
            connections = self.query_one("#connections-list", ListView)
            if connections.index is not None:
                self._last_connections_index = connections.index
        except NoMatches:
            pass

        try:
            temporal = self.query_one("#temporal-list", ListView)
            if temporal.index is not None:
                self._last_temporal_index = temporal.index
        except NoMatches:
            pass

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

        # O(1) lookup using UUID-to-index map
        idx = self._uuid_to_episode_idx.get(episode_uuid)
        if idx is None:
            return False
        ep_period_idx = self._get_period_for_episode(idx)
        return ep_period_idx == self.selected_period_index

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Handle ListView cursor movement (↑↓ navigation)."""
        # Only respond to entries list events, not connections/temporal lists
        if event.list_view.id != "episodes-list":
            return
        if self.episodes and event.list_view.index is not None:
            episode_idx = self.list_index_to_episode.get(event.list_view.index)
            if episode_idx is not None and 0 <= episode_idx < len(self.episodes):
                episode = self.episodes[episode_idx]
                self.selected_entry_uuid = episode["uuid"]

                # Update temporal pane if we crossed into a different period
                if self.periods:
                    new_period_idx = self._get_period_for_episode(episode_idx)
                    if new_period_idx != self.selected_period_index:
                        self.selected_period_index = new_period_idx

                # Check if we need to load more episodes
                self._check_load_more(episode_idx)

    def _handle_view_screen_result(self, episode_uuid: str | None) -> None:
        """Store episode UUID to select when screen resumes."""
        self._select_uuid_on_resume = episode_uuid

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle ListView selection (Enter key)."""
        # Handle entries list
        if event.list_view.id == "episodes-list":
            if self.episodes and event.list_view.index is not None:
                episode_idx = self.list_index_to_episode.get(event.list_view.index)
                if episode_idx is not None and 0 <= episode_idx < len(self.episodes):
                    episode = self.episodes[episode_idx]
                    self.app.push_screen(
                        ViewScreen(episode["uuid"], DEFAULT_JOURNAL, episodes=self.episodes),
                        self._handle_view_screen_result,
                    )
        # Handle connections list - navigate to entity browser
        elif event.list_view.id == "connections-list":
            if isinstance(event.item, EntityListItem) and event.item.entity_uuid:
                self.app.push_screen(EntityBrowserScreen(event.item.entity_uuid, DEFAULT_JOURNAL))
        # Handle temporal list - navigate to entity browser
        elif event.list_view.id == "temporal-list":
            if isinstance(event.item, EntityListItem) and event.item.entity_uuid:
                self.app.push_screen(EntityBrowserScreen(event.item.entity_uuid, DEFAULT_JOURNAL))

    async def load_episodes(self):
        # Reset pagination state at start
        self._current_offset = 0
        self._has_more = True
        self._is_loading_more = False
        self._pagination_generation += 1  # Invalidate in-flight requests
        self._last_period_label = None

        try:
            new_episodes, has_more = await get_home_screen_paginated(
                limit=self._page_size, offset=0
            )
            self._has_more = has_more
            self._current_offset = len(new_episodes)

            list_view = self.query_one("#episodes-list", ListView)
            empty_state = self.query_one("#empty-state", Static)
            main_container = self.query_one("#main-container", Horizontal)

            with self.app.batch_update():
                if not new_episodes:
                    self.episodes = []
                    self.periods = []
                    self.list_index_to_episode = {}
                    self._uuid_to_episode_idx = {}
                    await list_view.clear()
                    self._last_entries_index = None
                    main_container.display = False
                    empty_state.display = True
                    empty_state.focus()
                else:
                    old_index = list_view.index if list_view.index is not None else 0
                    await list_view.clear()
                    self._last_entries_index = None

                    grouped = group_entries_by_period(new_episodes)
                    items: list[ListItem] = []
                    self.list_index_to_episode = {}
                    self._uuid_to_episode_idx = {}
                    episode_idx = 0

                    for period_label, period_episodes in grouped:
                        divider = ListItem(PeriodDivider(period_label), disabled=True)
                        items.append(divider)
                        self._last_period_label = period_label

                        for episode in period_episodes:
                            date_str = self._format_date(episode["valid_at"])
                            preview = get_display_title(episode)
                            text = f"[bold]{date_str}[/bold]  {preview}"
                            entry_label = EntryLabel(text, episode["uuid"])
                            if (
                                self.processing_output.is_inferring
                                and episode["uuid"]
                                == self.processing_output.active_episode_uuid
                            ):
                                entry_label.set_processing(True)
                            item = ListItem(entry_label)
                            self.list_index_to_episode[len(items)] = episode_idx
                            self._uuid_to_episode_idx[episode["uuid"]] = episode_idx
                            items.append(item)
                            episode_idx += 1

                    await list_view.extend(items)
                    self.episodes = new_episodes
                    self.periods = calculate_periods(new_episodes)

                    empty_state.display = False
                    main_container.display = True

                    # Find index to select
                    new_index = 0
                    if self.list_index_to_episode:
                        first_selectable = min(self.list_index_to_episode.keys())
                        new_index = first_selectable

                        # Priority: stored UUID from ViewScreen > previous selection
                        if self._select_uuid_on_resume:
                            for list_idx, ep_idx in self.list_index_to_episode.items():
                                if new_episodes[ep_idx]["uuid"] == self._select_uuid_on_resume:
                                    new_index = list_idx
                                    break
                            self._select_uuid_on_resume = None
                        elif old_index in self.list_index_to_episode:
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
                        temporal_pane.border_title = f"{period['label']} [dim](3)[/dim]"
                        self.run_worker(
                            self._fetch_period_stats(period["start"], period["end"]),
                            exclusive=True,
                            group="period_stats",
                        )
        except Exception as e:
            logger.error("Failed to load episodes: %s", e, exc_info=True)
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
                if episode_idx is not None and 0 <= episode_idx < len(self.episodes):
                    episode = self.episodes[episode_idx]
                    self.app.push_screen(
                        ViewScreen(episode["uuid"], DEFAULT_JOURNAL, episodes=self.episodes),
                        self._handle_view_screen_result,
                    )
        except Exception as e:
            logger.error("Failed to open view screen: %s", e, exc_info=True)

    def action_quit(self):
        """Request graceful shutdown via unified async path (fire-and-forget)."""
        self.workers.cancel_group(self, "processing_poll")

        def handle_shutdown_done(task):
            try:
                exc = task.exception()
                if exc is not None:
                    logger.error("Shutdown failed: %s", exc, exc_info=exc)
            except asyncio.CancelledError:
                pass

        task = asyncio.create_task(self.app._async_shutdown())
        task.add_done_callback(handle_shutdown_done)

    def action_cursor_down(self):
        """Move cursor down in the focused ListView."""
        focused = self.app.focused
        if isinstance(focused, ListView):
            focused.action_cursor_down()
        elif self.episodes:
            # Fallback to episodes list if no ListView focused
            try:
                list_view = self.query_one("#episodes-list", ListView)
                list_view.action_cursor_down()
            except Exception as e:
                logger.debug("cursor_down failed: %s", e)

    def action_cursor_up(self):
        """Move cursor up in the focused ListView."""
        focused = self.app.focused
        if isinstance(focused, ListView):
            focused.action_cursor_up()
        elif self.episodes:
            # Fallback to episodes list if no ListView focused
            try:
                list_view = self.query_one("#episodes-list", ListView)
                list_view.action_cursor_up()
            except Exception as e:
                logger.debug("cursor_up failed: %s", e)

    def action_focus_entries(self) -> None:
        """Focus entries list (1) and restore position."""
        try:
            self._save_current_list_positions()
            list_view = self.query_one("#episodes-list", ListView)
            list_view.focus()
            if self._last_entries_index is not None:
                # Validate index is within current bounds
                if 0 <= self._last_entries_index < len(list_view.children):
                    list_view.index = self._last_entries_index
            elif len(list_view.children) > 0:
                # Select first item if no saved position
                list_view.index = 0
        except Exception as e:
            logger.debug("focus_entries failed: %s", e)

    def action_focus_connections(self) -> None:
        """Focus connections list (2) and restore position."""
        try:
            self._save_current_list_positions()
            entity_list = self.query_one("#connections-list", ListView)
            # Only focus if list is visible and has items
            if entity_list.display and len(entity_list.children) > 0:
                entity_list.focus()
                if self._last_connections_index is not None:
                    if 0 <= self._last_connections_index < len(entity_list.children):
                        entity_list.index = self._last_connections_index
                else:
                    # Select first item if no saved position
                    entity_list.index = 0
        except Exception as e:
            logger.debug("focus_connections failed: %s", e)

    def action_focus_temporal(self) -> None:
        """Focus temporal list (3) and restore position."""
        try:
            self._save_current_list_positions()
            entity_list = self.query_one("#temporal-list", ListView)
            # Only focus if list is visible and has items
            if entity_list.display and len(entity_list.children) > 0:
                entity_list.focus()
                if self._last_temporal_index is not None:
                    if 0 <= self._last_temporal_index < len(entity_list.children):
                        entity_list.index = self._last_temporal_index
                else:
                    # Select first item if no saved position
                    entity_list.index = 0
        except Exception as e:
            logger.debug("focus_temporal failed: %s", e)

    def action_open_settings(self):
        self.app.push_screen(SettingsScreen())

    def action_open_logs(self):
        self.app.push_screen(LogScreen())
