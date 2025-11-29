"""Entity Browser screen for viewing entity details and memories."""

import asyncio
import logging

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Footer, Header, ListItem, ListView, Markdown, Static

from backend.database import get_entity_browser_data, render_sparkline
from backend.database.queries import extract_entity_sentences, ENTITY_QUOTE_TARGET_LENGTH
from backend.settings import DEFAULT_JOURNAL

logger = logging.getLogger("charlie")


class DateListItem(ListItem):
    """Non-selectable list item showing just the date."""

    SCOPED_CSS = False
    DEFAULT_CSS = """
    DateListItem {
        height: 1;
        padding: 0 0 0 2;
    }

    DateListItem .date-text {
        color: $text-muted;
        text-style: bold;
    }
    """

    def __init__(self, date_str: str):
        super().__init__()
        self.date_str = date_str
        self.disabled = True

    def compose(self) -> ComposeResult:
        yield Static(self.date_str, classes="date-text")


class QuoteListItem(ListItem):
    """List item showing centered quote text."""

    SCOPED_CSS = False
    DEFAULT_CSS = """
    QuoteListItem {
        height: auto;
        min-height: 1;
        padding: 1 4;
    }

    QuoteListItem .quote-text {
        width: 100%;
        text-align: left;
    }
    """

    def __init__(self, quote_text: str, episode_uuid: str):
        super().__init__()
        self.quote_text = quote_text
        self.episode_uuid = episode_uuid

    def compose(self) -> ComposeResult:
        yield Static(self.quote_text, classes="quote-text")


class ConnectionChip(Static):
    """Clickable chip for navigating to connected entities."""

    DEFAULT_CSS = """
    ConnectionChip {
        width: auto;
        background: $panel;
        padding: 0 1;
        margin: 0 1 0 0;
    }

    ConnectionChip:hover {
        background: $accent;
    }
    """

    def __init__(self, name: str, entity_uuid: str, journal: str):
        super().__init__(name)
        self.entity_uuid = entity_uuid
        self.journal = journal

    def on_click(self) -> None:
        self.app.push_screen(EntityBrowserScreen(self.entity_uuid, self.journal))


class EntityBrowserScreen(Screen):
    """Screen for browsing entity details and memories."""

    BINDINGS = [
        Binding("escape", "back", "Back", show=True),
        Binding("enter", "read_entry", "Read", show=True),
        Binding("h", "go_home", "Home", show=True),
        Binding("q", "back", "Back", show=False),
    ]

    DEFAULT_CSS = """
    EntityBrowserScreen {
        background: $surface;
        layout: vertical;
    }

    EntityBrowserScreen #header-container {
        height: auto;
        padding: 1 2;
    }

    EntityBrowserScreen #entity-name {
        text-align: center;
        text-style: bold;
        width: 100%;
        margin-bottom: 1;
    }

    EntityBrowserScreen #sparkline-row {
        height: 1;
        align: center middle;
    }

    EntityBrowserScreen #start-date {
        width: auto;
        color: $text-muted;
        margin-right: 1;
    }

    EntityBrowserScreen #sparkline {
        width: auto;
    }

    EntityBrowserScreen #end-date {
        width: auto;
        color: $text-muted;
        margin-left: 1;
    }

    EntityBrowserScreen #body-container {
        height: 1fr;
        width: 100%;
    }

    EntityBrowserScreen #content-area {
        height: 1fr;
        width: 100%;
    }

    EntityBrowserScreen #quotes-container {
        height: 1fr;
        width: 100%;
    }

    EntityBrowserScreen #quotes-list {
        height: 100%;
        width: 100%;
        scrollbar-size: 0 0;
    }

    EntityBrowserScreen #connections-footer {
        height: 3;
        width: 100%;
        padding: 0 2;
        background: $panel;
    }

    EntityBrowserScreen #connections-label {
        width: 100%;
        text-align: center;
        color: $text-muted;
    }

    EntityBrowserScreen #connections-chips {
        width: 100%;
        align: center middle;
    }

    EntityBrowserScreen ConnectionChip {
        margin: 0 1 0 0;
    }

    /* Reader panel (hidden by default) */
    EntityBrowserScreen #reader-panel {
        display: none;
        width: 0;
        height: 1fr;
        padding: 1 2;
        scrollbar-size: 0 0;
    }

    EntityBrowserScreen #reader-date {
        color: $text-muted;
        text-align: center;
    }

    /* Reader mode - wide: 50/50 split */
    EntityBrowserScreen.reader-open #quotes-container {
        width: 1fr;
    }

    EntityBrowserScreen.reader-open #reader-panel {
        display: initial;
        width: 1fr;
    }

    EntityBrowserScreen.reader-open #connections-footer {
        display: none;
    }

    /* Reader mode - narrow: hide quotes, full-width reader */
    EntityBrowserScreen.reader-open.narrow #quotes-container {
        display: none;
    }

    EntityBrowserScreen.reader-open.narrow #reader-panel {
        width: 100%;
    }

    EntityBrowserScreen #loading {
        width: 100%;
        height: 100%;
        content-align: center middle;
    }
    """

    entity_data: reactive[dict | None] = reactive(None)
    reader_open: reactive[bool] = reactive(False)
    selected_episode_uuid: reactive[str | None] = reactive(None)

    def __init__(self, entity_uuid: str, journal: str = DEFAULT_JOURNAL):
        super().__init__()
        self.entity_uuid = entity_uuid
        self.journal = journal

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False, icon="")

        # Loading state
        yield Static("Loading...", id="loading")

        # Header container (frozen)
        header = Container(
            Static("", id="entity-name"),
            Horizontal(
                Static("", id="start-date"),
                Static("", id="sparkline"),
                Static("", id="end-date"),
                id="sparkline-row",
            ),
            id="header-container",
        )
        header.display = False
        yield header

        # Main body: content area (quotes + reader) + connections footer
        body = Vertical(
            Horizontal(
                Container(
                    ListView(id="quotes-list"),
                    id="quotes-container",
                ),
                VerticalScroll(
                    Static("", id="reader-date"),
                    Markdown("", id="reader-content"),
                    id="reader-panel",
                ),
                id="content-area",
            ),
            Container(
                Static("Also appears with", id="connections-label"),
                Horizontal(id="connections-chips"),
                id="connections-footer",
            ),
            id="body-container",
        )
        body.display = False
        yield body

        yield Footer()

    async def on_mount(self) -> None:
        self.run_worker(self._load_entity_data(), exclusive=True, group="entity-data")

    async def on_screen_resume(self) -> None:
        self.run_worker(self._load_entity_data(), exclusive=True, group="entity-data")

    def on_screen_suspend(self) -> None:
        self.workers.cancel_group(self, "entity-data")

    async def _load_entity_data(self) -> None:
        try:
            data = await get_entity_browser_data(self.entity_uuid, self.journal)
            self.entity_data = data
        except Exception as e:
            logger.error("Failed to load entity data: %s", e, exc_info=True)
            self.notify("Error loading entity", severity="error")
            self.app.pop_screen()

    def watch_entity_data(self, old: dict | None, new: dict | None) -> None:
        if new is None:
            return

        try:
            loading = self.query_one("#loading", Static)
            header = self.query_one("#header-container", Container)
            body = self.query_one("#body-container", Vertical)

            with self.app.batch_update():
                loading.display = False
                header.display = True
                body.display = True

                # Update header
                entity = new["entity"]
                entity_name = entity["name"]
                name_widget = self.query_one("#entity-name", Static)
                name_widget.update(entity_name)

                # Update sparkline
                sparkline_str = render_sparkline(new["sparkline_data"])
                sparkline_widget = self.query_one("#sparkline", Static)
                sparkline_widget.update(sparkline_str)

                # Update dates
                first = entity.get("first_mention")
                last = entity.get("last_mention")
                start_widget = self.query_one("#start-date", Static)
                end_widget = self.query_one("#end-date", Static)

                if first and last:
                    start_widget.update(first.strftime("%b %Y"))
                    end_widget.update(last.strftime("%b %Y"))
                else:
                    start_widget.update("")
                    end_widget.update("")

                # Populate quotes list with entity-specific sentences
                quotes_list = self.query_one("#quotes-list", ListView)
                quotes_list.clear()

                prev_date = None
                for entry in new["entries"]:
                    valid_at = entry["valid_at"]
                    date = valid_at.date() if hasattr(valid_at, "date") else valid_at

                    # Extract all sentences mentioning the entity
                    sentences = extract_entity_sentences(entry["content"], entity_name)
                    if not sentences:
                        continue

                    # Add date item when date changes
                    is_new_date = date != prev_date
                    if is_new_date:
                        date_str = (
                            valid_at.strftime("%b %-d")
                            if hasattr(valid_at, "strftime")
                            else str(date)
                        )
                        prev_date = date
                        quotes_list.append(DateListItem(date_str))

                    # Create quote items for each sentence
                    # Pad to consistent length (content + 2 for quotes)
                    quote_target = ENTITY_QUOTE_TARGET_LENGTH + 2
                    for sentence in sentences:
                        quote_text = f'"{sentence}"'.ljust(quote_target)
                        item = QuoteListItem(quote_text, entry["episode_uuid"])
                        quotes_list.append(item)

                # Populate footer connections (limit to 5)
                chips_container = self.query_one("#connections-chips", Horizontal)
                chips_container.remove_children()

                for conn in new["connections"][:5]:
                    chip = ConnectionChip(conn["name"], conn["uuid"], self.journal)
                    chips_container.mount(chip)

                # Focus the quotes list if it has items
                if new["entries"]:
                    quotes_list.index = 0
                    quotes_list.focus()

        except Exception as e:
            logger.error("Failed to update entity browser: %s", e, exc_info=True)

    def watch_reader_open(self, old: bool, new: bool) -> None:
        if new:
            self.add_class("reader-open")
        else:
            self.remove_class("reader-open")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.list_view.id != "quotes-list":
            return

        # Get the selected QuoteListItem
        if isinstance(event.item, QuoteListItem):
            self.selected_episode_uuid = event.item.episode_uuid
            self._open_reader(event.item.episode_uuid)

    def _open_reader(self, episode_uuid: str) -> None:
        if self.entity_data is None:
            return

        # Find the entry content
        entry = None
        for e in self.entity_data["entries"]:
            if e["episode_uuid"] == episode_uuid:
                entry = e
                break

        if entry is None:
            return

        try:
            reader_date = self.query_one("#reader-date", Static)
            reader_content = self.query_one("#reader-content", Markdown)

            valid_at = entry["valid_at"]
            if hasattr(valid_at, "strftime"):
                date_str = valid_at.strftime("%B %-d, %Y")
            else:
                date_str = str(valid_at)

            reader_date.update(date_str)
            reader_content.update(entry["content"])

            # Responsive: narrow mode if terminal < 100 columns
            if self.app.size.width < 100:
                self.add_class("narrow")
            else:
                self.remove_class("narrow")

            self.reader_open = True
            reader_content.focus()
        except Exception as e:
            logger.error("Failed to open reader: %s", e, exc_info=True)

    def action_back(self) -> None:
        if self.reader_open:
            self.reader_open = False
            self.remove_class("narrow")
            try:
                quotes_list = self.query_one("#quotes-list", ListView)
                quotes_list.focus()
            except Exception:
                pass
        else:
            self.app.pop_screen()

    def action_read_entry(self) -> None:
        if self.reader_open and self.selected_episode_uuid:
            from frontend.screens.view_screen import ViewScreen

            self.app.push_screen(ViewScreen(self.selected_episode_uuid, self.journal))

    def action_go_home(self) -> None:
        """Pop all screens to return to home."""
        while len(self.app.screen_stack) > 1:
            self.app.pop_screen()
