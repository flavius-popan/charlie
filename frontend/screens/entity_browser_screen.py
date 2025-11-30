"""Entity Browser screen for viewing entity details and memories."""

import logging

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Footer, Header, ListItem, ListView, Markdown, Static

from backend.database import get_entity_browser_data, get_entry_entities_with_counts
from backend.database.queries import (
    extract_entity_sentences,
    ENTITY_QUOTE_TARGET_LENGTH,
)
from backend.settings import DEFAULT_JOURNAL
from frontend.utils import (
    inject_entity_links,
    emphasize_text,
    emphasize_rich,
    get_display_title,
)

logger = logging.getLogger("charlie")


class DateListItem(ListItem):
    """Non-selectable list item showing just the date."""

    SCOPED_CSS = False
    DEFAULT_CSS = """
    DateListItem {
        height: auto;
        padding: 1 0 1 2;
    }

    DateListItem .date-text {
        color: $text-muted;
    }
    """

    def __init__(self, date_str: str):
        super().__init__()
        self.date_str = date_str
        self.disabled = True

    def compose(self) -> ComposeResult:
        yield Static(self.date_str, classes="date-text")


class QuoteSeparator(ListItem):
    """Thin separator between quotes from the same entry."""

    SCOPED_CSS = False
    DEFAULT_CSS = """
    QuoteSeparator {
        height: 1;
        min-height: 1;
        max-height: 1;
        padding: 0;
    }
    """

    def __init__(self):
        super().__init__()
        self.disabled = True

    def compose(self) -> ComposeResult:
        yield Static(" ")


class QuoteListItem(ListItem):
    """List item showing centered quote text."""

    SCOPED_CSS = False
    DEFAULT_CSS = """
    QuoteListItem {
        height: auto;
        min-height: 1;
        padding: 0 4;
    }

    QuoteListItem .quote-text {
        width: 100%;
        text-align: left;
    }
    """

    def __init__(self, quote_text: str, episode_uuid: str, markup: bool = False):
        super().__init__()
        self.quote_text = quote_text
        self.episode_uuid = episode_uuid
        self._markup = markup

    def compose(self) -> ComposeResult:
        yield Static(self.quote_text, classes="quote-text", markup=self._markup)


class TitleListItem(ListItem):
    """List item showing episode title for entries without extractable quotes."""

    SCOPED_CSS = False
    DEFAULT_CSS = """
    TitleListItem {
        height: auto;
        padding: 0 4;
        text-align: center;
    }

    TitleListItem .title-text {
        width: 100%;
        text-align: left;
    }
    """

    def __init__(self, title: str, episode_uuid: str):
        super().__init__()
        self.title = title
        self.episode_uuid = episode_uuid

    def compose(self) -> ComposeResult:
        yield Static(self.title, classes="title-text")


class ConnectionLink(Static):
    """Clickable text link for navigating to connected entities."""

    DEFAULT_CSS = """
    ConnectionLink {
        width: auto;
        color: $text-muted;
    }

    ConnectionLink:hover {
        color: $text;
        text-style: underline;
    }
    """

    def __init__(self, name: str, entity_uuid: str, journal: str):
        super().__init__(name)
        self.entity_uuid = entity_uuid
        self.journal = journal

    def on_click(self) -> None:
        current_reader_open = getattr(self.screen, 'reader_open', False)
        self.app.push_screen(EntityBrowserScreen(self.entity_uuid, self.journal, initial_reader_open=current_reader_open))


class ConnectionSeparator(Static):
    """Dot separator between connection links."""

    DEFAULT_CSS = """
    ConnectionSeparator {
        width: auto;
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self):
        super().__init__("·")


class MentionInfo(Static):
    """Clickable mention info that scrolls to oldest quote."""

    DEFAULT_CSS = """
    MentionInfo {
        width: 100%;
        text-align: center;
        color: $text-muted;
    }

    MentionInfo:hover {
        color: $text;
        text-style: underline;
    }
    """

    def __init__(self, text: str = "", oldest_index: int = 0, **kwargs):
        super().__init__(text, **kwargs)
        self.oldest_index = oldest_index

    def on_click(self) -> None:
        """Scroll to oldest quote when clicked."""
        try:
            screen = self.screen
            quotes_list = screen.query_one("#quotes-list", ListView)
            # Set index and scroll to that position
            quotes_list.index = self.oldest_index
            quotes_list.scroll_to_widget(quotes_list.children[self.oldest_index])
            quotes_list.focus()

            # Update viewer if already open
            if screen.reader_open:
                item = quotes_list.children[self.oldest_index]
                if isinstance(item, (QuoteListItem, TitleListItem)):
                    screen._open_reader(item.episode_uuid)
        except Exception:
            pass


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
        padding: 1 2 0 2;
        align: center top;
    }

    EntityBrowserScreen #entity-name {
        text-align: center;
        text-style: bold;
        width: 100%;
        height: auto;
    }

    EntityBrowserScreen #mention-info {
        height: auto;
    }

    EntityBrowserScreen #quotes-footer {
        width: 100%;
        height: auto;
        align: center middle;
    }

    EntityBrowserScreen #body-container {
        height: 1fr;
        width: 100%;
        border-top: solid $accent;
    }

    EntityBrowserScreen #content-area {
        height: 1fr;
        width: 100%;
    }

    EntityBrowserScreen #quotes-container {
        height: 1fr;
        width: 1fr;
    }

    EntityBrowserScreen #quotes-list {
        height: 1fr;
        width: 100%;
        scrollbar-size: 0 0;
    }


    /* Reader panel (hidden by default) */
    EntityBrowserScreen #reader-panel {
        display: none;
        width: 0;
        height: 1fr;
        padding: 0 2;
        scrollbar-size: 0 0;
    }

    EntityBrowserScreen #reader-content {
        padding: 0;
        margin: 0;
    }

    EntityBrowserScreen #reader-date {
        color: $text-muted;
        text-align: center;
    }

    /* Reader mode - wide: 50/50 split, hide date in reader */
    EntityBrowserScreen.reader-open #quotes-container {
        width: 1fr;
    }

    EntityBrowserScreen.reader-open #reader-panel {
        display: initial;
        width: 1fr;
    }

    EntityBrowserScreen.reader-open #reader-date {
        display: none;
        height: 0;
    }

    /* Reader mode - narrow: hide quotes, full-width reader, show date */
    EntityBrowserScreen.reader-open.narrow #quotes-container {
        display: none;
    }

    EntityBrowserScreen.reader-open.narrow #reader-date {
        display: block;
        height: auto;
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

    def __init__(self, entity_uuid: str, journal: str = DEFAULT_JOURNAL, *, initial_reader_open: bool = False):
        super().__init__()
        self.entity_uuid = entity_uuid
        self.journal = journal
        self._initial_reader_open = initial_reader_open
        self._has_been_suspended = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False, icon="")

        # Loading state
        yield Static("Loading...", id="loading")

        # Header container (frozen) with name, mention info, and connections
        header = Container(
            Static("", id="entity-name"),
            MentionInfo("", id="mention-info"),
            Horizontal(id="quotes-footer"),
            id="header-container",
        )
        header.display = False
        yield header

        # Main body: content area (quotes + reader)
        body = Vertical(
            Horizontal(
                Vertical(
                    ListView(id="quotes-list"),
                    id="quotes-container",
                ),
                VerticalScroll(
                    Static("", id="reader-date"),
                    Markdown("", id="reader-content", open_links=False),
                    id="reader-panel",
                ),
                id="content-area",
            ),
            id="body-container",
        )
        body.display = False
        yield body

        yield Footer()

    async def on_mount(self) -> None:
        self.run_worker(self._load_entity_data(), exclusive=True, group="entity-data")

    async def on_screen_resume(self) -> None:
        # Only close reader if we've actually been suspended (returning from another screen)
        if self._has_been_suspended:
            self.reader_open = False
            self.remove_class("narrow")
            self.run_worker(
                self._load_entity_data(), exclusive=True, group="entity-data"
            )

    def _is_wide_screen(self) -> bool:
        """Check if screen is wide enough for side-by-side layout."""
        return self.app.size.width >= 100

    def on_screen_suspend(self) -> None:
        self._has_been_suspended = True
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
            quotes_footer = self.query_one("#quotes-footer", Horizontal)

            with self.app.batch_update():
                loading.display = False
                header.display = True
                body.display = True

                # Update header - entity name
                entity = new["entity"]
                entity_name = entity["name"]
                name_widget = self.query_one("#entity-name", Static)
                name_widget.update(entity_name)

                # Populate quotes list with entity-specific sentences
                quotes_list = self.query_one("#quotes-list", ListView)
                quotes_list.clear()

                prev_date = None
                list_index = 0
                oldest_quote_index = 0  # Track index of oldest (last) quote
                total_quote_count = 0  # Count actual quotes for mention info

                for entry in new["entries"]:
                    valid_at = entry["valid_at"]
                    date = valid_at.date() if hasattr(valid_at, "date") else valid_at

                    # Extract all sentences mentioning the entity
                    sentences = extract_entity_sentences(entry["content"], entity_name)

                    # Add date item when date changes (with year)
                    if date != prev_date:
                        date_str = (
                            valid_at.strftime("%b %-d, %Y")
                            if hasattr(valid_at, "strftime")
                            else str(date)
                        )
                        prev_date = date
                        quotes_list.append(DateListItem(date_str))
                        list_index += 1

                    if sentences:
                        # Create quote items for each sentence with entity emphasized
                        quote_target = ENTITY_QUOTE_TARGET_LENGTH + 2
                        for i, sentence in enumerate(sentences):
                            # Add separator between quotes from same entry
                            if i > 0:
                                quotes_list.append(QuoteSeparator())
                                list_index += 1
                            emphasized = emphasize_rich(sentence, entity_name)
                            quote_text = f'"{emphasized}"'.ljust(quote_target)
                            item = QuoteListItem(
                                quote_text, entry["episode_uuid"], markup=True
                            )
                            quotes_list.append(item)
                            oldest_quote_index = list_index  # Last quote is oldest
                            list_index += 1
                            total_quote_count += 1
                    else:
                        # No extractable quotes - show episode title instead
                        title = get_display_title(entry, max_chars=60)
                        item = TitleListItem(title, entry["episode_uuid"])
                        quotes_list.append(item)
                        oldest_quote_index = list_index
                        list_index += 1

                # Update mention info in header (use quote count, not entry count)
                mention_info = self.query_one("#mention-info", MentionInfo)
                first_mention = entity.get("first_mention")

                if first_mention and hasattr(first_mention, "strftime"):
                    date_str = first_mention.strftime("%b %-d, %Y")
                    if total_quote_count == 1:
                        mention_text = f"mentioned once on {date_str}"
                    elif total_quote_count > 1:
                        mention_text = f"first mentioned {date_str}"
                    else:
                        # No quotes found, fall back to entry-based text
                        mention_text = f"appears in entry from {date_str}"
                else:
                    mention_text = ""

                mention_info.update(mention_text)
                mention_info.oldest_index = oldest_quote_index

                # Populate footer connections as text links (limit to 5)
                quotes_footer.remove_children()

                connections = new["connections"][:5]
                for i, conn in enumerate(connections):
                    link = ConnectionLink(conn["name"], conn["uuid"], self.journal)
                    quotes_footer.mount(link)
                    # Add separator after all but the last
                    if i < len(connections) - 1:
                        quotes_footer.mount(ConnectionSeparator())

                # Focus the quotes list on first selectable (non-disabled) item
                if quotes_list.children:
                    # Find first selectable item (QuoteListItem or TitleListItem)
                    for i, child in enumerate(quotes_list.children):
                        if isinstance(child, (QuoteListItem, TitleListItem)):
                            quotes_list.index = i
                            # Set selected_episode_uuid since programmatic index
                            # change doesn't fire Highlighted event
                            self.selected_episode_uuid = child.episode_uuid
                            break
                    quotes_list.focus()

            # Open reader if inherited from previous screen
            if self._initial_reader_open and self.selected_episode_uuid:
                self._open_reader(self.selected_episode_uuid)

        except Exception as e:
            logger.error("Failed to update entity browser: %s", e, exc_info=True)

    def watch_reader_open(self, old: bool, new: bool) -> None:
        if new:
            self.add_class("reader-open")
        else:
            self.remove_class("reader-open")

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Track highlighted item for selection (does not open reader)."""
        if event.list_view.id != "quotes-list":
            return
        if event.item is None:
            return

        # Get episode UUID from the highlighted item
        episode_uuid = None
        if isinstance(event.item, QuoteListItem):
            episode_uuid = event.item.episode_uuid
        elif isinstance(event.item, TitleListItem):
            episode_uuid = event.item.episode_uuid

        if episode_uuid is not None:
            self.selected_episode_uuid = episode_uuid

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle Enter key on ListView - open reader."""
        if event.list_view.id != "quotes-list":
            return
        if self.selected_episode_uuid:
            if self.reader_open and not self._is_wide_screen():
                # Narrow screen with reader open - push full ViewScreen
                from frontend.screens.view_screen import ViewScreen

                self.app.push_screen(
                    ViewScreen(self.selected_episode_uuid, self.journal)
                )
            else:
                # Wide screen or reader not open - open/update the reader panel
                self._open_reader(self.selected_episode_uuid)

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

            # Inject entity links into content
            content = entry["content"]
            self.run_worker(
                self._load_linked_content(episode_uuid, content, reader_content),
                exclusive=True,
                group="reader-links",
            )

            # Responsive: narrow mode if terminal < 100 columns
            if self.app.size.width < 100:
                self.add_class("narrow")
            else:
                self.remove_class("narrow")

            # Reset scroll position when switching entries
            reader_panel = self.query_one("#reader-panel", VerticalScroll)
            reader_panel.scroll_home(animate=False)

            self.reader_open = True
        except Exception as e:
            logger.error("Failed to open reader: %s", e, exc_info=True)

    async def _load_linked_content(
        self, episode_uuid: str, content: str, reader_content: Markdown
    ) -> None:
        """Load content with entity links injected and current entity emphasized."""
        try:
            entities = await get_entry_entities_with_counts(episode_uuid, self.journal)
            if entities:
                # Exclude current entity to avoid self-referential links
                content = inject_entity_links(
                    content, entities, min_mentions=2, exclude_uuids={self.entity_uuid}
                )
            # Emphasize the current entity name in the content
            if self.entity_data:
                entity_name = self.entity_data["entity"]["name"]
                content = emphasize_text(content, entity_name)
            await reader_content.update(content)
        except Exception as e:
            logger.error("Failed to load linked content: %s", e, exc_info=True)
            await reader_content.update(content)

    def on_markdown_link_clicked(self, event: Markdown.LinkClicked) -> None:
        """Handle link clicks, including custom entity:// protocol."""
        href = event.href

        if href.startswith("entity://"):
            entity_uuid = href.replace("entity://", "")
            self.app.push_screen(EntityBrowserScreen(entity_uuid, self.journal, initial_reader_open=self.reader_open))
        elif href.startswith(("http://", "https://")):
            self.app.open_url(href)

    def action_back(self) -> None:
        # On narrow screen with reader open: close reader first
        if self.reader_open and not self._is_wide_screen():
            self.reader_open = False
            self.remove_class("narrow")
            try:
                quotes_list = self.query_one("#quotes-list", ListView)
                quotes_list.focus()
            except Exception:
                pass
        else:
            # Wide screen or reader already closed: pop screen
            self.app.pop_screen()

    def action_read_entry(self) -> None:
        if not self.selected_episode_uuid:
            return
        if self.reader_open and not self._is_wide_screen():
            # Narrow screen with reader open - push full ViewScreen
            from frontend.screens.view_screen import ViewScreen

            self.app.push_screen(ViewScreen(self.selected_episode_uuid, self.journal))
        else:
            # Wide screen or reader not open - open/update the reader panel
            self._open_reader(self.selected_episode_uuid)

    async def action_go_home(self) -> None:
        """Pop all screens to return to home."""
        import asyncio
        from frontend.screens.home_screen import HomeScreen

        # Cancel workers before popping screens
        self.workers.cancel_group(self, "entity-data")
        self.workers.cancel_group(self, "reader-links")

        # Pop screens until we reach HomeScreen
        while len(self.app.screen_stack) > 1:
            if isinstance(self.app.screen, HomeScreen):
                break
            self.app.pop_screen()
            await asyncio.sleep(0)  # Yield to event loop between pops
