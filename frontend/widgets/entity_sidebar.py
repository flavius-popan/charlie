import asyncio
import json
import logging

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Label,
    ListItem,
    ListView,
    LoadingIndicator,
)

from backend.database.queries import delete_entity_mention
from backend.database.redis_ops import redis_ops
from frontend.utils import extract_title

logger = logging.getLogger("charlie")


class EntityListItem(ListItem):
    """A list item for displaying entity information."""

    DEFAULT_CSS = """
    EntityListItem {
        padding-left: 1;
    }
    """

    def __init__(self, label_text: str, **kwargs):
        super().__init__(**kwargs)
        self.label_text = label_text

    def compose(self) -> ComposeResult:
        yield Label(self.label_text)


class DeleteEntityModal(ModalScreen):
    """Confirmation modal for entity deletion."""

    DEFAULT_CSS = """
    DeleteEntityModal {
        align: center middle;
    }

    #delete-dialog {
        width: 80;
        height: auto;
        max-height: 15;
        border: thick $background 80%;
        background: $surface;
        padding: 2 4;
    }

    #delete-dialog Vertical {
        height: auto;
    }

    #delete-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    #delete-hint {
        color: $text-muted;
        margin-bottom: 2;
    }

    #delete-buttons {
        align: center middle;
    }

    #delete-dialog Button {
        margin-right: 2;
    }
    """

    def __init__(self, entity: dict):
        super().__init__()
        self.entity = entity

    def compose(self) -> ComposeResult:
        name = self.entity["name"]
        title = f"Remove '{name}'?"
        hint = "It won't appear again in any future entries."

        yield Vertical(
            Label(title, id="delete-title"),
            Label(hint, id="delete-hint"),
            Horizontal(
                Button("Cancel", id="cancel", variant="default"),
                Button("Remove", id="remove", variant="error"),
                id="delete-buttons",
            ),
            id="delete-dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "remove":
            self.dismiss(True)
        else:
            self.dismiss(False)

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(False)
            event.stop()
            event.prevent_default()


class EntitySidebar(Container):
    """Sidebar showing entities connected to current episode."""

    DEFAULT_CSS = """
    EntitySidebar {
        width: 1fr;
        border-left: solid $accent;

    }

    EntitySidebar #entity-content {
        padding-left: 0;
    }

    EntitySidebar .sidebar-header {
        color: $text-muted;
        text-align: center;
        width: 100%;
        height: 4%;
    }

    EntitySidebar .sidebar-footer {
        color: $text-muted;
        text-align: center;
        width: 100%;
        height: 2%;
    }
    """

    BINDINGS = [
        Binding("d", "delete_entity", "Delete", show=False),
    ]

    episode_uuid: reactive[str] = reactive("")
    journal: reactive[str] = reactive("")
    loading: reactive[bool] = reactive(True)
    entities: reactive[list[dict]] = reactive([])
    inference_enabled: reactive[bool] = reactive(True)
    status: reactive[str | None] = reactive(None)
    active_processing: reactive[bool] = reactive(False)
    user_override: reactive[bool] = reactive(False)

    def __init__(
        self,
        episode_uuid: str,
        journal: str,
        *,
        inference_enabled: bool = True,
        status: str | None = None,
        active_processing: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.episode_uuid = episode_uuid
        self.journal = journal
        self.set_reactive(EntitySidebar.inference_enabled, inference_enabled)
        self.set_reactive(EntitySidebar.status, status)
        self.set_reactive(EntitySidebar.active_processing, active_processing)

    def compose(self) -> ComposeResult:
        yield Label("Connections", classes="sidebar-header")
        yield Container(id="entity-content")
        yield Label("d: delete | ↑↓: navigate", classes="sidebar-footer")

    def on_mount(self) -> None:
        """Render initial content and attempt immediate cache fetch."""
        self._update_content()
        # Attempt immediate cache fetch - if data exists, show instantly
        self.run_worker(self.refresh_entities(), exclusive=True)

    def watch_loading(self, loading: bool) -> None:
        """Reactive: swap between loading indicator and entity list."""
        if not self.is_mounted:
            return
        self._update_content()

    def watch_active_processing(self, active_processing: bool) -> None:
        """Reactive: re-render when processing state changes."""
        if not self.is_mounted:
            return
        self._update_content()

    def watch_entities(self, entities: list[dict]) -> None:
        """Reactive: re-render when entities change."""
        if not self.is_mounted:
            return
        if not self.loading:
            self._update_content()

    def watch_status(self, status: str | None) -> None:
        """Reactive: re-render when status changes.

        Note: If user_override is True, don't update.
        This prevents background polling from overwriting
        user-initiated changes like entity deletion.
        """
        if not self.is_mounted:
            return
        if self.user_override:
            return
        if not self.loading:
            self._update_content()

    def _update_content(self) -> None:
        """Update container content based on loading state and entity data."""
        if not self.is_mounted:
            return

        content_container = self.query_one("#entity-content", Container)
        if not content_container.is_attached:
            # Defer until the container is attached to avoid mount errors.
            self.call_after_refresh(self._update_content)
            return
        content_container.remove_children()

        should_show_spinner = (
            self.loading
            and self.inference_enabled
            and self.status in ("pending_nodes", "pending_edges")
            and self.active_processing
        )

        if should_show_spinner:
            content_container.mount(LoadingIndicator())
            return

        if not self.entities:
            message = "No connections found"
            clear_loading = True

            if not self.inference_enabled:
                if self.status in ("pending_nodes", "pending_edges"):
                    message = "Inference disabled; extraction is paused."
                else:
                    message = "Inference disabled; enable inference to extract connections."
            elif self.status in ("pending_nodes", "pending_edges"):
                message = "Awaiting processing..."
                clear_loading = False  # keep loading True so polling/spinner can proceed

            content_container.mount(Label(message))
            if clear_loading:
                self.loading = False
        else:
            items = [
                EntityListItem(self._format_entity_label(entity))
                for entity in self.entities
            ]
            list_view = ListView(*items)
            content_container.mount(list_view)

            def focus_and_select():
                list_view.index = 0
                list_view.focus()

            self.call_after_refresh(focus_and_select)

    def _format_entity_label(self, entity: dict) -> str:
        """Format entity as 'Name [Type]'."""
        name = entity["name"]
        entity_type = entity.get("type", "Entity")
        return f"{name} [{entity_type}]"

    async def refresh_entities(self) -> None:
        """Fetch entity data from Redis cache (single attempt, no polling)."""
        try:
            cache_key = f"journal:{self.journal}:{self.episode_uuid}"

            def _fetch_nodes():
                with redis_ops() as r:
                    return r.hget(cache_key, "nodes")

            nodes_json = await asyncio.to_thread(_fetch_nodes)

            if nodes_json:
                nodes = json.loads(nodes_json.decode())
                filtered_nodes = [n for n in nodes if n["name"] != "I"]
                # Batch update to trigger only one render
                with self.app.batch_update():
                    self.entities = filtered_nodes
                    self.loading = False
            # If no data found, keep loading=True (ViewScreen will handle polling)
        except Exception as e:
            logger.error(f"Failed to fetch entities from Redis: {e}", exc_info=True)
            self.loading = False

    def action_delete_entity(self) -> None:
        """Show delete confirmation for selected entity."""
        list_view = self.query_one(ListView)
        # Deletion is only valid when the sidebar is visible and actively focused.
        if not self.display:
            return
        if not list_view.has_focus:
            return
        if list_view.index is None or list_view.index < 0:
            return

        entity = self.entities[list_view.index]
        self.app.push_screen(DeleteEntityModal(entity), self._handle_delete_result)

    async def _handle_delete_result(self, confirmed: bool) -> None:
        """Handle deletion confirmation result."""
        if not confirmed:
            return

        list_view = self.query_one(ListView)
        if list_view.index is None or list_view.index < 0:
            return

        entity = self.entities[list_view.index]

        try:
            # Delete from database
            await delete_entity_mention(self.episode_uuid, entity["uuid"], self.journal)

            # Remove from local state
            new_entities = [e for e in self.entities if e["uuid"] != entity["uuid"]]

            # If we deleted all entities, ensure UI shows "No connections found"
            # Set status/loading BEFORE updating entities to avoid race in watchers
            if len(new_entities) == 0:
                self.user_override = True
                self.status = "done"
                self.loading = False

            self.entities = new_entities

        except Exception as e:
            logger.error(f"Failed to delete entity mention: {e}", exc_info=True)
