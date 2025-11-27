import asyncio
import json
import logging
from typing import Callable

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
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

    class CacheCheckComplete(Message):
        """Posted when initial cache check completes."""

        def __init__(self, entities_found: bool) -> None:
            self.entities_found = entities_found
            super().__init__()

    _pending_render: bool = False

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
    cache_loading: reactive[bool] = reactive(True)
    entities: reactive[list[dict]] = reactive([])
    inference_enabled: reactive[bool] = reactive(True)
    status: reactive[str | None] = reactive(None)
    active_processing: reactive[bool] = reactive(False)
    active_episode_uuid: reactive[str | None] = reactive(None)

    def __init__(
        self,
        episode_uuid: str,
        journal: str,
        *,
        inference_enabled: bool = True,
        status: str | None = None,
        active_processing: bool = False,
        active_episode_uuid: str | None = None,
        on_entity_deleted: Callable[[bool], None] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.episode_uuid = episode_uuid
        self.journal = journal
        self.on_entity_deleted = on_entity_deleted
        self.set_reactive(EntitySidebar.inference_enabled, inference_enabled)
        self.set_reactive(EntitySidebar.status, status)
        self.set_reactive(EntitySidebar.active_processing, active_processing)
        self.set_reactive(EntitySidebar.active_episode_uuid, active_episode_uuid)

    def compose(self) -> ComposeResult:
        yield Label("Connections", classes="sidebar-header")
        yield Container(id="entity-content")
        yield Label("d: delete | ↑↓: navigate", classes="sidebar-footer")

    def on_mount(self) -> None:
        """Render initial content and attempt immediate cache fetch."""
        self._update_content()
        # Attempt immediate cache fetch - if data exists, show instantly
        self.run_worker(self.refresh_entities(), exclusive=True)

    def watch_cache_loading(self, loading: bool) -> None:
        """Reactive: re-render when cache loading flag changes."""
        if not self.is_mounted:
            return
        self._request_render()

    def watch_active_processing(self, active_processing: bool) -> None:
        """Reactive: re-render when processing state changes."""
        if not self.is_mounted:
            return
        self._request_render()

    def watch_entities(self, entities: list[dict]) -> None:
        """Reactive: re-render when entities change."""
        if not self.is_mounted:
            return
        if not self.cache_loading:
            self._request_render()

    def watch_status(self, status: str | None) -> None:
        """Reactive: re-render when status changes."""
        if not self.is_mounted:
            return
        if not self.cache_loading:
            self._request_render()

    def _request_render(self) -> None:
        """Coalesce multiple watcher triggers into a single render per refresh."""
        if self._pending_render:
            return
        self._pending_render = True
        # Schedule on next event loop tick to coalesce multiple reactive triggers.
        self.call_later(self._flush_render)

    def _flush_render(self) -> None:
        """Flush the scheduled render if sidebar is ready."""
        self._pending_render = False
        if self.is_mounted:
            self._update_content()

    def _update_content(self) -> None:
        """Update container content based on loading state and entity data.

        Skips DOM manipulation when content hasn't changed to avoid jank.
        """
        if not self.is_mounted:
            return

        content_container = self.query_one("#entity-content", Container)
        if not content_container.is_attached:
            # Defer until the container is attached to avoid mount errors.
            self.call_after_refresh(self._update_content)
            return

        current_children = list(content_container.children)

        should_show_spinner = (
            self.cache_loading
            and self.inference_enabled
            and self.status in ("pending_nodes", "pending_edges")
            and self.active_episode_uuid
            and self.active_episode_uuid == self.episode_uuid
        )

        if should_show_spinner:
            # Skip if already showing spinner
            if len(current_children) == 1 and isinstance(current_children[0], LoadingIndicator):
                return
            content_container.remove_children()
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
                clear_loading = False
                self.cache_loading = True

            # Skip if already showing same message
            if len(current_children) == 1 and isinstance(current_children[0], Label):
                existing_label = current_children[0]
                if existing_label.render() == message:
                    if clear_loading:
                        self.cache_loading = False
                    return

            content_container.remove_children()
            content_container.mount(Label(message))
            if clear_loading:
                self.cache_loading = False
        else:
            # Skip if already showing entities (ListView exists and count matches)
            if len(current_children) == 1 and isinstance(current_children[0], ListView):
                existing_list = current_children[0]
                if len(existing_list.children) == len(self.entities):
                    return

            items = [
                EntityListItem(self._format_entity_label(entity))
                for entity in self.entities
            ]
            list_view = ListView(*items)
            content_container.remove_children()
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
                    self.cache_loading = False
                self.post_message(self.CacheCheckComplete(entities_found=True))
            else:
                # No data found - notify parent, keep cache_loading=True for polling
                self.post_message(self.CacheCheckComplete(entities_found=False))
        except Exception as e:
            logger.error(f"Failed to fetch entities from Redis: {e}", exc_info=True)
            self.cache_loading = False
            self.post_message(self.CacheCheckComplete(entities_found=False))

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

            # Notify ViewScreen/machine about the deletion
            if self.on_entity_deleted:
                self.on_entity_deleted(len(new_entities) > 0)

            self.entities = new_entities

        except Exception as e:
            logger.error(f"Failed to delete entity mention: {e}", exc_info=True)
