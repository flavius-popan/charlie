"""Shared widget for entity list items."""

from textual.app import ComposeResult
from textual.widgets import Label, ListItem


class EntityListItem(ListItem):
    """List item for entity display with UUID for navigation."""

    def __init__(self, name: str, entity_uuid: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.entity_uuid = entity_uuid
        self._name = name

    def compose(self) -> ComposeResult:
        yield Label(self._name)
