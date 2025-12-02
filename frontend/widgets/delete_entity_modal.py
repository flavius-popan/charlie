"""Modal dialog for entity deletion with scope and blocking options."""

from dataclasses import dataclass
from typing import Literal

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Label, RadioButton, RadioSet


@dataclass
class DeleteEntityResult:
    """Result from the delete entity modal."""

    confirmed: bool
    scope: Literal["entry", "all"]
    block_future: bool


class DeleteEntityModal(ModalScreen[DeleteEntityResult]):
    """Modal for deleting entities with scope selection and blocking option."""

    BINDINGS = [
        Binding("left", "app.focus_previous", show=False),
        Binding("right", "app.focus_next", show=False),
        Binding("h", "app.focus_previous", show=False),
        Binding("l", "app.focus_next", show=False),
        Binding("j", "app.focus_next", show=False),
        Binding("k", "app.focus_previous", show=False),
    ]

    DEFAULT_CSS = """
    DeleteEntityModal {
        align: center middle;
    }

    #delete-dialog {
        width: 50;
        height: auto;
        border: thick $background 80%;
        background: $surface;
        padding: 1 3;
    }

    #delete-title {
        text-style: bold;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
    }

    #delete-scope-label {
        color: $text-muted;
        margin-bottom: 1;
    }

    #delete-scope {
        border: none;
        height: auto;
        width: 100%;
        margin: 0 0 1 2;
        padding: 0;
        background: transparent;
    }

    #delete-scope RadioButton {
        height: auto;
        width: 100%;
        padding: 0;
        margin: 0;
        background: transparent;
        color: $text;
    }

    #delete-scope RadioButton.-on {
        color: $text;
        text-style: bold;
    }

    #delete-block {
        border: none;
        height: auto;
        width: 100%;
        padding: 0;
        margin: 0 0 1 2;
        background: transparent;
        color: $text;
    }

    #delete-buttons {
        height: auto;
        width: 100%;
        align: center middle;
        margin-top: 1;
    }

    #delete-buttons Button {
        margin: 0 1;
    }
    """

    def __init__(
        self,
        entity_name: str,
        default_scope: Literal["entry", "all"],
        checkbox_default: bool = True,
        show_scope: bool = True,
    ):
        super().__init__()
        self._entity_name = entity_name
        self._default_scope = default_scope
        self._checkbox_default = checkbox_default
        self._show_scope = show_scope

    def compose(self) -> ComposeResult:
        widgets = [Label(f'Remove "{self._entity_name}"', id="delete-title")]

        if self._show_scope:
            widgets.append(Label("From:", id="delete-scope-label"))
            widgets.append(
                RadioSet(
                    RadioButton("This entry only", id="radio-entry"),
                    RadioButton("All entries", id="radio-all"),
                    id="delete-scope",
                )
            )

        widgets.append(
            Checkbox(
                "Block permanently", id="delete-block", value=self._checkbox_default
            )
        )
        widgets.append(
            Horizontal(
                Button("Cancel", id="cancel", variant="default"),
                Button("Remove", id="confirm", variant="error"),
                id="delete-buttons",
            )
        )

        yield Vertical(*widgets, id="delete-dialog")

    def on_mount(self) -> None:
        # Set default radio selection (only if scope radio exists)
        if self._show_scope:
            radio_set = self.query_one("#delete-scope", RadioSet)
            buttons = list(radio_set.query(RadioButton))
            if self._default_scope == "entry":
                buttons[0].value = True
            else:
                buttons[1].value = True

        # Focus Cancel button by default (safer default)
        cancel_button = self.query_one("#cancel", Button)
        cancel_button.focus()

    def _build_result(self, confirmed: bool) -> DeleteEntityResult:
        """Build a DeleteEntityResult with current modal state."""
        checkbox = self.query_one("#delete-block", Checkbox)

        # Determine scope based on whether scope radio exists
        if self._show_scope:
            radio_set = self.query_one("#delete-scope", RadioSet)
            if radio_set.pressed_button:
                scope = "all" if radio_set.pressed_button.id == "radio-all" else "entry"
            else:
                scope = self._default_scope
        else:
            # No scope selection - use default (should be "all" when hidden)
            scope = self._default_scope

        return DeleteEntityResult(
            confirmed=confirmed,
            scope=scope,  # type: ignore
            block_future=checkbox.value,
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm":
            self.dismiss(self._build_result(confirmed=True))
        else:
            self.dismiss(self._build_result(confirmed=False))

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.dismiss(self._build_result(confirmed=False))
            event.stop()
            event.prevent_default()
