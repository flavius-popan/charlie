"""Reusable confirmation modal dialog."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label


class ConfirmationModal(ModalScreen):
    """Generic confirmation modal for destructive actions."""

    BINDINGS = [
        Binding("left", "focus_previous", show=False),
        Binding("right", "focus_next", show=False),
    ]

    DEFAULT_CSS = """
    ConfirmationModal {
        align: center middle;
    }

    #confirm-dialog {
        width: 80;
        height: auto;
        max-height: 15;
        border: thick $background 80%;
        background: $surface;
        padding: 2 4;
    }

    #confirm-dialog Vertical {
        height: auto;
    }

    #confirm-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    #confirm-hint {
        color: $text-muted;
        margin-bottom: 2;
    }

    #confirm-buttons {
        align: center middle;
    }

    #confirm-dialog Button {
        margin-right: 2;
    }
    """

    def __init__(
        self,
        title: str,
        hint: str,
        confirm_label: str = "Confirm",
        confirm_variant: str = "error",
    ):
        super().__init__()
        self._title = title
        self._hint = hint
        self._confirm_label = confirm_label
        self._confirm_variant = confirm_variant

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label(self._title, id="confirm-title"),
            Label(self._hint, id="confirm-hint"),
            Horizontal(
                Button("Cancel", id="cancel", variant="default"),
                Button(self._confirm_label, id="confirm", variant=self._confirm_variant),
                id="confirm-buttons",
            ),
            id="confirm-dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm":
            self.dismiss(True)
        else:
            self.dismiss(False)

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(False)
            event.stop()
            event.prevent_default()
