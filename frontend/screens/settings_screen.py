import asyncio
import logging

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Label, Switch

from backend.database.redis_ops import (
    get_inference_enabled,
    set_inference_enabled,
)

logger = logging.getLogger("charlie")


class SettingsScreen(ModalScreen):
    """Modal screen for application settings."""

    DEFAULT_CSS = """
    #settings-dialog {
        width: 60;
        height: auto;
        padding: 2 4;
        background: $panel;
        border: thick $primary;
    }

    #settings-title {
        text-align: center;
        text-style: bold;
        color: $text;
        padding-bottom: 1;
    }

    SettingsScreen {
        align: center middle;
    }
    """

    def __init__(self):
        super().__init__()
        # Default to enabled; actual state loaded on mount from Redis
        try:
            self.inference_enabled = get_inference_enabled()
        except Exception:
            self.inference_enabled = True
        self._loading_toggle_state = False
        self._toggle_worker = None

    BINDINGS = [
        Binding("s", "dismiss_modal", "Close", show=True),
        Binding("escape", "dismiss_modal", "Close", show=False),
        Binding("j,down", "app.focus_next", "Next", show=False),
        Binding("k,up", "app.focus_previous", "Previous", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("Settings", id="settings-title"),
            Label(""),
            Label("Enable background inference"),
            Switch(id="inference-toggle", value=self.inference_enabled),
            id="settings-dialog",
        )
        yield Footer()

    def action_dismiss_modal(self) -> None:
        self.dismiss()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id != "inference-toggle":
            return

        if self._loading_toggle_state:
            event.switch.value = self.inference_enabled
            return

        desired = event.value
        previous = self.inference_enabled

        event.switch.disabled = True
        self._loading_toggle_state = True

        # Run persistence off the UI thread to avoid freezes during model load/enqueue
        self._toggle_worker = self.run_worker(
            self._persist_inference_toggle(event.switch, desired, previous),
            exclusive=True,
            name="inference-toggle",
        )

    async def _persist_inference_toggle(
        self, switch: Switch, desired: bool, previous: bool
    ):
        try:
            # Persist toggle in a thread to keep UI responsive
            # Don't clear model state on re-enable - let natural flow handle it
            # to avoid racing with ongoing unload operations
            await asyncio.to_thread(set_inference_enabled, desired)
            self.inference_enabled = desired
            self._on_toggle_success(switch, desired)
        except Exception as exc:
            logger.error("Failed to persist inference toggle: %s", exc, exc_info=True)
            self._on_toggle_failure(switch, previous)
        finally:
            self._clear_toggle_loading(switch)

    def _on_toggle_success(self, switch: Switch, value: bool) -> None:
        switch.value = value
        switch.disabled = False

    def _on_toggle_failure(self, switch: Switch, previous: bool) -> None:
        switch.value = previous
        self.inference_enabled = previous
        switch.disabled = False
        self.notify("ai ded 💀 check logz", severity="error")

    def _clear_toggle_loading(self, switch: Switch) -> None:
        self._loading_toggle_state = False
        switch.disabled = False
        switch.focus()
