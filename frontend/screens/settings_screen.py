import asyncio
import logging

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Label, Switch

from backend.database.redis_ops import (
    get_dead_episodes_count,
    get_inference_enabled,
    retry_dead_episodes,
    set_inference_enabled,
)
from backend.settings import DEFAULT_JOURNAL

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

    #retry-dead-btn {
        margin-top: 1;
        width: 100%;
    }

    #retry-dead-btn.hidden {
        display: none;
    }

    SettingsScreen {
        align: center middle;
    }
    """

    dead_count = reactive(0)

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
        Binding("escape", "dismiss_modal", "Close", show=True),
        Binding("s", "dismiss_modal", "Close", show=False),
        Binding("j,down", "app.focus_next", "Next", show=False),
        Binding("k,up", "app.focus_previous", "Previous", show=False),
        Binding("h", "go_home", "Home", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("Settings", id="settings-title"),
            Label(""),
            Label("Enable background inference"),
            Switch(id="inference-toggle", value=self.inference_enabled),
            Button("", id="retry-dead-btn", classes="hidden"),
            id="settings-dialog",
        )
        yield Footer()

    def on_mount(self) -> None:
        self._update_dead_count()

    def _update_dead_count(self) -> None:
        try:
            self.dead_count = get_dead_episodes_count(DEFAULT_JOURNAL)
        except Exception:
            self.dead_count = 0

    @property
    def _retry_button_label(self) -> str:
        if self.dead_count <= 0:
            return ""
        elif self.dead_count == 1:
            return "Retry 1 failed entry"
        return f"Retry {self.dead_count} failed entries"

    def watch_dead_count(self, dead_count: int) -> None:
        try:
            button = self.query_one("#retry-dead-btn", Button)
            if dead_count <= 0:
                button.add_class("hidden")
            else:
                button.remove_class("hidden")
                button.label = self._retry_button_label
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "retry-dead-btn":
            self.run_worker(self._retry_dead_episodes(), exclusive=True)

    async def _retry_dead_episodes(self) -> None:
        count = await retry_dead_episodes(DEFAULT_JOURNAL)
        if count == 0:
            self.notify("No failed entries to retry")
        else:
            entry_word = "entry" if count == 1 else "entries"
            self.notify(f"Retrying {count} {entry_word}...")
        self._update_dead_count()

    def action_dismiss_modal(self) -> None:
        self.dismiss()

    def action_go_home(self) -> None:
        """Pop all screens to return to home."""
        from frontend.screens.home_screen import HomeScreen

        while len(self.app.screen_stack) > 1:
            if isinstance(self.app.screen, HomeScreen):
                break
            self.app.pop_screen()

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
