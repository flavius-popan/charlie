import asyncio
import logging
import os
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Header, Log as LogWidget

LOGS_DIR = Path(__file__).parent.parent.parent / "logs"

logger = logging.getLogger("charlie")


class LogScreen(Screen):
    """Screen for quick log inspection using Textual's Log widget."""

    BINDINGS = [
        Binding("r", "reload", "Reload", show=True),
        Binding("l", "close", "Close", show=True),
        Binding("escape", "close", "Close", show=False),
        Binding("q", "close", "Close", show=False),
    ]

    DEFAULT_CSS = """
    LogScreen #log-view {
        scrollbar-size-vertical: 0;
        scrollbar-size-horizontal: 0;
    }
    """

    def __init__(self, log_path: Path | None = None):
        super().__init__()
        self.log_path = log_path or Path(
            os.environ.get("TEXTUAL_LOG", LOGS_DIR / "charlie.log")
        )

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False, icon="")
        yield LogWidget(id="log-view", auto_scroll=True, max_lines=2000)
        yield Footer()

    async def on_mount(self):
        await self._load_log()

    async def on_screen_resume(self):
        await self._load_log()

    async def action_reload(self):
        await self._load_log()

    def action_close(self):
        self.app.pop_screen()

    async def _load_log(self):
        log_widget = self.query_one("#log-view", LogWidget)
        log_widget.clear()

        if not self.log_path.exists():
            log_widget.write_line("Log file has not been created yet.")
            return

        try:
            lines = await asyncio.to_thread(self._tail_file, self.log_path, 2000)
            log_widget.write_lines(lines)
            log_widget.scroll_end(animate=False)
        except Exception as exc:
            log_widget.write_line(f"Failed to load log: {exc}")
            logger.error("Failed to load log view: %s", exc, exc_info=True)

    @staticmethod
    def _tail_file(path: Path, max_lines: int) -> list[str]:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return [line.rstrip("\n") for line in f.readlines()[-max_lines:]]
