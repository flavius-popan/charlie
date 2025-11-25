import asyncio
import atexit
import json
import logging
import os
from pathlib import Path

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.screen import ModalScreen, Screen
from textual.worker import WorkerCancelled
from textual.widgets import (
    Button,
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    LoadingIndicator,
    Log as LogWidget,
    Markdown,
    Static,
    Switch,
    TextArea,
)
from textual.logging import TextualHandler

from backend import add_journal_entry
from backend.database import (
    delete_episode,
    ensure_database_ready,
    get_home_screen,
    get_episode,
    get_tcp_server_endpoint,
    shutdown_database,
    update_episode,
)
from backend.database.queries import delete_entity_mention
from backend.database.redis_ops import (
    get_episode_status,
    get_inference_enabled,
    redis_ops,
    set_inference_enabled,
)
from backend.settings import DEFAULT_JOURNAL
from backend.services.queue import (
    is_huey_consumer_running,
    start_huey_consumer,
    stop_huey_consumer,
)

LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

os.environ.setdefault("TEXTUAL_LOG", str(LOGS_DIR / "charlie.log"))

# UI THREAD SAFETY MANDATE:
# No blocking or synchronous I/O on the UI thread. Offload Redis/DB/file/network
# work via run_worker(thread=True) or asyncio.to_thread. Keep this invariant when
# adding handlers or features.

class _NoActiveAppFilter(logging.Filter):
    """Allow records through only when no Textual app context is active."""

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        from textual._context import active_app

        try:
            active_app.get()
        except LookupError:
            return True
        return False


def _configure_logging() -> None:
    """Route all logging through Textual, with a file fallback for background threads."""
    log_path = Path(os.environ["TEXTUAL_LOG"])

    textual_handler = TextualHandler(stderr=False, stdout=False)
    textual_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    logging.basicConfig(
        level=logging.INFO,
        handlers=[textual_handler, file_handler],
        force=True,
    )

    # Quiet Huey per-task INFO logs; keep warnings/errors.
    for name in ("huey", "huey.consumer", "huey.api", "huey.signals", "huey.queue"):
        logging.getLogger(name).setLevel(logging.WARNING)


_configure_logging()

logger = logging.getLogger("charlie")

from frontend.screens.home_screen import HomeScreen
from frontend.screens.view_screen import ViewScreen
from frontend.screens.edit_screen import EditScreen
from frontend.screens.log_screen import LogScreen
from frontend.screens.settings_screen import SettingsScreen
from frontend.widgets.entity_sidebar import EntitySidebar



class CharlieApp(App):
    """A minimal journal TUI application."""

    TITLE = "Charlie"

    def __init__(self):
        super().__init__()

    async def on_mount(self):
        self.theme = "catppuccin-mocha"
        self.push_screen(HomeScreen())
        # Worker is started after DB readiness in HomeScreen._init_and_load.

    def _ensure_huey_worker_running(self):
        """Start Huey consumer in-process if not already running."""
        if is_huey_consumer_running():
            return

        try:
            # Ensure task registry is populated before starting the consumer.
            # Importing registers @huey.task functions with the shared Huey instance.
            import backend.services.tasks  # noqa: F401

            start_huey_consumer()
            endpoint = get_tcp_server_endpoint()
            if endpoint:
                host, port = endpoint
                logger.info(
                    "Huey consumer running in-process using embedded Redis at %s:%d",
                    host,
                    port,
                )
            else:
                logger.info(
                    "Huey consumer running in-process using embedded Redis (unix socket)"
                )
            atexit.register(self._shutdown_huey)

        except Exception as exc:
            logger.error("Failed to start Huey worker: %s", exc, exc_info=True)
            self.notify("huey ded 💀 check logz", severity="error", timeout=5)

    def _shutdown_huey(self):
        """Terminate Huey worker gracefully (finish current task first)."""
        try:
            stop_huey_consumer()
        except Exception as exc:
            logger.warning("Error while stopping Huey worker: %s", exc, exc_info=True)

    def stop_huey_worker(self):
        """Public wrapper to stop worker (used by UI handlers)."""
        self._shutdown_huey()

    def _graceful_shutdown(self):
        """Stop background worker before tearing down the database."""
        try:
            self.stop_huey_worker()
        finally:
            try:
                shutdown_database()
            except Exception as exc:
                logger.warning(
                    "Database shutdown encountered an error: %s", exc, exc_info=True
                )

    def on_unmount(self):
        # Covers exits triggered outside the key binding (e.g., cmd+q/ctrl+c)
        self._graceful_shutdown()


CharlieApp.CSS = """
Screen {
    background: $surface;
}

#empty-state {
    width: 100%;
    height: 100%;
    content-align: center middle;
    color: $text-muted;
}

#journal-content {
    padding: 1 2;
    height: 100%;
}

#editor {
    height: 100%;
    scrollbar-size-vertical: 0;
    scrollbar-size-horizontal: 0;
}

Footer {
    background: $panel;
}

Footer .footer--key {
    color: $text-muted;
}

#log-view {
    height: 100%;
}
"""


if __name__ == "__main__":
    app = CharlieApp()
    app.run()
