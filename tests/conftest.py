"""Test configuration.

## Known test warnings

RuntimeWarning about unawaited coroutines from Textual (Header._on_mount, Screen._watch_selections):
These warnings appear during Python interpreter shutdown and cannot be suppressed via
pytest's filterwarnings or Python's warnings module. They are emitted by Textual's
internal async cleanup and do not indicate test failures. See pyproject.toml filterwarnings
for attempted suppressions.
"""

import sys
from pathlib import Path
import pytest

# Ensure project root is importable when running tests directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Disable TCP listener in tests to avoid clashing with a running app instance.
from backend import settings as backend_settings
backend_settings.REDIS_TCP_ENABLED = False
backend_settings.TCP_PORT = 0

import backend.database.lifecycle as lifecycle
# Update the module-level _tcp_server dict that was already created
lifecycle._tcp_server["enabled"] = False  # type: ignore[index]
lifecycle.REDIS_TCP_ENABLED = False  # type: ignore[misc]

# Patch Textual's Header._on_mount to handle NoMatches exception
# The original code only catches NoScreen but not NoMatches, causing test failures
# when HeaderTitle hasn't been composed yet.
try:
    from textual.widgets._header import Header
    from textual.css.query import NoMatches

    original_on_mount = Header._on_mount

    def patched_on_mount(self, event):
        async def set_title():
            try:
                self.query_one("HeaderTitle").update(self.format_title())
            except (NoMatches, Exception):
                # Ignore errors when HeaderTitle hasn't been composed yet
                pass

        self.watch(self.app, "title", set_title)
        self.watch(self.app, "sub_title", set_title)
        self.watch(self.screen, "title", set_title)
        self.watch(self.screen, "sub_title", set_title)

    Header._on_mount = patched_on_mount
except (ImportError, AttributeError):
    pass

# Patch redislite cleanup at import time to prevent 15-second shutdown hang
try:
    import redislite.client

    def noop_cleanup(*args, **kwargs):
        """No-op cleanup - skip shutdown entirely for tests."""
        pass

    redislite.client.RedisMixin._cleanup = noop_cleanup
except (ImportError, AttributeError):
    pass


def pytest_configure(config):
    """Override default marker expression when -m all is specified."""
    markexpr = config.getoption("-m", default="")
    if markexpr == "all":
        config.option.markexpr = ""


@pytest.fixture
def mock_redis():
    """Provide a mock Redis client for tests that need Redis operations.

    Returns a Mock object configured for EntitySidebar.refresh_entities() and
    editing presence detection patterns without breaking other operations.
    """
    from unittest.mock import Mock

    mock_redis = Mock()
    mock_redis.hget.return_value = None
    return mock_redis


def assert_worker_running(screen, worker_name):
    """Helper to check if a named worker is currently running on a screen.

    Args:
        screen: The Textual screen to check for workers
        worker_name: The name of the worker to look for

    Returns:
        True if worker is found and running, False otherwise
    """
    return any(w.name == worker_name and w.is_running for w in screen.workers)

