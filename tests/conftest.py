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

# Ensure project root is importable when running tests directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Patch redislite cleanup at import time to prevent 15-second shutdown hang
try:
    import redislite.client

    def noop_cleanup(*args, **kwargs):
        """No-op cleanup - skip shutdown entirely for tests."""
        pass

    redislite.client.RedisMixin._cleanup = noop_cleanup
except (ImportError, AttributeError):
    pass

