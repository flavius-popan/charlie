"""Tests for enabling the FalkorDB/Redis TCP endpoint via settings flag."""

from __future__ import annotations

import importlib

import backend.settings as settings


def _reload_lifecycle(monkeypatch, *, enabled: bool):
    """Reload lifecycle module with patched TCP flag."""

    monkeypatch.setattr(settings, "REDIS_TCP_ENABLED", enabled, raising=False)

    import backend.database.lifecycle as lifecycle

    # Reload to rebuild the module-level TCP config from patched settings.
    return importlib.reload(lifecycle)


def test_tcp_server_disabled_when_flag_false(monkeypatch):
    lifecycle = _reload_lifecycle(monkeypatch, enabled=False)

    assert lifecycle.get_tcp_server_endpoint() is None
    assert lifecycle._build_serverconfig() is None

    # Restore default disabled state for other tests
    _reload_lifecycle(monkeypatch, enabled=False)
