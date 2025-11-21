from __future__ import annotations

import sys
from unittest.mock import patch

from charlie import CharlieApp
from backend.settings import HUEY_WORKER_TYPE, HUEY_WORKERS


def test_build_huey_command_prefers_console_script():
    app = CharlieApp()
    console_path = "/opt/bin/huey_consumer"

    with patch("charlie.shutil.which", return_value=console_path):
        cmd = app._build_huey_command()

    assert cmd[0] == console_path
    assert cmd[1] == "backend.services.tasks.huey"
    assert ["-k", HUEY_WORKER_TYPE, "-w", str(HUEY_WORKERS), "-q"] == cmd[2:]


def test_build_huey_command_module_fallback_when_console_missing():
    app = CharlieApp()

    with patch("charlie.shutil.which", return_value=None):
        cmd = app._build_huey_command()

    assert cmd[:3] == [sys.executable, "-m", "huey.bin.huey_consumer"]
    assert cmd[3] == "backend.services.tasks.huey"
    assert ["-k", HUEY_WORKER_TYPE, "-w", str(HUEY_WORKERS), "-q"] == cmd[4:]
