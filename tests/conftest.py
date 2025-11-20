"""Root pytest configuration for all tests."""

import sys
from pathlib import Path

import pytest

# Add project root to sys.path so we can import settings
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from settings import DEFAULT_MODEL_PATH


def pytest_addoption(parser):
    """Register custom CLI options for test configuration."""
    parser.addoption(
        "--model",
        action="store",
        default=DEFAULT_MODEL_PATH,
        help="HuggingFace model path (default: %(default)s)",
    )


@pytest.fixture
def model_path(request):
    """Provide the model path from CLI flag for any test that needs it."""
    return request.config.getoption("--model")
