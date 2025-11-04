"""Root pytest configuration for all tests."""

import pytest
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
