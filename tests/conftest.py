"""Root pytest configuration for all tests."""

import sys
from pathlib import Path

# Add project root to sys.path so we can import settings
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
