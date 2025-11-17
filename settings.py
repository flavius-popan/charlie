"""Configuration for graphiti-poc.py"""

from pathlib import Path
import os

# Disable tokenizers parallelism to avoid fork warnings when running optimizers in parallel
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _env_flag(name: str, default: bool = False) -> bool:
    """Return True if the environment variable is set to a truthy value."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Ensure DSPy uses a writable cache directory inside the repository.
_DSPY_CACHE_DIR = Path(__file__).resolve().parent / "prompts" / ".dspy_cache"
_DSPY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
for _env_var in ("DSPY_CACHEDIR", "DSPY_CACHE_DIR", "DSPY_CACHE"):
    os.environ.setdefault(_env_var, str(_DSPY_CACHE_DIR))

# Model
DEFAULT_MODEL_PATH = "mlx-community/Qwen3-4B-Instruct-2507-4bit-DWQ-2510"

# Database
DB_PATH = Path("data/graphiti-poc.db")
GRAPH_NAME = "phase1_poc"

# Optional FalkorDB Lite TCP debug endpoint (defaults to unix socket only).
FALKORLITE_TCP_HOST = os.getenv("FALKORLITE_TCP_HOST", "0.0.0.0")
FALKORLITE_TCP_PORT = int(os.getenv("FALKORLITE_TCP_PORT", "6380"))
FALKORLITE_TCP_PASSWORD = os.getenv("FALKORLITE_TCP_PASSWORD") or None
FALKORLITE_TCP_ENABLED_BY_DEFAULT = _env_flag("FALKORLITE_TCP_ENABLED", False)

GROUP_ID = "phase1-poc"
EPISODE_CONTEXT_WINDOW = 3  # Mirrors Graphiti's default (EPISODE_WINDOW_LEN)

# Model generation parameters
# Supported: temp, top_p, min_p, min_tokens_to_keep, top_k, max_tokens
MODEL_CONFIG = {
    "temp": 0.0,
    "top_p": 1.0,
    "min_p": 0.0,
    "max_tokens": 2048,
}

# GEPA Reflection Model Configuration
REFLECTION_MODEL = os.getenv("REFLECTION_MODEL", "gpt-4o-mini")
REFLECTION_TEMPERATURE = float(os.getenv("REFLECTION_TEMPERATURE", "0.7"))  # Standard for GEPA tutorials - balances diversity with focus
REFLECTION_MAX_TOKENS = int(
    os.getenv("REFLECTION_MAX_TOKENS", "2048")
)  # Standard for GEPA reflection feedback

GEPA_REFLECTION_MINIBATCH_SIZE = 3  # Number of examples per reflection iteration

# GEPA Optimization Configuration
GEPA_MAX_FULL_EVALS = 1  # Single iteration to prevent over-complication
