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


# Centralized prompt + cache directories live under pipeline/prompts so a single
# `rm -rf pipeline/prompts` wipes every optimizer artifact.
PROMPTS_DIR = Path(__file__).resolve().parent / "pipeline" / "prompts"
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

# Ensure DSPy uses a writable cache directory inside pipeline/prompts.
_DSPY_CACHE_DIR = PROMPTS_DIR / ".dspy_cache"
_DSPY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
for _env_var in ("DSPY_CACHEDIR", "DSPY_CACHE_DIR", "DSPY_CACHE"):
    os.environ.setdefault(_env_var, str(_DSPY_CACHE_DIR))

# Consolidated GEPA outputs live beneath pipeline/prompts as well.
GEPA_OUTPUT_DIR = PROMPTS_DIR / "gepa"
GEPA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model
DEFAULT_MODEL_PATH = "unsloth/Qwen3-4B-Instruct-2507-GGUF"

# PyInstaller: Set LLAMA_GPU_LAYERS=0 for CPU-only builds (macOS/Linux/Windows)
LLAMA_CPP_GPU_LAYERS = int(os.getenv("LLAMA_GPU_LAYERS", "-1"))  # -1=auto GPU

# PyInstaller: Adjust LLAMA_CTX_SIZE for memory/speed trade-offs
LLAMA_CPP_N_CTX = int(os.getenv("LLAMA_CTX_SIZE", "4096"))

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

# Model generation parameters (supported: temp, top_p, top_k, min_p, presence_penalty, max_tokens)
MODEL_CONFIG = {
    "temp": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "max_tokens": 2048,
}

# GEPA Reflection Model Configuration
REFLECTION_MODEL = os.getenv("REFLECTION_MODEL", "gpt-4o-mini")
REFLECTION_TEMPERATURE = float(
    os.getenv("REFLECTION_TEMPERATURE", "0.7")
)  # Standard for GEPA tutorials - balances diversity with focus
REFLECTION_MAX_TOKENS = int(
    os.getenv("REFLECTION_MAX_TOKENS", "2048")
)  # Standard for GEPA reflection feedback

GEPA_REFLECTION_MINIBATCH_SIZE = 3  # Number of examples per reflection iteration

# GEPA Optimization Configuration
GEPA_MAX_FULL_EVALS = 3  # Standard tutorial value - enough budget for demo selection
GEPA_NUM_THREADS = (
    1  # Disable multiprocessing - llama.cpp models cannot be safely forked
)
