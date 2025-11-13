"""Configuration for graphiti-poc.py"""

from pathlib import Path

# Model
DEFAULT_MODEL_PATH = "mlx-community/Qwen3-4B-Instruct-2507-4bit-DWQ-2510"

# Database
DB_PATH = Path("data/graphiti-poc.db")
GRAPH_NAME = "phase1_poc"

GROUP_ID = "phase1-poc"
EPISODE_CONTEXT_WINDOW = 3  # Mirrors Graphiti's default (EPISODE_WINDOW_LEN)

# Model generation parameters
# Supported: temp, top_p, min_p, min_tokens_to_keep, top_k, max_tokens
MODEL_CONFIG = {
    "temp": 0.0,
    "top_p": 1.0,
    "min_p": 0.0,
    "max_tokens": 4096,
}
