"""Backend configuration for Charlie."""

from __future__ import annotations

import os
from pathlib import Path

DB_PATH = Path(os.getenv("CHARLIE_DB_PATH", "data/charlie.db"))
DEFAULT_JOURNAL = "default"

# Single switch to expose FalkorDB/Redis over TCP. Set to True to enable.
REDIS_TCP_ENABLED = False

# TCP server defaults (bind to localhost for safety).
TCP_HOST = "127.0.0.1"
TCP_PORT = 6379
TCP_PASSWORD = None

# Model configuration
MODEL_REPO_ID = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
MODEL_QUANTIZATION = "Q8_0"

# llama.cpp inference settings (OS/hardware specific)
LLAMA_CPP_N_CTX = int(os.getenv("LLAMA_CTX_SIZE", "4096"))
LLAMA_CPP_GPU_LAYERS = int(os.getenv("LLAMA_GPU_LAYERS", "-1"))
LLAMA_CPP_VERBOSE = False

# Generation parameters
#
# Token budget for entity extraction (4096 context window):
#   - Output reserve: 1024 tokens (CoT reasoning + JSON entities)
#   - Prompt overhead: ~500 tokens (system prompt, entity types, field labels)
#   - Available for journal: ~2,500 tokens (~1,700 words)
#
# Entries over ~1,700 words will be silently truncated at the context boundary,
# potentially missing entities mentioned near the end. To increase capacity,
# set LLAMA_CTX_SIZE env var (model supports up to 32K).
#
# If extraction output exceeds max_tokens, JSON may be malformed. The task
# runner catches this and marks the episode as "dead" to prevent stuck state.
MODEL_CONFIG = {
    "temp": 0.0,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "max_tokens": 1024,
}

# Startup delay before model loading allowed (lets app initialize first)
MODEL_LOAD_GRACE_SECONDS = 10

# Orchestrator maintenance loop interval (1s for responsiveness)
ORCHESTRATOR_INTERVAL_SECONDS = 1

# Huey task queue configuration
HUEY_WORKER_TYPE = "thread"
HUEY_WORKERS = 1
