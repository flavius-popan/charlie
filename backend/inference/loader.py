"""Helpers for loading llama.cpp models."""

from __future__ import annotations

import contextlib
import logging
import os
import sys

from llama_cpp import Llama

from backend.settings import (
    MODEL_REPO_ID,
    MODEL_QUANTIZATION,
    LLAMA_CPP_GPU_LAYERS,
    LLAMA_CPP_N_CTX,
    LLAMA_CPP_VERBOSE,
)

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _suppress_stderr():
    """Temporarily redirect stderr to devnull to silence Metal backend warnings."""
    try:
        stderr_fd = sys.stderr.fileno()
        old_stderr = os.dup(stderr_fd)
    except OSError:
        # If stderr is unavailable (e.g., in some worker contexts), skip suppression.
        yield
        return

    with open(os.devnull, "w") as devnull:
        try:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
        finally:
            try:
                os.dup2(old_stderr, stderr_fd)
            finally:
                os.close(old_stderr)


def load_model(repo_id: str | None = None) -> Llama:
    """Load GGUF model via HuggingFace auto-download."""
    repo_id = repo_id or MODEL_REPO_ID
    filename_pattern = f"*{MODEL_QUANTIZATION}.gguf"

    logger.info("Loading llama.cpp model: repo=%s, quant=%s", repo_id, MODEL_QUANTIZATION)

    with _suppress_stderr():
        llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename_pattern,
            n_ctx=LLAMA_CPP_N_CTX,
            n_gpu_layers=LLAMA_CPP_GPU_LAYERS,
            verbose=LLAMA_CPP_VERBOSE,
        )

    logger.info("Loaded llama.cpp model successfully")
    return llm
