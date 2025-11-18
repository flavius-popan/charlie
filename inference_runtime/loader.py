"""Helpers for loading llama.cpp models."""

from __future__ import annotations

import logging

from llama_cpp import Llama

from settings import LLAMA_CPP_GPU_LAYERS, LLAMA_CPP_N_CTX

logger = logging.getLogger(__name__)


def load_model(model_path: str | None = None) -> Llama:
    """Load Qwen3 GGUF model via HF auto-download."""
    logger.info("Loading llama.cpp model from Hugging Face")
    # PyInstaller: Model auto-downloads to ~/.cache/huggingface/ on first run
    llm = Llama.from_pretrained(
        repo_id="unsloth/Qwen3-4B-Instruct-2507-GGUF",
        filename="*Q4_K_M.gguf",
        n_ctx=LLAMA_CPP_N_CTX,
        n_gpu_layers=LLAMA_CPP_GPU_LAYERS,  # -1=auto GPU, 0=CPU only
        verbose=False,
    )
    logger.info("Loaded llama.cpp model successfully")
    return llm
