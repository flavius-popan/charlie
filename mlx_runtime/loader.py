"""Helpers for loading MLX models."""

from __future__ import annotations

import logging
from typing import Tuple

import mlx_lm

from settings import DEFAULT_MODEL_PATH

logger = logging.getLogger(__name__)


def load_mlx_model(model_path: str | None = None) -> Tuple[object, object]:
    """Load an MLX model + tokenizer, defaulting to the configured path."""
    path = model_path or DEFAULT_MODEL_PATH
    logger.info("Loading MLX model: %s", path)
    model, tokenizer = mlx_lm.load(path)
    logger.info("Loaded MLX model successfully")
    return model, tokenizer
