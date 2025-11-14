"""MLX model loading utilities."""

import logging
import mlx_lm
import outlines

from settings import DEFAULT_MODEL_PATH

logger = logging.getLogger(__name__)


def load_mlx_model(model_path: str = None):
    """
    Load MLX model and tokenizer.

    Args:
        model_path: Path to MLX model directory (uses DEFAULT_MODEL_PATH if None)

    Returns:
        tuple: (mlx_model, mlx_tokenizer)
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    logger.info(f"Loading MLX model: {model_path}")
    # mlx_lm.load() automatically checks ~/.cache/huggingface/hub/ first
    # Only downloads if not cached
    mlx_model, mlx_tokenizer = mlx_lm.load(model_path)
    logger.info("MLX model loaded successfully")

    return mlx_model, mlx_tokenizer


def create_outlines_model(model_path: str = None, *, enable_prompt_cache: bool = False):
    """
    Create Outlines model wrapper around MLX.

    Args:
        model_path: Path to MLX model directory (uses DEFAULT_MODEL_PATH if None)
        enable_prompt_cache: Whether to build and return an MLX prompt cache

    Returns:
        tuple: (outlines_model, mlx_tokenizer, prompt_cache or None)
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    logger.info(f"Creating Outlines wrapper for: {model_path}")
    mlx_model, mlx_tokenizer = load_mlx_model(model_path)

    prompt_cache = None
    if enable_prompt_cache:
        from mlx_lm.models.cache import make_prompt_cache

        logger.info("Initializing MLX prompt cache")
        prompt_cache = make_prompt_cache(mlx_model)

    outlines_model = outlines.from_mlxlm(mlx_model, mlx_tokenizer)
    logger.info("Outlines model ready")

    return outlines_model, mlx_tokenizer, prompt_cache
