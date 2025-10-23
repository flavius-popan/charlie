"""MLX model loading utilities."""

import logging
import mlx_lm
import outlines

logger = logging.getLogger(__name__)

# Model path (matches settings.py)
DEFAULT_MODEL_PATH = ".models/mlx-community--Qwen3-4B-Instruct-2507-8bit"

def load_mlx_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Load MLX model and tokenizer.

    Args:
        model_path: Path to MLX model directory

    Returns:
        tuple: (mlx_model, mlx_tokenizer)
    """
    logger.info(f"Loading MLX model: {model_path}")
    mlx_model, mlx_tokenizer = mlx_lm.load(model_path)
    logger.info("MLX model loaded successfully")

    return mlx_model, mlx_tokenizer

def create_outlines_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Create Outlines model wrapper around MLX.

    Args:
        model_path: Path to MLX model directory

    Returns:
        outlines model ready for structured generation
    """
    logger.info(f"Creating Outlines wrapper for: {model_path}")
    mlx_model, mlx_tokenizer = load_mlx_model(model_path)

    outlines_model = outlines.from_mlxlm(mlx_model, mlx_tokenizer)
    logger.info("Outlines model ready")

    return outlines_model, mlx_tokenizer
