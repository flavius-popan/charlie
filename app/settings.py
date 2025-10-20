"""
Charlie Application Settings

Central configuration for database paths and other application-wide settings.
"""

# Database configuration
BRAIN_DIR: str = "brain"
DB_PATH: str = f"{BRAIN_DIR}/charlie.kuzu"

# Journal loading defaults
JOURNAL_FILE_PATH: str = "raw_data/Journal.json"
SOURCE_DESCRIPTION: str = "journal entry"

# LLM Backend Configuration (Phase 1: MLX only)
MLX_MODEL_NAME: str = ".models/mlx-community--Qwen3-4B-Instruct-2507-8bit"
