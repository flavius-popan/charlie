#!/usr/bin/env python3
"""Local model management for MLX models."""

import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import mlx_lm.manage

MODELS_DIR = Path(__file__).parent / ".models"
MODELS_DIR.mkdir(exist_ok=True)

PATTERNS = [
    "*.json",
    "model*.safetensors",
    "*.py",
    "tokenizer.model",
    "*.tiktoken",
    "*.txt",
    "*.jsonl",
    "*.jinja",
]


def download(repo: str) -> Path:
    """Download model to .models directory."""
    name = repo.replace("/", "--")
    local = MODELS_DIR / name
    if local.exists():
        print(f"Model exists: {local}")
        return local
    print(f"Downloading {repo}...")
    path = snapshot_download(repo, local_dir=str(local), allow_patterns=PATTERNS)
    print(f"Saved to: {path}")
    return Path(path)


def list_models():
    """List local models."""
    if not any(MODELS_DIR.iterdir()):
        print("No local models")
        return
    for model in sorted(MODELS_DIR.iterdir()):
        if model.is_dir():
            size = sum(f.stat().st_size for f in model.rglob("*") if f.is_file())
            print(f"{model.name:<50} {size / 1e6:>8.1f}M")


def scan_cache():
    """Scan HuggingFace cache."""
    mlx_lm.manage.main()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"

    if cmd == "download" and len(sys.argv) > 2:
        download(sys.argv[2])
    elif cmd == "list":
        list_models()
    elif cmd == "cache":
        sys.argv = ["models.py", "--scan"]
        scan_cache()
    else:
        print("Usage:")
        print("  python models.py download <repo>  # Download to .models/")
        print("  python models.py list              # List local models")
        print("  python models.py cache             # Scan HF cache")
