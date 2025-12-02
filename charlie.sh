#!/usr/bin/env bash
set -euo pipefail

# Charlie Run Script
# Simple wrapper that activates venv and runs Charlie

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

source "$VENV_DIR/bin/activate"
exec python "$SCRIPT_DIR/charlie.py" "$@"
