#!/usr/bin/env bash
set -euo pipefail

# Charlie Setup Script
# Handles first-run setup: venv, dependencies, model download, and verification

step() { echo -e "\n==> $1"; }
fail() { echo "ERROR: $1" >&2; exit 1; }

# =============================================================================
# 0. Pre-flight Checks
# =============================================================================

step "Pre-flight checks"

# Find suitable Python (3.12+)
find_python() {
    for cmd in python3.13 python3.12 python3 python; do
        if command -v "$cmd" &> /dev/null; then
            version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
            major=$(echo "$version" | cut -d. -f1)
            minor=$(echo "$version" | cut -d. -f2)
            if [[ "$major" -ge 3 && "$minor" -ge 12 ]]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON_CMD=$(find_python) || true

if [[ -z "$PYTHON_CMD" ]]; then
    echo "Python 3.12+ required but not found."

    # Offer to install via uv if available
    if command -v uv &> /dev/null; then
        read -p "Install Python 3.12 via uv? [Y/n] " install_python
        if [[ ! "$install_python" =~ ^[Nn]$ ]]; then
            echo "Installing Python 3.12..."
            uv python install 3.12
            PYTHON_CMD=$(find_python) || true
        fi
    fi

    if [[ -z "$PYTHON_CMD" ]]; then
        fail "Python 3.12+ required. Install via: uv python install 3.12, pyenv, or brew install python@3.12"
    fi
fi

echo "Using Python: $PYTHON_CMD ($($PYTHON_CMD --version))"

# Check internet connectivity (lightweight check before 4.5GB download)
if ! curl -s --head --connect-timeout 5 https://huggingface.co > /dev/null; then
    fail "Cannot reach huggingface.co. Check your internet connection."
fi
echo "Internet connectivity: OK"

# Check for port conflict from previous crash (hard fail - data corruption risk)
if lsof -i :6379 &>/dev/null 2>&1; then
    echo "ERROR: Port 6379 already in use."
    echo "This could be from a previous Charlie crash or another Redis instance."
    echo ""
    echo "To fix, find and kill the process:"
    echo "  lsof -i :6379  # Find the PID"
    echo "  kill <PID>     # Terminate it"
    fail "Cannot proceed with port 6379 in use (risk of data corruption)"
fi

# Check if Charlie is already running
if pgrep -f "charlie.py" > /dev/null 2>&1; then
    fail "Charlie appears to be already running. Stop it before running setup.sh"
fi

# =============================================================================
# 1. Virtual Environment Setup
# =============================================================================

step "Setting up virtual environment"

VENV_DIR=".venv"

# Check if venv exists and is complete
VENV_EXISTS=false
VENV_COMPLETE=false
if [[ -d "$VENV_DIR" ]]; then
    VENV_EXISTS=true
    # Check if installation is complete by trying to import backend
    if "$VENV_DIR/bin/python" -c "import backend" 2>/dev/null; then
        VENV_COMPLETE=true
    fi
fi

if [[ "$VENV_COMPLETE" == "true" ]]; then
    echo "Existing complete installation found, reusing venv"
    source "$VENV_DIR/bin/activate"
else
    if [[ "$VENV_EXISTS" == "true" ]]; then
        echo "Incomplete installation detected, recreating venv..."
        rm -rf "$VENV_DIR"
    fi

    # Prefer uv, fall back to pip
    if command -v uv &> /dev/null; then
        echo "Using uv for dependency management"
        uv venv "$VENV_DIR" --python "$PYTHON_CMD"
        source "$VENV_DIR/bin/activate"
        uv pip install -e ".[dev]"
    else
        echo "Using pip for dependency management"
        "$PYTHON_CMD" -m venv "$VENV_DIR"
        source "$VENV_DIR/bin/activate"
        pip install -e ".[dev]"
    fi
fi

# Verify activation succeeded
if [[ -z "$VIRTUAL_ENV" ]]; then
    fail "Failed to activate virtual environment"
fi

# Get venv bin path dynamically
VENV_BIN=$(python -c "import sys; print(sys.prefix)")/bin
echo "Virtual environment: $VIRTUAL_ENV"

# =============================================================================
# 2. FalkorDB Binary Fix
# =============================================================================

step "Applying FalkorDB binary fix (macOS workaround)"

# Per https://github.com/FalkorDB/falkordblite/issues/27
# Remove broken binaries with incorrect @loader_path references (macOS issue)
if [[ -f "$VENV_BIN/falkordb.so" ]] || [[ -f "$VENV_BIN/redis-cli" ]] || [[ -f "$VENV_BIN/redis-server" ]]; then
    rm -f "$VENV_BIN/falkordb.so" "$VENV_BIN/redis-cli" "$VENV_BIN/redis-server"
    echo "Removed problematic binaries from venv/bin"
else
    echo "No problematic binaries found (already cleaned or not macOS)"
fi

# =============================================================================
# 3. Model Download + Inference Verification
# =============================================================================

step "Downloading model and verifying inference"

echo "This may take several minutes on first run (~4.5GB download)"
echo "Model will be cached in ~/.cache/huggingface/"

python -c "
from backend.inference.loader import load_model

print('Loading model (downloading if needed)...')
llm = load_model()

print('Running inference test...')
result = llm.create_chat_completion(
    messages=[{'role': 'user', 'content': 'Say OK'}],
    max_tokens=8
)
content = result['choices'][0]['message']['content']
assert content.strip(), 'Empty response from model'
print(f'Inference OK: {content.strip()[:50]}')
" || fail "Model download or inference failed"

# =============================================================================
# 4. Database Verification
# =============================================================================

step "Initializing and verifying database"

echo "Setting up production database (safe to rerun - uses idempotent operations)"

python -c "
import asyncio
from backend.database import ensure_database_ready, get_falkordb_graph, shutdown_database

async def verify():
    # Initialize prod DB with 'default' journal (creates SELF entity 'I')
    await ensure_database_ready('default')

    # Verify graph responds to queries
    graph = get_falkordb_graph('default')
    result = graph.query('RETURN 1 as n')
    assert result.result_set[0][0] == 1

    print('Database OK: FalkorDB initialized and responding')
    shutdown_database()

asyncio.run(verify())
" || fail "Database initialization failed"

# =============================================================================
# 5. Journal Import (Optional)
# =============================================================================

step "Journal import"

echo "Available importers:"
echo "  - Text files (.txt, .md, .rtf): python importers/files.py <directory>"
echo "  - Day One export: python importers/dayone.py <export.zip>"
echo ""
read -p "Import text files now? [y/N] " import_choice
if [[ "$import_choice" =~ ^[Yy]$ ]]; then
    read -p "Path to journal directory: " journal_path
    if [[ -n "$journal_path" ]]; then
        echo ""
        echo "Options: --recursive, --date-source created|modified, --dry-run"
        read -p "Additional options (or press Enter for defaults): " import_opts
        python importers/files.py "$journal_path" $import_opts
    fi
fi

# =============================================================================
# 6. Launch Charlie
# =============================================================================

step "Setup complete!"

echo ""
echo "Charlie is ready to run."
echo ""
read -p "Launch Charlie now? [Y/n] " launch_choice
if [[ ! "$launch_choice" =~ ^[Nn]$ ]]; then
    python charlie.py
else
    echo ""
    echo "To start Charlie later, run:"
    echo "  source .venv/bin/activate"
    echo "  python charlie.py"
fi
