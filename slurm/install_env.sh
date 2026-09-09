#!/usr/bin/env bash
# Idempotent environment setup for ANTS SLURM cluster.
# Called from inside $JOBSCRATCH after code has been rsynced there.
# Loads CUDA, creates/activates a venv, and installs project deps via uv.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Ensure ~/.local/bin is in PATH (uv lives there on ANTS)
export PATH="$HOME/.local/bin:$PATH"

# ── CUDA setup ────────────────────────────────────────────────────────────────
# ANTS GPU nodes have CUDA at /usr/local/cuda but don't set LD_LIBRARY_PATH
# in non-interactive SLURM sessions, causing driver/library mismatches.
if [ -d /usr/local/cuda ]; then
    export CUDA_HOME="/usr/local/cuda"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

echo "CUDA: $(nvcc --version 2>/dev/null | head -1 || echo 'not found via nvcc')"
echo "Python: $(python3 --version)"

# ── Virtual environment ───────────────────────────────────────────────────────
VENV="$PROJECT_ROOT/.venv"

if [ ! -f "$VENV/bin/activate" ]; then
    echo "Creating venv at $VENV..."
    python3 -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

# ── Dependencies ──────────────────────────────────────────────────────────────
echo "Installing deps with uv..."
uv pip install --quiet -r "$PROJECT_ROOT/requirements.txt"

echo "Environment ready. Python: $(python3 -c 'import sys; print(sys.executable)')"
echo "PyTorch CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
