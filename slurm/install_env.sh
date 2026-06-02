#!/usr/bin/env bash
# Idempotent environment setup for ANTS SLURM cluster.
# Called from inside $JOBSCRATCH after code has been rsynced there.
# Loads CUDA, creates/activates a venv, and installs project deps via uv.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── CUDA / modules ────────────────────────────────────────────────────────────
module purge
module load cuda/12.4 2>/dev/null || module load cuda 2>/dev/null || true
module load python/3.13 2>/dev/null || true

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
