#!/usr/bin/env bash
# SLURM job: profile RAM and battery of the LiteRT mobile model.
#
# Usage:
#   sbatch slurm/profile_mobile.sbatch
#   sbatch --export=ALL,MODEL_PATH=models/mobile-lite/horizon-mobile-int4_q4_block128_ekv1280.litertlm slurm/profile_mobile.sbatch
#   sbatch --export=ALL,MODEL_PATH=models/mobile-standard/horizon-mobile-int8_q8_ekv1280.litertlm,RUNS=20 slurm/profile_mobile.sbatch

#SBATCH --job-name=horizon-profile
#SBATCH --partition=gpuMin
#SBATCH --gres=gpu:nvidia_l4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/slurm/home/%u/output/%j/terminal.out
#SBATCH --error=/slurm/home/%u/output/%j/terminal.err

set -euo pipefail

OUTDIR="/slurm/home/$USER/output"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

MODEL_PATH="${MODEL_PATH:-models/mobile-standard/horizon-mobile-int8_q8_ekv1280.litertlm}"
RUNS="${RUNS:-10}"

echo "=== Profile mobile model: $MODEL_PATH ==="
echo "=== Runs: $RUNS ==="
echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo ""

if [ ! -f "$SUBMIT_DIR/$MODEL_PATH" ]; then
    echo "ERROR: model not found: $SUBMIT_DIR/$MODEL_PATH"
    echo "Available .litertlm files:"
    find "$SUBMIT_DIR/models" -name "*.litertlm" 2>/dev/null || echo "  (none)"
    exit 1
fi

# ── Environment ───────────────────────────────────────────────────────────────
echo "=== Setting up environment ==="
source "$SUBMIT_DIR/slurm/install_env.sh"

if [ -f "$SUBMIT_DIR/.env" ]; then
    set -a; source "$SUBMIT_DIR/.env"; set +a
fi

export PYTHONUNBUFFERED=1

# ── Run profiler ──────────────────────────────────────────────────────────────
cd "$SUBMIT_DIR"

echo "=== Running profiler ==="
python -m evaluation.profile_mobile \
    --model "$MODEL_PATH" \
    --runs "$RUNS" \
    --output "evaluation/reports/profile.json" \
    2>&1 | tee "$OUTDIR/profile.log"

echo "=== Profiling finished ==="
