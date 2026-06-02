#!/usr/bin/env bash
# SLURM job: train horizon main model on H100 NVL (96 GB VRAM).
# Usage:
#   sbatch slurm/train_h100.sbatch
#   sbatch slurm/train_h100.sbatch --export=RESUME_CHECKPOINT=experiments/run-xyz/checkpoints/step-5000
#
# Run from: /slurm/home/$USER/safecircle/horizon

#SBATCH --job-name=horizon-h100
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=38
#SBATCH --gres=H_100_NVL
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=/slurm/home/%u/output/%j/terminal.out
#SBATCH --error=/slurm/home/%u/output/%j/terminal.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTDIR="/slurm/home/$USER/output/$SLURM_JOB_ID"
mkdir -p "$OUTDIR"

SCRATCH="${SLURM_TMPDIR:-/scratch/slurm/$USER/$SLURM_JOB_ID/tmp}"
JOBSCRATCH="${SCRATCH}/horizon-${SLURM_JOB_ID}"
mkdir -p "$JOBSCRATCH"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

# ── Cleanup trap — copy results back on exit/timeout ─────────────────────────
cleanup() {
    echo "=== Syncing experiments back to home ==="
    rsync -a --ignore-missing-args \
        "$JOBSCRATCH/experiments/" "$SUBMIT_DIR/experiments/" 2>/dev/null || true
    rsync -a --ignore-missing-args \
        "$JOBSCRATCH/data/.tokenized_cache/" "$SUBMIT_DIR/data/.tokenized_cache/" 2>/dev/null || true
    echo "=== Sync complete ==="
}
trap cleanup EXIT INT TERM

# ── Copy project to fast scratch ──────────────────────────────────────────────
echo "=== Copying project to scratch ==="
rsync -a --exclude='.venv' --exclude='experiments' --exclude='data/raw' \
    "$SUBMIT_DIR/" "$JOBSCRATCH/"

mkdir -p "$JOBSCRATCH/data/processed"
rsync -a "$SUBMIT_DIR/data/processed/" "$JOBSCRATCH/data/processed/"

# Copy existing checkpoint if resuming
if [ -n "${RESUME_CHECKPOINT:-}" ]; then
    echo "=== Copying checkpoint for resume: $RESUME_CHECKPOINT ==="
    mkdir -p "$JOBSCRATCH/$(dirname "$RESUME_CHECKPOINT")"
    rsync -a "$SUBMIT_DIR/$RESUME_CHECKPOINT/" "$JOBSCRATCH/$RESUME_CHECKPOINT/"
fi

# Copy tokenized cache if it exists (avoids re-tokenization)
if [ -d "$SUBMIT_DIR/data/.tokenized_cache" ]; then
    rsync -a "$SUBMIT_DIR/data/.tokenized_cache/" "$JOBSCRATCH/data/.tokenized_cache/"
fi

# ── Environment ───────────────────────────────────────────────────────────────
echo "=== Setting up environment ==="
source "$JOBSCRATCH/slurm/install_env.sh"

if [ -f "$SUBMIT_DIR/.env" ]; then
    set -a; source "$SUBMIT_DIR/.env"; set +a
fi

export HF_HOME="$JOBSCRATCH/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "$HF_HOME"

# ── Launch training ───────────────────────────────────────────────────────────
cd "$JOBSCRATCH"

RESUME_FLAG=""
if [ -n "${RESUME_CHECKPOINT:-}" ]; then
    RESUME_FLAG="--resume $RESUME_CHECKPOINT"
fi

echo "=== Starting H100 training (Job $SLURM_JOB_ID) ==="
python -m training.scripts.train \
    --config training/configs/h100.yaml \
    $RESUME_FLAG \
    2>&1 | tee "$OUTDIR/training.log"

echo "=== Training finished ==="
