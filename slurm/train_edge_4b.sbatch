#!/usr/bin/env bash
# SLURM job: fine-tune Gemma 4 E4B → horizon-edge-4b on H100 (80GB VRAM).
#
# Usage:
#   sbatch slurm/train_edge_4b.sbatch
#   sbatch --export=RESUME_CHECKPOINT=experiments/edge-4b-xyz/checkpoint-1000 slurm/train_edge_4b.sbatch
#
# Run from: /slurm/home/$USER/safecircle/horizon

#SBATCH --job-name=horizon-edge-4b
#SBATCH --partition=gpuMax
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=38
#SBATCH --gres=gpu:nvidia_h100_nvl
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/slurm/home/%u/output/%j/terminal.out
#SBATCH --error=/slurm/home/%u/output/%j/terminal.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTDIR="/slurm/home/$USER/output/$SLURM_JOB_ID"
mkdir -p "$OUTDIR"

SCRATCH="${SLURM_TMPDIR:-/scratch/slurm/$USER/$SLURM_JOB_ID/tmp}"
JOBSCRATCH="${SCRATCH}/horizon-edge-4b-${SLURM_JOB_ID}"
mkdir -p "$JOBSCRATCH"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PERSISTENT_ARTIFACTS_DIR="/slurm/home/$USER/artifacts"

# ── Cleanup trap ──────────────────────────────────────────────────────────────
cleanup() {
    kill "${SYNC_PID:-}" 2>/dev/null || true
    echo "=== Final sync to home and persistent storage ==="
    rsync -a --ignore-missing-args \
        "$JOBSCRATCH/experiments/" "$SUBMIT_DIR/experiments/" 2>/dev/null || true
    rsync -a --ignore-missing-args \
        "$JOBSCRATCH/experiments/" "$PERSISTENT_ARTIFACTS_DIR/experiments/" 2>/dev/null || true
    echo "=== Sync complete ==="
}
trap cleanup EXIT INT TERM

# ── Copy project to scratch ───────────────────────────────────────────────────
echo "=== Copying project to scratch ==="
rsync -a --exclude='.venv' --exclude='experiments' --exclude='data/raw' \
    "$SUBMIT_DIR/" "$JOBSCRATCH/"

mkdir -p "$JOBSCRATCH/data/processed"
rsync -a "$SUBMIT_DIR/data/processed/" "$JOBSCRATCH/data/processed/"

if [ -n "${RESUME_CHECKPOINT:-}" ]; then
    echo "=== Copying checkpoint for resume: $RESUME_CHECKPOINT ==="
    mkdir -p "$JOBSCRATCH/$(dirname "$RESUME_CHECKPOINT")"
    if [ -d "$SUBMIT_DIR/$RESUME_CHECKPOINT" ]; then
        rsync -a "$SUBMIT_DIR/$RESUME_CHECKPOINT/" "$JOBSCRATCH/$RESUME_CHECKPOINT/"
    else
        rsync -a "$PERSISTENT_ARTIFACTS_DIR/$RESUME_CHECKPOINT/" "$JOBSCRATCH/$RESUME_CHECKPOINT/"
    fi
fi

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
export ARTIFACTS_DIR="$JOBSCRATCH/experiments"
export PERSISTENT_ARTIFACTS_DIR="/slurm/home/$USER/artifacts"
mkdir -p "$HF_HOME" "$ARTIFACTS_DIR" "$PERSISTENT_ARTIFACTS_DIR"

# ── Train ─────────────────────────────────────────────────────────────────────
cd "$JOBSCRATCH"

RESUME_FLAG=""
if [ -n "${RESUME_CHECKPOINT:-}" ]; then
    RESUME_FLAG="--resume $RESUME_CHECKPOINT"
fi

while true; do
    sleep 600
    rsync -a --ignore-missing-args \
        "$JOBSCRATCH/experiments/" "$PERSISTENT_ARTIFACTS_DIR/experiments/" 2>/dev/null || true
done &
SYNC_PID=$!

echo "=== Starting Gemma 4 E4B fine-tune → horizon-edge-4b (Job $SLURM_JOB_ID) ==="
python -m training.scripts.train \
    --config training/configs/gemma4_e4b_h100.yaml \
    $RESUME_FLAG \
    2>&1 | tee "$OUTDIR/training.log"

echo "=== Training finished ==="
echo ""
echo "Next steps:"
echo "  1. Merge LoRA:  python scripts/merge_lora.py --checkpoint experiments/edge-4b-<ts>/final --output models/horizon-edge-4b-merged"
echo "  2. Upload:      python scripts/upload_to_hf.py --what edge-4b --edge-4b-dir models/horizon-edge-4b-merged"
