#!/usr/bin/env bash
# SLURM job: fine-tune Gemma 3 1B for LiteRT-LM mobile deployment (L40s GPU).
# After training completes, run export_litert.py on a workstation to produce
# the .litertlm file (ai_edge_torch requires Linux x86-64).
#
# Usage:
#   sbatch slurm/train_mobile.sbatch
#   sbatch --export=RESUME_CHECKPOINT=experiments/mobile-xyz/checkpoint-500 slurm/train_mobile.sbatch
#
# Run from: /slurm/home/$USER/safecircle/horizon

#SBATCH --job-name=horizon-mobile
#SBATCH --partition=gpu
#SBATCH --nodelist=slurm-gpu08
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:nvidia_l40s
#SBATCH --mem=60G
#SBATCH --time=20:00:00
#SBATCH --output=/slurm/home/%u/output/%j/terminal.out
#SBATCH --error=/slurm/home/%u/output/%j/terminal.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTDIR="/slurm/home/$USER/output/$SLURM_JOB_ID"
mkdir -p "$OUTDIR"

SCRATCH="${SLURM_TMPDIR:-/scratch/slurm/$USER/$SLURM_JOB_ID/tmp}"
JOBSCRATCH="${SCRATCH}/horizon-mobile-${SLURM_JOB_ID}"
mkdir -p "$JOBSCRATCH"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

# ── Cleanup trap — copy results back on exit/timeout/SIGTERM ─────────────────
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
    rsync -a "$SUBMIT_DIR/$RESUME_CHECKPOINT/" "$JOBSCRATCH/$RESUME_CHECKPOINT/"
fi

# ── Environment ───────────────────────────────────────────────────────────────
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

# ── Background checkpoint sync every 10 min ──────────────────────────────────
while true; do
    sleep 600
    rsync -a --ignore-missing-args \
        "$JOBSCRATCH/experiments/" "$PERSISTENT_ARTIFACTS_DIR/experiments/" 2>/dev/null || true
done &
SYNC_PID=$!

echo "=== Starting mobile fine-tune (Job $SLURM_JOB_ID) ==="
python -m training.scripts.train \
    --config training/configs/mobile.yaml \
    $RESUME_FLAG \
    2>&1 | tee "$OUTDIR/training.log"

echo "=== Training finished ==="
echo ""
echo "Next: export to LiteRT-LM on your workstation (requires ai-edge-torch):"
echo "  uv pip install ai-edge-torch litert-lm-builder"
echo "  python -m training.scripts.export_litert \\"
echo "      --checkpoint experiments/mobile-<timestamp>/final \\"
echo "      --output models/mobile"
