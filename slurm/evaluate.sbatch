#!/usr/bin/env bash
# SLURM job: evaluate a trained checkpoint.
# Usage:
#   sbatch slurm/evaluate.sbatch --export=CHECKPOINT=experiments/h100-20250601/checkpoints/step-5000
#
# Run from: /slurm/home/$USER/safecircle/horizon

#SBATCH --job-name=horizon-eval
#SBATCH --partition=gpuUltra
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=/slurm/home/%u/output/%j/terminal.out
#SBATCH --error=/slurm/home/%u/output/%j/terminal.err

set -euo pipefail

if [ -z "${CHECKPOINT:-}" ]; then
    echo "Error: CHECKPOINT is required."
    echo "Usage: sbatch slurm/evaluate.sbatch --export=CHECKPOINT=experiments/run-xxx/checkpoints/step-5000"
    exit 1
fi

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTDIR="/slurm/home/$USER/output"

SCRATCH="${SLURM_TMPDIR:-/scratch/slurm/$USER/$SLURM_JOB_ID/tmp}"
JOBSCRATCH="${SCRATCH}/horizon-eval-${SLURM_JOB_ID}"
mkdir -p "$JOBSCRATCH"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

cleanup() {
    echo "=== Syncing eval results back to home ==="
    rsync -a --ignore-missing-args \
        "$JOBSCRATCH/evaluation/reports/" "$SUBMIT_DIR/evaluation/reports/" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── Copy project + checkpoint to scratch ─────────────────────────────────────
echo "=== Copying project to scratch ==="
rsync -a --exclude='.venv' --exclude='data' \
    "$SUBMIT_DIR/" "$JOBSCRATCH/"

echo "=== Copying eval dataset ==="
mkdir -p "$JOBSCRATCH/data/processed"
rsync -a "$SUBMIT_DIR/data/processed/eval.jsonl" "$JOBSCRATCH/data/processed/" || {
    echo "ERROR: data/processed/eval.jsonl not found — run the data pipeline first"
    exit 1
}

echo "=== Copying checkpoint: $CHECKPOINT ==="
mkdir -p "$JOBSCRATCH/$(dirname "$CHECKPOINT")"
rsync -a "$SUBMIT_DIR/$CHECKPOINT/" "$JOBSCRATCH/$CHECKPOINT/"

# ── Environment ───────────────────────────────────────────────────────────────
echo "=== Setting up environment ==="
source "$JOBSCRATCH/slurm/install_env.sh"

if [ -f "$SUBMIT_DIR/.env" ]; then
    set -a; source "$SUBMIT_DIR/.env"; set +a
fi

export HF_HOME="$JOBSCRATCH/.cache/huggingface"
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME"

# ── Run evaluation ────────────────────────────────────────────────────────────
cd "$JOBSCRATCH"

echo "=== Evaluating checkpoint: $CHECKPOINT (Job $SLURM_JOB_ID) ==="
python -m evaluation.metrics.evaluate \
    --checkpoint "$CHECKPOINT" \
    --test-set data/processed/eval.jsonl \
    --output evaluation/reports \
    2>&1 | tee "$OUTDIR/eval.log"

echo "=== Evaluation finished ==="
