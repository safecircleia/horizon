#!/usr/bin/env bash
# SLURM job: merge LoRA adapter into base model (CPU only, 32GB RAM).
# Usage:
#   sbatch --export=CHECKPOINT=experiments/h100-combined-<timestamp>/final slurm/merge_lora.sbatch
#
# Run from: /slurm/home/$USER/safecircle/horizon

#SBATCH --job-name=horizon-merge
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/slurm/home/%u/output/%j/terminal.out
#SBATCH --error=/slurm/home/%u/output/%j/terminal.err

set -euo pipefail

if [ -z "${CHECKPOINT:-}" ]; then
    echo "Error: CHECKPOINT is required."
    echo "Usage: sbatch --export=CHECKPOINT=experiments/h100-combined-<ts>/final slurm/merge_lora.sbatch"
    exit 1
fi

OUTPUT="${OUTPUT:-models/horizon-full-merged}"

OUTDIR="/slurm/home/$USER/output"

SCRATCH="${SLURM_TMPDIR:-/scratch/slurm/$USER/$SLURM_JOB_ID/tmp}"
JOBSCRATCH="${SCRATCH}/horizon-merge-${SLURM_JOB_ID}"
mkdir -p "$JOBSCRATCH"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

cleanup() {
    echo "=== Syncing merged model back to home ==="
    rsync -a --ignore-missing-args \
        "$JOBSCRATCH/$OUTPUT/" "$SUBMIT_DIR/$OUTPUT/" 2>/dev/null || true
    echo "=== Sync complete ==="
}
trap cleanup EXIT INT TERM

# ── Copy project + checkpoint to scratch ─────────────────────────────────────
rsync -a --exclude='.venv' --exclude='data' --exclude='experiments' \
    "$SUBMIT_DIR/" "$JOBSCRATCH/"

mkdir -p "$JOBSCRATCH/$(dirname "$CHECKPOINT")"
rsync -a "$SUBMIT_DIR/$CHECKPOINT/" "$JOBSCRATCH/$CHECKPOINT/"

# ── Environment ───────────────────────────────────────────────────────────────
source "$JOBSCRATCH/slurm/install_env.sh"

if [ -f "$SUBMIT_DIR/.env" ]; then
    set -a; source "$SUBMIT_DIR/.env"; set +a
fi

export HF_HOME="$JOBSCRATCH/.cache/huggingface"
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME"

cd "$JOBSCRATCH"

# ── Merge ─────────────────────────────────────────────────────────────────────
echo "=== Merging LoRA: $CHECKPOINT → $OUTPUT ==="
python scripts/merge_lora.py \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    2>&1 | tee "$OUTDIR/merge.log"

echo "=== Merge complete ==="
ls -lh "$OUTPUT/"
