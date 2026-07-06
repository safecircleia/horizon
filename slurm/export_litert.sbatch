#!/usr/bin/env bash
# SLURM job: export mobile LoRA checkpoint to both LiteRT-LM variants.
# Produces horizon-mobile-int8.litertlm (6GB+ phones) and
#          horizon-mobile-int4.litertlm (4GB phones).
#
# Usage:
#   sbatch --export=CHECKPOINT=experiments/mobile-<timestamp>/final slurm/export_litert.sbatch
#
# Run from: /slurm/home/$USER/safecircle/horizon

#SBATCH --job-name=horizon-export
#SBATCH --partition=gpuMin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/slurm/home/%u/output/%j.out
#SBATCH --error=/slurm/home/%u/output/%j.err

set -euo pipefail

if [ -z "${CHECKPOINT:-}" ]; then
    echo "Error: CHECKPOINT is required."
    echo "Usage: sbatch --export=CHECKPOINT=experiments/mobile-<ts>/final slurm/export_litert.sbatch"
    exit 1
fi

OUTDIR="/slurm/home/$USER/output"

SCRATCH="${SLURM_TMPDIR:-/scratch/slurm/$USER/$SLURM_JOB_ID/tmp}"
JOBSCRATCH="${SCRATCH}/horizon-export-${SLURM_JOB_ID}"
mkdir -p "$JOBSCRATCH"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

cleanup() {
    echo "=== Syncing exported models back to home ==="
    rsync -a --ignore-missing-args \
        "$JOBSCRATCH/models/mobile-standard/" "$SUBMIT_DIR/models/mobile-standard/" 2>/dev/null || true
    rsync -a --ignore-missing-args \
        "$JOBSCRATCH/models/mobile-lite/" "$SUBMIT_DIR/models/mobile-lite/" 2>/dev/null || true
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

uv pip install --quiet tensorflow litert-torch litert-lm-builder

cd "$JOBSCRATCH"

# ── Export standard variant (INT8, 6GB+ phones) ───────────────────────────────
echo "=== Exporting INT8 variant (standard) ==="
python -m training.scripts.export_litert \
    --checkpoint "$CHECKPOINT" \
    --output models/mobile-standard \
    --quantization int8 \
    2>&1 | tee "$OUTDIR/export-int8.log"

# ── Export lite variant (INT4, 4GB phones) ────────────────────────────────────
echo "=== Exporting INT4 variant (lite) ==="
python -m training.scripts.export_litert \
    --checkpoint "$CHECKPOINT" \
    --output models/mobile-lite \
    --quantization int4 \
    --skip-merge \
    --merged-dir models/mobile-standard/merged \
    2>&1 | tee "$OUTDIR/export-int4.log"

echo ""
echo "=== Export complete ==="
echo "  INT8 (standard): models/mobile-standard/horizon-mobile-int8.litertlm"
echo "  INT4 (lite):     models/mobile-lite/horizon-mobile-int4.litertlm"
