#!/usr/bin/env bash
# SLURM job: export merged Gemma 4 E2B/E4B to all LiteRT-LM variants.
#
# Usage:
#   sbatch --export=MODEL_SIZE=e2b slurm/export_edge.sbatch
#   sbatch --export=MODEL_SIZE=e4b slurm/export_edge.sbatch
#
# Expects the merged model at models/horizon-edge-{MODEL_SIZE}-merged/
# Run from: /slurm/home/$USER/safecircle/horizon

#SBATCH --job-name=horizon-export-edge
#SBATCH --partition=gpuUltra
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:nvidia_h100_nvl
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/slurm/home/%u/output/%j/terminal.out
#SBATCH --error=/slurm/home/%u/output/%j/terminal.err

set -euo pipefail

MODEL_SIZE="${MODEL_SIZE:-e2b}"
# Map size code to directory names (merge uses "2b"/"4b", not "e2b"/"e4b")
if [ "$MODEL_SIZE" = "e2b" ]; then
    MODEL_DIR="models/horizon-edge-2b-merged"
    OUTPUT_DIR="models/horizon-edge-2b-litert"
elif [ "$MODEL_SIZE" = "e4b" ]; then
    MODEL_DIR="models/horizon-edge-4b-merged"
    OUTPUT_DIR="models/horizon-edge-4b-litert"
else
    MODEL_DIR="models/horizon-edge-${MODEL_SIZE}-merged"
    OUTPUT_DIR="models/horizon-edge-${MODEL_SIZE}-litert"
fi

if [ -z "${MODEL_SIZE:-}" ]; then
    echo "Error: MODEL_SIZE is required (e2b or e4b)."
    echo "Usage: sbatch --export=MODEL_SIZE=e2b slurm/export_edge.sbatch"
    exit 1
fi

OUTDIR="/slurm/home/$USER/output/$SLURM_JOB_ID"
mkdir -p "$OUTDIR"

SCRATCH="${SLURM_TMPDIR:-/scratch/slurm/$USER/$SLURM_JOB_ID/tmp}"
JOBSCRATCH="${SCRATCH}/horizon-export-edge-${SLURM_JOB_ID}"
mkdir -p "$JOBSCRATCH"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

cleanup() {
    echo "=== Syncing exported models back to home ==="
    rsync -a --ignore-missing-args \
        "$JOBSCRATCH/$OUTPUT_DIR/" "$SUBMIT_DIR/$OUTPUT_DIR/" 2>/dev/null || true
    echo "=== Sync complete ==="
}
trap cleanup EXIT INT TERM

# ── Copy project + merged model to scratch ────────────────────────────────────
echo "=== Copying project to scratch ==="
rsync -a --exclude='.venv' --exclude='data' --exclude='experiments' \
    "$SUBMIT_DIR/" "$JOBSCRATCH/"

echo "=== Copying merged model ($MODEL_DIR) ==="
mkdir -p "$JOBSCRATCH/$MODEL_DIR"
rsync -a "$SUBMIT_DIR/$MODEL_DIR/" "$JOBSCRATCH/$MODEL_DIR/"

# ── Environment ───────────────────────────────────────────────────────────────
source "$JOBSCRATCH/slurm/install_env.sh"

if [ -f "$SUBMIT_DIR/.env" ]; then
    set -a; source "$SUBMIT_DIR/.env"; set +a
fi

export HF_HOME="$JOBSCRATCH/.cache/huggingface"
mkdir -p "$HF_HOME"

echo "=== Installing litert-torch ==="
uv pip install litert-torch-nightly
PYEOF

cd "$JOBSCRATCH"

# ── Export all variants ───────────────────────────────────────────────────────
echo "=== Exporting all LiteRT-LM variants for horizon-edge-${MODEL_SIZE} ==="
python -m training.scripts.export_edge \
    --model-dir "$MODEL_DIR" \
    --output "$OUTPUT_DIR" \
    --model-size "$MODEL_SIZE" \
    2>&1 | tee "$OUTDIR/export.log"

echo ""
echo "=== Export complete ==="
echo "Next: python scripts/upload_to_hf.py --what edge-${MODEL_SIZE} --edge-${MODEL_SIZE}-dir $OUTPUT_DIR"
