#!/usr/bin/env bash
# SLURM job: run accuracy benchmark against held-out benchmark set.
# Exits non-zero if targets are missed or regression > 0.5% detected.
#
# Usage:
#   sbatch slurm/benchmark.sbatch --export=CHECKPOINT=experiments/run-xxx/final
#   sbatch slurm/benchmark.sbatch --export=CHECKPOINT=experiments/run-xxx/final,SAVE_BASELINE=1

#SBATCH --job-name=horizon-benchmark
#SBATCH --partition=gpuUltra
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --output=/slurm/home/%u/output/%j/terminal.out
#SBATCH --error=/slurm/home/%u/output/%j/terminal.err

set -euo pipefail

if [ -z "${CHECKPOINT:-}" ]; then
    echo "Error: CHECKPOINT is required."
    echo "Usage: sbatch slurm/benchmark.sbatch --export=CHECKPOINT=experiments/run-xxx/final"
    exit 1
fi

OUTDIR="/slurm/home/$USER/output"

SCRATCH="${SLURM_TMPDIR:-/scratch/slurm/$USER/$SLURM_JOB_ID/tmp}"
JOBSCRATCH="${SCRATCH}/horizon-bench-${SLURM_JOB_ID}"
mkdir -p "$JOBSCRATCH"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

cleanup() {
    echo "=== Syncing benchmark results back to home ==="
    rsync -a --ignore-missing-args \
        "$JOBSCRATCH/evaluation/reports/benchmark/" "$SUBMIT_DIR/evaluation/reports/benchmark/" 2>/dev/null || true
    # Sync baseline if updated
    rsync -a --ignore-missing-args \
        "$JOBSCRATCH/evaluation/reports/baseline.json" "$SUBMIT_DIR/evaluation/reports/baseline.json" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── Copy project + data to scratch ───────────────────────────────────────────
echo "=== Copying project to scratch ==="
rsync -a --exclude='.venv' --exclude='data' --exclude='experiments' --exclude='models' \
    "$SUBMIT_DIR/" "$JOBSCRATCH/"

echo "=== Copying benchmark split ==="
mkdir -p "$JOBSCRATCH/data/evaluation"
rsync -a "$SUBMIT_DIR/data/evaluation/benchmark.jsonl" "$JOBSCRATCH/data/evaluation/" 2>/dev/null || true

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

cd "$JOBSCRATCH"

# Create benchmark split if it doesn't exist yet
if [ ! -f "data/evaluation/benchmark.jsonl" ]; then
    echo "=== Creating benchmark split ==="
    python -m data.scripts.create_benchmark_split
fi

SAVE_FLAG=""
if [ "${SAVE_BASELINE:-0}" = "1" ]; then
    SAVE_FLAG="--save-baseline"
fi

echo "=== Running benchmark: $CHECKPOINT (Job $SLURM_JOB_ID) ==="
python -m evaluation.benchmark \
    --checkpoint "$CHECKPOINT" \
    --batch-size 16 \
    $SAVE_FLAG \
    2>&1 | tee "$OUTDIR/benchmark.log"

echo "=== Benchmark finished ==="
