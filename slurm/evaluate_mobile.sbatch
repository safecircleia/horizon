#!/bin/bash
#SBATCH --job-name=evaluate_mobile
#SBATCH --output=/slurm/home/jtpalma/output/%j/stdout.log
#SBATCH --error=/slurm/home/jtpalma/output/%j/stderr.log
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jtpalma@safecircle.tech

# ---------------------------------------------------------------------------
# Mobile model evaluation (LiteRT-LM accuracy)
#
# Env vars:
#   MODEL_PATH  — path to .litertlm file (required)
#   TEST_SET    — path to eval JSONL (default: data/processed/eval.jsonl)
#   MAX_SAMPLES — limit examples (default: all)
# ---------------------------------------------------------------------------

set -euo pipefail

REPO="/slurm/home/jtpalma/safecircle/horizon"
SCRATCH="${SLURM_TMPDIR:-/tmp}/horizon-$$"

echo "Copying project to scratch: $SCRATCH"
mkdir -p "$SCRATCH"
rsync -a --exclude '.git' --exclude 'experiments' --exclude 'models' "$REPO/" "$SCRATCH/"

cleanup() {
    echo "Syncing results back to repo..."
    rsync -a "$SCRATCH/evaluation/reports/" "$REPO/evaluation/reports/" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$SCRATCH"

source slurm/install_env.sh
source .env 2>/dev/null || true

MODEL_PATH="${MODEL_PATH:?ERROR: MODEL_PATH must be set}"
TEST_SET="${TEST_SET:-data/processed/eval.jsonl}"

CMD="python -m evaluation.metrics.evaluate_mobile --model $MODEL_PATH --test-set $TEST_SET"

if [ -n "${MAX_SAMPLES:-}" ]; then
    CMD="$CMD --max-samples $MAX_SAMPLES"
fi

echo "Running: $CMD"
echo "=========================================="
eval "$CMD" 2>&1 | tee "$SCRATCH/evaluation/reports/mobile/run.log"
exit_code=${PIPESTATUS[0]}

echo "=========================================="
echo "Evaluation exited with code: $exit_code"
exit "$exit_code"
