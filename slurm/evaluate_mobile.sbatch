#!/bin/bash
#SBATCH --job-name=evaluate_mobile
#SBATCH --output=/slurm/home/%u/output/%j/stdout.log
#SBATCH --error=/slurm/home/%u/output/%j/stderr.log
#SBATCH --partition=gpuMin
#SBATCH --gres=gpu:nvidia_l4:1
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

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
SCRATCH="${SLURM_TMPDIR:-/tmp}/horizon-$$"

MODEL_PATH="${MODEL_PATH:?ERROR: MODEL_PATH must be set}"
TEST_SET="${TEST_SET:-data/processed/eval.jsonl}"

# Resolve relative paths against REPO before the model dir is excluded below.
[[ "$MODEL_PATH" = /* ]] || MODEL_PATH="$REPO/$MODEL_PATH"
[[ "$TEST_SET" = /* ]] || TEST_SET="$REPO/$TEST_SET"

echo "Copying project to scratch: $SCRATCH"
mkdir -p "$SCRATCH"
rsync -a --exclude '.git' --exclude 'experiments' --exclude 'models' "$REPO/" "$SCRATCH/"

cleanup() {
    echo "Syncing results back to repo..."
    rsync -a "$SCRATCH/evaluation/reports/" "$REPO/evaluation/reports/" 2>/dev/null || true
    rm -rf "$SCRATCH"
}
trap cleanup EXIT INT TERM

cd "$SCRATCH"

source slurm/install_env.sh
source .env 2>/dev/null || true

cmd=(python -m evaluation.metrics.evaluate_mobile --model "$MODEL_PATH" --test-set "$TEST_SET")

if [ -n "${MAX_SAMPLES:-}" ]; then
    cmd+=(--max-samples "$MAX_SAMPLES")
fi

mkdir -p "$SCRATCH/evaluation/reports/mobile"

echo "Running: ${cmd[*]}"
echo "=========================================="
"${cmd[@]}" 2>&1 | tee "$SCRATCH/evaluation/reports/mobile/run.log"
exit_code=${PIPESTATUS[0]}

echo "=========================================="
echo "Evaluation exited with code: $exit_code"
exit "$exit_code"
