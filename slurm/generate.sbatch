#!/usr/bin/env bash
# SLURM job: bulk dataset generation using local vLLM on H100.
# Starts vLLM, runs all 8 categories in parallel, syncs results back on exit.
#
# Usage:
#   sbatch slurm/generate.sbatch
#   sbatch slurm/generate.sbatch --export=LANGUAGE=es
#   sbatch slurm/generate.sbatch --export=LANGUAGE=mixed,CONCURRENCY=150
#
# Run from: /slurm/home/$USER/safecircle/horizon

#SBATCH --job-name=horizon-generate
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

CONCURRENCY="${CONCURRENCY:-150}"
LANGUAGE="${LANGUAGE:-mixed}"   # en | es | mixed

OUTDIR="/slurm/home/$USER/output/$SLURM_JOB_ID"
mkdir -p "$OUTDIR"

SCRATCH="${SLURM_TMPDIR:-/scratch/slurm/$USER/$SLURM_JOB_ID/tmp}"
JOBSCRATCH="${SCRATCH}/horizon-gen-${SLURM_JOB_ID}"
mkdir -p "$JOBSCRATCH"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

# ── Cleanup: sync raw data back on exit/timeout ───────────────────────────────
cleanup() {
    echo "=== Syncing data/raw back to home ==="
    rsync -a --ignore-missing-args \
        "$JOBSCRATCH/data/raw/" "$SUBMIT_DIR/data/raw/" 2>/dev/null || true
    echo "=== Sync complete ==="
}
trap cleanup EXIT INT TERM

# ── Copy project to scratch ───────────────────────────────────────────────────
rsync -a --exclude='.venv' --exclude='data/raw' --exclude='experiments' \
    "$SUBMIT_DIR/" "$JOBSCRATCH/"
mkdir -p "$JOBSCRATCH/data/raw"

# Copy existing partial data so --resume works correctly
if [ -d "$SUBMIT_DIR/data/raw" ]; then
    rsync -a "$SUBMIT_DIR/data/raw/" "$JOBSCRATCH/data/raw/"
fi

# ── Environment ───────────────────────────────────────────────────────────────
source "$JOBSCRATCH/slurm/install_env.sh"

if [ -f "$SUBMIT_DIR/.env" ]; then
    set -a; source "$SUBMIT_DIR/.env"; set +a
fi

export HF_HOME="$JOBSCRATCH/.cache/huggingface"
mkdir -p "$HF_HOME"

cd "$JOBSCRATCH"

# ── Start vLLM server ─────────────────────────────────────────────────────────
echo "=== Starting vLLM server ==="
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 16384 \
    --port 8000 &
VLLM_PID=$!

echo "Waiting for vLLM to be ready..."
for i in $(seq 1 60); do
    if curl -sf "http://localhost:8000/health" >/dev/null 2>&1; then
        echo "vLLM ready (${i}x5s)"
        break
    fi
    sleep 5
done

# ── Generate all categories in parallel ──────────────────────────────────────
CATEGORIES=(grooming bullying sexual_content isolation personal_info platform_migration threats benign)

declare -A CAT_COUNT=(
    [grooming]=80000 [bullying]=70000 [sexual_content]=70000
    [isolation]=50000 [personal_info]=50000 [platform_migration]=30000
    [threats]=50000 [benign]=100000
)

echo "=== Generating: LANGUAGE=${LANGUAGE}, CONCURRENCY=${CONCURRENCY} ==="

pids=()
for cat in "${CATEGORIES[@]}"; do
    target="${CAT_COUNT[$cat]}"
    echo "  → $cat (target=${target})"
    python -m data.generation.scripts.generate \
        --category "$cat" \
        --count "$target" \
        --generator vllm \
        --language "$LANGUAGE" \
        --concurrency "$CONCURRENCY" \
        --config data/generation/config.yaml \
        --resume \
        2>&1 | tee "$OUTDIR/${cat}.log" &
    pids+=($!)
done

failed=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "WARNING: ${CATEGORIES[$i]} failed or incomplete"
        failed=$((failed + 1))
    fi
done

kill "$VLLM_PID" 2>/dev/null || true

echo ""
echo "=== Done. Failed: ${failed}/${#CATEGORIES[@]} — logs in $OUTDIR/ ==="
