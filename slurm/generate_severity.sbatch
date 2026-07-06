#!/usr/bin/env bash
# SLURM job: generate severity top-up dataset (35K critical/high-heavy examples).
# Output goes to data/raw_severity/ — separate from the main 500K dataset.
# Run from: /slurm/home/$USER/safecircle/horizon

#SBATCH --job-name=horizon-severity
#SBATCH --partition=gpuMax
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=38
#SBATCH --gres=gpu:nvidia_h100_nvl
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/slurm/home/%u/output/%j.out
#SBATCH --error=/slurm/home/%u/output/%j.err

set -euo pipefail

CONCURRENCY="${CONCURRENCY:-150}"
GEN_LANGUAGE="${GEN_LANGUAGE:-mixed}"

OUTDIR="/slurm/home/$USER/output"

SCRATCH="${SLURM_TMPDIR:-/scratch/slurm/$USER/$SLURM_JOB_ID/tmp}"
JOBSCRATCH="${SCRATCH}/horizon-severity-${SLURM_JOB_ID}"
mkdir -p "$JOBSCRATCH"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

cleanup() {
    echo "=== Syncing data/raw_severity back to home ==="
    rsync -a --ignore-missing-args \
        "$JOBSCRATCH/data/raw_severity/" "$SUBMIT_DIR/data/raw_severity/" 2>/dev/null || true
    echo "=== Sync complete ==="
}
trap cleanup EXIT INT TERM

# ── Copy project to scratch ───────────────────────────────────────────────────
rsync -a --exclude='.venv' --exclude='data/raw' --exclude='data/raw_severity' \
    --exclude='experiments' "$SUBMIT_DIR/" "$JOBSCRATCH/"
mkdir -p "$JOBSCRATCH/data/raw_severity"

if [ -d "$SUBMIT_DIR/data/raw_severity" ]; then
    rsync -a "$SUBMIT_DIR/data/raw_severity/" "$JOBSCRATCH/data/raw_severity/"
fi

# ── Environment ───────────────────────────────────────────────────────────────
source "$JOBSCRATCH/slurm/install_env.sh"

if [ -f "$SUBMIT_DIR/.env" ]; then
    set -a; source "$SUBMIT_DIR/.env"; set +a
fi

export HF_HOME="$JOBSCRATCH/.cache/huggingface"
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME"

cd "$JOBSCRATCH"

# ── Start vLLM ────────────────────────────────────────────────────────────────
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
        echo "vLLM ready (${i}x5s)"; break
    fi
    sleep 5
done

# ── Generate all 7 risk categories in parallel ───────────────────────────────
CATEGORIES=(grooming bullying sexual_content isolation personal_info platform_migration threats)

echo "=== Generating severity top-up: GEN_LANGUAGE=${GEN_LANGUAGE}, CONCURRENCY=${CONCURRENCY} ==="

pids=()
for cat in "${CATEGORIES[@]}"; do
    echo "  → $cat (target=5000)"
    python -m data.generation.scripts.generate \
        --category "$cat" \
        --count 5000 \
        --generator vllm \
        --language "$GEN_LANGUAGE" \
        --concurrency "$CONCURRENCY" \
        --config data/generation/config_severity.yaml \
        --output "data/raw_severity/${cat}.jsonl" \
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
echo ""
echo "Next: preprocess the top-up data:"
echo "  python -m training.scripts.preprocess \\"
echo "      --input data/raw_severity --output data/processed_severity"
