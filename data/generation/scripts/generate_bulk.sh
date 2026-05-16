#!/usr/bin/env bash
# Bulk dataset generation using local vLLM server on H100.
#
# Target: ~100GB raw JSONL across all 8 risk categories.
# Each category gets COUNT conversations (default 200 000 = ~12.5GB each → ~100GB total).
#
# Usage:
#   # First: start vLLM server on H100
#   vllm serve Qwen/Qwen2.5-7B-Instruct \
#       --max-model-len 4096 \
#       --gpu-memory-utilization 0.85 \
#       --enable-chunked-prefill \
#       --max-num-batched-tokens 16384 \
#       --port 8000
#
#   # Then run this script (from project root):
#   bash data/generation/scripts/generate_bulk.sh
#
#   # Resume a partial run (skips already-generated convs):
#   bash data/generation/scripts/generate_bulk.sh --resume
#
# Throughput estimate (Qwen2.5-72B-AWQ, H100, concurrency=150):
#   ~500 conversations/min → 200 000 convs ≈ 7h per category
#   Run all 8 categories in parallel → limited only by disk I/O and GPU
#   Swap to Qwen2.5-7B-Instruct (model_vllm in config) for ~4x speed at lower quality
#
# Environment variables:
#   COUNT           - conversations per category (default: 200000)
#   CONCURRENCY     - async workers per category process (default: 150)
#   VLLM_BASE_URL   - vLLM server URL (default: http://localhost:8000/v1)
#   CONFIG          - path to generation config (default: data/generation/config.yaml)

set -euo pipefail

COUNT="${COUNT:-200000}"
CONCURRENCY="${CONCURRENCY:-150}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
CONFIG="${CONFIG:-data/generation/config.yaml}"
RESUME_FLAG=""

for arg in "$@"; do
    if [[ "$arg" == "--resume" ]]; then
        RESUME_FLAG="--resume"
    fi
done

CATEGORIES=(grooming bullying sexual_content isolation personal_info platform_migration threats benign)

echo "=========================================="
echo "SafeCircle Bulk Dataset Generation"
echo "=========================================="
echo "Target:      ${COUNT} conversations × ${#CATEGORIES[@]} categories"
echo "Concurrency: ${CONCURRENCY} workers/category"
echo "vLLM URL:    ${VLLM_BASE_URL}"
echo "Config:      ${CONFIG}"
echo "Resume:      ${RESUME_FLAG:-no}"
echo "=========================================="
echo ""

# Verify vLLM server is reachable before starting
echo "Checking vLLM server..."
if ! curl -sf "${VLLM_BASE_URL}/models" > /dev/null; then
    echo "ERROR: vLLM server not reachable at ${VLLM_BASE_URL}"
    echo "Start it with:"
    echo "  vllm serve Qwen/Qwen2.5-7B-Instruct \\"
    echo "      --max-model-len 4096 \\"
    echo "      --gpu-memory-utilization 0.85 \\"
    echo "      --enable-chunked-prefill \\"
    echo "      --max-num-batched-tokens 16384 \\"
    echo "      --port 8000"
    exit 1
fi
echo "vLLM server OK"
echo ""

# Patch concurrency into config via env so we don't need to edit the YAML
export VLLM_BASE_URL

pids=()
log_dir="logs/generation"
mkdir -p "$log_dir"

for category in "${CATEGORIES[@]}"; do
    output="data/raw/${category}.jsonl"
    log="${log_dir}/$(date +%Y%m%d-%H%M%S)-${category}.log"

    echo "Starting: ${category} → ${output}"

    PYTHONPATH=. python -m data.generation.scripts.generate \
        --category "${category}" \
        --count "${COUNT}" \
        --generator vllm \
        --output "${output}" \
        --config "${CONFIG}" \
        ${RESUME_FLAG} \
        2>&1 | tee "${log}" &

    pids+=($!)
    # Stagger starts by 2s to avoid thundering-herd on the vLLM server
    sleep 2
done

echo ""
echo "All ${#CATEGORIES[@]} generation jobs started (PIDs: ${pids[*]})"
echo "Logs: ${log_dir}/"
echo ""
echo "Monitor progress:"
echo "  tail -f ${log_dir}/*.log"
echo ""

# Wait for all jobs and report failures
failed=0
for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    cat="${CATEGORIES[$i]}"
    if wait "$pid"; then
        echo "✓ ${cat} done"
    else
        echo "✗ ${cat} FAILED (exit $?)"
        failed=$((failed + 1))
    fi
done

echo ""
echo "=========================================="
if [[ $failed -eq 0 ]]; then
    echo "All categories generated successfully."
    echo "Dataset size:"
    du -sh data/raw/*.jsonl 2>/dev/null || true
    du -sh data/raw/ 2>/dev/null || true
else
    echo "${failed} categories failed. Check logs in ${log_dir}/"
    exit 1
fi
