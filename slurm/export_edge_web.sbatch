#!/usr/bin/env bash
# SLURM job: export ONLY the Web (WebGPU) LiteRT-LM variant.
#
# Usage:
#   sbatch --export=MODEL_SIZE=e2b slurm/export_edge_web.sbatch
#   sbatch --export=MODEL_SIZE=e4b slurm/export_edge_web.sbatch
#
# Expects the merged model at models/horizon-edge-{MODEL_SIZE}-merged/
# Produces: horizon-edge-{MODEL_SIZE}-web.litertlm (tf_lite_artisan_text_decoder format)

#SBATCH --job-name=horizon-export-web
#SBATCH --partition=gpuUltra
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:nvidia_h100_nvl
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/slurm/home/%u/output/%j/terminal.out
#SBATCH --error=/slurm/home/%u/output/%j/terminal.err

set -euo pipefail

MODEL_SIZE="${MODEL_SIZE:-e2b}"
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

OUTDIR="/slurm/home/$USER/output"

SCRATCH="${SLURM_TMPDIR:-/scratch/slurm/$USER/$SLURM_JOB_ID/tmp}"
JOBSCRATCH="${SCRATCH}/horizon-export-web-${SLURM_JOB_ID}"
mkdir -p "$JOBSCRATCH"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

cleanup() {
    echo "=== Syncing exported web model back to home ==="
    rsync -a --ignore-missing-args \
        "$JOBSCRATCH/$OUTPUT_DIR/" "$SUBMIT_DIR/$OUTPUT_DIR/" 2>/dev/null || true
    echo "=== Sync complete ==="
}
trap cleanup EXIT INT TERM

# ── Copy only what's needed to scratch ────────────────────────────────────────
echo "=== Copying project (minimal) to scratch ==="
mkdir -p "$JOBSCRATCH"
rsync -a \
    --exclude='.venv' --exclude='data' --exclude='experiments' \
    --exclude='models' --exclude='notebooks' --exclude='.git' \
    --exclude='.codegraph' --exclude='.serena' --exclude='.playwright-mcp' \
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
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME"

echo "=== Installing litert-torch + transformers 4.52 (gemma4 without heterogeneity) ==="
uv pip install --quiet litert-torch==0.9.1
uv pip install --quiet "tokenizers>=0.21,<0.22"
uv pip install --quiet --force-reinstall --no-deps "transformers==4.52.*"

# Patch upstream bug: LiteRTLMCacheLayerForGemma4 missing get_max_length.
python3 - <<'PYEOF'
import sys, pathlib
sp = pathlib.Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
p = sp / "litert_torch/generative/export_hf/model_ext/gemma4/cache.py"
if not p.exists():
    print(f"WARNING: {p} not found — patch skipped"); sys.exit(0)
src = p.read_text()
if "get_max_length" in src:
    print("Patch already applied"); sys.exit(0)
patch = (
    "\n# patch: missing abstract method (upstream bug in 0.9.1)\n"
    "LiteRTLMCacheLayerForGemma4.get_max_length = lambda self: self.max_cache_len\n"
    "LiteRTLMCacheLayerForGemma4.__abstractmethods__ = frozenset(\n"
    "    m for m in LiteRTLMCacheLayerForGemma4.__abstractmethods__ if m != 'get_max_length'\n"
    ")\n"
)
p.write_text(src + patch)
for c in (p.parent / "__pycache__").glob(f"{p.stem}.*.pyc"):
    c.unlink()
print(f"Patched {p}")
PYEOF

cd "$JOBSCRATCH"


# ── Export ONLY the web variant ──────────────────────────────────────────────
echo "=== Exporting Web (WebGPU) variant for horizon-edge-${MODEL_SIZE} ==="
python -m training.scripts.export_edge \
    --model-dir "$MODEL_DIR" \
    --output "$OUTPUT_DIR" \
    --model-size "$MODEL_SIZE" \
    --variants web \
    2>&1 | tee "$OUTDIR/export-web.log"

echo ""
echo "=== Export complete ==="
echo "Output: $OUTPUT_DIR/horizon-edge-${MODEL_SIZE}-web.litertlm"
echo ""
echo "Next steps:"
echo "  1. Verify model_type is tf_lite_artisan_text_decoder (web-compatible)"
echo "  2. Upload: python scripts/upload.py --what edge-${MODEL_SIZE} --version <version>"
