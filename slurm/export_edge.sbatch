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
uv pip install --quiet litert-torch-nightly

echo "=== Patching litert-torch nightly ==="
SITE_PACKAGES=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")

# Fix 1: create missing experimental subpackage
EXP_DIR="$SITE_PACKAGES/litert_torch/generative/export_hf/experimental"
mkdir -p "$EXP_DIR"
echo "" > "$EXP_DIR/__init__.py"
echo "def transpile(*a, **k): raise NotImplementedError" > "$EXP_DIR/minijinja_transpile.py"
echo "Fix 1: experimental dir ready at $EXP_DIR"

# Fix 2: patch cache.py directly — add get_max_length and cumulative_length to
# LiteRTLMCacheLayer. Use Python only for the text transformation, no imports.
CACHE_PY="$SITE_PACKAGES/litert_torch/generative/export_hf/core/cache.py"
python3 - "$CACHE_PY" << 'PYEOF'
import sys, pathlib, re

cache_py = pathlib.Path(sys.argv[1])
src = cache_py.read_text()
marker = "    self.head_dim = self.v_cache_shape[v_head_dim_idx]"

if marker not in src:
    print(f"WARNING: marker not found in {cache_py}")
    sys.exit(0)

addition = ""
cls_block = re.search(r'class LiteRTLMCacheLayer\b.*?def update\b', src, re.DOTALL)
if cls_block and "self.cumulative_length" not in cls_block.group():
    addition += "\n    self.cumulative_length = 0"
if "def get_max_length" not in src:
    addition += "\n\n  def get_max_length(self):\n    return self.max_cache_len"

if not addition:
    print("Fix 2: cache.py already patched")
    sys.exit(0)

patched = src.replace(marker, marker + addition, 1)
cache_py.write_text(patched)

# Remove stale bytecode
for pyc in (cache_py.parent / "__pycache__").glob("cache*.pyc"):
    pyc.unlink()

print(f"Fix 2: cache.py patched ({addition.strip()!r})")
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
