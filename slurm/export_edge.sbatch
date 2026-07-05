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
uv pip install --quiet litert-torch-nightly litert-lm-builder

# ponytail: nightly has two bugs — missing experimental subpackage and
# LiteRTLMCacheLayer missing get_max_length (abstract in base, not implemented
# in subclass). Patch both directly in the installed package.
python3 - <<'PYEOF'
import pathlib, sys

import litert_torch.generative.export_hf as _pkg
pkg_dir = pathlib.Path(_pkg.__file__).parent

# Fix 1: missing experimental subpackage
exp_dir = pkg_dir / "experimental"
exp_dir.mkdir(exist_ok=True)
(exp_dir / "__init__.py").write_text("")
(exp_dir / "minijinja_transpile.py").write_text(
    "# shim\ndef transpile(*args, **kwargs):\n"
    "    raise NotImplementedError('minijinja_transpile shim')\n"
)
print(f"Fix 1: experimental shim written to {exp_dir}")

# Fix 2: LiteRTLMCacheLayer missing get_max_length + cumulative_length
import re
cache_py = pkg_dir / "core" / "cache.py"
src = cache_py.read_text()
changed = False

# 2a: inject cumulative_length=0 into __init__ (separate from get_max_length)
init_marker = "    self.head_dim = self.v_cache_shape[v_head_dim_idx]"
cls_block = re.search(r'class LiteRTLMCacheLayer\b.*?def update\b', src, re.DOTALL)
if cls_block and "cumulative_length" not in cls_block.group() and init_marker in src:
    src = src.replace(init_marker, init_marker + "\n    self.cumulative_length = 0", 1)
    changed = True
    print("Fix 2a: cumulative_length=0 added to LiteRTLMCacheLayer.__init__")

# 2b: add get_max_length method after __init__
if "def get_max_length" not in src and init_marker in src:
    src = src.replace(
        init_marker + "\n    self.cumulative_length = 0",
        init_marker + "\n    self.cumulative_length = 0\n\n  def get_max_length(self):\n    return self.max_cache_len",
        1,
    )
    changed = True
    print("Fix 2b: get_max_length added to LiteRTLMCacheLayer")

if changed:
    cache_py.write_text(src)
else:
    print("Fix 2: already complete, skipping")
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
