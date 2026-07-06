#!/usr/bin/env bash
# SLURM job: debug why LiteRT-LM export doesn't preserve fine-tuned weights.
# Checks: weight loading, checkpoint validity, litert-torch builder behavior.
#
# Usage:
#   sbatch slurm/debug_export.sbatch

#SBATCH --job-name=horizon-debug-export
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/slurm/home/%u/output/%j.out
#SBATCH --error=/slurm/home/%u/output/%j.err

set -euo pipefail

OUTDIR="/slurm/home/$USER/output"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$SUBMIT_DIR"

source .venv/bin/activate

if [ -f "$SUBMIT_DIR/.env" ]; then
    set -a; source "$SUBMIT_DIR/.env"; set +a
fi

export HF_HOME="$SUBMIT_DIR/.cache/huggingface"
export PYTHONUNBUFFERED=1
uv pip install --quiet tensorflow litert-torch litert-lm-builder

# Find the latest checkpoint
CHECKPOINT_DIR=$(find experiments/ -name "final" -type d 2>/dev/null | sort | tail -1)
MERGED_DIR="models/mobile-standard/merged"

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "ERROR: No checkpoint found"
    exit 1
fi

echo "=== Debug Export: Weight Loading ==="
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Merged dir: $MERGED_DIR"
echo ""

python3 << 'PYTHON'
import torch
import sys
from pathlib import Path

# Step 1: Check merged weights exist and have expected shape
print("=" * 60)
print("STEP 1: Verify merged checkpoint")
print("=" * 60)

merged_dir = "models/mobile-standard/merged"
if not Path(merged_dir).exists():
    print(f"ERROR: {merged_dir} does not exist. Running merge first...")
    from training.scripts.export_litert import merge_lora
    import glob
    checkpoint = glob.glob("experiments/*/final")[0]
    merge_lora(checkpoint, merged_dir)

# Load merged weights and check a few layers
from safetensors import safe_open
import glob

safetensor_files = glob.glob(f"{merged_dir}/*.safetensors")
print(f"Found {len(safetensor_files)} safetensor files")

# Check first layer weights (if fine-tuned, they should differ from base)
sample_keys = []
with safe_open(safetensor_files[0], framework="pt") as f:
    keys = list(f.keys())
    print(f"Total keys in first shard: {len(keys)}")
    sample_keys = keys[:5]
    for k in sample_keys:
        tensor = f.get_tensor(k)
        print(f"  {k}: shape={tensor.shape}, mean={tensor.float().mean():.6f}, std={tensor.float().std():.6f}")

print()
print("=" * 60)
print("STEP 2: Load via litert-torch build_model_1b")
print("=" * 60)

try:
    from litert_torch.generative.examples.gemma3 import decoder as gemma3_decoder

    # Check what build_model_1b actually does with checkpoint_path
    import inspect
    src = inspect.getsource(gemma3_decoder.build_model_1b)
    print("build_model_1b source (first 2000 chars):")
    print(src[:2000])
    print("..." if len(src) > 2000 else "")
    print()

    # Now actually build it
    print("Building model with checkpoint_path=merged_dir...")
    pytorch_model = gemma3_decoder.build_model_1b(
        checkpoint_path=merged_dir,
        mask_cache_size=0,
    )

    # Check if it loaded our weights or base weights
    print("\nModel loaded. Checking sample weights...")
    state_dict = pytorch_model.state_dict()
    sd_keys = list(state_dict.keys())
    print(f"Model state_dict keys: {len(sd_keys)}")
    for k in sd_keys[:5]:
        t = state_dict[k]
        print(f"  {k}: shape={t.shape}, mean={t.float().mean():.6f}, std={t.float().std():.6f}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("STEP 3: Compare weights")
print("=" * 60)

# Load base model weights for comparison
try:
    from transformers import AutoModelForCausalLM
    print("Loading base model (google/gemma-3-1b-it) for comparison...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-it",
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    base_sd = base_model.state_dict()

    # Load merged weights
    merged_sd = {}
    for sf in safetensor_files:
        with safe_open(sf, framework="pt") as f:
            for k in f.keys():
                merged_sd[k] = f.get_tensor(k)

    # Compare a few layers
    print("\nWeight comparison (merged vs base):")
    compared = 0
    different = 0
    for k in list(base_sd.keys())[:20]:
        if k in merged_sd:
            base_t = base_sd[k].float()
            merged_t = merged_sd[k].float()
            if base_t.shape == merged_t.shape:
                diff = (base_t - merged_t).abs().mean().item()
                is_diff = diff > 1e-6
                if is_diff:
                    different += 1
                    print(f"  DIFFERENT {k}: mean_abs_diff={diff:.6f}")
                compared += 1

    print(f"\nCompared {compared} layers: {different} different, {compared - different} identical")
    if different == 0:
        print("WARNING: Merged weights are IDENTICAL to base — LoRA merge may have failed!")
    else:
        print("OK: Merged weights differ from base (fine-tuning preserved)")

except Exception as e:
    print(f"Comparison failed: {e}")

print()
print("=" * 60)
print("STEP 4: Check litert-torch weight mapping")
print("=" * 60)

# Check if build_model_1b's state_dict matches the merged weights
try:
    litert_sd = pytorch_model.state_dict()

    # Sample comparison
    print("Comparing litert model weights vs merged checkpoint...")
    # The key names will differ — litert-torch uses its own naming
    # Just check if values are non-zero and have reasonable stats
    all_zeros = 0
    for k, v in list(litert_sd.items())[:10]:
        is_zero = v.abs().max().item() < 1e-10
        if is_zero:
            all_zeros += 1
        print(f"  {'ZERO!' if is_zero else 'OK   '} {k}: max={v.float().abs().max():.6f}")

    if all_zeros > 3:
        print("WARNING: Many zero layers — model may not have loaded weights!")
except Exception as e:
    print(f"Step 4 failed: {e}")

print("\n=== Debug complete ===")
PYTHON
