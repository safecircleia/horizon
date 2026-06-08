#!/usr/bin/env python3
"""Export a fine-tuned Gemma 3 1B LoRA checkpoint to LiteRT-LM format.

Supports two quantization variants for different device capabilities:
  --quantization int8   ~1.2 GB, for 6 GB+ RAM phones (default)
  --quantization int4   ~0.7 GB, for 4 GB RAM phones (budget/older)

Pipeline:
  1. Merge LoRA weights into the base model (fp32, CPU)
  2. Convert to TFLite via ai_edge_torch with selected quantization
  3. Package into a .litertlm container via litert-lm-builder

Usage:
    # Standard variant (INT8, 6GB+ phones)
    python -m training.scripts.export_litert \\
        --checkpoint experiments/mobile-<ts>/final \\
        --output models/mobile-standard

    # Lite variant (INT4, 4GB phones) — reuses merged/ from standard export
    python -m training.scripts.export_litert \\
        --checkpoint experiments/mobile-<ts>/final \\
        --output models/mobile-lite \\
        --quantization int4 \\
        --skip-merge \\
        --merged-dir models/mobile-standard/merged

Requirements:
    uv pip install ai-edge-torch litert-lm-builder
"""

import argparse
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.model.mobile import MOBILE_BASE_MODEL, SYSTEM_PROMPT

QUANT_OPTIONS = ("int8", "int4")


# ── Step 1: merge LoRA ────────────────────────────────────────────────────────

def merge_lora(checkpoint_path: str, output_dir: str) -> None:
    print(f"Loading LoRA checkpoint: {checkpoint_path}")
    config = PeftConfig.from_pretrained(checkpoint_path)
    base = config.base_model_name_or_path or MOBILE_BASE_MODEL

    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model = model.merge_and_unload()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Merged checkpoint saved to: {output_dir}")


# ── Step 2: convert to TFLite ─────────────────────────────────────────────────

def convert_to_tflite(
    merged_dir: str,
    tflite_path: str,
    max_seq_length: int,
    quantization: str,
) -> None:
    try:
        import ai_edge_torch
        from ai_edge_torch.generative.utilities import model_builder
        from ai_edge_torch.quantize.quant_recipe import (
            Dtype, GenerativeQuantRecipe, QuantRecipe,
        )
    except ImportError:
        print(
            "ERROR: ai_edge_torch is not installed.\n"
            "Install with: uv pip install ai-edge-torch\n"
            "See: https://ai.google.dev/edge/litert/conversion/pytorch/genai"
        )
        sys.exit(1)

    quant_dtype = Dtype.INT8 if quantization == "int8" else Dtype.INT4
    print(f"Converting to TFLite ({quantization.upper()}, max_seq_length={max_seq_length})...")

    edge_model = model_builder.build_model(merged_dir, max_seq_length=max_seq_length)

    sample_ids = torch.zeros((1, max_seq_length), dtype=torch.long)
    sample_mask = torch.ones((1, max_seq_length), dtype=torch.long)
    sample_pos = torch.arange(max_seq_length, dtype=torch.long).unsqueeze(0)

    converted = ai_edge_torch.convert(
        edge_model.eval(),
        (sample_ids, sample_mask, sample_pos),
        quant_config=GenerativeQuantRecipe(
            default=QuantRecipe(weight_dtype=quant_dtype)
        ),
    )

    Path(tflite_path).parent.mkdir(parents=True, exist_ok=True)
    converted.export(tflite_path)
    size_mb = Path(tflite_path).stat().st_size / 1024 / 1024
    print(f"TFLite model saved: {tflite_path} ({size_mb:.1f} MB)")


# ── Step 3: package into .litertlm ───────────────────────────────────────────

def build_litertlm(
    tflite_path: str,
    tokenizer_dir: str,
    output_path: str,
    variant: str,
    version: str,
) -> None:
    sp_model = next(Path(tokenizer_dir).glob("*.model"), None)
    if sp_model is None:
        sp_model = next(Path(tokenizer_dir).glob("tokenizer*"), None)
    if sp_model is None:
        print(f"ERROR: No SentencePiece .model file found in: {tokenizer_dir}")
        sys.exit(1)

    cmd = [
        "litert-lm-builder",
        "system_metadata",
        "--str", "model_name", f"horizon-mobile-{variant}",
        "--str", "base_model", MOBILE_BASE_MODEL,
        "--str", "variant", variant,
        "--str", "system_prompt", SYSTEM_PROMPT,
        "sp_tokenizer",
        "--path", str(sp_model),
        "tflite_model",
        "--path", str(tflite_path),
        "--model_type", "prefill_decode",
        "--str_metadata", "model_version", version,
        "--str_metadata", "quantization", variant,
        "output",
        "--path", str(output_path),
    ]

    print("Building .litertlm container...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: litert-lm-builder failed:")
        print(result.stderr)
        sys.exit(1)

    if result.stdout:
        print(result.stdout)

    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f".litertlm ready: {output_path} ({size_mb:.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export fine-tuned Gemma 3 1B LoRA to a LiteRT-LM .litertlm container"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Fine-tuned LoRA checkpoint (e.g. experiments/mobile-<ts>/final)"
    )
    parser.add_argument(
        "--output", default="models/mobile-standard",
        help="Output directory (default: models/mobile-standard)"
    )
    parser.add_argument(
        "--quantization", default="int8", choices=QUANT_OPTIONS,
        help="int8 (~1.2GB, 6GB+ phones) or int4 (~0.7GB, 4GB phones) (default: int8)"
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=512,
        help="Max sequence length for TFLite export (default: 512)"
    )
    parser.add_argument(
        "--version", default="1.0.0",
        help="Version string embedded in the container (default: 1.0.0)"
    )
    parser.add_argument(
        "--skip-merge", action="store_true",
        help="Skip LoRA merge — reuse an existing merged directory"
    )
    parser.add_argument(
        "--merged-dir",
        help="Path to existing merged checkpoint (used with --skip-merge)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    variant = args.quantization

    merged_dir = args.merged_dir or str(output_dir / "merged")
    tflite_path = str(output_dir / "model.tflite")
    litertlm_path = str(output_dir / f"horizon-mobile-{variant}.litertlm")

    if args.skip_merge and Path(merged_dir).exists():
        print(f"Skipping merge — using existing: {merged_dir}")
    else:
        merge_lora(args.checkpoint, merged_dir)

    convert_to_tflite(merged_dir, tflite_path, args.max_seq_length, variant)
    build_litertlm(tflite_path, merged_dir, litertlm_path, variant, args.version)

    print("\nExport complete.")
    print(f"  Variant            : {variant}")
    print(f"  Merged checkpoint  : {merged_dir}")
    print(f"  TFLite model       : {tflite_path}")
    print(f"  LiteRT-LM container: {litertlm_path}")
    print(f"\nTest locally:")
    print(f"  uvx litert-lm run {litertlm_path} --prompt 'Hello'")


if __name__ == "__main__":
    main()
