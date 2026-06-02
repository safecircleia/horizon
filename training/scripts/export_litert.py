#!/usr/bin/env python3
"""Export a fine-tuned Gemma 3 1B LoRA checkpoint to LiteRT-LM format.

Pipeline:
  1. Merge LoRA weights into the base model (fp32, CPU)
  2. Convert to TFLite via ai_edge_torch (INT8 dynamic-range quantization)
  3. Package into a .litertlm container via litert-lm-builder

Usage:
    python -m training.scripts.export_litert \\
        --checkpoint experiments/mobile-20250601/final \\
        --output models/mobile

Output layout:
    models/mobile/
        merged/                  # merged HuggingFace checkpoint
        model.tflite             # quantised TFLite flatbuffer
        horizon-mobile.litertlm  # deployable LiteRT-LM container

Requirements (in addition to base requirements.txt):
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


# ── Step 1: merge LoRA ────────────────────────────────────────────────────────

def merge_lora(checkpoint_path: str, output_dir: str) -> None:
    """Merge LoRA adapters into base weights and save a plain HF checkpoint."""
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

def convert_to_tflite(merged_dir: str, tflite_path: str, max_seq_length: int) -> None:
    """Convert merged HF checkpoint to TFLite via ai_edge_torch GenAI builder."""
    try:
        import ai_edge_torch
        from ai_edge_torch.generative.utilities import model_builder
    except ImportError:
        print(
            "ERROR: ai_edge_torch is not installed.\n"
            "Install with: uv pip install ai-edge-torch\n"
            "See: https://ai.google.dev/edge/litert/conversion/pytorch/genai"
        )
        sys.exit(1)

    print(f"Converting to TFLite (max_seq_length={max_seq_length})...")

    edge_model = model_builder.build_model(merged_dir, max_seq_length=max_seq_length)

    sample_ids = torch.zeros((1, max_seq_length), dtype=torch.long)
    sample_mask = torch.ones((1, max_seq_length), dtype=torch.long)
    sample_pos = torch.arange(max_seq_length, dtype=torch.long).unsqueeze(0)

    converted = ai_edge_torch.convert(
        edge_model.eval(),
        (sample_ids, sample_mask, sample_pos),
        quant_config=ai_edge_torch.quantize.quant_recipe.GenerativeQuantRecipe(
            default=ai_edge_torch.quantize.quant_recipe.QuantRecipe(
                weight_dtype=ai_edge_torch.quantize.quant_recipe.Dtype.INT8
            )
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
    version: str = "1.0.0",
) -> None:
    """Package TFLite + SentencePiece tokenizer into a .litertlm container."""
    sp_model = next(Path(tokenizer_dir).glob("*.model"), None)
    if sp_model is None:
        sp_model = next(Path(tokenizer_dir).glob("tokenizer*"), None)
    if sp_model is None:
        print(
            f"ERROR: No SentencePiece .model file found in: {tokenizer_dir}"
        )
        sys.exit(1)

    cmd = [
        "litert-lm-builder",
        "system_metadata",
        "--str", "model_name", "horizon-mobile",
        "--str", "base_model", MOBILE_BASE_MODEL,
        "--str", "system_prompt", SYSTEM_PROMPT,
        "sp_tokenizer",
        "--path", str(sp_model),
        "tflite_model",
        "--path", str(tflite_path),
        "--model_type", "prefill_decode",
        "--str_metadata", "model_version", version,
        "output",
        "--path", str(output_path),
    ]

    print(f"Building .litertlm container...")
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
        help="Fine-tuned LoRA checkpoint path (e.g. experiments/mobile-<ts>/final)"
    )
    parser.add_argument(
        "--output", default="models/mobile",
        help="Output directory (default: models/mobile)"
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
        help="Skip LoRA merge step if merged/ already exists"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    merged_dir = str(output_dir / "merged")
    tflite_path = str(output_dir / "model.tflite")
    litertlm_path = str(output_dir / "horizon-mobile.litertlm")

    if args.skip_merge and (output_dir / "merged").exists():
        print(f"Skipping merge — using existing: {merged_dir}")
    else:
        merge_lora(args.checkpoint, merged_dir)

    convert_to_tflite(merged_dir, tflite_path, args.max_seq_length)
    build_litertlm(tflite_path, merged_dir, litertlm_path, args.version)

    print("\nExport complete.")
    print(f"  Merged checkpoint  : {merged_dir}")
    print(f"  TFLite model       : {tflite_path}")
    print(f"  LiteRT-LM container: {litertlm_path}")
    print(f"\nTest locally:")
    print(f"  uvx litert-lm run {litertlm_path} --prompt 'Hello'")


if __name__ == "__main__":
    main()
