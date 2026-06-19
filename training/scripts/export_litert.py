#!/usr/bin/env python3
"""Export a fine-tuned Gemma 3 1B LoRA checkpoint to LiteRT-LM format.

Supports two quantization variants for different device capabilities:
  --quantization int8   ~1.2 GB, for 6 GB+ RAM phones (default)
  --quantization int4   ~0.7 GB, for 4 GB RAM phones (budget/older)

Pipeline:
  1. Merge LoRA weights into a standard HF safetensors checkpoint (CPU)
  2. Load merged checkpoint into litert-torch's Gemma3 Decoder
  3. Convert + quantize to TFLite via litert-torch
  4. Package into a .litertlm container (includes tokenizer + metadata)

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
    uv pip install tensorflow litert-torch litert-lm-builder
"""

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.model.mobile import MOBILE_BASE_MODEL, SYSTEM_PROMPT

QUANT_MAP = {
    "int8": "dynamic_int8",
    "int4": "dynamic_int4_block128",
}


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


# ── Step 2+3+4: convert via litert-torch ──────────────────────────────────────

def convert_and_package(
    merged_dir: str,
    output_dir: str,
    quantization: str,
    kv_cache_max_len: int,
    prefill_seq_lens: list,
    variant: str,
    version: str,
) -> str:
    try:
        from litert_torch.generative.examples.gemma3 import decoder as gemma3_decoder
        from litert_torch.generative.utilities import converter
        from litert_torch.generative.utilities.export_config import ExportConfig
        from litert_torch.generative.layers import kv_cache as kv_utils
    except ImportError as e:
        print(f"ERROR: litert-torch import failed: {e}")
        print("Install with: uv pip install tensorflow litert-torch litert-lm-builder")
        sys.exit(1)

    litert_quant = QUANT_MAP[quantization]
    print(f"Loading merged model from: {merged_dir}")
    pytorch_model = gemma3_decoder.build_model_1b(
        checkpoint_path=merged_dir,
        mask_cache_size=0,
    )

    export_config = ExportConfig(
        mask_as_input=True,
        kvcache_layout=kv_utils.KV_LAYOUT_TRANSPOSED,
    )

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Converting to LiteRT-LM ({quantization.upper()}, kv={kv_cache_max_len}, prefill={prefill_seq_lens})...")
    converter.convert_to_litert(
        pytorch_model=pytorch_model,
        output_path=str(output_dir_path),
        output_name_prefix=f"horizon-mobile-{variant}",
        prefill_seq_len=prefill_seq_lens,
        kv_cache_max_len=kv_cache_max_len,
        quantize=litert_quant,
        export_config=export_config,
        output_format="litertlm",
        hf_tokenizer_model_path=str(Path(merged_dir) / "tokenizer.json"),
        llm_model_type="gemma3",
        model_prompt_prefix="<start_of_turn>model\n",
        model_prompt_suffix="<end_of_turn>\n",
        user_prompt_prefix=f"<bos>{SYSTEM_PROMPT}<start_of_turn>user\n",
        user_prompt_suffix="<end_of_turn>\n<start_of_turn>model\n",
        stop_token_ids=[1, 107],  # <eos>=1, <end_of_turn>=107
    )

    litertlm_files = list(output_dir_path.glob("*.litertlm"))
    if litertlm_files:
        out = litertlm_files[0]
        size_mb = out.stat().st_size / 1024 / 1024
        print(f".litertlm ready: {out} ({size_mb:.1f} MB)")
        return str(out)

    # Fallback: convert_to_litert may have produced a .tflite only
    tflite_files = list(output_dir_path.glob("*.tflite"))
    if tflite_files:
        out = tflite_files[0]
        size_mb = out.stat().st_size / 1024 / 1024
        print(f"TFLite model ready: {out} ({size_mb:.1f} MB)")
        return str(out)

    print("WARNING: no output file found in output directory")
    return str(output_dir_path)


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
        "--quantization", default="int8", choices=list(QUANT_MAP.keys()),
        help="int8 (~1.2GB, 6GB+ phones) or int4 (~0.7GB, 4GB phones) (default: int8)"
    )
    parser.add_argument(
        "--kv-cache-max-len", type=int, default=1280,
        help="KV cache size — max tokens (prefill + decode). Default: 1280"
    )
    parser.add_argument(
        "--prefill-seq-lens", type=int, nargs="+", default=[8, 64, 128, 256, 512],
        help="Prefill sequence lengths exported as separate signatures"
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

    variant = args.quantization
    merged_dir = args.merged_dir or str(Path(args.output) / "merged")

    if args.skip_merge and Path(merged_dir).exists():
        print(f"Skipping merge — using existing: {merged_dir}")
    else:
        merge_lora(args.checkpoint, merged_dir)

    out = convert_and_package(
        merged_dir=merged_dir,
        output_dir=args.output,
        quantization=variant,
        kv_cache_max_len=args.kv_cache_max_len,
        prefill_seq_lens=args.prefill_seq_lens,
        variant=variant,
        version=args.version,
    )

    print("\nExport complete.")
    print(f"  Variant         : {variant}")
    print(f"  Merged checkpoint: {merged_dir}")
    print(f"  Output           : {out}")
    print(f"\nTest locally:")
    print(f"  uvx litert-lm run {out} --prompt 'Hello'")


if __name__ == "__main__":
    main()
