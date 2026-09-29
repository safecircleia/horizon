#!/usr/bin/env python3
"""Merge LoRA adapter weights into the base model for standalone inference.

Usage:
    python scripts/merge_lora.py --checkpoint experiments/edge-2b-20260619-231638/final --output models/horizon-edge-2b-merged
    python scripts/merge_lora.py --checkpoint experiments/h100-20260514-134855/final --output models/horizon-full-merged
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _detect_base_model(checkpoint_path: Path) -> str:
    """Read base_model_name_or_path from adapter_config.json."""
    config_file = checkpoint_path / "adapter_config.json"
    if not config_file.exists():
        raise FileNotFoundError(
            f"No adapter_config.json found at {checkpoint_path}. "
            "Is this a valid LoRA checkpoint?"
        )
    with open(config_file) as f:
        cfg = json.load(f)
    base = cfg.get("base_model_name_or_path")
    if not base:
        raise ValueError("adapter_config.json has no base_model_name_or_path field")
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", required=True, help="Path to LoRA checkpoint (adapter weights)"
    )
    parser.add_argument(
        "--output",
        default="models/horizon-full-merged",
        help="Output path for merged model",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model ID (auto-detected from adapter_config.json if omitted)",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    base_model = args.base_model or _detect_base_model(checkpoint)
    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print(f"Loading LoRA adapter: {checkpoint}")
    model = PeftModel.from_pretrained(model, str(checkpoint))

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {out}")
    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    print("Done.")


if __name__ == "__main__":
    main()
