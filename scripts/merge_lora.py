#!/usr/bin/env python3
"""Merge LoRA adapter weights into the base model for standalone inference.

Usage:
    python scripts/merge_lora.py --checkpoint experiments/h100-20260514-134855/final --output models/horizon-full-merged
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to LoRA checkpoint (adapter weights)")
    parser.add_argument("--output", default="models/horizon-full-merged", help="Output path for merged model")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.2-3B-Instruct")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading LoRA adapter: {args.checkpoint}")
    model = PeftModel.from_pretrained(model, args.checkpoint)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {out}")
    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    print("Done.")


if __name__ == "__main__":
    main()
