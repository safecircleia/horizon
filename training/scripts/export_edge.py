#!/usr/bin/env python3
"""Export a fine-tuned Gemma 4 E2B/E4B merged model to LiteRT-LM .litertlm format.

Produces all hardware-specific variants matching litert-community naming:
  - General (CPU/GPU cross-platform)
  - Google Tensor G5 (Pixel 10)
  - Intel Lunar Lake (LNL)
  - Intel Panther Lake (PTL)
  - Qualcomm QCS8275 (Dragonwing IQ8)
  - Qualcomm SM8750 (Snapdragon 8 Elite)
  - Web (WebGPU-optimized)

Usage:
    python -m training.scripts.export_edge \
        --model-dir models/horizon-edge-2b-merged \
        --output models/horizon-edge-2b-litert \
        --model-size e2b

    python -m training.scripts.export_edge \
        --model-dir models/horizon-edge-4b-merged \
        --output models/horizon-edge-4b-litert \
        --model-size e4b

Requirements:
    uv pip install litert-torch-nightly
"""

import argparse
import subprocess
import sys
from pathlib import Path

CHAT_TEMPLATE_REPO = {
    "e2b": "litert-community/gemma-4-E2B-it-litert-lm",
    "e4b": "litert-community/gemma-4-E4B-it-litert-lm",
}

VARIANTS = [
    {"suffix": "", "label": "general"},
    {"suffix": "_Google_Tensor_G5", "label": "Google Tensor G5"},
    {"suffix": "_intel_LNL", "label": "Intel Lunar Lake"},
    {"suffix": "_intel_PTL", "label": "Intel Panther Lake"},
    {"suffix": "_qualcomm_qcs8275", "label": "Qualcomm QCS8275"},
    {"suffix": "_qualcomm_sm8750", "label": "Qualcomm SM8750"},
    {"suffix": "-web", "label": "Web (WebGPU)"},
]


def export_variant(model_dir: str, output_dir: str, model_size: str, suffix: str, label: str) -> bool:
    """Export one variant using litert-torch export_hf."""
    out_path = Path(output_dir) / f"variant{suffix}"
    out_path.mkdir(parents=True, exist_ok=True)

    chat_template_repo = CHAT_TEMPLATE_REPO[model_size]
    is_web = suffix == "-web"

    cmd = [
        "litert-torch", "export_hf",
        f"--model={model_dir}",
        f"--output_dir={out_path}",
        "--externalize_embedder",
        f"--jinja_chat_template_override={chat_template_repo}",
    ]


    if suffix:
        target = suffix.lstrip("_").lstrip("-")
        cmd.append(f"--target={target}")

    print(f"\n{'='*60}")
    print(f"Exporting: {label} (suffix={suffix or 'none'})")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"WARNING: export failed for variant '{label}' (exit {result.returncode})")
        return False

    litertlm_files = list(out_path.glob("*.litertlm"))
    if litertlm_files:
        final_name = f"horizon-edge-{model_size}{suffix}.litertlm"
        final_path = Path(output_dir) / final_name
        litertlm_files[0].rename(final_path)
        size_mb = final_path.stat().st_size / 1024 / 1024
        print(f"  -> {final_path} ({size_mb:.0f} MB)")
        return True

    print(f"  WARNING: no .litertlm output for {label}")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Export Gemma 4 E2B/E4B to LiteRT-LM variants"
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="Path to merged safetensors model (e.g. models/horizon-edge-2b-merged)"
    )
    parser.add_argument(
        "--output", default="models/horizon-edge-2b-litert",
        help="Output directory for .litertlm files"
    )
    parser.add_argument(
        "--model-size", required=True, choices=["e2b", "e4b"],
        help="Model size (e2b or e4b)"
    )
    parser.add_argument(
        "--variants", nargs="*", default=None,
        help="Specific variant suffixes to export (default: all). "
             "Options: general, Google_Tensor_G5, intel_LNL, intel_PTL, qualcomm_qcs8275, qualcomm_sm8750, web"
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: model directory not found: {model_dir}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants_to_export = VARIANTS
    if args.variants:
        requested = set(args.variants)
        variants_to_export = [v for v in VARIANTS if v["label"].split(" ")[0].lower() in requested
                              or v["suffix"].lstrip("_").lstrip("-") in requested
                              or v["label"] in requested]
        if not variants_to_export:
            variants_to_export = [v for v in VARIANTS if any(r in v["suffix"] or r in v["label"] for r in requested)]

    print(f"Model: {model_dir}")
    print(f"Output: {output_dir}")
    print(f"Size: {args.model_size.upper()}")
    print(f"Variants: {len(variants_to_export)}")

    results = []
    for variant in variants_to_export:
        ok = export_variant(
            model_dir=str(model_dir),
            output_dir=str(output_dir),
            model_size=args.model_size,
            suffix=variant["suffix"],
            label=variant["label"],
        )
        results.append((variant["label"], ok))

    print(f"\n{'='*60}")
    print("Export Summary:")
    print(f"{'='*60}")
    for label, ok in results:
        status = "OK" if ok else "FAILED"
        print(f"  [{status}] {label}")

    litertlm_files = sorted(output_dir.glob("*.litertlm"))
    if litertlm_files:
        print(f"\nProduced {len(litertlm_files)} .litertlm files:")
        for f in litertlm_files:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.name} ({size_mb:.0f} MB)")
    else:
        print("\nWARNING: no .litertlm files produced")
        sys.exit(1)


if __name__ == "__main__":
    main()
