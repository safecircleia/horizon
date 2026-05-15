# training/scripts/export_onnx.py
"""Export horizon-mobile to ONNX with optional INT8 quantization."""

import argparse
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import AutoTokenizer

from training.model.mobile import HorizonMobileModel


class _ModelWrapper(torch.nn.Module):
    """Wraps HorizonMobileModel to return a tuple for ONNX export."""
    def __init__(self, model: HorizonMobileModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out["category_logits"], out["severity_logits"]


def export_to_onnx(
    model: HorizonMobileModel,
    output_path: str,
    max_seq_length: int = 256,
) -> None:
    wrapper = _ModelWrapper(model)
    wrapper.eval()
    dummy_input_ids = torch.zeros(1, max_seq_length, dtype=torch.long)
    dummy_attention = torch.ones(1, max_seq_length, dtype=torch.long)

    batch = torch.export.Dim("batch", min=1, max=64)
    seq = torch.export.Dim("seq", min=1, max=512)
    dynamic_shapes = {
        "input_ids": {0: batch, 1: seq},
        "attention_mask": {0: batch, 1: seq},
    }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_attention),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["category_logits", "severity_logits"],
            dynamic_shapes=dynamic_shapes,
            opset_version=18,
            dynamo=True,
        )


def quantize_onnx(input_path: str, output_path: str) -> None:
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
    )


def verify_onnx_output(
    onnx_path: str,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
):
    session = ort.InferenceSession(onnx_path)
    cat_logits, sev_logits = session.run(
        None,
        {"input_ids": input_ids, "attention_mask": attention_mask},
    )
    return cat_logits, sev_logits


def main():
    parser = argparse.ArgumentParser(description="Export horizon-mobile to ONNX")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="models/mobile")
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--quantize", action="store_true", default=False)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading mobile model...")
    model = HorizonMobileModel.from_pretrained(args.checkpoint)
    model.eval()

    raw_path = str(out_dir / "horizon-mobile-raw.onnx")
    final_path = str(out_dir / "horizon-mobile.onnx")

    print("Exporting to ONNX...")
    export_to_onnx(model, raw_path, max_seq_length=args.max_seq_length)

    if args.quantize:
        print("Quantizing to INT8...")
        quantize_onnx(raw_path, final_path)
        Path(raw_path).unlink()
        print(f"Quantized ONNX: {final_path} ({Path(final_path).stat().st_size / 1e6:.1f} MB)")
    else:
        Path(raw_path).rename(final_path)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
    tokenizer.save_pretrained(str(out_dir / "tokenizer"))

    print("Verifying output...")
    dummy_ids = np.zeros((1, 32), dtype=np.int64)
    dummy_mask = np.ones((1, 32), dtype=np.int64)
    cat_logits, sev_logits = verify_onnx_output(final_path, dummy_ids, dummy_mask)
    print(f"Output shapes: category={cat_logits.shape}, severity={sev_logits.shape}")
    print("Export complete.")


if __name__ == "__main__":
    main()
