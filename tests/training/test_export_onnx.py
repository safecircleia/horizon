# tests/training/test_export_onnx.py
import torch
import tempfile
from pathlib import Path
from training.model.mobile import HorizonMobileModel
from training.scripts.export_onnx import export_to_onnx, verify_onnx_output


def test_export_produces_onnx_file():
    model = HorizonMobileModel(pretrained=False)
    model.eval()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "model.onnx"
        export_to_onnx(model, output_path=str(out_path), max_seq_length=64)
        assert out_path.exists()
        assert out_path.stat().st_size > 0


def test_onnx_output_matches_pytorch():
    model = HorizonMobileModel(pretrained=False)
    model.eval()
    input_ids = torch.randint(0, 1000, (1, 32))
    attention_mask = torch.ones(1, 32, dtype=torch.long)

    with torch.no_grad():
        pt_out = model(input_ids=input_ids, attention_mask=attention_mask)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "model.onnx"
        export_to_onnx(model, output_path=str(out_path), max_seq_length=64)
        cat_logits, sev_logits = verify_onnx_output(
            str(out_path),
            input_ids=input_ids.numpy(),
            attention_mask=attention_mask.numpy(),
        )

    assert cat_logits.shape == (1, 8)
    assert sev_logits.shape == (1, 5)
