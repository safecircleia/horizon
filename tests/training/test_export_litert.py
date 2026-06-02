# tests/training/test_export_litert.py
"""Unit tests for export_litert helpers — no GPU or ai_edge_torch required."""

import sys
from unittest.mock import MagicMock, patch

import pytest


def test_build_litertlm_exits_without_sp_model(tmp_path):
    """build_litertlm exits if no SentencePiece .model file is found."""
    from training.scripts.export_litert import build_litertlm

    tflite = tmp_path / "model.tflite"
    tflite.write_bytes(b"dummy")
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()

    with pytest.raises(SystemExit):
        build_litertlm(str(tflite), str(tokenizer_dir), str(tmp_path / "out.litertlm"))


def test_build_litertlm_calls_builder_with_correct_args(tmp_path):
    """build_litertlm passes tflite path, sp_tokenizer, and version to litert-lm-builder."""
    from training.scripts.export_litert import build_litertlm

    tflite = tmp_path / "model.tflite"
    tflite.write_bytes(b"dummy")
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    sp_file = tokenizer_dir / "tokenizer.model"
    sp_file.write_bytes(b"dummy_sp")

    out = tmp_path / "out.litertlm"
    out.write_bytes(b"x" * 1024)  # fake file so stat() works

    ok = MagicMock()
    ok.returncode = 0
    ok.stdout = ""

    with patch("subprocess.run", return_value=ok) as mock_run:
        build_litertlm(str(tflite), str(tokenizer_dir), str(out), version="2.0.0")

    cmd = mock_run.call_args[0][0]
    assert "litert-lm-builder" in cmd
    assert "sp_tokenizer" in cmd
    assert str(sp_file) in cmd
    assert "tflite_model" in cmd
    assert str(tflite) in cmd
    assert "prefill_decode" in cmd
    assert "2.0.0" in cmd


def test_build_litertlm_exits_on_builder_failure(tmp_path):
    """build_litertlm exits when litert-lm-builder returns non-zero."""
    from training.scripts.export_litert import build_litertlm

    tflite = tmp_path / "model.tflite"
    tflite.write_bytes(b"dummy")
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.model").write_bytes(b"sp")

    fail = MagicMock()
    fail.returncode = 1
    fail.stderr = "builder error"

    with patch("subprocess.run", return_value=fail):
        with pytest.raises(SystemExit):
            build_litertlm(str(tflite), str(tokenizer_dir), str(tmp_path / "out.litertlm"))


def test_convert_to_tflite_exits_without_ai_edge_torch(tmp_path):
    """convert_to_tflite exits cleanly when ai_edge_torch is not installed."""
    from training.scripts.export_litert import convert_to_tflite

    broken_modules = {
        "ai_edge_torch": None,
        "ai_edge_torch.generative": None,
        "ai_edge_torch.generative.utilities": None,
        "ai_edge_torch.generative.utilities.model_builder": None,
    }
    with patch.dict(sys.modules, broken_modules):
        with pytest.raises((SystemExit, ImportError)):
            convert_to_tflite(str(tmp_path), str(tmp_path / "model.tflite"), 512)
