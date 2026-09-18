# tests/training/test_export_litert.py
"""Unit tests for export_litert helpers — no GPU or litert-torch required."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_merge_lora_saves_to_output_dir(tmp_path):
    """merge_lora merges LoRA weights and saves to the given output directory."""
    from training.scripts.export_litert import merge_lora

    checkpoint = str(tmp_path / "checkpoint")
    merged_dir = str(tmp_path / "merged")

    fake_tokenizer = MagicMock()
    fake_model = MagicMock()
    fake_model.merge_and_unload.return_value = fake_model
    fake_peft_config = MagicMock()
    fake_peft_config.base_model_name_or_path = "google/gemma-3-1b-it"

    with (
        patch(
            "training.scripts.export_litert.PeftConfig.from_pretrained",
            return_value=fake_peft_config,
        ),
        patch(
            "training.scripts.export_litert.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ),
        patch(
            "training.scripts.export_litert.AutoModelForCausalLM.from_pretrained",
            return_value=fake_model,
        ),
        patch(
            "training.scripts.export_litert.PeftModel.from_pretrained",
            return_value=fake_model,
        ),
    ):
        merge_lora(checkpoint, merged_dir)

    fake_model.save_pretrained.assert_called_once_with(merged_dir)
    fake_tokenizer.save_pretrained.assert_called_once_with(merged_dir)


def test_merge_lora_creates_output_dir(tmp_path):
    """merge_lora creates the output directory if it doesn't exist."""
    from training.scripts.export_litert import merge_lora

    checkpoint = str(tmp_path / "checkpoint")
    merged_dir = str(tmp_path / "deep" / "nested" / "merged")

    fake_tokenizer = MagicMock()
    fake_model = MagicMock()
    fake_model.merge_and_unload.return_value = fake_model
    fake_peft_config = MagicMock()
    fake_peft_config.base_model_name_or_path = "google/gemma-3-1b-it"

    with (
        patch(
            "training.scripts.export_litert.PeftConfig.from_pretrained",
            return_value=fake_peft_config,
        ),
        patch(
            "training.scripts.export_litert.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ),
        patch(
            "training.scripts.export_litert.AutoModelForCausalLM.from_pretrained",
            return_value=fake_model,
        ),
        patch(
            "training.scripts.export_litert.PeftModel.from_pretrained",
            return_value=fake_model,
        ),
    ):
        merge_lora(checkpoint, merged_dir)

    assert Path(merged_dir).exists()


def test_convert_and_package_exits_without_litert_torch(tmp_path):
    """convert_and_package exits cleanly when litert-torch is not installed."""
    from training.scripts.export_litert import convert_and_package

    broken_modules = {
        "litert_torch": None,
        "litert_torch.generative": None,
        "litert_torch.generative.examples": None,
        "litert_torch.generative.examples.gemma3": None,
        "litert_torch.generative.examples.gemma3.decoder": None,
        "litert_torch.generative.utilities": None,
        "litert_torch.generative.utilities.converter": None,
        "litert_torch.generative.utilities.export_config": None,
        "litert_torch.generative.layers": None,
        "litert_torch.generative.layers.kv_cache": None,
    }
    with patch.dict(sys.modules, broken_modules):
        with pytest.raises(SystemExit):
            convert_and_package(
                merged_dir=str(tmp_path),
                output_dir=str(tmp_path / "out"),
                quantization="int8",
                kv_cache_max_len=1280,
                prefill_seq_lens=[8, 64, 128, 256, 512],
                variant="int8",
                version="1.0.0",
            )


def test_quant_map_has_expected_keys():
    """QUANT_MAP contains int8 and int4 entries."""
    from training.scripts.export_litert import QUANT_MAP

    assert "int8" in QUANT_MAP
    assert "int4" in QUANT_MAP
    assert QUANT_MAP["int8"] == "dynamic_int8"
    assert QUANT_MAP["int4"] == "dynamic_int4_block128"
