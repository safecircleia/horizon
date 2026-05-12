# tests/training/test_mobile_model.py
import torch
from training.model.mobile import HorizonMobileModel, CATEGORIES, SEVERITIES


def test_categories_and_severities():
    assert len(CATEGORIES) == 8
    assert "benign" in CATEGORIES
    assert len(SEVERITIES) == 5
    assert "none" in SEVERITIES


def test_forward_output_shapes():
    model = HorizonMobileModel(pretrained=False)
    input_ids = torch.randint(0, 1000, (2, 64))
    attention_mask = torch.ones(2, 64, dtype=torch.long)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    assert out["category_logits"].shape == (2, 8)
    assert out["severity_logits"].shape == (2, 5)


def test_predict_returns_labels():
    model = HorizonMobileModel(pretrained=False)
    model.eval()
    input_ids = torch.randint(0, 1000, (1, 32))
    attention_mask = torch.ones(1, 32, dtype=torch.long)
    result = model.predict(input_ids=input_ids, attention_mask=attention_mask)
    assert result["category"] in CATEGORIES
    assert result["severity"] in SEVERITIES
    assert 0.0 <= result["confidence"] <= 1.0
