# training/model/mobile.py
"""MobileBERT-based classifier for on-device risk detection."""

from typing import Dict
import json
import os
import torch
import torch.nn as nn
from transformers import MobileBertModel, MobileBertConfig

CATEGORIES = [
    "grooming", "bullying", "sexual_content", "isolation",
    "personal_info", "platform_migration", "threats", "benign",
]

SEVERITIES = ["none", "low", "medium", "high", "critical"]

MOBILEBERT_MODEL = "google/mobilebert-uncased"


class HorizonMobileModel(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        if pretrained:
            self.encoder = MobileBertModel.from_pretrained(MOBILEBERT_MODEL)
        else:
            config = MobileBertConfig()
            self.encoder = MobileBertModel(config)
        hidden = self.encoder.config.hidden_size
        self.category_head = nn.Linear(hidden, len(CATEGORIES))
        self.severity_head = nn.Linear(hidden, len(SEVERITIES))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output
        return {
            "category_logits": self.category_head(pooled),
            "severity_logits": self.severity_head(pooled),
        }

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict:
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        cat_probs = torch.softmax(logits["category_logits"], dim=-1)[0]
        sev_probs = torch.softmax(logits["severity_logits"], dim=-1)[0]
        cat_idx = cat_probs.argmax().item()
        sev_idx = sev_probs.argmax().item()
        return {
            "category": CATEGORIES[cat_idx],
            "severity": SEVERITIES[sev_idx],
            "confidence": round(cat_probs[cat_idx].item(), 4),
        }

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f"{path}/pytorch_model.bin")
        with open(f"{path}/config.json", "w") as f:
            json.dump({"model_type": "horizon_mobile"}, f)

    @classmethod
    def from_pretrained(cls, path: str) -> "HorizonMobileModel":
        model = cls(pretrained=False)
        bin_path = f"{path}/pytorch_model.bin"
        safetensors_path = f"{path}/model.safetensors"
        if os.path.exists(bin_path):
            state = torch.load(bin_path, map_location="cpu", weights_only=True)
        elif os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state = load_file(safetensors_path)
        else:
            raise FileNotFoundError(f"No model weights found in {path}")
        model.load_state_dict(state, strict=False)
        return model
