# training/model/mobile.py
"""MobileBERT-based classifier for on-device risk detection."""

from typing import Dict
import json
import os
import torch
import torch.nn as nn
from transformers import MobileBertModel, MobileBertConfig

LABELS = ["safe", "risk"]

# Kept for backward compatibility with distill.py
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
        self.classifier = nn.Linear(hidden, len(LABELS))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output
        return {"logits": self.classifier(pooled)}

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict:
        self.train(False)
        with torch.no_grad():
            out = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(out["logits"], dim=-1)[0]
        idx = probs.argmax().item()
        return {"label": LABELS[idx], "confidence": round(probs[idx].item(), 4)}

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
