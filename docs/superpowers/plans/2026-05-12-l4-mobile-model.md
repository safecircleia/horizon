# L4 GPU Optimization + Mobile Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade training to fully utilize the NVIDIA L4 (24GB VRAM) and produce `horizon-mobile`, a ~25MB on-device ONNX classifier with explainability for iOS and Android.

**Architecture:** Train `horizon-full` (Llama 3.2 3B, full bfloat16 + Flash Attention 2) on the L4, then distill it into a MobileBERT classifier with category + severity heads, exported to INT8 ONNX. Explainability via Integrated Gradients exported as a second ONNX graph.

**Tech Stack:** PyTorch, HuggingFace Transformers, PEFT, BitsAndBytes, captum (Integrated Gradients), onnxruntime, marimo

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `training/configs/l4.yaml` | Create | L4-optimized full model training config |
| `training/configs/mobile.yaml` | Create | MobileBERT distillation training config |
| `training/model/loader.py` | Modify | Add Flash Attention 2 + full bfloat16 branch |
| `training/model/mobile.py` | Create | MobileBERT classifier with category + severity heads |
| `training/scripts/distill.py` | Create | Soft label generation + distillation training |
| `training/scripts/export_onnx.py` | Create | ONNX export + INT8 quantization |
| `notebooks/horizon_training.py` | Modify | L4 config option, distillation + export sections |
| `tests/training/test_mobile_model.py` | Create | Unit tests for MobileBERT classifier |
| `tests/training/test_distill.py` | Create | Unit tests for distillation loss + soft label generation |
| `tests/training/test_export_onnx.py` | Create | Unit tests for ONNX export shapes and output validity |

---

## Task 1: L4 Training Config

**Files:**
- Create: `training/configs/l4.yaml`

- [ ] **Step 1: Create the config**

```yaml
# training/configs/l4.yaml
model:
  base_model: "meta-llama/Llama-3.2-3B-Instruct"
  torch_dtype: "bfloat16"
  max_seq_length: 4096
  attn_implementation: "flash_attention_2"

lora:
  rank: 128
  alpha: 256
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

quantization:
  load_in_4bit: false

training:
  output_dir: "experiments/l4-{timestamp}"
  max_steps: 15000
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 16
  gradient_accumulation_steps: 2
  gradient_checkpointing: false
  learning_rate: 2.0e-4
  lr_scheduler_type: "cosine"
  warmup_steps: 200
  weight_decay: 0.01
  max_grad_norm: 1.0
  fp16: false
  bf16: true
  logging_steps: 50
  eval_steps: 500
  save_steps: 500
  save_total_limit: 3
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
  report_to: "tensorboard"
  optim: "adamw_torch_fused"
  dataloader_num_workers: 4

data:
  train_file: "data/processed/train.jsonl"
  eval_file: "data/processed/eval.jsonl"
  train_split: 0.9
  max_seq_length: 4096

stages:
  stage1:
    name: "category_recognition"
    steps: 5000
  stage2:
    name: "severity_calibration"
    steps: 5000
  stage3:
    name: "false_positive_reduction"
    steps: 5000
```

- [ ] **Step 2: Commit**

```bash
git add training/configs/l4.yaml
git commit -m "feat: add L4 GPU training config (full bfloat16, Flash Attention 2)"
```

---

## Task 2: Flash Attention 2 in Model Loader

**Files:**
- Modify: `training/model/loader.py`

- [ ] **Step 1: Replace the `if use_4bit: ... else:` block in `loader.py` (lines 33–58) with:**

```python
    if use_4bit:
        compute_dtype = TORCH_DTYPE_MAP[quant_cfg["bnb_4bit_compute_dtype"]]
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
            llm_int8_enable_fp32_cpu_offload=quant_cfg.get("llm_int8_enable_fp32_cpu_offload", False),
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["base_model"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=model_dtype,
            max_memory=quant_cfg.get("max_memory", None),
        )
        model = prepare_model_for_kbit_training(model)
    elif torch.cuda.is_available():
        # Full precision GPU path (L4 / high-VRAM GPUs)
        kwargs = dict(
            device_map="cuda:0",
            trust_remote_code=True,
            torch_dtype=model_dtype,
        )
        attn_impl = model_cfg.get("attn_implementation")
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl
        model = AutoModelForCausalLM.from_pretrained(model_cfg["base_model"], **kwargs)
    else:
        # CPU fallback
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["base_model"],
            device_map="cpu",
            trust_remote_code=True,
            torch_dtype=model_dtype,
        )
```

- [ ] **Step 2: Run existing tests**

```bash
python -m pytest tests/ -v -x
```

Expected: all existing tests pass.

- [ ] **Step 3: Commit**

```bash
git add training/model/loader.py
git commit -m "feat: add full bfloat16 + Flash Attention 2 loader branch for high-VRAM GPUs"
```

---

## Task 3: MobileBERT Classifier Model

**Files:**
- Create: `training/model/mobile.py`
- Create: `tests/training/test_mobile_model.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/training/test_mobile_model.py -v
```

Expected: `ModuleNotFoundError` for `training.model.mobile`.

- [ ] **Step 3: Implement the model**

```python
# training/model/mobile.py
"""MobileBERT-based classifier for on-device risk detection."""

from typing import Dict
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
        import os, json
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f"{path}/pytorch_model.bin")
        with open(f"{path}/config.json", "w") as f:
            json.dump({"model_type": "horizon_mobile"}, f)

    @classmethod
    def from_pretrained(cls, path: str) -> "HorizonMobileModel":
        model = cls(pretrained=False)
        state = torch.load(f"{path}/pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state)
        return model
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/training/test_mobile_model.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add training/model/mobile.py tests/training/test_mobile_model.py
git commit -m "feat: add MobileBERT classifier with category and severity heads"
```

---

## Task 4: Mobile Training Config

**Files:**
- Create: `training/configs/mobile.yaml`

- [ ] **Step 1: Create the config**

```yaml
# training/configs/mobile.yaml
model:
  base_model: "google/mobilebert-uncased"
  max_seq_length: 256

distillation:
  temperature: 4.0
  alpha: 0.7

training:
  output_dir: "experiments/mobile-{timestamp}"
  max_steps: 5000
  per_device_train_batch_size: 32
  per_device_eval_batch_size: 32
  gradient_accumulation_steps: 1
  learning_rate: 2.0e-5
  lr_scheduler_type: "cosine"
  warmup_steps: 200
  weight_decay: 0.01
  max_grad_norm: 1.0
  fp16: false
  bf16: true
  logging_steps: 50
  eval_steps: 250
  save_steps: 250
  save_total_limit: 3
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
  report_to: "tensorboard"
  optim: "adamw_torch_fused"
  dataloader_num_workers: 4

data:
  train_file: "data/processed/train.jsonl"
  eval_file: "data/processed/eval.jsonl"
  max_seq_length: 256
```

- [ ] **Step 2: Commit**

```bash
git add training/configs/mobile.yaml
git commit -m "feat: add mobile distillation training config"
```

---

## Task 5: Knowledge Distillation Script

**Files:**
- Create: `training/scripts/distill.py`
- Create: `tests/training/test_distill.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/training/test_distill.py
import torch
from training.scripts.distill import distillation_loss


def test_distillation_loss_shape():
    student_cat_logits = torch.randn(4, 8)
    student_sev_logits = torch.randn(4, 5)
    soft_labels = torch.softmax(torch.randn(4, 8), dim=-1)
    hard_cat_labels = torch.randint(0, 8, (4,))
    hard_sev_labels = torch.randint(0, 5, (4,))

    loss = distillation_loss(
        student_cat_logits=student_cat_logits,
        student_sev_logits=student_sev_logits,
        soft_labels=soft_labels,
        hard_cat_labels=hard_cat_labels,
        hard_sev_labels=hard_sev_labels,
        temperature=4.0,
        alpha=0.7,
    )
    assert loss.shape == torch.Size([])
    assert loss.item() > 0


def test_distillation_loss_alpha_boundary():
    student_logits = torch.randn(2, 8)
    soft = torch.softmax(torch.randn(2, 8), dim=-1)
    hard = torch.zeros(2, dtype=torch.long)
    sev = torch.zeros(2, dtype=torch.long)
    sev_logits = torch.randn(2, 5)

    loss_full = distillation_loss(student_logits, sev_logits, soft, hard, sev, 4.0, alpha=1.0)
    loss_none = distillation_loss(student_logits, sev_logits, soft, hard, sev, 4.0, alpha=0.0)
    assert loss_full.item() != loss_none.item()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/training/test_distill.py -v
```

Expected: `ImportError` for `training.scripts.distill`.

- [ ] **Step 3: Implement distill.py**

```python
# training/scripts/distill.py
"""Knowledge distillation: generate soft labels from horizon-full, train horizon-mobile."""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import torch
import torch.nn.functional as F
import yaml
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from training.model.mobile import HorizonMobileModel, CATEGORIES, SEVERITIES
from training.model.loader import load_for_inference

CATEGORY_TO_IDX = {c: i for i, c in enumerate(CATEGORIES)}
SEVERITY_TO_IDX = {s: i for i, s in enumerate(SEVERITIES)}


def distillation_loss(
    student_cat_logits: torch.Tensor,
    student_sev_logits: torch.Tensor,
    soft_labels: torch.Tensor,
    hard_cat_labels: torch.Tensor,
    hard_sev_labels: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    """Combined KL divergence (soft) + CrossEntropy (hard) loss."""
    scaled_student = student_cat_logits / temperature
    kl_loss = F.kl_div(
        F.log_softmax(scaled_student, dim=-1),
        soft_labels,
        reduction="batchmean",
    ) * (temperature ** 2)

    ce_cat = F.cross_entropy(student_cat_logits, hard_cat_labels)
    ce_sev = F.cross_entropy(student_sev_logits, hard_sev_labels)

    return alpha * kl_loss + (1 - alpha) * (ce_cat + ce_sev) / 2


def generate_soft_labels(
    teacher_model,
    teacher_tokenizer,
    examples: List[Dict],
    max_seq_length: int,
    batch_size: int = 8,
    device: str = "cuda",
) -> List[Dict]:
    """Run teacher over examples and collect soft category probability distributions."""
    teacher_model.eval()
    augmented = []

    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        texts = [ex["text"] for ex in batch]
        enc = teacher_tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = teacher_model(**enc)

        # Use last-token logits, map first N vocab positions to category probabilities
        logits = out.logits[:, -1, :]
        probs = torch.softmax(logits[:, :len(CATEGORIES)], dim=-1).cpu()

        for j, ex in enumerate(batch):
            augmented.append({**ex, "soft_labels": probs[j].tolist()})

        if i % 100 == 0:
            print(f"  Soft labels: {i}/{len(examples)}")

    return augmented


def load_jsonl(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Distill horizon-full into horizon-mobile")
    parser.add_argument("--teacher", required=True, help="Path to horizon-full checkpoint")
    parser.add_argument("--config", required=True, help="Path to mobile.yaml config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dist_cfg = cfg["distillation"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    max_seq = data_cfg["max_seq_length"]
    temperature = dist_cfg["temperature"]
    alpha = dist_cfg["alpha"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading teacher model...")
    teacher, teacher_tokenizer = load_for_inference(args.teacher)
    teacher = teacher.to(device)

    print("Loading training data...")
    train_examples = load_jsonl(data_cfg["train_file"])
    eval_examples = load_jsonl(data_cfg["eval_file"])

    print(f"Generating soft labels for {len(train_examples)} train examples...")
    train_with_soft = generate_soft_labels(
        teacher, teacher_tokenizer, train_examples, max_seq, device=device
    )

    del teacher
    if device == "cuda":
        torch.cuda.empty_cache()

    student = HorizonMobileModel(pretrained=True).to(device)
    student_tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

    class DistillationTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            soft = torch.tensor(inputs.pop("soft_labels"), dtype=torch.float32).to(device)
            hard_cat = inputs.pop("hard_cat_labels").to(device)
            hard_sev = inputs.pop("hard_sev_labels").to(device)
            out = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            loss = distillation_loss(
                out["category_logits"], out["severity_logits"],
                soft, hard_cat, hard_sev,
                temperature=temperature, alpha=alpha,
            )
            return (loss, out) if return_outputs else loss

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = train_cfg["output_dir"].replace("{timestamp}", timestamp)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=train_cfg["max_steps"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_steps=train_cfg["warmup_steps"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        bf16=train_cfg.get("bf16", False),
        fp16=train_cfg.get("fp16", False),
        logging_steps=train_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        report_to=train_cfg.get("report_to", "tensorboard"),
        optim=train_cfg.get("optim", "adamw_torch_fused"),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 2),
    )

    train_dataset = Dataset.from_list(train_with_soft)
    eval_dataset = Dataset.from_list(eval_examples)

    trainer = DistillationTrainer(
        model=student,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(student_tokenizer),
    )

    print("Starting distillation training...")
    trainer.train()

    student.save_pretrained(f"{output_dir}/final/")
    student_tokenizer.save_pretrained(f"{output_dir}/final/")
    print(f"Mobile model saved to {output_dir}/final/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/training/test_distill.py -v
```

Expected: all 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add training/scripts/distill.py tests/training/test_distill.py
git commit -m "feat: add knowledge distillation script for horizon-mobile"
```

---

## Task 6: ONNX Export Script

**Files:**
- Create: `training/scripts/export_onnx.py`
- Create: `tests/training/test_export_onnx.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/training/test_export_onnx.py -v
```

Expected: `ImportError` for `training.scripts.export_onnx`.

- [ ] **Step 3: Implement export_onnx.py**

```python
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
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["category_logits", "severity_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "category_logits": {0: "batch_size"},
            "severity_logits": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
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
    parser.add_argument("--quantize", action="store_true", default=True)
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/training/test_export_onnx.py -v
```

Expected: all 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add training/scripts/export_onnx.py tests/training/test_export_onnx.py
git commit -m "feat: add ONNX export script with INT8 quantization for horizon-mobile"
```

---

## Task 7: Update Marimo Notebook

**Files:**
- Modify: `notebooks/horizon_training.py`

- [ ] **Step 1: Update hardware detection cell — replace the cell body (lines 50–64) with:**

```python
@app.cell
def _(mo):
    import torch

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        hw_info = f"**GPU detected:** {gpu_name} ({gpu_mem:.1f} GB VRAM)"
        if gpu_mem >= 20:
            hw_info += "\n\n✅ High-VRAM GPU detected — use **L4 config** for full bfloat16 training (no 4-bit needed)."
            hw_color = "success"
        else:
            hw_info += f"\n\n⚠️ {gpu_mem:.1f} GB VRAM — use **Quick** or **Base** config with 4-bit quantization."
            hw_color = "warn"
    else:
        hw_info = "**No GPU found** — training will run on CPU (slow, not recommended)"
        hw_color = "warn"

    mo.callout(mo.md(hw_info), kind=hw_color)
    return gpu_mem, gpu_name, has_cuda, torch
```

- [ ] **Step 2: Update training config dropdown — replace the `config_choice` cell (lines 172–181) with:**

```python
@app.cell
def _(mo):
    config_choice = mo.ui.dropdown(
        options={
            "Quick (500 steps, for testing)": "training/configs/quick.yaml",
            "Base (10,000 steps, production)": "training/configs/base.yaml",
            "L4 GPU (15,000 steps, full bfloat16)": "training/configs/l4.yaml",
            "Mobile distillation (MobileBERT)": "training/configs/mobile.yaml",
        },
        value="Quick (500 steps, for testing)",
        label="Training config",
    )
    resume_path = mo.ui.text(placeholder="experiments/run-xxx/checkpoints/step-500 (optional)", label="Resume from checkpoint")
    mo.vstack([config_choice, resume_path])
    return config_choice, resume_path
```

- [ ] **Step 3: Add Step 3.5 — Knowledge Distillation section — insert before `## 📋 Experiment History` (before line 258):**

```python
@app.cell
def _(mo):
    mo.md("## 🧪 Step 3.5: Knowledge Distillation (Mobile Model)")
    return


@app.cell
def _(Path, mo):
    _runs = sorted(Path("experiments").glob("*/final"), key=lambda p: p.stat().st_mtime, reverse=True) if Path("experiments").exists() else []
    _options = {str(p): str(p) for p in _runs} if _runs else {"No checkpoints found": ""}
    teacher_select = mo.ui.dropdown(options=_options, label="Teacher checkpoint (horizon-full)")
    mo.vstack([mo.md("Select the trained `horizon-full` checkpoint to distill from:"), teacher_select])
    return (teacher_select,)


@app.cell
def _(mo):
    run_distill_btn = mo.ui.run_button(label="▶ Run Distillation")
    run_distill_btn
    return (run_distill_btn,)


@app.cell
def _(mo, run_distill_btn, subprocess, teacher_select):
    mo.stop(not run_distill_btn.value)
    mo.stop(not teacher_select.value)
    _result = subprocess.run(
        ["python", "-m", "training.scripts.distill",
         "--teacher", teacher_select.value,
         "--config", "training/configs/mobile.yaml"],
        capture_output=True, text=True
    )
    if _result.returncode == 0:
        mo.callout(mo.md(f"✅ Distillation complete\n```\n{_result.stdout[-3000:]}\n```"), kind="success")
    else:
        mo.callout(mo.md(f"❌ Distillation failed\n```\n{_result.stderr[-3000:]}\n```"), kind="danger")
    return
```

- [ ] **Step 4: Add Step 4.5 — ONNX Export section — insert after distillation section, before `## 📋 Experiment History`:**

```python
@app.cell
def _(mo):
    mo.md("## 📦 Step 4.5: Export Mobile Model to ONNX")
    return


@app.cell
def _(Path, mo):
    _mobile_runs = sorted(Path("experiments").glob("mobile-*/final"), key=lambda p: p.stat().st_mtime, reverse=True) if Path("experiments").exists() else []
    _options = {str(p): str(p) for p in _mobile_runs} if _mobile_runs else {"No mobile checkpoints found": ""}
    mobile_checkpoint_select = mo.ui.dropdown(options=_options, label="Mobile checkpoint to export")
    mo.vstack([mo.md("Select a trained mobile checkpoint to export:"), mobile_checkpoint_select])
    return (mobile_checkpoint_select,)


@app.cell
def _(mo):
    run_export_btn = mo.ui.run_button(label="▶ Export to ONNX")
    run_export_btn
    return (run_export_btn,)


@app.cell
def _(mo, mobile_checkpoint_select, run_export_btn, subprocess):
    mo.stop(not run_export_btn.value)
    mo.stop(not mobile_checkpoint_select.value)
    _result = subprocess.run(
        ["python", "-m", "training.scripts.export_onnx",
         "--checkpoint", mobile_checkpoint_select.value,
         "--output", "models/mobile",
         "--quantize"],
        capture_output=True, text=True
    )
    if _result.returncode == 0:
        from pathlib import Path as _Path
        _sizes = {p.name: f"{p.stat().st_size / 1e6:.1f} MB" for p in _Path("models/mobile").glob("*.onnx")} if _Path("models/mobile").exists() else {}
        _size_info = "\n".join(f"- `{k}`: {v}" for k, v in _sizes.items()) or "No ONNX files found"
        mo.callout(mo.md(f"✅ Export complete\n\n**Output files:**\n{_size_info}\n\n```\n{_result.stdout[-2000:]}\n```"), kind="success")
    else:
        mo.callout(mo.md(f"❌ Export failed\n```\n{_result.stderr[-3000:]}\n```"), kind="danger")
    return
```

- [ ] **Step 5: Verify the notebook starts without errors**

```bash
cd /home/tomas/Documents/Github/SafeCircle/horizon && python -c "import ast; ast.parse(open('notebooks/horizon_training.py').read()); print('Syntax OK')"
```

Expected: `Syntax OK`

- [ ] **Step 6: Commit**

```bash
git add notebooks/horizon_training.py
git commit -m "feat: update notebook with L4 config, distillation, and ONNX export sections"
```

---

## Task 8: Makefile Targets

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add these targets to the Makefile after the existing `train` target:**

```makefile
train-l4:
	python -m training.scripts.train --config training/configs/l4.yaml

distill:
	python -m training.scripts.distill \
		--teacher $(TEACHER) \
		--config training/configs/mobile.yaml

export-mobile:
	python -m training.scripts.export_onnx \
		--checkpoint $(CHECKPOINT) \
		--output models/mobile \
		--quantize
```

- [ ] **Step 2: Commit**

```bash
git add Makefile
git commit -m "feat: add Makefile targets for L4 training, distillation, and ONNX export"
```

---

## Task 9: Final Integration Test

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 2: Verify configs load correctly**

```bash
python -c "
import yaml
for cfg in ['training/configs/l4.yaml', 'training/configs/mobile.yaml']:
    with open(cfg) as f:
        data = yaml.safe_load(f)
    print(f'{cfg}: OK — {list(data.keys())}')
"
```

Expected:
```
training/configs/l4.yaml: OK — ['model', 'lora', 'quantization', 'training', 'data', 'stages']
training/configs/mobile.yaml: OK — ['model', 'distillation', 'training', 'data']
```

- [ ] **Step 3: Verify mobile model forward pass**

```bash
python -c "
import torch
from training.model.mobile import HorizonMobileModel
m = HorizonMobileModel(pretrained=False)
ids = torch.zeros(1, 32, dtype=torch.long)
mask = torch.ones(1, 32, dtype=torch.long)
out = m(ids, mask)
print('category_logits:', out['category_logits'].shape)
print('severity_logits:', out['severity_logits'].shape)
result = m.predict(ids, mask)
print('predict:', result)
"
```

Expected: shapes `(1, 8)` and `(1, 5)`, predict returns dict with `category`, `severity`, `confidence`.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete L4 GPU optimization and horizon-mobile implementation"
```
