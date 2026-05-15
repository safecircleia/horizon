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
    val_examples = load_jsonl(data_cfg["eval_file"])

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
            soft_raw = inputs.pop("soft_labels", None)
            hard_cat = inputs.pop("hard_cat_labels").to(device)
            hard_sev = inputs.pop("hard_sev_labels").to(device)
            if soft_raw is None:
                soft_raw = torch.zeros(hard_cat.shape[0], len(CATEGORIES))
            soft = torch.tensor(soft_raw, dtype=torch.float32).to(device) if not isinstance(soft_raw, torch.Tensor) else soft_raw.to(device)
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

    def tokenize_and_label(examples_list, include_soft=False):
        texts = [ex["text"] for ex in examples_list]
        enc = student_tokenizer(texts, truncation=True, max_length=max_seq, padding=False)
        records = []
        for i, ex in enumerate(examples_list):
            label = ex.get("label", {})
            if isinstance(label, str):
                label = json.loads(label)
            cat_str = (label.get("categories") or [ex.get("category", "benign")])[0]
            sev_str = label.get("risk_level", "none")
            record = {
                "input_ids": enc["input_ids"][i],
                "attention_mask": enc["attention_mask"][i],
                "hard_cat_labels": CATEGORY_TO_IDX.get(cat_str, 0),
                "hard_sev_labels": SEVERITY_TO_IDX.get(sev_str, 0),
            }
            if include_soft:
                record["soft_labels"] = ex["soft_labels"]
            records.append(record)
        return records

    train_dataset = Dataset.from_list(tokenize_and_label(train_with_soft, include_soft=True))
    val_dataset = Dataset.from_list(tokenize_and_label(val_examples, include_soft=False))

    def distill_collator(features):
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"]) for f in features]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=student_tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "hard_cat_labels": torch.tensor([f["hard_cat_labels"] for f in features], dtype=torch.long),
            "hard_sev_labels": torch.tensor([f["hard_sev_labels"] for f in features], dtype=torch.long),
        }
        if "soft_labels" in features[0]:
            batch["soft_labels"] = torch.tensor([f["soft_labels"] for f in features], dtype=torch.float32)
        return batch

    trainer = DistillationTrainer(
        model=student,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=distill_collator,
    )

    print("Starting distillation training...")
    trainer.train()

    student.save_pretrained(f"{output_dir}/final/")
    student_tokenizer.save_pretrained(f"{output_dir}/final/")
    print(f"Mobile model saved to {output_dir}/final/")


if __name__ == "__main__":
    main()
