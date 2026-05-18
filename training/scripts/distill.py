# training/scripts/distill.py
"""Knowledge distillation: generate soft labels from horizon-full, train horizon-mobile."""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
import yaml
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

from training.model.mobile import HorizonMobileModel, CATEGORIES, SEVERITIES
from training.model.loader import load_for_inference

CATEGORY_TO_IDX = {c: i for i, c in enumerate(CATEGORIES)}
SEVERITY_TO_IDX = {s: i for i, s in enumerate(SEVERITIES)}
ASSISTANT_TAG = "<|start_header_id|>assistant<|end_header_id|>"


def distillation_loss(
    student_cat_logits: torch.Tensor,
    student_sev_logits: torch.Tensor,
    soft_labels: torch.Tensor,
    hard_cat_labels: torch.Tensor,
    hard_sev_labels: torch.Tensor,
    temperature: float,
    alpha: float,
    cat_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    scaled_student = student_cat_logits.clamp(-30, 30) / temperature
    kl_loss = F.kl_div(
        F.log_softmax(scaled_student, dim=-1),
        soft_labels,
        reduction="batchmean",
    )
    ce_cat = F.cross_entropy(student_cat_logits, hard_cat_labels, weight=cat_weights)
    ce_sev = F.cross_entropy(student_sev_logits, hard_sev_labels)
    return alpha * kl_loss + (1 - alpha) * (ce_cat + ce_sev) / 2


def _extract_prompt(text: str) -> str:
    if ASSISTANT_TAG in text:
        return text[:text.rindex(ASSISTANT_TAG) + len(ASSISTANT_TAG)] + "\n"
    return text


def _parse_teacher_output(generated: str) -> Optional[Dict]:
    first_block = generated.split("\nassistant")[0].strip()
    try:
        return json.loads(first_block)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", first_block, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None


def _build_soft_label(parsed: Optional[Dict]) -> List[float]:
    """Convert a teacher JSON prediction into a soft label probability vector."""
    n = len(CATEGORIES)
    benign_idx = CATEGORY_TO_IDX.get("benign", n - 1)

    if parsed is None:
        # Unknown — peak on benign with low confidence
        soft = [0.05 / (n - 1)] * n
        soft[benign_idx] = 0.95
        return soft

    pred_cats = parsed.get("categories", [])
    confidence = min(max(float(parsed.get("confidence", 0.5)), 0.0), 1.0)
    matched = [c for c in pred_cats if c in CATEGORY_TO_IDX and c != "benign"]

    if not matched:
        # Benign example — peak strongly on benign class
        soft = [0.05 / (n - 1)] * n
        soft[benign_idx] = 0.95
        return soft

    soft = [0.0] * n
    share = confidence / len(matched)
    remainder = (1.0 - confidence) / max(n - len(matched), 1)
    for k, cat in enumerate(CATEGORIES):
        soft[k] = share if cat in matched else remainder
    total = sum(soft)
    return [max(s / total, 1e-8) for s in soft]


def generate_soft_labels(
    teacher_model,
    teacher_tokenizer,
    examples: List[Dict],
    max_seq_length: int,
    batch_size: int = 16,
    device: str = "cuda",
) -> List[Dict]:
    """Run teacher generation and build soft labels from predicted category + confidence."""
    teacher_tokenizer.padding_side = "left"
    teacher_model.eval()
    augmented = []
    failures = 0

    with tqdm(total=len(examples), desc="Generating soft labels", unit="ex") as pbar:
        for i in range(0, len(examples), batch_size):
            batch = examples[i : i + batch_size]
            prompts = [_extract_prompt(ex["text"]) for ex in batch]
            enc = teacher_tokenizer(
                prompts,
                truncation=True,
                max_length=max_seq_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            input_len = enc["input_ids"].shape[1]

            with torch.no_grad():
                out = teacher_model.generate(
                    **enc,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=teacher_tokenizer.eos_token_id,
                )

            for j, ex in enumerate(batch):
                generated = teacher_tokenizer.decode(out[j][input_len:], skip_special_tokens=True)
                parsed = _parse_teacher_output(generated)
                if parsed is None:
                    failures += 1
                augmented.append({**ex, "soft_labels": _build_soft_label(parsed)})

            pbar.update(len(batch))

    if failures:
        print(f"Warning: {failures}/{len(examples)} teacher parse failures (used uniform fallback)")
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

    # Convert messages format → text format if needed
    def _ensure_text(examples, tokenizer):
        if examples and "messages" in examples[0] and "text" not in examples[0]:
            for ex in examples:
                ex["text"] = tokenizer.apply_chat_template(
                    ex["messages"], tokenize=False, add_generation_prompt=False
                )
                if "label" not in ex:
                    for msg in reversed(ex["messages"]):
                        if msg["role"] == "assistant":
                            try:
                                ex["label"] = json.loads(msg["content"])
                            except json.JSONDecodeError:
                                ex["label"] = {"risk_level": "none", "categories": []}
                            break
        return examples

    train_examples = _ensure_text(train_examples, teacher_tokenizer)
    val_examples = _ensure_text(val_examples, teacher_tokenizer)

    print("Building soft labels from stored labels (no inference needed)...")
    train_with_soft = []
    for ex in tqdm(train_examples, desc="Soft labels", unit="ex"):
        label = ex.get("label", {})
        if isinstance(label, str):
            label = json.loads(label)
        parsed = {"categories": label.get("categories", []), "confidence": label.get("severity_score", label.get("confidence", 0.5))}
        train_with_soft.append({**ex, "soft_labels": _build_soft_label(parsed)})

    del teacher
    if device == "cuda":
        torch.cuda.empty_cache()

    student = HorizonMobileModel(pretrained=True).to(device)
    student_tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

    # Binary classification: benign=0, any risk=1
    # Weight the risk class 2x since benign is 2x more common (10k vs 40k risk)
    binary_weights = torch.tensor([1.0, 2.5], dtype=torch.float32, device=device)

    class DistillationTrainer(Trainer):
        def _model_inputs(self, inputs):
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            hard_cat = inputs["hard_cat_labels"].to(device)
            binary_labels = (hard_cat != CATEGORY_TO_IDX["benign"]).long()
            out = model(**self._model_inputs(inputs))
            loss = F.cross_entropy(out["logits"], binary_labels, weight=binary_weights)
            return (loss, out) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            with torch.no_grad():
                loss = self.compute_loss(model, inputs)
            return loss.detach(), None, None

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = train_cfg["output_dir"].replace("{timestamp}", timestamp)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    def _extract_conversation(text: str) -> str:
        """Strip Llama chat template boilerplate, keep only the conversation content."""
        user_tag = "<|start_header_id|>user<|end_header_id|>"
        asst_tag = "<|start_header_id|>assistant<|end_header_id|>"
        if user_tag in text:
            text = text[text.index(user_tag) + len(user_tag):]
        if asst_tag in text:
            text = text[:text.index(asst_tag)]
        return text.replace("<|eot_id|>", "").strip()

    def tokenize_and_label(examples_list, include_soft=False):
        texts = [_extract_conversation(ex["text"]) for ex in examples_list]
        enc = student_tokenizer(texts, truncation=True, max_length=max_seq, padding=False)
        records = []
        for i, ex in enumerate(examples_list):
            label = ex.get("label", {})
            if isinstance(label, str):
                label = json.loads(label)
            cats = label.get("categories") or [ex.get("category", "benign")]
            cat_str = cats[0] if cats else "benign"
            sev_str = label.get("risk_level", "none")
            record = {
                "input_ids": enc["input_ids"][i],
                "attention_mask": enc["attention_mask"][i],
                "hard_cat_labels": CATEGORY_TO_IDX.get(cat_str, CATEGORY_TO_IDX["benign"]),
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
        remove_unused_columns=False,
    )

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
