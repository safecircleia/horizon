#!/usr/bin/env python3
"""QLoRA fine-tuning script for SafeCircle risk detection model.

Usage:
    python -m training.scripts.train --config training/configs/base.yaml
    python -m training.scripts.train --config training/configs/quick.yaml
    python -m training.scripts.train --config training/configs/base.yaml --resume experiments/run-foo/checkpoints/step-500
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from datasets import Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

load_dotenv()

if os.getenv("HF_TOKEN"):
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HF_TOKEN")

from training.model.loader import load_model_and_tokenizer


def detect_hardware() -> dict:
    """Return hardware capabilities to override config where needed."""
    has_cuda = torch.cuda.is_available()
    has_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    has_fp16 = has_cuda and not has_bf16
    return {
        "has_cuda": has_cuda,
        "use_bf16": has_bf16,
        "use_fp16": has_fp16,
        "use_cpu": not has_cuda,
        "use_4bit": has_cuda,
    }


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_dataset_from_jsonl(path: str, tokenizer, max_seq_length: int) -> Dataset:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    dataset = Dataset.from_list(examples)
    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: len(x["input_ids"]) > 0)
    return dataset


def resolve_output_dir(template: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return template.replace("{timestamp}", timestamp)


def main():
    parser = argparse.ArgumentParser(description="Train SafeCircle risk detection model")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--resume", help="Resume from checkpoint path")
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config["training"]
    data_cfg = config["data"]

    hw = detect_hardware()
    print(f"Hardware: {'GPU (CUDA)' if hw['has_cuda'] else 'CPU'} | "
          f"bf16={hw['use_bf16']} | fp16={hw['use_fp16']} | 4bit={hw['use_4bit']}")

    # Override precision/quantization settings for CPU
    if not hw["has_cuda"]:
        train_cfg["bf16"] = False
        train_cfg["fp16"] = False
        config["quantization"]["load_in_4bit"] = False
        config["model"]["torch_dtype"] = "float32"
        print("Warning: training on CPU — this will be very slow. Use a GPU for real training.")

    output_dir = resolve_output_dir(train_cfg["output_dir"])
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    with open(f"{output_dir}/training_config.yaml", "w") as f:
        yaml.dump(config, f)

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)

    max_seq = data_cfg.get("max_seq_length", 2048)

    print(f"Loading training data from {data_cfg['train_file']}...")
    train_dataset = load_dataset_from_jsonl(data_cfg["train_file"], tokenizer, max_seq)

    print(f"Loading eval data from {data_cfg['eval_file']}...")
    eval_dataset = load_dataset_from_jsonl(data_cfg["eval_file"], tokenizer, max_seq)

    print(f"Train: {len(train_dataset)} examples | Eval: {len(eval_dataset)} examples")

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
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", False),
        use_cpu=hw["use_cpu"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        report_to=train_cfg.get("report_to", "tensorboard"),
        dataloader_pin_memory=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume)

    print(f"Saving final model to {output_dir}/final/")
    model.save_pretrained(f"{output_dir}/final/")
    tokenizer.save_pretrained(f"{output_dir}/final/")

    print("Training complete.")


if __name__ == "__main__":
    main()
