#!/usr/bin/env python3
"""QLoRA fine-tuning entry point for SafeCircle risk detection model.

Usage:
    python -m training.scripts.train --config training/configs/h100.yaml
    python -m training.scripts.train --config training/configs/l4.yaml
    python -m training.scripts.train --config training/configs/h100.yaml --resume experiments/run-foo/checkpoints/step-500
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml
from datasets import Dataset
from dotenv import load_dotenv
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

load_dotenv()

if os.getenv("HF_TOKEN"):
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from training.model.loader import load_model_and_tokenizer


def _detect_hardware() -> dict:
    has_cuda = torch.cuda.is_available()
    has_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    return {
        "has_cuda": has_cuda,
        "use_bf16": has_bf16,
        "use_fp16": has_cuda and not has_bf16,
        "use_cpu": not has_cuda,
    }


def _load_jsonl_dataset(path: str, tokenizer, max_seq_length: int) -> Dataset:
    with open(path) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    dataset = Dataset.from_list(examples)
    dataset = dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_seq_length, padding=False),
        batched=True,
        remove_columns=dataset.column_names,
    )
    return dataset.filter(lambda x: len(x["input_ids"]) > 0)


def main():
    parser = argparse.ArgumentParser(description="Train SafeCircle risk detection model")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--resume", help="Resume from checkpoint path")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]
    data_cfg = config["data"]
    hw = _detect_hardware()

    print(f"Hardware: {'GPU (CUDA)' if hw['has_cuda'] else 'CPU'} | "
          f"bf16={hw['use_bf16']} | fp16={hw['use_fp16']}")

    if not hw["has_cuda"]:
        train_cfg.update({"bf16": False, "fp16": False})
        config["quantization"]["load_in_4bit"] = False
        config["model"]["torch_dtype"] = "float32"
        print("Warning: no GPU detected — training on CPU will be extremely slow.")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = train_cfg["output_dir"].replace("{timestamp}", timestamp)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{output_dir}/training_config.yaml", "w") as f:
        yaml.dump(config, f)

    print(f"Output directory: {output_dir}")
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)

    if train_cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()

    max_seq = data_cfg.get("max_seq_length", 2048)
    print(f"Loading datasets (max_seq={max_seq})...")
    train_dataset = _load_jsonl_dataset(data_cfg["train_file"], tokenizer, max_seq)
    eval_dataset = _load_jsonl_dataset(data_cfg["eval_file"], tokenizer, max_seq)
    print(f"Train: {len(train_dataset)} examples | Eval: {len(eval_dataset)} examples")

    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=train_cfg["max_steps"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
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
        dataloader_pin_memory=train_cfg.get("dataloader_pin_memory", False),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        torch_compile=train_cfg.get("torch_compile", False),
        optim=train_cfg.get("optim", "adamw_torch"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume)

    final_dir = f"{output_dir}/final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Training complete. Model saved to {final_dir}")


if __name__ == "__main__":
    main()
