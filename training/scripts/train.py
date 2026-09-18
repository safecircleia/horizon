#!/usr/bin/env python3
"""QLoRA fine-tuning entry point for SafeCircle risk detection model.

Uses TRL SFTTrainer with assistant_only_loss=True so loss is computed
only on assistant completions, not on system/user prompt tokens.

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

load_dotenv()

if os.getenv("HF_TOKEN"):
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from trl import SFTConfig, SFTTrainer

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


def _load_jsonl_dataset(path: str) -> Dataset:
    """Load a messages-format JSONL file into a HuggingFace Dataset."""
    with open(path) as f:
        examples = [json.loads(line) for line in f if line.strip()]
    return Dataset.from_list(examples)


def main():
    parser = argparse.ArgumentParser(
        description="Train SafeCircle risk detection model"
    )
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--resume", help="Resume from checkpoint path")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]
    data_cfg = config["data"]
    hw = _detect_hardware()

    print(
        f"Hardware: {'GPU (CUDA)' if hw['has_cuda'] else 'CPU'} | "
        f"bf16={hw['use_bf16']} | fp16={hw['use_fp16']}"
    )

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

    max_seq = data_cfg.get("max_seq_length", 2048)
    print(f"Loading datasets (max_seq={max_seq})...")
    train_dataset = _load_jsonl_dataset(data_cfg["train_file"])
    eval_dataset = _load_jsonl_dataset(data_cfg["eval_file"])
    print(f"Train: {len(train_dataset)} examples | Eval: {len(eval_dataset)} examples")

    sft_config = SFTConfig(
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
        dataloader_prefetch_factor=(
            train_cfg.get("dataloader_prefetch_factor", 2)
            if train_cfg.get("dataloader_num_workers", 0) > 0
            else None
        ),
        torch_compile=train_cfg.get("torch_compile", False),
        torch_compile_backend=train_cfg.get("torch_compile_backend", "inductor"),
        optim=train_cfg.get("optim", "adamw_torch"),
        # Compute loss only on assistant turns.
        # assistant_only_loss requires {% generation %} markers (not in Llama-3.2 template);
        # completion_only_loss with the Llama-3.2 assistant header achieves the same effect.
        # Explicitly pin loss_type="nll" — trl>=1.7 changed the default to "chunked_nll"
        # which alters training behavior. Remove this pin only if you want chunked NLL.
        loss_type="nll",
        max_length=max_seq,
        completion_only_loss=True,
        dataset_text_field=None,  # use messages format, not a single text field
        dataset_kwargs={
            "cache_dir": data_cfg.get("cache_dir", "data/.tokenized_cache")
        },
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    resume = str(Path(args.resume).resolve()) if args.resume else None
    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume)

    final_dir = f"{output_dir}/final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Training complete. Model saved to {final_dir}")


if __name__ == "__main__":
    main()
