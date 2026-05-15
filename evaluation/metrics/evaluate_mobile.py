#!/usr/bin/env python3
"""Evaluation script for the horizon-mobile ONNX model.

Usage:
    python -m evaluation.metrics.evaluate_mobile --model models/mobile --test-set data/processed/eval.jsonl
    python -m evaluation.metrics.evaluate_mobile --model models/mobile --test-set data/processed/eval.jsonl --batch-size 64
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

from training.model.mobile import CATEGORIES, SEVERITIES
from evaluation.metrics.evaluate import compute_metrics, print_report

RISK_LEVELS = ["none", "low", "medium", "high", "critical"]
SEV_TO_RISK = {
    "none": "none",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "critical": "critical",
}


def load_test_set(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_batch(session, tokenizer, texts: list[str], max_seq_length: int) -> tuple[list[str], list[list[str]]]:
    pred_levels = []
    pred_cats = []
    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        cat_logits, sev_logits = session.run(None, {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        })
        cat_idx = int(cat_logits[0].argmax())
        sev_idx = int(sev_logits[0].argmax())
        cat = CATEGORIES[cat_idx]
        sev = SEVERITIES[sev_idx]
        pred_levels.append(sev)
        pred_cats.append([] if cat == "benign" else [cat])
    return pred_levels, pred_cats


def main():
    parser = argparse.ArgumentParser(description="Evaluate horizon-mobile ONNX model")
    parser.add_argument("--model", required=True, help="Path to models/mobile directory (contains .onnx + tokenizer/)")
    parser.add_argument("--test-set", required=True, help="Path to eval JSONL")
    parser.add_argument("--output", default="evaluation/reports/mobile", help="Output directory")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=256)
    args = parser.parse_args()

    model_dir = Path(args.model)
    onnx_path = model_dir / "horizon-mobile.onnx"
    tokenizer_path = model_dir / "tokenizer"

    if not onnx_path.exists():
        print(f"Error: ONNX model not found at {onnx_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading ONNX model: {onnx_path}")
    session = ort.InferenceSession(str(onnx_path))

    print(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    print(f"Loading test set: {args.test_set}")
    examples = load_test_set(args.test_set)
    if args.max_samples:
        examples = examples[:args.max_samples]

    y_true_levels, y_pred_levels = [], []
    y_true_cats, y_pred_cats = [], []

    batches = [examples[i:i + args.batch_size] for i in range(0, len(examples), args.batch_size)]

    def _extract_conversation(text: str) -> str:
        user_tag = "<|start_header_id|>user<|end_header_id|>"
        asst_tag = "<|start_header_id|>assistant<|end_header_id|>"
        if user_tag in text:
            text = text[text.index(user_tag) + len(user_tag):]
        if asst_tag in text:
            text = text[:text.index(asst_tag)]
        return text.replace("<|eot_id|>", "").strip()

    with tqdm(total=len(examples), unit="ex", desc="Evaluating") as pbar:
        for batch in batches:
            texts = [_extract_conversation(ex["text"]) for ex in batch]

            pred_levels, pred_cats = run_batch(session, tokenizer, texts, args.max_seq_length)

            for ex, pred_level, pred_cat in zip(batch, pred_levels, pred_cats):
                label = ex["label"]
                if isinstance(label, str):
                    label = json.loads(label)
                true_level = label.get("risk_level", "none")
                true_cats = [c for c in label.get("categories", []) if c != "benign"]

                if pred_level not in RISK_LEVELS:
                    pred_level = "none"

                y_true_levels.append(true_level)
                y_pred_levels.append(pred_level)
                y_true_cats.append(true_cats)
                y_pred_cats.append(pred_cat)

            pbar.update(len(batch))

    results = compute_metrics(y_true_levels, y_pred_levels, y_true_cats, y_pred_cats)
    print_report(results)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()
