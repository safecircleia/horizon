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
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from training.model.mobile import LABELS


def load_test_set(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_batch(session, tokenizer, texts: list[str], max_seq_length: int) -> list[str]:
    predictions = []
    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        logits, = session.run(None, {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        })
        predictions.append(LABELS[int(logits[0].argmax())])
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate horizon-mobile ONNX model")
    parser.add_argument("--model", required=True, help="Path to models/mobile directory (contains .onnx + tokenizer/)")
    parser.add_argument("--test-set", required=True, help="Path to eval JSONL")
    parser.add_argument("--output", default="evaluation/reports/mobile", help="Output directory")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=512)
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

    y_true, y_pred = [], []
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
            predictions = run_batch(session, tokenizer, texts, args.max_seq_length)
            for ex, pred in zip(batch, predictions):
                label = ex["label"]
                if isinstance(label, str):
                    label = json.loads(label)
                is_risk = label.get("risk_level", "none") != "none"
                y_true.append("risk" if is_risk else "safe")
                y_pred.append(pred)
            pbar.update(len(batch))

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "risk" and p == "risk")
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == "safe" and p == "safe")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == "safe" and p == "risk")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "risk" and p == "safe")
    total_risk = tp + fn
    total_safe = tn + fp
    fpr = round(fp / total_safe, 4) if total_safe else 0
    fnr = round(fn / total_risk, 4) if total_risk else 0
    f1 = round(f1_score(y_true, y_pred, pos_label="risk"), 4)
    precision = round(precision_score(y_true, y_pred, pos_label="risk"), 4)
    recall = round(recall_score(y_true, y_pred, pos_label="risk"), 4)

    results = {"fpr": fpr, "fnr": fnr, "f1": f1, "precision": precision, "recall": recall,
               "tp": tp, "tn": tn, "fp": fp, "fn": fn}

    print("\n" + "=" * 60)
    print("MOBILE MODEL BINARY EVALUATION REPORT")
    print("=" * 60)
    print(f"\n  F1:        {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"\n  False Positive Rate: {fpr:.2%}  ({fp} FPs / {total_safe} safe examples)")
    print(f"  False Negative Rate: {fnr:.2%}  ({fn} FNs / {total_risk} risk examples)")
    print("=" * 60)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()
