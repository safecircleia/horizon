#!/usr/bin/env python3
"""Evaluation script for SafeCircle risk detection model.

Runs inference on a test set and computes per-category F1, precision, recall,
confusion matrix, false positive/negative rates.

Usage:
    python -m evaluation.metrics.evaluate --checkpoint experiments/run-foo/final --test-set data/evaluation/test.jsonl
    python -m evaluation.metrics.evaluate --checkpoint experiments/run-foo/final --test-set data/evaluation/test.jsonl --batch-size 32
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

RISK_LEVELS = ["none", "low", "medium", "high", "critical"]
CATEGORIES = ["grooming", "bullying", "sexual_content", "isolation", "personal_info", "platform_migration", "threats"]
ASSISTANT_TAG = "<|start_header_id|>assistant<|end_header_id|>"


def load_test_set(path: str) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _extract_prompt(text: str) -> str:
    if ASSISTANT_TAG in text:
        return text[:text.rindex(ASSISTANT_TAG) + len(ASSISTANT_TAG)] + "\n"
    return text


def _parse_prediction(generated: str) -> Optional[dict]:
    try:
        return json.loads(generated.strip())
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", generated, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None


def run_inference_batch(model, tokenizer, prompts: list[str], max_new_tokens: int = 256) -> list[Optional[dict]]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    for output in outputs:
        generated = tokenizer.decode(output[input_len:], skip_special_tokens=True)
        results.append(_parse_prediction(generated))
    return results


def run_inference_batch_debug(model, tokenizer, prompts: list[str], max_new_tokens: int = 256) -> list[str]:
    """Return raw decoded strings for debugging."""
    inputs = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=2048, padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return [tokenizer.decode(o[input_len:], skip_special_tokens=True) for o in outputs]


def compute_metrics(
    y_true_levels: list[str],
    y_pred_levels: list[str],
    y_true_cats: list[list[str]],
    y_pred_cats: list[list[str]],
) -> dict:
    results = {}

    results["risk_level"] = {
        "classification_report": classification_report(
            y_true_levels, y_pred_levels, labels=RISK_LEVELS, output_dict=True, zero_division=0
        ),
        "macro_f1": f1_score(y_true_levels, y_pred_levels, labels=RISK_LEVELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true_levels, y_pred_levels, labels=RISK_LEVELS, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true_levels, y_pred_levels, labels=RISK_LEVELS).tolist(),
    }

    y_true_binary = ["benign" if l == "none" else "risk" for l in y_true_levels]
    y_pred_binary = ["benign" if l == "none" else "risk" for l in y_pred_levels]
    total_benign = y_true_binary.count("benign")
    total_risk = y_true_binary.count("risk")
    fp = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == "benign" and p == "risk")
    fn = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == "risk" and p == "benign")

    results["binary"] = {
        "false_positive_rate": round(fp / total_benign, 4) if total_benign else 0,
        "false_negative_rate": round(fn / total_risk, 4) if total_risk else 0,
        "false_positives": fp,
        "false_negatives": fn,
    }

    cat_results = {}
    for cat in CATEGORIES:
        t = [1 if cat in cats else 0 for cats in y_true_cats]
        p = [1 if cat in cats else 0 for cats in y_pred_cats]
        if sum(t) == 0:
            continue
        cat_results[cat] = {
            "f1": round(f1_score(t, p, zero_division=0), 4),
            "precision": round(precision_score(t, p, zero_division=0), 4),
            "recall": round(recall_score(t, p, zero_division=0), 4),
            "support": sum(t),
        }
    results["per_category"] = cat_results

    return results


def print_report(results: dict):
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    print(f"\nRisk Level Classification:")
    print(f"  Macro F1:    {results['risk_level']['macro_f1']:.4f}")
    print(f"  Weighted F1: {results['risk_level']['weighted_f1']:.4f}")

    print(f"\nBinary (Benign vs Risk):")
    b = results["binary"]
    print(f"  False Positive Rate: {b['false_positive_rate']:.2%}  ({b['false_positives']} FPs)")
    print(f"  False Negative Rate: {b['false_negative_rate']:.2%}  ({b['false_negatives']} FNs)")

    print(f"\nPer-Category F1:")
    for cat, m in results["per_category"].items():
        print(f"  {cat:25s}: F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  (n={m['support']})")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SafeCircle model checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--test-set", required=True, help="Path to test JSONL")
    parser.add_argument("--output", default="evaluation/reports", help="Output directory for results")
    parser.add_argument("--max-samples", type=int, help="Limit evaluation to N samples")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size (default: 16)")
    parser.add_argument("--debug", type=int, default=0, metavar="N", help="Print raw output for first N examples and exit")
    args = parser.parse_args()

    from training.model.loader import load_for_inference

    if not Path(args.test_set).exists():
        print(f"Error: test set not found: {args.test_set}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, tokenizer = load_for_inference(args.checkpoint)
    tokenizer.padding_side = "left"  # required for batched generation

    print(f"Loading test set: {args.test_set}")
    examples = load_test_set(args.test_set)
    if args.max_samples:
        examples = examples[:args.max_samples]

    if args.debug:
        sample = examples[:args.debug]
        prompts = [_extract_prompt(ex["text"]) for ex in sample]
        print(f"\n--- PROMPT (example 0) ---\n{prompts[0]}\n--- END PROMPT ---\n")
        raw_outputs = run_inference_batch_debug(model, tokenizer, prompts)
        for i, raw in enumerate(raw_outputs):
            print(f"\n--- RAW OUTPUT {i} ---\n{raw}\n--- END ---")
        sys.exit(0)

    y_true_levels, y_pred_levels = [], []
    y_true_cats, y_pred_cats = [], []
    failures = 0

    batches = [examples[i:i + args.batch_size] for i in range(0, len(examples), args.batch_size)]

    with tqdm(total=len(examples), unit="ex", desc="Evaluating") as pbar:
        for batch in batches:
            prompts = [_extract_prompt(ex["text"]) for ex in batch]

            labels = []
            for ex in batch:
                label = ex["label"]
                if isinstance(label, str):
                    label = json.loads(label)
                labels.append(label)

            predictions = run_inference_batch(model, tokenizer, prompts)

            for label, prediction in zip(labels, predictions):
                true_level = label.get("risk_level", "none")
                true_cats = [c for c in label.get("categories", []) if c != "benign"]

                if prediction is None:
                    failures += 1
                    pred_level = "none"
                    pred_cats = []
                else:
                    pred_level = prediction.get("risk_level", "none")
                    if pred_level not in RISK_LEVELS:
                        pred_level = "none"
                    pred_cats = [c for c in prediction.get("categories", []) if c in CATEGORIES]

                y_true_levels.append(true_level)
                y_pred_levels.append(pred_level)
                y_true_cats.append(true_cats)
                y_pred_cats.append(pred_cats)

            pbar.update(len(batch))

    if failures:
        print(f"Warning: {failures}/{len(examples)} inference failures (defaulted to 'none')")

    results = compute_metrics(y_true_levels, y_pred_levels, y_true_cats, y_pred_cats)
    print_report(results)

    output_dir = Path(args.output) / "latest"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()
