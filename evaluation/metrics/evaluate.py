#!/usr/bin/env python3
"""Evaluation script for SafeCircle risk detection model.

Runs inference on a test set and computes per-category F1, precision, recall,
confusion matrix, false positive/negative rates.

Usage:
    python -m evaluation.metrics.evaluate --checkpoint experiments/run-foo/final --test-set data/evaluation/test.jsonl
"""

import argparse
import json
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

RISK_LEVELS = ["none", "low", "medium", "high", "critical"]
CATEGORIES = ["grooming", "bullying", "sexual_content", "isolation", "personal_info", "platform_migration", "threats"]


def load_test_set(path: str) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def run_inference(model, tokenizer, text: str, max_new_tokens: int = 256) -> Optional[dict]:
    """Run model inference and parse JSON output."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    try:
        return json.loads(generated.strip())
    except json.JSONDecodeError:
        # Try extracting JSON from response
        import re
        match = re.search(r"\{.*\}", generated, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None


def compute_metrics(
    y_true_levels: list[str],
    y_pred_levels: list[str],
    y_true_cats: list[list[str]],
    y_pred_cats: list[list[str]],
) -> dict:
    """Compute all evaluation metrics."""
    results = {}

    # Risk level classification
    results["risk_level"] = {
        "classification_report": classification_report(
            y_true_levels, y_pred_levels, labels=RISK_LEVELS, output_dict=True, zero_division=0
        ),
        "macro_f1": f1_score(y_true_levels, y_pred_levels, labels=RISK_LEVELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true_levels, y_pred_levels, labels=RISK_LEVELS, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true_levels, y_pred_levels, labels=RISK_LEVELS).tolist(),
    }

    # False positive / negative on benign vs risk
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

    # Per-category multi-label metrics
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
    args = parser.parse_args()

    from training.model.loader import load_for_inference
    from training.scripts.preprocess import conversation_to_text, SYSTEM_PROMPT

    if not Path(args.test_set).exists():
        print(f"Error: test set not found: {args.test_set}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, tokenizer = load_for_inference(args.checkpoint)

    print(f"Loading test set: {args.test_set}")
    examples = load_test_set(args.test_set)
    if args.max_samples:
        examples = examples[:args.max_samples]

    y_true_levels, y_pred_levels = [], []
    y_true_cats, y_pred_cats = [], []
    failures = 0

    for i, ex in enumerate(examples):
        if i % 50 == 0:
            print(f"  {i}/{len(examples)}...")

        conversation_text = conversation_to_text(ex["messages"])
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{SYSTEM_PROMPT}\n"
            f"<|eot_id|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"Analyze this conversation:\n{conversation_text}\n"
            f"<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )

        true_level = ex["label"]["risk_level"]
        true_cats = [c for c in ex["label"].get("categories", []) if c != "benign"]

        prediction = run_inference(model, tokenizer, prompt)
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
