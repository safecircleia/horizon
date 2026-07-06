#!/usr/bin/env python3
"""Benchmark runner for SafeCircle risk detection model.

Runs the full evaluation pipeline against the held-out benchmark set and checks
results against defined accuracy targets. Exits 1 if any target is missed.

Usage:
    python -m evaluation.benchmark --checkpoint experiments/run-foo/final
    python -m evaluation.benchmark --checkpoint experiments/run-foo/final --benchmark-set data/evaluation/benchmark.jsonl
    python -m evaluation.benchmark --checkpoint experiments/run-foo/final --baseline evaluation/reports/baseline.json
"""

import argparse
import json
import sys
from pathlib import Path

from evaluation.metrics.evaluate import (
    load_test_set,
    run_inference_batch,
    compute_metrics,
    RISK_LEVELS,
    CATEGORIES,
)

# Targets from issue #11
TARGETS = {
    "recall": 0.97,       # true positive rate ≥ 97%
    "fpr": 0.03,          # false positive rate ≤ 3%
    "precision": 0.95,    # precision ≥ 95%
    "f1": 0.96,           # F1 ≥ 0.96
}

BENCHMARK_SET = Path("data/evaluation/benchmark.jsonl")
BASELINE_PATH = Path("evaluation/reports/baseline.json")
REGRESSION_THRESHOLD = 0.005  # block if accuracy drops > 0.5%


def _summary(results: dict) -> dict:
    """Flatten to a simple dict of headline metrics."""
    rl = results["risk_level"]
    b = results["binary"]
    recall = 1.0 - b["false_negative_rate"]
    return {
        "recall": round(recall, 4),
        "fpr": b["false_positive_rate"],
        "precision": round(rl["classification_report"].get("risk", {}).get("precision", 0), 4),
        "f1": round(rl["weighted_f1"], 4),
        "macro_f1": round(rl["macro_f1"], 4),
    }


def check_targets(summary: dict) -> list[str]:
    failures = []
    if summary["recall"] < TARGETS["recall"]:
        failures.append(f"recall {summary['recall']:.4f} < target {TARGETS['recall']}")
    if summary["fpr"] > TARGETS["fpr"]:
        failures.append(f"FPR {summary['fpr']:.4f} > target {TARGETS['fpr']}")
    if summary["precision"] < TARGETS["precision"]:
        failures.append(f"precision {summary['precision']:.4f} < target {TARGETS['precision']}")
    if summary["f1"] < TARGETS["f1"]:
        failures.append(f"F1 {summary['f1']:.4f} < target {TARGETS['f1']}")
    return failures


def check_regression(summary: dict, baseline: dict) -> list[str]:
    regressions = []
    for metric in ("recall", "precision", "f1", "macro_f1"):
        if metric not in baseline:
            continue
        drop = baseline[metric] - summary[metric]
        if drop > REGRESSION_THRESHOLD:
            regressions.append(
                f"{metric} dropped {drop:.4f} (was {baseline[metric]:.4f}, now {summary[metric]:.4f})"
            )
    return regressions


def print_benchmark_report(summary: dict, target_failures: list[str], regressions: list[str]) -> None:
    print("\n" + "=" * 60)
    print("BENCHMARK REPORT")
    print("=" * 60)
    print(f"\n  Recall (TPR): {summary['recall']:.4f}  [target ≥ {TARGETS['recall']}]  {'✓' if summary['recall'] >= TARGETS['recall'] else '✗'}")
    print(f"  False Positive Rate: {summary['fpr']:.4f}  [target ≤ {TARGETS['fpr']}]  {'✓' if summary['fpr'] <= TARGETS['fpr'] else '✗'}")
    print(f"  Precision:    {summary['precision']:.4f}  [target ≥ {TARGETS['precision']}]  {'✓' if summary['precision'] >= TARGETS['precision'] else '✗'}")
    print(f"  Weighted F1:  {summary['f1']:.4f}  [target ≥ {TARGETS['f1']}]  {'✓' if summary['f1'] >= TARGETS['f1'] else '✗'}")
    print(f"  Macro F1:     {summary['macro_f1']:.4f}")

    if regressions:
        print(f"\n  REGRESSIONS vs baseline:")
        for r in regressions:
            print(f"    ✗ {r}")
    else:
        print(f"\n  No regressions vs baseline.")

    if target_failures:
        print(f"\n  TARGET FAILURES:")
        for f in target_failures:
            print(f"    ✗ {f}")
        print("\n  BENCHMARK: FAILED")
    else:
        print("\n  BENCHMARK: PASSED")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SafeCircle accuracy benchmark")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--benchmark-set", default=str(BENCHMARK_SET), help="Path to benchmark JSONL")
    parser.add_argument("--baseline", default=str(BASELINE_PATH), help="Path to baseline JSON for regression check")
    parser.add_argument("--output", default="evaluation/reports/benchmark", help="Output directory")
    parser.add_argument("--max-samples", type=int, help="Limit to N samples")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-baseline", action="store_true", help="Save current results as the new baseline")
    args = parser.parse_args()

    bench_path = Path(args.benchmark_set)
    if not bench_path.exists():
        print(f"Error: benchmark set not found: {bench_path}", file=sys.stderr)
        print("Generate it with: python -m data.scripts.create_benchmark_split", file=sys.stderr)
        sys.exit(1)

    from training.model.loader import load_for_inference

    print(f"Loading checkpoint: {args.checkpoint}")
    model, tokenizer = load_for_inference(args.checkpoint)
    tokenizer.padding_side = "left"

    try:
        from peft import PeftConfig
        base_model = PeftConfig.from_pretrained(args.checkpoint).base_model_name_or_path
    except Exception:
        base_model = ""
    print(f"  Base model: {base_model or '(unknown)'}")

    print(f"Loading benchmark set: {bench_path}")
    examples = load_test_set(str(bench_path))
    if args.max_samples:
        examples = examples[:args.max_samples]
    print(f"  {len(examples)} examples")

    for ex in examples:
        if "messages" in ex:
            prompt_messages = [m for m in ex["messages"] if m["role"] != "assistant"]
            ex["text"] = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            if "label" not in ex:
                for msg in reversed(ex["messages"]):
                    if msg["role"] == "assistant":
                        try:
                            ex["label"] = json.loads(msg["content"])
                        except json.JSONDecodeError:
                            ex["label"] = {"risk_level": "none", "categories": []}
                        break

    y_true_levels, y_pred_levels = [], []
    y_true_cats, y_pred_cats = [], []
    failures = 0
    batches = [examples[i:i + args.batch_size] for i in range(0, len(examples), args.batch_size)]
    total = len(examples)
    done = 0
    print_every = max(1, len(batches) // 10)  # ~10 progress lines total

    for batch_idx, batch in enumerate(batches):
        prompts = [ex["text"] for ex in batch]
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
                pred_level, pred_cats = "none", []
            else:
                pred_level = prediction.get("risk_level", "none")
                if pred_level not in RISK_LEVELS:
                    pred_level = "none"
                pred_cats = [c for c in prediction.get("categories", []) if c in CATEGORIES]

            y_true_levels.append(true_level)
            y_pred_levels.append(pred_level)
            y_true_cats.append(true_cats)
            y_pred_cats.append(pred_cats)

        done += len(batch)
        if (batch_idx + 1) % print_every == 0 or done == total:
            pct = done / total * 100
            print(f"  [{done:4d}/{total}  {pct:5.1f}%]  failures so far: {failures}")

    if failures:
        print(f"Warning: {failures}/{len(examples)} inference failures")

    results = compute_metrics(y_true_levels, y_pred_levels, y_true_cats, y_pred_cats)
    summary = _summary(results)

    baseline = {}
    baseline_path = Path(args.baseline)
    if baseline_path.exists():
        with open(baseline_path) as f:
            data = json.load(f)
            baseline = data.get("summary", data)

    target_failures = check_targets(summary)
    regressions = check_regression(summary, baseline)
    print_benchmark_report(summary, target_failures, regressions)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"summary": summary, "full": results, "checkpoint": args.checkpoint}
    with open(output_dir / "results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {output_dir}/results.json")

    if args.save_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_path, "w") as f:
            json.dump({"summary": summary, "checkpoint": args.checkpoint}, f, indent=2)
        print(f"Baseline saved to {baseline_path}")

    if target_failures or regressions:
        sys.exit(1)


if __name__ == "__main__":
    main()
