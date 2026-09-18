#!/usr/bin/env python3
"""Evaluation script for the horizon-mobile LiteRT-LM model.

Runs inference on a test set via ``litert-lm`` CLI and computes per-category
F1, precision, recall, false-positive / false-negative rates — matching the
structure of the main ``evaluate.py`` report.

Usage:
    python -m evaluation.metrics.evaluate_mobile \
        --model models/mobile-standard/horizon-mobile-int8_q8_ekv1280.litertlm \
        --test-set data/processed/eval.jsonl
    python -m evaluation.metrics.evaluate_mobile \
        --model models/mobile-standard/horizon-mobile-int8_q8_ekv1280.litertlm \
        --test-set data/processed/eval.jsonl --max-samples 200
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from training.model.mobile import RISK_CATEGORIES, RISK_LEVELS, SYSTEM_PROMPT

# Categories that count as "risk" (everything except benign)
_RISK_CATS = [c for c in RISK_CATEGORIES if c != "benign"]


# ---------------------------------------------------------------------------
# LiteRT-LM inference via subprocess
# ---------------------------------------------------------------------------


def _build_prompt(conversation_text: str) -> str:
    """Wrap raw conversation text in the Gemma chat template used at training time.

    Includes the system prompt so the model knows to respond with JSON.
    """
    return (
        f"<start_of_turn>user\n"
        f"{SYSTEM_PROMPT}\n\n"
        f"Analyze this conversation:\n{conversation_text}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def _run_litert(model_path: str, prompt: str, timeout: int = 120) -> str | None:
    """Run a single litert-lm inference and return the raw stdout text."""
    cmd = ["uvx", "litert-lm", "run", model_path, "--prompt", prompt]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        return None


def _parse_prediction(raw: str | None) -> dict | None:
    """Extract the first JSON object from the model's raw stdout."""
    if raw is None:
        return None
    # Take only the first JSON block (model may echo or repeat)
    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    # Fallback: try the whole string
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def load_test_set(path: str) -> list[dict]:
    """Load a JSONL test set (messages-format or text-format)."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _extract_conversation_from_messages(example: dict) -> tuple[str, dict]:
    """Extract conversation text and ground-truth label from a messages-format example.

    Returns (conversation_text, label_dict).
    """
    messages = example.get("messages", [])
    label = None
    conversation_parts: list[str] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            # The user turn usually starts with "Analyze this conversation:\n"
            text = content
            if text.startswith("Analyze this conversation:"):
                text = text[len("Analyze this conversation:") :].strip()
            if text:
                conversation_parts.append(text)
        elif role == "assistant":
            try:
                label = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                label = {"risk_level": "none", "categories": ["benign"]}

    if label is None:
        # Fallback: check top-level label field
        lab = example.get("label", {})
        if isinstance(lab, str):
            try:
                label = json.loads(lab)
            except json.JSONDecodeError:
                label = {"risk_level": "none", "categories": ["benign"]}
        else:
            label = lab if lab else {"risk_level": "none", "categories": ["benign"]}

    return "\n".join(conversation_parts), label


def _extract_conversation_from_text(example: dict) -> tuple[str, dict]:
    """Extract conversation text and label from a text-format example (Gemma tags)."""
    text = example.get("text", "")
    # Strip Gemma chat tags to get the conversation body
    user_tag = "<start_of_turn>user"
    model_tag = "<start_of_turn>model"

    conversation = text
    if user_tag in text:
        conversation = text[text.index(user_tag) + len(user_tag) :]
    if model_tag in conversation:
        conversation = conversation[: conversation.index(model_tag)]
    conversation = conversation.replace("<end_of_turn>", "").strip()
    if conversation.startswith("Analyze this conversation:"):
        conversation = conversation[len("Analyze this conversation:") :].strip()

    label = example.get("label", {})
    if isinstance(label, str):
        try:
            label = json.loads(label)
        except json.JSONDecodeError:
            label = {"risk_level": "none", "categories": ["benign"]}

    return conversation, label


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true_levels: list[str],
    y_pred_levels: list[str],
    y_true_cats: list[list[str]],
    y_pred_cats: list[list[str]],
) -> dict:
    """Compute multi-level and binary metrics, mirroring evaluate.py."""
    results: dict = {}

    # Risk-level classification
    results["risk_level"] = {
        "classification_report": classification_report(
            y_true_levels,
            y_pred_levels,
            labels=RISK_LEVELS,
            output_dict=True,
            zero_division=0,
        ),
        "macro_f1": round(
            f1_score(
                y_true_levels,
                y_pred_levels,
                labels=RISK_LEVELS,
                average="macro",
                zero_division=0,
            ),
            4,
        ),
        "weighted_f1": round(
            f1_score(
                y_true_levels,
                y_pred_levels,
                labels=RISK_LEVELS,
                average="weighted",
                zero_division=0,
            ),
            4,
        ),
        "confusion_matrix": confusion_matrix(
            y_true_levels, y_pred_levels, labels=RISK_LEVELS
        ).tolist(),
    }

    # Binary (benign vs risk)
    y_true_bin = ["benign" if lv == "none" else "risk" for lv in y_true_levels]
    y_pred_bin = ["benign" if lv == "none" else "risk" for lv in y_pred_levels]
    total_benign = y_true_bin.count("benign")
    total_risk = y_true_bin.count("risk")
    fp = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == "benign" and p == "risk")
    fn = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == "risk" and p == "benign")
    tp = total_risk - fn

    results["binary"] = {
        "false_positive_rate": round(fp / total_benign, 4) if total_benign else 0,
        "false_negative_rate": round(fn / total_risk, 4) if total_risk else 0,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0,
        "recall": round(tp / total_risk, 4) if total_risk else 0.0,
        "f1": round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) > 0 else 0.0,
        "false_positives": fp,
        "false_negatives": fn,
    }

    # Per-category
    cat_results: dict = {}
    for cat in _RISK_CATS:
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


def print_report(results: dict) -> None:
    print("\n" + "=" * 60)
    print("MOBILE MODEL EVALUATION REPORT  (LiteRT-LM)")
    print("=" * 60)

    print("\nRisk Level Classification:")
    print(f"  Macro F1:    {results['risk_level']['macro_f1']:.4f}")
    print(f"  Weighted F1: {results['risk_level']['weighted_f1']:.4f}")

    print("\nBinary (Benign vs Risk):")
    b = results["binary"]
    print(f"  Recall (TPR):  {b['recall']:.2%}")
    print(f"  Precision:     {b['precision']:.2%}")
    print(f"  F1:            {b['f1']:.4f}")
    print(
        f"  FPR:           {b['false_positive_rate']:.2%}  ({b['false_positives']} FPs)"
    )
    print(
        f"  FNR:           {b['false_negative_rate']:.2%}  ({b['false_negatives']} FNs)"
    )

    if results.get("per_category"):
        print("\nPer-Category F1:")
        for cat, m in results["per_category"].items():
            print(
                f"  {cat:25s}: F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  (n={m['support']})"
            )

    # Confusion matrix
    cm = results["risk_level"]["confusion_matrix"]
    col_w = 10
    header = f"{'':12s}" + "".join(f"{lv:>{col_w}}" for lv in RISK_LEVELS)
    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(header)
    for i, row in enumerate(cm):
        print(f"  {RISK_LEVELS[i]:10s}" + "".join(f"{v:>{col_w}}" for v in row))

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate horizon-mobile LiteRT-LM model"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to .litertlm model file",
    )
    parser.add_argument("--test-set", required=True, help="Path to eval JSONL")
    parser.add_argument(
        "--output",
        default="evaluation/reports/mobile",
        help="Output directory",
    )
    parser.add_argument("--max-samples", type=int, help="Limit to N examples")
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-inference timeout in seconds (default: 120)",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: model not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Model:    {model_path}")
    print(f"Test set: {args.test_set}")

    examples = load_test_set(args.test_set)
    if args.max_samples:
        examples = examples[: args.max_samples]
    print(f"Examples: {len(examples)}")

    y_true_levels: list[str] = []
    y_pred_levels: list[str] = []
    y_true_cats: list[list[str]] = []
    y_pred_cats: list[list[str]] = []
    failures = 0
    latencies: list[float] = []
    failure_log: list[dict] = []

    for ex in tqdm(examples, desc="Evaluating", unit="ex"):
        # Extract conversation and label
        if "messages" in ex:
            conversation, label = _extract_conversation_from_messages(ex)
        else:
            conversation, label = _extract_conversation_from_text(ex)

        prompt = _build_prompt(conversation)

        # Run inference
        t0 = time.perf_counter()
        raw_output = _run_litert(str(model_path), prompt, timeout=args.timeout)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        prediction = _parse_prediction(raw_output)

        # Ground truth
        true_level = label.get("risk_level", "none")
        true_cats = [c for c in label.get("categories", []) if c in _RISK_CATS]

        # Predicted
        if prediction is None:
            failures += 1
            pred_level = "none"
            pred_cats = []
            failure_log.append({"raw_output": raw_output, "elapsed_s": round(elapsed, 2)})
        else:
            pred_level = prediction.get("risk_level", "none")
            if pred_level not in RISK_LEVELS:
                pred_level = "none"
            pred_cats = [c for c in prediction.get("categories", []) if c in _RISK_CATS]

        y_true_levels.append(true_level)
        y_pred_levels.append(pred_level)
        y_true_cats.append(true_cats)
        y_pred_cats.append(pred_cats)

    if failures:
        print(
            f"\nWarning: {failures}/{len(examples)} inference failures (defaulted to 'none')"
        )
        failures_path = Path(args.output) / "failures.jsonl"
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        with open(failures_path, "w") as f:
            for entry in failure_log:
                f.write(json.dumps(entry) + "\n")
        print(f"Failure details written to {failures_path}")

    # Compute and display
    results = compute_metrics(y_true_levels, y_pred_levels, y_true_cats, y_pred_cats)

    # Attach latency stats
    if latencies:
        s = sorted(latencies)
        results["latency_s"] = {
            "p50": round(s[len(s) // 2], 3),
            "p95": round(s[int(len(s) * 0.95)], 3) if len(s) >= 20 else round(s[-1], 3),
            "mean": round(sum(s) / len(s), 3),
        }
        results["inference_failures"] = failures

    print_report(results)

    if latencies:
        lat = results["latency_s"]
        print(f"\nLatency  P50={lat['p50']}s  P95={lat['p95']}s  mean={lat['mean']}s")

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()
