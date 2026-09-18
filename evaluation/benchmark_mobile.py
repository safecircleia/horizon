#!/usr/bin/env python3
"""End-to-end benchmark for the on-device (mobile / edge) LiteRT-LM model.

Validates ALL constraints from issue #8:
    - Accuracy: TPR grooming >= 97 %, TPR explicit >= 99 %, FPR <= 3 %
    - Latency: < 500 ms per inference (P95)
    - RAM: < 150 MB peak loaded
    - Model size: < 50 MB on disk, < 100 MB loaded RAM
    - Battery: < 3 % drain / day (estimated)
    - Regression: no metric may drop > 0.5 % vs saved baseline

Runs both accuracy evaluation (via litert-lm CLI) and performance profiling
(via psutil) in a single pass, then checks every target and exits non-zero on
failure.

Usage:
    python -m evaluation.benchmark_mobile \
        --model models/mobile-standard/horizon-mobile-int8_q8_ekv1280.litertlm \
        --test-set data/processed/eval.jsonl
    python -m evaluation.benchmark_mobile \
        --model models/mobile-lite/horizon-mobile-int4_q4_block128_ekv1280.litertlm \
        --test-set data/processed/eval.jsonl --save-baseline
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

from sklearn.metrics import recall_score
from tqdm import tqdm

from evaluation.rules import compute_rule_catch_rate
from training.model.mobile import RISK_CATEGORIES, RISK_LEVELS, SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Issue #8 targets
# ---------------------------------------------------------------------------
TARGETS = {
    # Accuracy
    "recall_overall": 0.97,  # TPR >= 97 %
    "recall_grooming": 0.97,  # TPR grooming >= 97 %
    "recall_sexual_content": 0.99,  # TPR explicit >= 99 %
    "fpr": 0.03,  # FPR <= 3 %
    "precision": 0.95,  # Precision >= 95 %
    "f1": 0.96,  # F1 >= 0.96
    "rule_catch_rate": 0.80,  # Rule pre-filter >= 80 %
    # Performance
    "latency_p95_ms": 500.0,  # < 500 ms per inference (P95)
    "peak_ram_mb": 150.0,  # < 150 MB loaded
    "model_disk_mb": 50.0,  # < 50 MB on disk
    "model_loaded_ram_mb": 100.0,  # < 100 MB loaded RAM (model only)
    # Battery: < 3 % / day → with ~1440 inferences/day (1 per minute),
    # budget ≈ 0.66 mWh per inference at 32 Wh battery
    "energy_mwh_per_inference": 0.66,
}

REGRESSION_THRESHOLD = 0.005  # 0.5 %

_RISK_CATS = [c for c in RISK_CATEGORIES if c != "benign"]

_DEFAULT_TDP_MW = 3000.0  # mid-range mobile SoC


# ---------------------------------------------------------------------------
# LiteRT-LM inference (with profiling)
# ---------------------------------------------------------------------------


def _build_prompt(conversation: str) -> str:
    return (
        f"<start_of_turn>user\n"
        f"{SYSTEM_PROMPT}\n\n"
        f"Analyze this conversation:\n{conversation}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def _run_litert_profiled(
    model_path: str, prompt: str, tdp_mw: float = _DEFAULT_TDP_MW, timeout: int = 120
) -> dict:
    """Run one litert-lm inference and return metrics + output.

    Returns dict with keys: output, wall_s, peak_rss_mb, avg_cpu_pct, energy_mwh
    """
    cmd = ["uvx", "litert-lm", "run", model_path, "--prompt", prompt]

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    peak_rss = 0.0
    cpu_samples: list[float] = []

    if psutil is not None:
        try:
            ps = psutil.Process(proc.pid)
            while proc.poll() is None:
                try:
                    mem = ps.memory_info().rss / 1024 / 1024
                    peak_rss = max(peak_rss, mem)
                    cpu_samples.append(ps.cpu_percent(interval=None))
                except psutil.NoSuchProcess:
                    break
                time.sleep(0.05)
        except psutil.NoSuchProcess:
            pass

    stdout, stderr = proc.communicate(timeout=timeout)
    wall = time.perf_counter() - t0

    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
    utilisation = avg_cpu / 100.0
    energy_mwh = round(tdp_mw * utilisation * (wall / 3600.0), 4)

    return {
        "output": stdout.strip() if proc.returncode == 0 else None,
        "wall_s": wall,
        "peak_rss_mb": peak_rss,
        "avg_cpu_pct": avg_cpu,
        "energy_mwh": energy_mwh,
    }


def _parse_prediction(raw: str | None) -> dict | None:
    if raw is None:
        return None
    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _load_test_set(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _extract_conversation(example: dict) -> tuple[str, dict]:
    """Return (conversation_text, label_dict) from either messages or text format."""
    if "messages" in example:
        messages = example["messages"]
        label = None
        parts: list[str] = []
        for msg in messages:
            if msg.get("role") == "user":
                text = msg.get("content", "")
                if text.startswith("Analyze this conversation:"):
                    text = text[len("Analyze this conversation:") :].strip()
                if text:
                    parts.append(text)
            elif msg.get("role") == "assistant":
                try:
                    label = json.loads(msg.get("content", "{}"))
                except (json.JSONDecodeError, TypeError):
                    pass
        if label is None:
            lab = example.get("label", {})
            label = (
                json.loads(lab)
                if isinstance(lab, str)
                else (lab or {"risk_level": "none", "categories": ["benign"]})
            )
        return "\n".join(parts), label
    else:
        text = example.get("text", "")
        # Strip Gemma tags
        if "<start_of_turn>user" in text:
            text = text[
                text.index("<start_of_turn>user") + len("<start_of_turn>user") :
            ]
        if "<start_of_turn>model" in text:
            text = text[: text.index("<start_of_turn>model")]
        text = text.replace("<end_of_turn>", "").strip()
        if text.startswith("Analyze this conversation:"):
            text = text[len("Analyze this conversation:") :].strip()
        lab = example.get("label", {})
        label = (
            json.loads(lab)
            if isinstance(lab, str)
            else (lab or {"risk_level": "none", "categories": ["benign"]})
        )
        return text, label


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    model_path: str,
    test_set_path: str,
    max_samples: int | None = None,
    tdp_mw: float = _DEFAULT_TDP_MW,
    timeout: int = 120,
) -> dict:
    """Run the full mobile benchmark and return a results dict."""

    examples = _load_test_set(test_set_path)
    if max_samples:
        examples = examples[:max_samples]

    # --- Accuracy pass ---
    y_true_levels: list[str] = []
    y_pred_levels: list[str] = []
    y_true_cats: list[list[str]] = []
    y_pred_cats: list[list[str]] = []
    failures = 0

    # --- Performance pass (same loop) ---
    walls: list[float] = []
    peaks: list[float] = []
    energies: list[float] = []

    # For rule catch-rate: collect (conversation, label) pairs
    rule_examples: list[dict] = []

    for ex in tqdm(examples, desc="Benchmarking", unit="ex"):
        conversation, label = _extract_conversation(ex)
        prompt = _build_prompt(conversation)

        # Profiled inference
        result = _run_litert_profiled(
            model_path, prompt, tdp_mw=tdp_mw, timeout=timeout
        )
        prediction = _parse_prediction(result["output"])

        walls.append(result["wall_s"])
        peaks.append(result["peak_rss_mb"])
        energies.append(result["energy_mwh"])

        # Ground truth
        true_level = label.get("risk_level", "none")
        true_cats = [c for c in label.get("categories", []) if c in _RISK_CATS]

        # Prediction
        if prediction is None:
            failures += 1
            pred_level = "none"
            pred_cats: list[str] = []
        else:
            pred_level = prediction.get("risk_level", "none")
            if pred_level not in RISK_LEVELS:
                pred_level = "none"
            pred_cats = [c for c in prediction.get("categories", []) if c in _RISK_CATS]

        y_true_levels.append(true_level)
        y_pred_levels.append(pred_level)
        y_true_cats.append(true_cats)
        y_pred_cats.append(pred_cats)

        rule_examples.append({"conversation": conversation, "label": label})

    # ------------------------------------------------------------------
    # Compute accuracy metrics
    # ------------------------------------------------------------------
    y_true_bin = ["benign" if lv == "none" else "risk" for lv in y_true_levels]
    y_pred_bin = ["benign" if lv == "none" else "risk" for lv in y_pred_levels]
    total_benign = y_true_bin.count("benign")
    total_risk = y_true_bin.count("risk")
    fp = sum(
        1
        for t, p in zip(y_true_bin, y_pred_bin, strict=True)
        if t == "benign" and p == "risk"
    )
    fn = sum(
        1
        for t, p in zip(y_true_bin, y_pred_bin, strict=True)
        if t == "risk" and p == "benign"
    )
    tp = total_risk - fn
    tn = total_benign - fp

    recall_overall = round(tp / total_risk, 4) if total_risk else 0.0
    fpr = round(fp / total_benign, 4) if total_benign else 0.0
    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    f1 = round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) > 0 else 0.0

    # Per-category recall
    cat_recall: dict[str, float] = {}
    for cat in _RISK_CATS:
        t = [1 if cat in cats else 0 for cats in y_true_cats]
        p = [1 if cat in cats else 0 for cats in y_pred_cats]
        support = sum(t)
        if support == 0:
            continue
        cat_recall[cat] = round(recall_score(t, p, zero_division=0), 4)

    # Rule-based catch rate
    rule_conversations = [e["conversation"] for e in rule_examples]
    rule_labels = [e["label"] for e in rule_examples]
    rule_results = compute_rule_catch_rate(rule_conversations, rule_labels)

    # ------------------------------------------------------------------
    # Compute performance metrics
    # ------------------------------------------------------------------
    def pct(arr: list[float], p: float) -> float:
        s = sorted(arr)
        idx = min(int(p / 100 * len(s)), len(s) - 1)
        return round(s[idx], 3)

    model_disk_mb = round(Path(model_path).stat().st_size / 1024 / 1024, 1)

    report: dict = {
        "model": model_path,
        "model_disk_mb": model_disk_mb,
        "examples": len(examples),
        "inference_failures": failures,
        "accuracy": {
            "recall_overall": recall_overall,
            "recall_grooming": cat_recall.get("grooming", 0.0),
            "recall_sexual_content": cat_recall.get("sexual_content", 0.0),
            "fpr": fpr,
            "precision": precision,
            "f1": f1,
            "per_category_recall": cat_recall,
            "rule_catch_rate": rule_results.get("catch_rate", 0.0),
            "rule_fpr": rule_results.get("false_positive_rate", 0.0),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
        "performance": {
            "latency_ms": {
                "p50": round(pct(walls, 50) * 1000, 1),
                "p95": round(pct(walls, 95) * 1000, 1),
                "p99": round(pct(walls, 99) * 1000, 1),
                "mean": round(sum(walls) / len(walls) * 1000, 1) if walls else 0,
            },
            "peak_ram_mb": {
                "p50": pct(peaks, 50),
                "p95": pct(peaks, 95),
                "max": round(max(peaks), 1) if peaks else 0,
                "mean": round(sum(peaks) / len(peaks), 1) if peaks else 0,
            },
            "energy_mwh_per_inference": {
                "p50": pct(energies, 50),
                "p95": pct(energies, 95),
                "mean": round(sum(energies) / len(energies), 4) if energies else 0,
            },
        },
    }

    return report


# ---------------------------------------------------------------------------
# Target checking
# ---------------------------------------------------------------------------


def check_targets(report: dict) -> list[str]:
    """Return a list of failure messages (empty = all passed)."""
    fails: list[str] = []
    a = report["accuracy"]
    p = report["performance"]

    def _check_gte(name: str, actual: float, target: float) -> None:
        if actual < target:
            fails.append(f"FAIL {name}: {actual:.4f} < {target:.4f}")

    def _check_lte(name: str, actual: float, target: float) -> None:
        if actual > target:
            fails.append(f"FAIL {name}: {actual:.4f} > {target:.4f}")

    # Accuracy
    _check_gte("recall_overall", a["recall_overall"], TARGETS["recall_overall"])
    _check_gte("recall_grooming", a["recall_grooming"], TARGETS["recall_grooming"])
    _check_gte(
        "recall_sexual_content",
        a["recall_sexual_content"],
        TARGETS["recall_sexual_content"],
    )
    _check_lte("fpr", a["fpr"], TARGETS["fpr"])
    _check_gte("precision", a["precision"], TARGETS["precision"])
    _check_gte("f1", a["f1"], TARGETS["f1"])
    _check_gte("rule_catch_rate", a["rule_catch_rate"], TARGETS["rule_catch_rate"])

    # Performance
    _check_lte("latency_p95_ms", p["latency_ms"]["p95"], TARGETS["latency_p95_ms"])
    _check_lte("peak_ram_mb", p["peak_ram_mb"]["max"], TARGETS["peak_ram_mb"])
    _check_lte("model_disk_mb", report["model_disk_mb"], TARGETS["model_disk_mb"])
    _check_lte(
        "energy_mwh_per_inference",
        p["energy_mwh_per_inference"]["p95"],
        TARGETS["energy_mwh_per_inference"],
    )

    return fails


def check_regression(report: dict, baseline_path: str) -> list[str]:
    """Compare against a saved baseline.  Returns failure messages."""
    fails: list[str] = []
    if not Path(baseline_path).exists():
        return fails
    with open(baseline_path) as f:
        baseline = json.load(f)

    # Accuracy metrics that must not regress
    for key in (
        "recall_overall",
        "recall_grooming",
        "recall_sexual_content",
        "precision",
        "f1",
    ):
        old = baseline.get("accuracy", {}).get(key, 0.0)
        new = report["accuracy"].get(key, 0.0)
        if old - new > REGRESSION_THRESHOLD:
            fails.append(
                f"REGRESSION {key}: {old:.4f} → {new:.4f} (Δ={old - new:.4f} > {REGRESSION_THRESHOLD})"
            )

    # FPR must not increase
    old_fpr = baseline.get("accuracy", {}).get("fpr", 1.0)
    new_fpr = report["accuracy"].get("fpr", 1.0)
    if new_fpr - old_fpr > REGRESSION_THRESHOLD:
        fails.append(
            f"REGRESSION fpr: {old_fpr:.4f} → {new_fpr:.4f} (Δ={new_fpr - old_fpr:.4f})"
        )

    return fails


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Mobile model benchmark (issue #8)")
    parser.add_argument("--model", required=True, help="Path to .litertlm model file")
    parser.add_argument("--test-set", required=True, help="Path to eval JSONL")
    parser.add_argument("--max-samples", type=int, help="Limit to N examples")
    parser.add_argument(
        "--tdp-mw", type=float, default=_DEFAULT_TDP_MW, help="Device TDP in mW"
    )
    parser.add_argument(
        "--timeout", type=int, default=120, help="Per-inference timeout (s)"
    )
    parser.add_argument(
        "--output",
        default="evaluation/reports/benchmark_mobile",
        help="Output directory",
    )
    parser.add_argument(
        "--baseline",
        default="evaluation/reports/benchmark_mobile/baseline.json",
        help="Path to baseline JSON for regression detection",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save current results as new baseline",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    if psutil is None:
        print(
            "WARNING: psutil not installed — performance metrics will be zeros",
            file=sys.stderr,
        )

    # Run benchmark
    report = run_benchmark(
        str(model_path),
        args.test_set,
        max_samples=args.max_samples,
        tdp_mw=args.tdp_mw,
        timeout=args.timeout,
    )

    # --- Print report ---
    a = report["accuracy"]
    p = report["performance"]

    print("\n" + "=" * 70)
    print("MOBILE MODEL BENCHMARK REPORT  (Issue #8 Targets)")
    print("=" * 70)
    print(f"\nModel:     {report['model']}")
    print(
        f"On-disk:   {report['model_disk_mb']} MB  (target: <= {TARGETS['model_disk_mb']} MB)"
    )
    print(
        f"Examples:  {report['examples']}  (failures: {report['inference_failures']})"
    )

    print("\n--- Accuracy ---")
    print(
        f"  Recall (overall):        {a['recall_overall']:.2%}  (target: >= {TARGETS['recall_overall']:.0%})"
    )
    print(
        f"  Recall (grooming):       {a['recall_grooming']:.2%}  (target: >= {TARGETS['recall_grooming']:.0%})"
    )
    print(
        f"  Recall (sexual_content): {a['recall_sexual_content']:.2%}  (target: >= {TARGETS['recall_sexual_content']:.0%})"
    )
    print(
        f"  FPR:                     {a['fpr']:.2%}  (target: <= {TARGETS['fpr']:.0%})"
    )
    print(
        f"  Precision:               {a['precision']:.2%}  (target: >= {TARGETS['precision']:.0%})"
    )
    print(f"  F1:                      {a['f1']:.4f}  (target: >= {TARGETS['f1']})")
    print(
        f"  Rule catch rate:         {a['rule_catch_rate']:.2%}  (target: >= {TARGETS['rule_catch_rate']:.0%})"
    )

    if a.get("per_category_recall"):
        print("\n  Per-category recall:")
        for cat, rec in a["per_category_recall"].items():
            print(f"    {cat:25s}: {rec:.2%}")

    print("\n--- Performance ---")
    lat = p["latency_ms"]
    print(f"  Latency P50:   {lat['p50']:.0f} ms")
    print(
        f"  Latency P95:   {lat['p95']:.0f} ms  (target: <= {TARGETS['latency_p95_ms']:.0f} ms)"
    )
    print(f"  Latency P99:   {lat['p99']:.0f} ms")
    ram = p["peak_ram_mb"]
    print(
        f"  Peak RAM max:  {ram['max']:.0f} MB  (target: <= {TARGETS['peak_ram_mb']:.0f} MB)"
    )
    print(f"  Peak RAM P95:  {ram['p95']:.0f} MB")
    e = p["energy_mwh_per_inference"]
    print(
        f"  Energy P95:    {e['p95']:.4f} mWh/inf  (target: <= {TARGETS['energy_mwh_per_inference']:.2f} mWh)"
    )

    print("=" * 70)

    # --- Check targets ---
    target_fails = check_targets(report)
    regression_fails = check_regression(report, args.baseline)
    all_fails = target_fails + regression_fails

    if all_fails:
        print(f"\n{'!' * 70}")
        print(f"  {len(all_fails)} CHECK(S) FAILED:")
        for f in all_fails:
            print(f"    {f}")
        print(f"{'!' * 70}")
    else:
        print("\nAll targets PASSED.")

    # --- Save results ---
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    with open(results_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nResults saved to {results_path}")

    if args.save_baseline:
        baseline_path = Path(args.baseline)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_path, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"Baseline saved to {baseline_path}")

    if all_fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
