#!/usr/bin/env python3
"""RAM and battery profiling for the mobile (LiteRT) model.

Measures peak RAM usage and estimates energy drain per inference call by
running the LiteRT model via `uvx litert-lm` and monitoring the process.

Includes pass/fail target checks from issue #8:
    - Latency P95 < 500 ms
    - Peak RAM   < 150 MB
    - Model disk < 50 MB
    - Energy P95 < 0.66 mWh / inference (≈ 3 % battery / day at 1 inv/min)

Usage:
    python -m evaluation.profile_mobile \
        --model models/mobile-standard/horizon-mobile-int8_q8_ekv1280.litertlm \
        --prompt "Analyze this conversation:\\nChild: hey wanna hang out?\\nOther: sure where?"
    python -m evaluation.profile_mobile \
        --model models/mobile-lite/horizon-mobile-int4_q4_block128_ekv1280.litertlm --runs 20

Requirements:
    pip install psutil
    uvx litert-lm (installed via uv tool)
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    import psutil
except ImportError:
    print("ERROR: psutil not installed. Run: pip install psutil", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Issue #8 performance targets
# ---------------------------------------------------------------------------
TARGETS = {
    "latency_p95_ms": 500.0,  # < 500 ms per inference (P95)
    "peak_ram_mb": 150.0,  # < 150 MB loaded
    "model_disk_mb": 50.0,  # < 50 MB on disk
    # Battery budget: < 3 % / day → ~1440 invocations/day (1 per minute)
    # 32 Wh battery → 960 mWh budget → 0.66 mWh per inference
    "energy_mwh_per_inference": 0.66,
}


_DEFAULT_PROMPT = (
    "Analyze this conversation:\n"
    "Child: hey, wanna play later?\n"
    "Other: sure! add me on discord first tho, just us"
)


def _run_litert(model_path: str, prompt: str) -> tuple[float, float, float]:
    """Run one litert-lm inference and return (wall_s, peak_rss_mb, cpu_pct)."""
    cmd = ["uvx", "litert-lm", "run", "--model", model_path, "--prompt", prompt]

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    peak_rss = 0.0
    cpu_samples: list[float] = []

    try:
        ps = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                mem = ps.memory_info().rss / 1024 / 1024  # MB
                if mem > peak_rss:
                    peak_rss = mem
                cpu_samples.append(ps.cpu_percent(interval=None))
            except psutil.NoSuchProcess:
                break
            time.sleep(0.05)
    except psutil.NoSuchProcess:
        pass

    stdout, stderr = proc.communicate(timeout=120)
    wall = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"  litert-lm stderr: {stderr.strip()}", file=sys.stderr)

    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
    return wall, peak_rss, avg_cpu


def _estimate_energy_mwh(
    wall_s: float, avg_cpu_pct: float, tdp_mw: float = 3000.0
) -> float:
    """Rough energy estimate: TDP x utilisation x time (mWh).

    Default TDP=3W is a conservative figure for a mid-range mobile SoC.
    Pass --tdp-mw to adjust for your device.
    """
    utilisation = avg_cpu_pct / 100.0
    energy_mwh = tdp_mw * utilisation * (wall_s / 3600.0)
    return round(energy_mwh, 4)


def _check_targets(report: dict) -> list[str]:
    """Check profiling results against issue #8 targets.  Returns failure messages."""
    fails: list[str] = []

    # Latency P95
    lat_p95_ms = report["latency_s"]["p95"] * 1000
    if lat_p95_ms > TARGETS["latency_p95_ms"]:
        fails.append(
            f"FAIL latency_p95: {lat_p95_ms:.0f} ms > {TARGETS['latency_p95_ms']:.0f} ms"
        )

    # Peak RAM (max across all runs)
    ram_max = report["peak_ram_mb"]["max"]
    if ram_max > TARGETS["peak_ram_mb"]:
        fails.append(
            f"FAIL peak_ram: {ram_max:.0f} MB > {TARGETS['peak_ram_mb']:.0f} MB"
        )

    # Model on disk
    disk_mb = report.get("model_disk_mb")
    if disk_mb is not None and disk_mb > TARGETS["model_disk_mb"]:
        fails.append(
            f"FAIL model_disk: {disk_mb:.1f} MB > {TARGETS['model_disk_mb']:.0f} MB"
        )

    # Energy per inference (P95)
    energy_p95 = report["energy_mwh_per_inference"]["p95"]
    if energy_p95 > TARGETS["energy_mwh_per_inference"]:
        fails.append(
            f"FAIL energy_p95: {energy_p95:.4f} mWh > {TARGETS['energy_mwh_per_inference']:.2f} mWh"
        )

    return fails


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile LiteRT mobile model RAM and energy"
    )
    parser.add_argument("--model", required=True, help="Path to .litertlm model file")
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT, help="Prompt to run")
    parser.add_argument(
        "--runs", type=int, default=10, help="Number of inference runs to average"
    )
    parser.add_argument(
        "--tdp-mw",
        type=float,
        default=3000.0,
        help="Device TDP in milliwatts (default: 3000)",
    )
    parser.add_argument(
        "--output", default="evaluation/reports/profile.json", help="Output JSON path"
    )
    parser.add_argument(
        "--check-targets",
        action="store_true",
        help="Exit non-zero if any issue #8 target is missed",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Model: {model_path.name}")
    print(f"Runs:  {args.runs}")
    print(f"TDP:   {args.tdp_mw} mW\n")

    walls, peaks, cpus = [], [], []
    for i in range(args.runs):
        wall, peak_rss, avg_cpu = _run_litert(str(model_path), args.prompt)
        walls.append(wall)
        peaks.append(peak_rss)
        cpus.append(avg_cpu)
        print(
            f"  Run {i + 1:2d}/{args.runs}  wall={wall:.2f}s  peak_rss={peak_rss:.1f}MB  avg_cpu={avg_cpu:.1f}%"
        )

    def pct(arr: list[float], p: float) -> float:
        s = sorted(arr)
        return round(s[min(int(p / 100 * len(s)), len(s) - 1)], 3)

    # Model size on disk
    model_disk_mb = round(model_path.stat().st_size / 1024 / 1024, 1)

    report = {
        "model": str(model_path),
        "model_disk_mb": model_disk_mb,
        "runs": args.runs,
        "latency_s": {
            "p50": pct(walls, 50),
            "p95": pct(walls, 95),
            "p99": pct(walls, 99),
            "mean": round(sum(walls) / len(walls), 3),
        },
        "peak_ram_mb": {
            "p50": pct(peaks, 50),
            "p95": pct(peaks, 95),
            "max": round(max(peaks), 1),
            "mean": round(sum(peaks) / len(peaks), 1),
        },
        "energy_mwh_per_inference": {
            "p50": _estimate_energy_mwh(
                pct(walls, 50), sum(cpus) / len(cpus), args.tdp_mw
            ),
            "p95": _estimate_energy_mwh(
                pct(walls, 95), sum(cpus) / len(cpus), args.tdp_mw
            ),
            "note": f"estimate based on {args.tdp_mw}mW TDP x avg CPU utilisation x wall time",
        },
    }

    print(f"\n{'=' * 60}")
    print("PROFILE REPORT")
    print(f"{'=' * 60}")
    print(
        f"  Model disk:  {model_disk_mb} MB  (target: <= {TARGETS['model_disk_mb']:.0f} MB)"
    )
    print(
        f"  Latency  P50={report['latency_s']['p50']}s  "
        f"P95={report['latency_s']['p95']}s  "
        f"(target: <= {TARGETS['latency_p95_ms']:.0f} ms)"
    )
    print(
        f"  Peak RAM P50={report['peak_ram_mb']['p50']}MB  "
        f"max={report['peak_ram_mb']['max']}MB  "
        f"(target: <= {TARGETS['peak_ram_mb']:.0f} MB)"
    )
    print(
        f"  Energy   P50={report['energy_mwh_per_inference']['p50']} mWh/inference  "
        f"(target: <= {TARGETS['energy_mwh_per_inference']:.2f} mWh)"
    )
    print(f"  ({report['energy_mwh_per_inference']['note']})")
    print(f"{'=' * 60}")

    # Target checks
    target_fails = _check_targets(report)
    if target_fails:
        print(f"\n{'!' * 60}")
        print(f"  {len(target_fails)} TARGET(S) MISSED:")
        for f in target_fails:
            print(f"    {f}")
        print(f"{'!' * 60}")
    else:
        print("\nAll performance targets PASSED.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {out}")

    if args.check_targets and target_fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
