#!/usr/bin/env python3
"""RAM and battery profiling for the mobile (LiteRT) model.

Measures peak RAM usage and estimates energy drain per inference call by
running the LiteRT model via `uvx litert-lm` and monitoring the process.

Usage:
    python -m evaluation.profile_mobile \
        --model models/mobile-standard/horizon-mobile-int8_q8_ekv1280.litertlm \
        --prompt "Analyze this conversation:\nChild: hey wanna hang out?\nOther: sure where?"
    python -m evaluation.profile_mobile --model models/mobile-lite/horizon-mobile-int4_q4_block128_ekv1280.litertlm --runs 20

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


def _estimate_energy_mwh(wall_s: float, avg_cpu_pct: float, tdp_mw: float = 3000.0) -> float:
    """Rough energy estimate: TDP × utilisation × time (mWh).

    Default TDP=3W is a conservative figure for a mid-range mobile SoC.
    Pass --tdp-mw to adjust for your device.
    """
    utilisation = avg_cpu_pct / 100.0
    energy_mwh = tdp_mw * utilisation * (wall_s / 3600.0)
    return round(energy_mwh, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile LiteRT mobile model RAM and energy")
    parser.add_argument("--model", required=True, help="Path to .litertlm model file")
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT, help="Prompt to run")
    parser.add_argument("--runs", type=int, default=10, help="Number of inference runs to average")
    parser.add_argument("--tdp-mw", type=float, default=3000.0, help="Device TDP in milliwatts (default: 3000)")
    parser.add_argument("--output", default="evaluation/reports/profile.json", help="Output JSON path")
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
        print(f"  Run {i+1:2d}/{args.runs}  wall={wall:.2f}s  peak_rss={peak_rss:.1f}MB  avg_cpu={avg_cpu:.1f}%")

    def pct(arr: list[float], p: float) -> float:
        s = sorted(arr)
        return round(s[min(int(p / 100 * len(s)), len(s) - 1)], 3)

    report = {
        "model": str(model_path),
        "runs": args.runs,
        "latency_s": {"p50": pct(walls, 50), "p95": pct(walls, 95), "p99": pct(walls, 99), "mean": round(sum(walls)/len(walls), 3)},
        "peak_ram_mb": {"p50": pct(peaks, 50), "p95": pct(peaks, 95), "max": round(max(peaks), 1), "mean": round(sum(peaks)/len(peaks), 1)},
        "energy_mwh_per_inference": {
            "p50": _estimate_energy_mwh(pct(walls, 50), sum(cpus)/len(cpus), args.tdp_mw),
            "p95": _estimate_energy_mwh(pct(walls, 95), sum(cpus)/len(cpus), args.tdp_mw),
            "note": f"estimate based on {args.tdp_mw}mW TDP × avg CPU utilisation × wall time",
        },
    }

    print(f"\n{'='*50}")
    print("PROFILE REPORT")
    print(f"{'='*50}")
    print(f"  Latency  P50={report['latency_s']['p50']}s  P95={report['latency_s']['p95']}s")
    print(f"  Peak RAM P50={report['peak_ram_mb']['p50']}MB  max={report['peak_ram_mb']['max']}MB")
    print(f"  Energy   P50≈{report['energy_mwh_per_inference']['p50']} mWh/inference")
    print(f"  ({report['energy_mwh_per_inference']['note']})")
    print(f"{'='*50}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
