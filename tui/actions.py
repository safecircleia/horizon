"""All TUI actions — maps menu items to SLURM jobs or local commands."""

import os
import subprocess
from pathlib import Path

from .slurm import sbatch, disk_usage

PROJECT_ROOT = Path(__file__).parent.parent


# ── Training ─────────────────────────────────────────────────────────────────

TRAIN_CONFIGS = {
    "Edge 2B (Gemma 4 E2B)": "slurm/train_edge_2b.sbatch",
    "Edge 4B (Gemma 4 E4B)": "slurm/train_edge_4b.sbatch",
    "Full (Llama 3.2 3B, H100)": "slurm/train_h100.sbatch",
    "Full (Llama 3.2 3B, L4)": "slurm/train_l4.sbatch",
    "Mobile (Gemma 3 1B)": "slurm/train_mobile.sbatch",
}


def submit_training(config_name: str, resume_checkpoint: str = None) -> tuple[bool, str]:
    script = TRAIN_CONFIGS[config_name]
    export_vars = {}
    if resume_checkpoint:
        export_vars["RESUME_CHECKPOINT"] = resume_checkpoint
    return sbatch(str(PROJECT_ROOT / script), export_vars or None)


# ── Merge LoRA ───────────────────────────────────────────────────────────────

def submit_merge(checkpoint: str, output: str) -> tuple[bool, str]:
    return sbatch(
        str(PROJECT_ROOT / "slurm/merge_lora.sbatch"),
        export_vars={"CHECKPOINT": checkpoint, "OUTPUT": output},
    )


# ── Export LiteRT-LM ─────────────────────────────────────────────────────────

def submit_export_edge(model_size: str) -> tuple[bool, str]:
    return sbatch(
        str(PROJECT_ROOT / "slurm/export_edge.sbatch"),
        export_vars={"MODEL_SIZE": model_size},
    )


def submit_export_mobile(checkpoint: str) -> tuple[bool, str]:
    return sbatch(
        str(PROJECT_ROOT / "slurm/export_litert.sbatch"),
        export_vars={"CHECKPOINT": checkpoint},
    )


# ── Evaluate ─────────────────────────────────────────────────────────────────

def submit_evaluate(checkpoint: str) -> tuple[bool, str]:
    return sbatch(
        str(PROJECT_ROOT / "slurm/evaluate.sbatch"),
        export_vars={"CHECKPOINT": checkpoint},
    )


def submit_benchmark(checkpoint: str, save_baseline: bool = False) -> tuple[bool, str]:
    export_vars: dict = {"CHECKPOINT": checkpoint}
    if save_baseline:
        export_vars["SAVE_BASELINE"] = "1"
    return sbatch(
        str(PROJECT_ROOT / "slurm/benchmark.sbatch"),
        export_vars=export_vars,
    )


def create_benchmark_split() -> tuple[bool, str]:
    result = subprocess.run(
        ["python", "-m", "data.scripts.create_benchmark_split"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    return result.returncode == 0, result.stdout + result.stderr


def load_latest_benchmark_report() -> dict | None:
    report_path = PROJECT_ROOT / "evaluation/reports/benchmark/results.json"
    if not report_path.exists():
        return None
    import json
    with open(report_path) as f:
        return json.load(f)


def run_profile_mobile(model_path: str, runs: int = 10) -> tuple[bool, str]:
    result = subprocess.run(
        ["python", "-m", "evaluation.profile_mobile", "--model", model_path, "--runs", str(runs)],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    return result.returncode == 0, result.stdout + result.stderr


# ── Test LiteRT-LM ──────────────────────────────────────────────────────────

def submit_test_litert(model_path: str = None) -> tuple[bool, str]:
    export_vars = {}
    if model_path:
        export_vars["MODEL_PATH"] = model_path
    return sbatch(
        str(PROJECT_ROOT / "slurm/test_litert.sbatch"),
        export_vars or None,
    )


# ── Upload ───────────────────────────────────────────────────────────────────

def run_upload(what: str, version: str, skip_hf: bool = False, skip_r2: bool = False) -> tuple[bool, str]:
    cmd = ["python", str(PROJECT_ROOT / "scripts/upload.py"), "--what", what, "--version", version]
    if skip_hf:
        cmd.append("--skip-hf")
    if skip_r2:
        cmd.append("--skip-r2")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    success = result.returncode == 0
    output = result.stdout + result.stderr
    return success, output


# ── Cleanup ──────────────────────────────────────────────────────────────────

def list_experiments() -> list[dict]:
    """List experiment directories with size info."""
    exp_dir = PROJECT_ROOT / "experiments"
    if not exp_dir.exists():
        return []
    experiments = []
    for d in sorted(exp_dir.iterdir()):
        if d.is_dir():
            experiments.append({
                "name": d.name,
                "path": str(d),
                "size": disk_usage(str(d)),
            })
    return experiments


def delete_experiment(path: str) -> tuple[bool, str]:
    import shutil
    try:
        shutil.rmtree(path)
        return True, f"Deleted {path}"
    except Exception as e:
        return False, str(e)


def list_models() -> list[dict]:
    """List model directories with size info."""
    models_dir = PROJECT_ROOT / "models"
    if not models_dir.exists():
        return []
    models = []
    for d in sorted(models_dir.iterdir()):
        if d.is_dir():
            models.append({
                "name": d.name,
                "path": str(d),
                "size": disk_usage(str(d)),
            })
    return models


# ── Maintenance ──────────────────────────────────────────────────────────────

def update_deps() -> tuple[bool, str]:
    result = subprocess.run(
        ["uv", "pip", "install", "-r", "requirements.txt"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    return result.returncode == 0, result.stdout + result.stderr


def clear_tokenized_cache() -> tuple[bool, str]:
    import shutil
    cache = PROJECT_ROOT / "data/.tokenized_cache"
    if cache.exists():
        shutil.rmtree(cache)
        return True, "Cleared tokenized cache"
    return True, "Cache already empty"
