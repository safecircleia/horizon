"""Remote action wrappers — mirrors tui/actions.py but runs jobs on the cluster.

Each function:
  1. Calls ensure_committed_and_pushed() to guarantee the cluster has the
     latest code before submitting anything.
  2. Delegates to the matching ssh_sbatch call in remote_slurm.py.

AI agents should import from this module to submit and monitor jobs:

    from tui.remote_actions import (
        remote_submit_training,
        remote_submit_merge,
        remote_submit_evaluate,
        remote_submit_benchmark,
        remote_job_status,
        remote_tail_log,
        remote_recent_jobs,
    )
"""

from __future__ import annotations

from .config import ClusterConfig, get_cluster_config
from .remote_slurm import (
    ssh_sacct_recent,
    ssh_sbatch,
    ssh_squeue,
    ssh_tail_err_log,
    ssh_tail_log,
)
from .ssh import ensure_committed_and_pushed

# ── Shared sbatch scripts (relative to repo root) ─────────────────────────────

_SBATCH = {
    "train_edge_2b": "slurm/train_edge_2b.sbatch",
    "train_edge_4b": "slurm/train_edge_4b.sbatch",
    "train_h100": "slurm/train_h100.sbatch",
    "train_l4": "slurm/train_l4.sbatch",
    "train_mobile": "slurm/train_mobile.sbatch",
    "merge_lora": "slurm/merge_lora.sbatch",
    "export_edge": "slurm/export_edge.sbatch",
    "export_edge_web": "slurm/export_edge_web.sbatch",
    "export_litert": "slurm/export_litert.sbatch",
    "evaluate": "slurm/evaluate.sbatch",
    "benchmark": "slurm/benchmark.sbatch",
    "profile_mobile": "slurm/profile_mobile.sbatch",
    "test_litert": "slurm/test_litert.sbatch",
    "benchmark_mobile": "slurm/benchmark_mobile.sbatch",
    "evaluate_mobile": "slurm/evaluate_mobile.sbatch",
}

# Maps the human-readable TRAIN_CONFIGS keys to internal sbatch keys
_TRAIN_KEY_MAP = {
    "Edge 2B (Gemma 4 E2B)": "train_edge_2b",
    "Edge 4B (Gemma 4 E4B)": "train_edge_4b",
    "Full (Llama 3.2 3B, H100)": "train_h100",
    "Full (Llama 3.2 3B, L4)": "train_l4",
    "Mobile (Gemma 3 1B)": "train_mobile",
}


def _remote_script(key: str, cfg: ClusterConfig) -> str:
    return f"{cfg.repo_path}/{_SBATCH[key]}"


def _guard(auto_message: str | None = None) -> tuple[bool, str]:
    """Run the git commit/push/pull guard. Returns (ok, msg)."""
    return ensure_committed_and_pushed(auto_message=auto_message)


# ── Training ──────────────────────────────────────────────────────────────────


def remote_submit_training(
    config_name: str,
    resume_checkpoint: str | None = None,
    auto_commit_message: str | None = None,
    cfg: ClusterConfig | None = None,
) -> tuple[bool, str]:
    """Commit+push local changes, pull on cluster, then submit a training job.

    Args:
        config_name: One of the TRAIN_CONFIGS keys (e.g. "Edge 2B (Gemma 4 E2B)").
        resume_checkpoint: Optional path on the cluster to resume from.
        auto_commit_message: If set, uncommitted changes are auto-committed with
            this message (useful for AI agents). If None and there are uncommitted
            changes, the call fails with a descriptive error.
        cfg: Optional cluster config override.

    Returns:
        (True, job_id) on success, (False, error_message) on failure.
    """
    cfg = cfg or get_cluster_config()
    ok, msg = _guard(auto_message=auto_commit_message)
    if not ok:
        return False, msg

    key = _TRAIN_KEY_MAP.get(config_name)
    if key is None:
        return False, f"Unknown config '{config_name}'. Valid: {list(_TRAIN_KEY_MAP)}"

    export_vars: dict[str, str] = {}
    if resume_checkpoint:
        export_vars["RESUME_CHECKPOINT"] = resume_checkpoint

    return ssh_sbatch(_remote_script(key, cfg), export_vars or None, cfg)


# ── Merge LoRA ────────────────────────────────────────────────────────────────


def remote_submit_merge(
    checkpoint: str,
    output: str,
    auto_commit_message: str | None = None,
    cfg: ClusterConfig | None = None,
) -> tuple[bool, str]:
    cfg = cfg or get_cluster_config()
    ok, msg = _guard(auto_message=auto_commit_message)
    if not ok:
        return False, msg
    return ssh_sbatch(
        _remote_script("merge_lora", cfg),
        {"CHECKPOINT": checkpoint, "OUTPUT": output},
        cfg,
    )


# ── Export ────────────────────────────────────────────────────────────────────


def remote_submit_export_edge(
    model_size: str,
    auto_commit_message: str | None = None,
    cfg: ClusterConfig | None = None,
) -> tuple[bool, str]:
    cfg = cfg or get_cluster_config()
    ok, msg = _guard(auto_message=auto_commit_message)
    if not ok:
        return False, msg
    return ssh_sbatch(
        _remote_script("export_edge", cfg), {"MODEL_SIZE": model_size}, cfg
    )


def remote_submit_export_edge_web(
    model_size: str,
    auto_commit_message: str | None = None,
    cfg: ClusterConfig | None = None,
) -> tuple[bool, str]:
    cfg = cfg or get_cluster_config()
    ok, msg = _guard(auto_message=auto_commit_message)
    if not ok:
        return False, msg
    return ssh_sbatch(
        _remote_script("export_edge_web", cfg), {"MODEL_SIZE": model_size}, cfg
    )


def remote_submit_export_mobile(
    checkpoint: str,
    auto_commit_message: str | None = None,
    cfg: ClusterConfig | None = None,
) -> tuple[bool, str]:
    cfg = cfg or get_cluster_config()
    ok, msg = _guard(auto_message=auto_commit_message)
    if not ok:
        return False, msg
    return ssh_sbatch(
        _remote_script("export_litert", cfg), {"CHECKPOINT": checkpoint}, cfg
    )


# ── Evaluate ──────────────────────────────────────────────────────────────────


def remote_submit_evaluate(
    checkpoint: str,
    auto_commit_message: str | None = None,
    cfg: ClusterConfig | None = None,
) -> tuple[bool, str]:
    cfg = cfg or get_cluster_config()
    ok, msg = _guard(auto_message=auto_commit_message)
    if not ok:
        return False, msg
    return ssh_sbatch(_remote_script("evaluate", cfg), {"CHECKPOINT": checkpoint}, cfg)


def remote_submit_benchmark(
    checkpoint: str,
    save_baseline: bool = False,
    auto_commit_message: str | None = None,
    cfg: ClusterConfig | None = None,
) -> tuple[bool, str]:
    cfg = cfg or get_cluster_config()
    ok, msg = _guard(auto_message=auto_commit_message)
    if not ok:
        return False, msg
    export_vars: dict[str, str] = {"CHECKPOINT": checkpoint}
    if save_baseline:
        export_vars["SAVE_BASELINE"] = "1"
    return ssh_sbatch(_remote_script("benchmark", cfg), export_vars, cfg)


def remote_submit_profile_mobile(
    model_path: str,
    runs: int = 10,
    auto_commit_message: str | None = None,
    cfg: ClusterConfig | None = None,
) -> tuple[bool, str]:
    cfg = cfg or get_cluster_config()
    ok, msg = _guard(auto_message=auto_commit_message)
    if not ok:
        return False, msg
    return ssh_sbatch(
        _remote_script("profile_mobile", cfg),
        {"MODEL_PATH": model_path, "RUNS": str(runs)},
        cfg,
    )


def remote_submit_test_litert(
    model_path: str | None = None,
    auto_commit_message: str | None = None,
    cfg: ClusterConfig | None = None,
) -> tuple[bool, str]:
    cfg = cfg or get_cluster_config()
    ok, msg = _guard(auto_message=auto_commit_message)
    if not ok:
        return False, msg
    export_vars: dict[str, str] = {}
    if model_path:
        export_vars["MODEL_PATH"] = model_path
    return ssh_sbatch(_remote_script("test_litert", cfg), export_vars or None, cfg)


# ── Mobile benchmark / evaluate ───────────────────────────────────────────────


def remote_submit_benchmark_mobile(
    model_path: str,
    test_set: str = "data/processed/eval.jsonl",
    save_baseline: bool = False,
    max_samples: int | None = None,
    auto_commit_message: str | None = None,
    cfg: ClusterConfig | None = None,
) -> tuple[bool, str]:
    """Submit the mobile model benchmark (issue #8 targets) on the cluster."""
    cfg = cfg or get_cluster_config()
    ok, msg = _guard(auto_message=auto_commit_message)
    if not ok:
        return False, msg
    export_vars: dict[str, str] = {
        "MODEL_PATH": model_path,
        "TEST_SET": test_set,
    }
    if save_baseline:
        export_vars["SAVE_BASELINE"] = "1"
    if max_samples is not None:
        export_vars["MAX_SAMPLES"] = str(max_samples)
    return ssh_sbatch(_remote_script("benchmark_mobile", cfg), export_vars, cfg)


def remote_submit_evaluate_mobile(
    model_path: str,
    test_set: str = "data/processed/eval.jsonl",
    max_samples: int | None = None,
    auto_commit_message: str | None = None,
    cfg: ClusterConfig | None = None,
) -> tuple[bool, str]:
    """Submit mobile model accuracy evaluation on the cluster."""
    cfg = cfg or get_cluster_config()
    ok, msg = _guard(auto_message=auto_commit_message)
    if not ok:
        return False, msg
    export_vars: dict[str, str] = {
        "MODEL_PATH": model_path,
        "TEST_SET": test_set,
    }
    if max_samples is not None:
        export_vars["MAX_SAMPLES"] = str(max_samples)
    return ssh_sbatch(_remote_script("evaluate_mobile", cfg), export_vars, cfg)


# ── Monitoring (agent-friendly) ───────────────────────────────────────────────


def remote_job_status(cfg: ClusterConfig | None = None) -> list[dict]:
    """Return running/pending jobs as a list of plain dicts (agent-friendly).

    Each dict has keys: job_id, name, state, time, partition, node.
    """
    jobs = ssh_squeue(cfg)
    return [
        {
            "job_id": j.job_id,
            "name": j.name,
            "state": j.state,
            "time": j.time,
            "partition": j.partition,
            "node": j.node,
        }
        for j in jobs
    ]


def remote_recent_jobs(count: int = 10, cfg: ClusterConfig | None = None) -> list[dict]:
    """Return recently completed/failed jobs as plain dicts.

    Each dict has keys: job_id, name, state, exit_code, end_time, elapsed.
    """
    jobs = ssh_sacct_recent(count, cfg)
    return [
        {
            "job_id": j.job_id,
            "name": j.name,
            "state": j.state,
            "exit_code": j.exit_code,
            "end_time": j.end_time,
            "elapsed": j.elapsed,
        }
        for j in jobs
    ]


def remote_tail_log(
    job_id: str,
    lines: int = 100,
    stderr: bool = False,
    cfg: ClusterConfig | None = None,
) -> str:
    """Read the last *lines* of a job's stdout (or stderr) log from the cluster."""
    if stderr:
        return ssh_tail_err_log(job_id, lines, cfg)
    return ssh_tail_log(job_id, lines, cfg)


def remote_wait_for_job(
    job_id: str,
    poll_seconds: int = 30,
    timeout_minutes: int = 720,
    cfg: ClusterConfig | None = None,
) -> dict:
    """Block until a job finishes (or timeout). Returns final job dict.

    Useful for AI agent workflows that need to wait for training to complete
    before proceeding to the next step.

    Returns a dict with keys: job_id, name, state, exit_code, end_time, elapsed.
    On timeout returns {"job_id": job_id, "state": "TIMEOUT_WAITING", ...}.
    """
    import time

    cfg = cfg or get_cluster_config()
    deadline = time.time() + timeout_minutes * 60
    terminal_states = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}

    while time.time() < deadline:
        # Check squeue first (running/pending)
        running = ssh_squeue(cfg)
        running_ids = {j.job_id for j in running}
        if job_id not in running_ids:
            # Job left the queue — look it up in sacct
            recent = ssh_sacct_recent(50, cfg)
            for j in recent:
                if j.job_id == job_id:
                    return {
                        "job_id": j.job_id,
                        "name": j.name,
                        "state": j.state,
                        "exit_code": j.exit_code,
                        "end_time": j.end_time,
                        "elapsed": j.elapsed,
                    }
            # sacct may not have it yet — give it a moment
            time.sleep(poll_seconds)
            continue

        # Check current state of running job
        for j in running:
            if j.job_id == job_id and j.state in terminal_states:
                return {
                    "job_id": j.job_id,
                    "name": j.name,
                    "state": j.state,
                    "time": j.time,
                    "exit_code": "?",
                    "end_time": "",
                    "elapsed": j.time,
                }

        time.sleep(poll_seconds)

    return {
        "job_id": job_id,
        "name": "",
        "state": "TIMEOUT_WAITING",
        "exit_code": "",
        "end_time": "",
        "elapsed": "",
    }
