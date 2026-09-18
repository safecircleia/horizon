"""Remote SLURM wrappers — mirrors tui/slurm.py but executes via SSH.

Every function matches the signature of its local counterpart so the TUI
can call either version transparently.
"""

from __future__ import annotations

from .config import ClusterConfig, get_cluster_config
from .slurm import Job, NodeInfo, RecentJob
from .ssh import run_remote

# ── Job submission ────────────────────────────────────────────────────────────


def ssh_sbatch(
    script: str,
    export_vars: dict | None = None,
    cfg: ClusterConfig | None = None,
) -> tuple[bool, str]:
    """Submit a SLURM job on the cluster. Returns (success, job_id_or_error)."""
    cfg = cfg or get_cluster_config()
    repo = cfg.repo_path
    venv_python = f"{repo}/.venv/bin/python"

    # Build --export= string
    export_part = ""
    if export_vars:
        export_str = ",".join(f"{k}={v}" for k, v in export_vars.items())
        export_part = f"--export=ALL,{export_str} "

    # Map the local script path to the cluster equivalent:
    # /…/horizon/slurm/foo.sbatch → $repo/slurm/foo.sbatch
    remote_script = _local_to_remote_path(script, repo)

    cmd = (
        f"cd {repo} && "
        f"VIRTUAL_ENV={repo}/.venv "
        f"PATH={repo}/.venv/bin:$PATH "
        f"PYTHON={venv_python} "
        f"sbatch {export_part}{remote_script}"
    )
    r = run_remote(cmd, cfg, timeout=30)
    if r.ok:
        # "Submitted batch job 1234"
        job_id = r.stdout.strip().split()[-1]
        return True, job_id
    return False, (r.stderr.strip() or r.stdout.strip() or "sbatch failed")


def _local_to_remote_path(local_script: str, repo_path: str) -> str:
    """Convert an absolute local path under the project root to the cluster path."""
    from pathlib import Path

    local = Path(local_script)
    # Walk up to find the project root (directory containing slurm/)
    for parent in [local.parent, local.parent.parent]:
        try:
            rel = local.relative_to(parent)
            # If the parent name matches what we expect, use it
            if (parent / "slurm").exists() or str(parent).endswith("horizon"):
                return f"{repo_path}/{rel}"
        except ValueError:
            pass
    # Fallback: try to strip everything up to /horizon/
    parts = local.parts
    for i, p in enumerate(parts):
        if p == "horizon":
            return repo_path + "/" + "/".join(parts[i + 1 :])
    # Last resort: just use the basename inside slurm/
    return f"{repo_path}/slurm/{local.name}"


# ── Queue / status ────────────────────────────────────────────────────────────


def ssh_squeue(cfg: ClusterConfig | None = None) -> list[Job]:
    """Return running/pending jobs for the cluster user."""
    cfg = cfg or get_cluster_config()
    fmt = "%i|%j|%T|%M|%P|%N"
    r = run_remote(
        f"squeue -u {cfg.user} -h -o '{fmt}' 2>/dev/null",
        cfg,
        timeout=15,
    )
    jobs: list[Job] = []
    for line in r.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 6:
            jobs.append(
                Job(
                    job_id=parts[0].strip(),
                    name=parts[1].strip(),
                    state=parts[2].strip(),
                    time=parts[3].strip(),
                    partition=parts[4].strip(),
                    node=parts[5].strip(),
                )
            )
    return jobs


def ssh_scancel(job_id: str, cfg: ClusterConfig | None = None) -> tuple[bool, str]:
    cfg = cfg or get_cluster_config()
    r = run_remote(f"scancel {job_id}", cfg, timeout=15)
    if r.ok:
        return True, f"Cancelled job {job_id}"
    return False, r.stderr.strip()


def ssh_sacct_recent(
    count: int = 10, cfg: ClusterConfig | None = None
) -> list[RecentJob]:
    cfg = cfg or get_cluster_config()
    fmt = "%i|%j|%T|%e|%E|%M"
    r = run_remote(
        f"sacct -u {cfg.user} -X -n -o '{fmt}' --state=ALL "
        f"--starttime=now-7days 2>/dev/null | tail -n {count * 2}",
        cfg,
        timeout=20,
    )
    jobs: list[RecentJob] = []
    for line in r.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 6:
            continue
        job_id = parts[0].strip()
        if "." in job_id:
            continue
        jobs.append(
            RecentJob(
                job_id=job_id,
                name=parts[1].strip(),
                state=parts[2].strip(),
                exit_code=parts[3].strip(),
                end_time=parts[4].strip(),
                elapsed=parts[5].strip(),
            )
        )
    return jobs[-count:][::-1]


def ssh_sinfo(cfg: ClusterConfig | None = None) -> list[NodeInfo]:
    cfg = cfg or get_cluster_config()
    r = run_remote(
        "sinfo -h -o '%N|%P|%T|%C|%m|%G' 2>/dev/null",
        cfg,
        timeout=15,
    )
    nodes: list[NodeInfo] = []
    seen: set[str] = set()
    for line in r.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 6:
            continue
        name = parts[0].strip()
        if name in seen:
            continue
        seen.add(name)
        # CPU field is "A/I/O/T" — take total
        cpu_field = parts[3].strip()
        cpus = cpu_field.split("/")[-1] if "/" in cpu_field else cpu_field
        nodes.append(
            NodeInfo(
                name=name,
                partition=parts[1].strip().rstrip("*"),
                state=parts[2].strip(),
                cpus=cpus,
                memory=parts[4].strip(),
                gres=parts[5].strip(),
            )
        )
    return nodes


# ── Log access ───────────────────────────────────────────────────────────────


def ssh_tail_log(
    job_id: str,
    lines: int = 50,
    cfg: ClusterConfig | None = None,
) -> str:
    """Read the last N lines of a job's stdout log from the cluster."""
    cfg = cfg or get_cluster_config()
    log_path = f"/slurm/home/{cfg.user}/output/{job_id}/terminal.out"
    r = run_remote(f"tail -n{lines} {log_path} 2>/dev/null", cfg, timeout=20)
    if r.ok and r.stdout:
        return r.stdout
    return f"(cannot read log: {log_path})"


def ssh_tail_err_log(
    job_id: str,
    lines: int = 50,
    cfg: ClusterConfig | None = None,
) -> str:
    """Read the last N lines of a job's stderr log from the cluster."""
    cfg = cfg or get_cluster_config()
    log_path = f"/slurm/home/{cfg.user}/output/{job_id}/terminal.err"
    r = run_remote(f"tail -n{lines} {log_path} 2>/dev/null", cfg, timeout=20)
    if r.ok and r.stdout:
        return r.stdout
    return f"(cannot read log: {log_path})"


# ── Actions that parallel tui/actions.py ─────────────────────────────────────


def ssh_submit_training(
    config_name: str,
    resume_checkpoint: str | None = None,
    cfg: ClusterConfig | None = None,
) -> tuple[bool, str]:
    from . import actions

    script = str(actions.PROJECT_ROOT / actions.TRAIN_CONFIGS[config_name])
    export_vars = {}
    if resume_checkpoint:
        export_vars["RESUME_CHECKPOINT"] = resume_checkpoint
    return ssh_sbatch(script, export_vars or None, cfg)


def ssh_submit_generic(
    sbatch_path: str,
    export_vars: dict | None = None,
    cfg: ClusterConfig | None = None,
) -> tuple[bool, str]:
    """Generic wrapper for any sbatch script path (absolute local path)."""
    return ssh_sbatch(sbatch_path, export_vars, cfg)
