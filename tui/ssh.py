"""SSH helpers for remote cluster access.

All operations run through the system `ssh` binary using the key already
configured in ~/.ssh.  No passwords, no extra dependencies.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import NamedTuple

from .config import ClusterConfig, get_cluster_config

# SSH flags: batch mode (never prompt), compression, 10s connect timeout
_SSH_FLAGS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "StrictHostKeyChecking=accept-new",
    "-C",
]


class RemoteResult(NamedTuple):
    ok: bool
    stdout: str
    stderr: str
    returncode: int


def run_remote(
    command: str,
    cfg: ClusterConfig | None = None,
    timeout: int = 60,
) -> RemoteResult:
    """Run *command* on the cluster via SSH and return the result."""
    cfg = cfg or get_cluster_config()
    ssh_cmd = ["ssh"] + _SSH_FLAGS + [cfg.remote, command]
    try:
        r = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return RemoteResult(r.returncode == 0, r.stdout, r.stderr, r.returncode)
    except subprocess.TimeoutExpired:
        return RemoteResult(False, "", "SSH command timed out", -1)
    except FileNotFoundError:
        return RemoteResult(False, "", "ssh binary not found", -1)


# ── Git sync guard ────────────────────────────────────────────────────────────


def git_local_status() -> dict:
    """Return a dict with keys: clean (bool), branch (str), ahead (int), status_lines (list[str])."""
    root = Path(__file__).parent.parent

    def _run(*cmd: str) -> str:
        try:
            r = subprocess.run(list(cmd), capture_output=True, text=True, cwd=str(root))
            return r.stdout.strip()
        except FileNotFoundError:
            return ""

    branch = _run("git", "rev-parse", "--abbrev-ref", "HEAD")
    status_out = _run("git", "status", "--porcelain")
    status_lines = [ln for ln in status_out.splitlines() if ln.strip()]

    # How many commits ahead of remote?
    ahead_out = _run("git", "rev-list", "--count", "@{u}..HEAD")
    try:
        ahead = int(ahead_out)
    except ValueError:
        ahead = 0

    return {
        "clean": len(status_lines) == 0,
        "branch": branch,
        "ahead": ahead,
        "status_lines": status_lines,
    }


def git_commit_and_push(message: str) -> tuple[bool, str]:
    """Stage all changes, commit with *message*, and push to origin."""
    root = Path(__file__).parent.parent

    def _run(*cmd: str) -> tuple[bool, str]:
        try:
            r = subprocess.run(list(cmd), capture_output=True, text=True, cwd=str(root))
            return r.returncode == 0, (r.stdout + r.stderr).strip()
        except FileNotFoundError as e:
            return False, str(e)

    ok, out = _run("git", "add", "-A")
    if not ok:
        return False, f"git add failed: {out}"

    ok, out = _run("git", "commit", "-m", message)
    if not ok:
        # Nothing to commit is fine
        if "nothing to commit" in out:
            pass  # will still push if ahead
        else:
            return False, f"git commit failed: {out}"
    commit_out = out

    ok, out = _run("git", "push")
    if not ok:
        return False, f"git push failed: {out}"

    return True, f"{commit_out}\n{out}".strip()


def git_push_existing() -> tuple[bool, str]:
    """Push already-committed (but unpushed) commits to origin."""
    root = Path(__file__).parent.parent
    try:
        r = subprocess.run(
            ["git", "push"],
            capture_output=True,
            text=True,
            cwd=str(root),
        )
        return r.returncode == 0, (r.stdout + r.stderr).strip()
    except FileNotFoundError as e:
        return False, str(e)


def ensure_committed_and_pushed(auto_message: str | None = None) -> tuple[bool, str]:
    """Ensure the working tree is committed and pushed before submitting a job.

    If *auto_message* is provided, uncommitted changes are automatically staged
    and committed with that message (useful for AI agent calls).
    Returns (ok, description).
    """
    status = git_local_status()
    lines = []

    if not status["clean"]:
        if auto_message:
            ok, out = git_commit_and_push(auto_message)
            if not ok:
                return False, f"Auto-commit failed: {out}"
            lines.append(f"Auto-committed & pushed: {auto_message}")
            lines.append(out)
        else:
            dirty = "\n".join(f"  {ln}" for ln in status["status_lines"][:10])
            return (
                False,
                f"Uncommitted changes — commit and push before submitting:\n{dirty}",
            )
    elif status["ahead"] > 0:
        ok, out = git_push_existing()
        if not ok:
            return False, f"git push failed: {out}"
        lines.append(f"Pushed {status['ahead']} commit(s) to origin")
        lines.append(out)
    else:
        lines.append("Working tree clean and up-to-date.")

    # Now pull on the cluster
    ok, msg = sync_repo()
    if not ok:
        return False, f"Cluster git pull failed: {msg}"
    lines.append(f"Cluster synced: {msg.strip()}")

    return True, "\n".join(lines)


# ── Cluster management ────────────────────────────────────────────────────────


def check_connection(cfg: ClusterConfig | None = None) -> tuple[bool, str]:
    """Return (ok, message) — checks that SSH works and SLURM is available."""
    cfg = cfg or get_cluster_config()
    r = run_remote("squeue --version 2>&1 || echo SLURM_NOT_FOUND", cfg, timeout=15)
    if not r.ok and r.returncode != 0:
        return False, r.stderr.strip() or "Connection failed"
    if "SLURM_NOT_FOUND" in r.stdout:
        return False, "Connected but SLURM not available on remote"
    return True, f"Connected to {cfg.remote} — SLURM ready"


def sync_repo(cfg: ClusterConfig | None = None) -> tuple[bool, str]:
    """Pull latest changes on the cluster; clone only if the repo is absent."""
    cfg = cfg or get_cluster_config()
    script = (
        f"set -e; "
        f"if [ -d {cfg.repo_path}/.git ]; then "
        f"  cd {cfg.repo_path} && git fetch --quiet && git pull --rebase --quiet && "
        f'  echo "Synced — $(git log -1 --oneline)"; '
        f"else "
        f"  mkdir -p $(dirname {cfg.repo_path}) && "
        f"  git clone --quiet {cfg.github_repo} {cfg.repo_path} && "
        f"  echo 'Cloned {cfg.github_repo} → {cfg.repo_path}'; "
        f"fi"
    )
    r = run_remote(script, cfg, timeout=120)
    if r.ok:
        return True, f"Repo synced at {cfg.repo_path}\n{r.stdout}"
    return False, r.stderr.strip() or r.stdout.strip() or "Sync failed"


def ensure_venv(cfg: ClusterConfig | None = None) -> tuple[bool, str]:
    """Create venv and install requirements with uv on the cluster (idempotent).

    If the venv already exists and is functional, just confirms it's ready
    without re-running the full install.
    """
    cfg = cfg or get_cluster_config()
    venv = f"{cfg.repo_path}/.venv"
    script = (
        f"set -e; "
        f"cd {cfg.repo_path}; "
        # If venv already exists and Python is usable, we're done
        f"if [ -f {venv}/bin/python ]; then "
        f"  echo 'venv already present'; "
        f"  {venv}/bin/python --version; "
        f"  exit 0; "
        f"fi; "
        # Otherwise install uv if needed and create venv
        f"if ! command -v uv &>/dev/null; then "
        f"  pip install --quiet uv 2>/dev/null || pip3 install --quiet uv; "
        f"fi; "
        f"uv venv {venv} --quiet; "
        f"uv pip install --quiet -r {cfg.repo_path}/requirements.txt "
        f"  --python {venv}/bin/python"
    )
    r = run_remote(script, cfg, timeout=300)
    if r.ok:
        return True, f"Virtual environment ready at {venv}\n{r.stdout}"
    return False, r.stderr.strip() or r.stdout.strip() or "venv setup failed"


def cluster_status(cfg: ClusterConfig | None = None) -> str:
    """Return a human-readable cluster status string."""
    cfg = cfg or get_cluster_config()
    ok, msg = check_connection(cfg)
    if not ok:
        return f"OFFLINE — {msg}"
    r = run_remote(
        "squeue -u $USER -h -o '%i|%j|%T|%M' 2>/dev/null | wc -l",
        cfg,
        timeout=10,
    )
    n = r.stdout.strip() if r.ok else "?"
    return f"ONLINE — {n} job(s) running"
