"""SLURM interaction layer — wraps squeue, sbatch, scancel, tail."""

import subprocess
from dataclasses import dataclass


@dataclass
class Job:
    job_id: str
    name: str
    state: str
    time: str
    partition: str
    node: str


def squeue(user: str = None) -> list[Job]:
    """Get running/pending jobs for the current user."""
    cmd = ["squeue", "--format=%i|%j|%T|%M|%P|%N", "--noheader"]
    if user:
        cmd += ["-u", user]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    jobs = []
    for line in result.stdout.strip().splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 6:
            jobs.append(Job(
                job_id=parts[0].strip(),
                name=parts[1].strip(),
                state=parts[2].strip(),
                time=parts[3].strip(),
                partition=parts[4].strip(),
                node=parts[5].strip(),
            ))
    return jobs


def sbatch(script: str, export_vars: dict = None) -> tuple[bool, str]:
    """Submit a SLURM job. Returns (success, job_id_or_error)."""
    cmd = ["sbatch"]
    if export_vars:
        export_str = ",".join(f"{k}={v}" for k, v in export_vars.items())
        cmd += [f"--export={export_str}"]
    cmd.append(script)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        # "Submitted batch job 1234"
        job_id = result.stdout.strip().split()[-1]
        return True, job_id
    return False, result.stderr.strip()


def scancel(job_id: str) -> tuple[bool, str]:
    result = subprocess.run(["scancel", job_id], capture_output=True, text=True)
    if result.returncode == 0:
        return True, f"Cancelled job {job_id}"
    return False, result.stderr.strip()


def tail_log(job_id: str, lines: int = 50) -> str:
    """Read the last N lines of a job's stdout log."""
    import os
    user = os.environ.get("USER", "")
    log_path = f"/slurm/home/{user}/output/{job_id}/terminal.out"
    result = subprocess.run(["tail", f"-n{lines}", log_path], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout
    return f"(cannot read log: {log_path})"


def disk_usage(path: str) -> str:
    result = subprocess.run(["du", "-sh", path], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip().split("\t")[0]
    return "?"
