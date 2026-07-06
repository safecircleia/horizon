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
        cmd += [f"--export=ALL,{export_str}"]
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
    log_path = f"/slurm/home/{user}/output/{job_id}.out"
    result = subprocess.run(["tail", f"-n{lines}", log_path], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout
    return f"(cannot read log: {log_path})"


def tail_err_log(job_id: str, lines: int = 50) -> str:
    """Read the last N lines of a job's stderr log."""
    import os
    user = os.environ.get("USER", "")
    log_path = f"/slurm/home/{user}/output/{job_id}.err"
    result = subprocess.run(["tail", f"-n{lines}", log_path], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout or "(stderr is empty)"
    return f"(cannot read log: {log_path})"


@dataclass
class RecentJob:
    job_id: str
    name: str
    state: str
    exit_code: str
    end_time: str
    elapsed: str


def sacct_recent(count: int = 10) -> list[RecentJob]:
    """Get recently completed/failed jobs from sacct."""
    cmd = [
        "sacct",
        "--format=JobID,JobName%20,State,ExitCode,End,Elapsed",
        "--noheader",
        "--parsable2",
        "--starttime=now-2days",
        "--endtime=now",
        "--state=COMPLETED,FAILED,TIMEOUT,CANCELLED,OUT_OF_MEMORY",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    jobs = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 6:
            continue
        job_id = parts[0].strip()
        # Skip sub-steps (e.g. "1234.batch", "1234.extern")
        if "." in job_id:
            continue
        jobs.append(RecentJob(
            job_id=job_id,
            name=parts[1].strip(),
            state=parts[2].strip(),
            exit_code=parts[3].strip(),
            end_time=parts[4].strip(),
            elapsed=parts[5].strip(),
        ))
    # Most recent first, capped
    return jobs[-count:][::-1]


def disk_usage(path: str) -> str:
    result = subprocess.run(["du", "-sh", path], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip().split("\t")[0]
    return "?"


@dataclass
class NodeInfo:
    name: str
    partition: str
    state: str
    cpus: str
    memory: str
    gres: str  # e.g. "gpu:nvidia_h100_nvl:1"


def sinfo() -> list[NodeInfo]:
    """Get cluster node info (name, partition, state, CPUs, memory, GPUs)."""
    cmd = ["sinfo", "--format=%n|%P|%T|%c|%m|%G", "--noheader"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    nodes = []
    seen = set()
    for line in result.stdout.strip().splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 6:
            name = parts[0].strip()
            if name in seen:
                continue
            seen.add(name)
            nodes.append(NodeInfo(
                name=name,
                partition=parts[1].strip().rstrip("*"),
                state=parts[2].strip(),
                cpus=parts[3].strip(),
                memory=parts[4].strip(),
                gres=parts[5].strip(),
            ))
    return nodes


def format_node_gpu(gres: str) -> str:
    """Parse GRES string like 'gpu:nvidia_h100_nvl:1' into readable form."""
    if not gres or gres == "(null)":
        return "—"
    parts = gres.split(":")
    if len(parts) >= 3:
        gpu_name = parts[1].replace("nvidia_", "").replace("_", " ").upper()
        count = parts[2]
        return f"{count}x {gpu_name}"
    return gres


def format_node_memory(mem_mb: str) -> str:
    """Convert memory in MB to human-readable."""
    try:
        mb = int(mem_mb)
        return f"{mb // 1024} GB"
    except ValueError:
        return mem_mb
