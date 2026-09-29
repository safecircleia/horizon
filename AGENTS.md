# Horizon — Agent Guide

This document is for AI agents (OpenCode, Claude, etc.) working in this repo.
It covers the cluster setup, the job submission API, and the required git
workflow.

---

## Cluster

| Field | Value |
|---|---|
| Host | `155.54.210.99` |
| User | `jtpalma` |
| SSH | `jtpalma@155.54.210.99` (key-based, no password) |
| Repo path | `/slurm/home/jtpalma/safecircle/horizon` |
| GitHub | `https://github.com/safecircleia/horizon.git` |
| Python | `.venv` managed by `uv` at `<repo_path>/.venv` |

The cluster runs SLURM (ANTS). Jobs are submitted via `sbatch`. Log files land
in `/slurm/home/jtpalma/output/<JOBID>/`.

---

## Critical workflow: commit before submitting

**The cluster receives code only through git.** There is no rsync, no scp.

The only way to get local changes onto the cluster is:

1. Commit them locally
2. Push to GitHub
3. The cluster does `git pull` before the job starts

Every submission function in `tui/remote_actions.py` enforces this
automatically. If you call any `remote_submit_*` function with
`auto_commit_message`, it will:

1. Stage all changes (`git add -A`)
2. Commit with your message
3. Push to `origin`
4. SSH into the cluster and `git pull --rebase`
5. Then submit the sbatch job

**Never skip this.** Code changes that are not pushed will not be on the
cluster.

---

## Job submission API

All agent-facing functions live in `tui/remote_actions.py`. Import directly:

```python
from tui.remote_actions import (
    remote_submit_training,
    remote_submit_merge,
    remote_submit_export_edge,
    remote_submit_export_edge_web,
    remote_submit_export_mobile,
    remote_submit_evaluate,
    remote_submit_benchmark,
    remote_submit_profile_mobile,
    remote_submit_test_litert,
    remote_job_status,
    remote_recent_jobs,
    remote_tail_log,
    remote_wait_for_job,
)
```

### Submit a training job

```python
ok, job_id = remote_submit_training(
    config_name="Edge 2B (Gemma 4 E2B)",
    auto_commit_message="feat: updated training config",
)
# ok=False means either git sync failed or sbatch failed; job_id is an error msg in that case
```

Valid `config_name` values:

| Name | GPU | Script |
|---|---|---|
| `"Edge 2B (Gemma 4 E2B)"` | L4 | `slurm/train_edge_2b.sbatch` |
| `"Edge 4B (Gemma 4 E4B)"` | L4 | `slurm/train_edge_4b.sbatch` |
| `"Full (Llama 3.2 3B, H100)"` | H100 | `slurm/train_h100.sbatch` |
| `"Full (Llama 3.2 3B, L4)"` | L4 | `slurm/train_l4.sbatch` |
| `"Mobile (Gemma 3 1B)"` | L4 | `slurm/train_mobile.sbatch` |

To resume from a checkpoint, pass `resume_checkpoint`:

```python
ok, job_id = remote_submit_training(
    "Full (Llama 3.2 3B, H100)",
    resume_checkpoint="/slurm/home/jtpalma/safecircle/horizon/experiments/h100-2026-09/checkpoint-5000",
    auto_commit_message="chore: resume run",
)
```

### Other submit functions

```python
# Merge LoRA into base model
ok, job_id = remote_submit_merge(
    checkpoint="/slurm/home/jtpalma/safecircle/horizon/experiments/h100-2026-09/final",
    output="/slurm/home/jtpalma/safecircle/horizon/models/merged",
    auto_commit_message="chore: merge lora",
)

# Evaluate a checkpoint
ok, job_id = remote_submit_evaluate(
    checkpoint="/slurm/home/jtpalma/safecircle/horizon/experiments/h100-2026-09/final",
    auto_commit_message="chore: evaluate",
)

# Benchmark
ok, job_id = remote_submit_benchmark(
    checkpoint="/slurm/home/jtpalma/safecircle/horizon/experiments/h100-2026-09/final",
    save_baseline=False,
    auto_commit_message="chore: benchmark",
)

# Export edge model
ok, job_id = remote_submit_export_edge(model_size="2b", auto_commit_message="chore: export")
ok, job_id = remote_submit_export_edge_web(model_size="2b", auto_commit_message="chore: export web")

# Export mobile (LiteRT-LM)
ok, job_id = remote_submit_export_mobile(
    checkpoint="/slurm/home/jtpalma/safecircle/horizon/experiments/mobile-2026-09/final",
    auto_commit_message="chore: export mobile",
)
```

### Monitor jobs

```python
# Currently running / pending
jobs = remote_job_status()
# Returns: [{"job_id": "123", "name": "train_edge_2b", "state": "RUNNING",
#             "time": "0:10:02", "partition": "gpu", "node": "slurm-gpu08"}, ...]

# Recently finished
recent = remote_recent_jobs(count=10)
# Returns: [{"job_id": "120", "name": "evaluate", "state": "COMPLETED",
#             "exit_code": "0:0", "end_time": "2026-09-08T10:00:00", "elapsed": "0:45:12"}, ...]
```

### Wait for a job to finish

```python
result = remote_wait_for_job(
    job_id="123",
    poll_seconds=30,       # how often to poll squeue / sacct
    timeout_minutes=720,   # give up after 12 hours
)
# result["state"] is one of: COMPLETED, FAILED, CANCELLED, TIMEOUT, OUT_OF_MEMORY, TIMEOUT_WAITING
```

`TIMEOUT_WAITING` means the function gave up waiting (the job may still be
running). All other states come directly from SLURM.

### Read logs

```python
# Last 100 lines of stdout
log = remote_tail_log(job_id="123", lines=100)

# Last 50 lines of stderr
err = remote_tail_log(job_id="123", lines=50, stderr=True)
```

Logs live at `/slurm/home/jtpalma/output/<JOBID>/` on the cluster.

---

## Full end-to-end example

```python
from tui.remote_actions import (
    remote_submit_training,
    remote_wait_for_job,
    remote_submit_evaluate,
    remote_tail_log,
)

# 1. Submit training (auto-commits any pending changes first)
ok, job_id = remote_submit_training(
    "Edge 2B (Gemma 4 E2B)",
    auto_commit_message="feat: new training hyperparams",
)
assert ok, f"Submission failed: {job_id}"
print(f"Training job submitted: {job_id}")

# 2. Wait for it to complete
result = remote_wait_for_job(job_id)
assert result["state"] == "COMPLETED", f"Training failed: {result}"

# 3. Evaluate the result
checkpoint = f"/slurm/home/jtpalma/safecircle/horizon/experiments/edge2b-{job_id}/final"
ok, eval_job_id = remote_submit_evaluate(checkpoint)
assert ok, f"Eval submission failed: {eval_job_id}"
result = remote_wait_for_job(eval_job_id)

# 4. Read the log
print(remote_tail_log(eval_job_id, lines=50))
```

---

## Low-level SSH helpers

For anything not covered by `remote_actions.py`, use `tui/ssh.py`:

```python
from tui.ssh import run_remote, check_connection, sync_repo, ensure_venv, git_local_status

# Test connectivity
ok, msg = check_connection()

# Pull latest code on cluster manually
ok, msg = sync_repo()

# Check local git state before doing anything
status = git_local_status()
# {"clean": True/False, "branch": "main", "ahead": 0, "status_lines": [...]}
```

---

## Cluster config

The cluster config is a singleton loaded from `~/.config/horizon/config.json`
(created automatically with defaults on first use). Override programmatically:

```python
from tui.config import ClusterConfig

cfg = ClusterConfig(
    host="155.54.210.99",
    user="jtpalma",
    repo_path="/slurm/home/jtpalma/safecircle/horizon",
    github_repo="https://github.com/safecircleia/horizon.git",
)
# Pass cfg= to any remote_* function to override the singleton
ok, job_id = remote_submit_training("Edge 2B (Gemma 4 E2B)", cfg=cfg)
```

---

## Prerequisites

Before any job submission works:

1. **SSH key** — your public key must be in `jtpalma@155.54.210.99:~/.ssh/authorized_keys`.
   Test with: `ssh -o BatchMode=yes jtpalma@155.54.210.99 echo ok`

2. **Repo cloned on cluster** — happens automatically on first `sync_repo()` call,
   or manually: `ssh jtpalma@155.54.210.99 "git clone https://github.com/safecircleia/horizon.git /slurm/home/jtpalma/safecircle/horizon"`

3. **venv on cluster** — happens automatically via `ensure_venv()`, or from the
   TUI → Cluster → Setup venv.

---

## TUI

Run the interactive TUI locally:

```bash
python -m tui
```

The TUI submits all jobs to the cluster over SSH (same `remote_actions.py` API).
The **Cluster (SSH)** menu provides: connection status, git status, commit &
push, repo sync, venv setup, and remote job/node monitoring.
