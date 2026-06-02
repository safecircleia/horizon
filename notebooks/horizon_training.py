import marimo

__generated_with = "0.9.0"
app = marimo.App(width="full", app_title="Horizon Training Dashboard")


@app.cell
def _():
    import marimo as mo
    mo.md("# 🛡️ Horizon Training Dashboard\n**SafeCircle Risk Detection Model** — end-to-end pipeline from data → training → evaluation")
    return (mo,)


@app.cell
def _(mo):
    mo.md("## ⚙️ Environment Setup")
    return


@app.cell
def _():
    import os
    import sys
    import subprocess
    from pathlib import Path

    # Resolve root from this notebook's location (notebooks/ sits one level below project root)
    ROOT = Path(__file__).resolve().parent.parent

    if not (ROOT / "training").exists():
        raise RuntimeError(f"Project root not found at {ROOT}. Run this notebook from the horizon/ directory.")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)

    from dotenv import load_dotenv
    load_dotenv()

    print(f"Project root:    {ROOT}")
    print(f"data/raw exists: {(ROOT / 'data' / 'raw').exists()}")
    print(f"HF_TOKEN set:    {'yes' if os.getenv('HF_TOKEN') else 'NO — set in .env'}")
    print(f"BEDROCK_API_KEY: {'yes' if os.getenv('BEDROCK_API_KEY') else 'NO — set in .env'}")
    return Path, ROOT, os, subprocess, sys


@app.cell
def _(ROOT, mo, subprocess):
    import datetime

    def stream(cmd, log_name=None):
        """Run a command, stream output live with in-place progress updates, persist to log."""
        cmd_str = " ".join(str(c) for c in cmd)
        mo.output.append(mo.md(f"```\n$ {cmd_str}\n```"))

        logs_dir = ROOT / "logs"
        logs_dir.mkdir(exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        label = log_name or (cmd[2] if len(cmd) > 2 else cmd[0])
        label = str(label).replace("/", "_").replace(".", "_")
        log_path = logs_dir / f"{ts}-{label}.log"

        # Buffer for display — supports \r in-place updates (tqdm progress bars)
        display_lines = []
        raw_lines = [f"$ {cmd_str}\n"]

        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=ROOT,
        ) as proc:
            for raw in proc.stdout:
                raw_lines.append(raw)
                # \r without \n means in-place progress update — overwrite last line
                if "\r" in raw and not raw.endswith("\n"):
                    last = raw.split("\r")[-1].rstrip()
                    if display_lines:
                        display_lines[-1] = last
                    else:
                        display_lines.append(last)
                else:
                    # Handle mixed \r\n (e.g. "...100%|\r\n")
                    clean = raw.split("\r")[-1].rstrip()
                    if clean:
                        display_lines.append(clean)
                mo.output.clear()
                mo.output.append(mo.plain_text("\n".join(display_lines[-200:])))
            proc.wait()

        log_path.write_text("".join(raw_lines))

        if proc.returncode != 0:
            mo.output.append(mo.callout(
                mo.md(f"❌ Exited with code {proc.returncode} — log: `{log_path.relative_to(ROOT)}`"),
                kind="danger",
            ))
        else:
            mo.output.append(mo.callout(
                mo.md(f"✅ Done — log: `{log_path.relative_to(ROOT)}`"),
                kind="success",
            ))
        return proc.returncode
    return datetime, stream


@app.cell
def _(mo):
    mo.md("## 🖥️ Hardware Detection")
    return


@app.cell
def _(mo):
    import torch

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        hw_info = f"**GPU:** {gpu_name} ({gpu_mem:.1f} GB VRAM) | CUDA {torch.version.cuda} | PyTorch {torch.__version__}"
        if gpu_mem >= 70:
            hw_info += "\n\n✅ H100 / A100 detected — use **H100 config** for rank-256 LoRA, Flash Attention 4 (Hopper-optimized), seq 4096, batch 32."
            hw_color = "success"
        elif gpu_mem >= 20:
            hw_info += "\n\n✅ High-VRAM GPU — use **L4 config** for full bfloat16 training."
            hw_color = "success"
        else:
            hw_info += f"\n\n⚠️ {gpu_mem:.1f} GB VRAM — use **Quick** or **Base** config with 4-bit quantization."
            hw_color = "warn"
    else:
        hw_info = "**No GPU found** — training will run on CPU (slow, not recommended)"
        hw_color = "warn"

    mo.callout(mo.md(hw_info), kind=hw_color)
    return gpu_mem, gpu_name, has_cuda, torch


@app.cell
def _(mo):
    mo.md("## 📊 Dataset Statistics")
    return


@app.cell
def _(ROOT, mo):
    raw_dir = ROOT / "data" / "raw"
    categories = ["grooming", "bullying", "sexual_content", "isolation",
                  "personal_info", "platform_migration", "threats", "benign"]

    stats = []
    total = 0
    for cat in categories:
        f = raw_dir / f"{cat}.jsonl"
        if f.exists():
            count = sum(1 for line in open(f) if line.strip())
            size_mb = f.stat().st_size / 1e6
            stats.append({"Category": cat, "Count": count, "Size (MB)": f"{size_mb:.1f}", "Status": "✓"})
            total += count
        else:
            stats.append({"Category": cat, "Count": 0, "Size (MB)": "—", "Status": "missing"})

    stats.append({"Category": "TOTAL", "Count": total, "Size (MB)": "", "Status": ""})

    mo.vstack([
        mo.callout(mo.md(f"Scanning `{raw_dir}`"), kind="info"),
        mo.ui.table(stats, label=f"Raw dataset — {total:,} conversations"),
    ])
    return categories, raw_dir, stats, total


@app.cell
def _(mo):
    mo.md("## ⬇️ Step 0: Download Dataset")
    return


@app.cell
def _(mo, os):
    hf_token = os.getenv("HF_TOKEN", "")
    _token_status = (
        mo.callout(mo.md("✅ `HF_TOKEN` found in environment — ready to download."), kind="success")
        if hf_token
        else mo.callout(mo.md("⚠️ `HF_TOKEN` not set. Add it to your `.env` file before downloading."), kind="warn")
    )
    _split_opts = {"All (raw + processed)": "all", "Processed only (train/eval)": "processed", "Raw only": "raw"}
    download_split = mo.ui.dropdown(options=_split_opts, value="All (raw + processed)", label="Dataset split")
    mo.vstack([_token_status, download_split])
    return download_split, hf_token


@app.cell
def _(mo):
    run_download_btn = mo.ui.run_button(label="⬇️ Download Dataset from HuggingFace")
    run_download_btn
    return (run_download_btn,)


@app.cell
def _(download_split, mo, run_download_btn, stream):
    mo.stop(not run_download_btn.value)
    stream(["python", "data/scripts/download_from_hub.py", "--split", download_split.value])
    return


@app.cell
def _(mo):
    mo.md("## 🔄 Step 1: Preprocess Data")
    return


@app.cell
def _(ROOT, mo):
    _train = ROOT / "data" / "processed" / "train.jsonl"
    _eval  = ROOT / "data" / "processed" / "eval.jsonl"

    _train_count = sum(1 for l in open(_train) if l.strip()) if _train.exists() else 0
    _eval_count  = sum(1 for l in open(_eval)  if l.strip()) if _eval.exists()  else 0

    if _train_count > 0:
        mo.callout(
            mo.md(f"✅ Preprocessed data found: **{_train_count:,} train** / **{_eval_count:,} eval**"),
            kind="success"
        )
    else:
        mo.callout(mo.md("⚠️ No preprocessed data found. Run preprocessing below."), kind="warn")
    return


@app.cell
def _(mo):
    split_ratio = mo.ui.slider(0.8, 0.95, value=0.9, step=0.05, label="Train/eval split")
    seed = mo.ui.number(value=42, label="Random seed")
    mo.vstack([split_ratio, seed])
    return seed, split_ratio


@app.cell
def _(mo):
    run_preprocess_btn = mo.ui.run_button(label="▶ Run Preprocessing")
    run_preprocess_btn
    return (run_preprocess_btn,)


@app.cell
def _(mo, run_preprocess_btn, seed, split_ratio, stream):
    mo.stop(not run_preprocess_btn.value)
    stream(["python", "-m", "training.scripts.preprocess",
            "--input", "data/raw",
            "--output", "data/processed",
            "--split", str(split_ratio.value),
            "--seed", str(int(seed.value))])
    return


@app.cell
def _(mo):
    mo.md("## 🏋️ Step 2: Train Model")
    return


@app.cell
def _(mo):
    config_choice = mo.ui.dropdown(
        options={
            "Quick (500 steps, for testing)": "training/configs/quick.yaml",
            "Base (10,000 steps, production)": "training/configs/base.yaml",
            "L4 GPU (15,000 steps, full bfloat16)": "training/configs/l4.yaml",
            "H100 (25,000 steps, rank-256 LoRA, FA2, batch 32)": "training/configs/h100.yaml",
            "Mobile distillation (MobileBERT)": "training/configs/mobile.yaml",
        },
        value="H100 (25,000 steps, rank-256 LoRA, FA2, batch 32)",
        label="Training config",
    )
    resume_path = mo.ui.text(
        placeholder="experiments/run-xxx/checkpoints/step-500 (optional)",
        label="Resume from checkpoint",
    )
    mo.vstack([config_choice, resume_path])
    return config_choice, resume_path


@app.cell
def _(mo):
    run_train_btn = mo.ui.run_button(label="▶ Start Training")
    run_train_btn
    return (run_train_btn,)


@app.cell
def _(config_choice, mo, resume_path, run_train_btn, stream):
    mo.stop(not run_train_btn.value)
    _cmd = ["python", "-m", "training.scripts.train", "--config", config_choice.value]
    if resume_path.value.strip():
        _cmd += ["--resume", resume_path.value.strip()]
    stream(_cmd)
    return


@app.cell
def _(mo):
    mo.md("## 📈 Step 3: Evaluate Model")
    return


@app.cell
def _(ROOT, mo):
    _runs = sorted((ROOT / "experiments").glob("*/final"),
                   key=lambda p: p.stat().st_mtime, reverse=True) \
            if (ROOT / "experiments").exists() else []
    _options = {str(p): str(p) for p in _runs} if _runs else {"No checkpoints found": ""}
    checkpoint_select = mo.ui.dropdown(options=_options, label="Select checkpoint")
    mo.vstack([mo.md("Choose a trained checkpoint to evaluate:"), checkpoint_select])
    return (checkpoint_select,)


@app.cell
def _(mo):
    run_eval_btn = mo.ui.run_button(label="▶ Run Evaluation")
    run_eval_btn
    return (run_eval_btn,)


@app.cell
def _(checkpoint_select, mo, run_eval_btn, stream):
    mo.stop(not run_eval_btn.value)
    mo.stop(not checkpoint_select.value)
    stream(["python", "-m", "evaluation.metrics.evaluate",
            "--checkpoint", checkpoint_select.value,
            "--test-set", "data/evaluation/test.jsonl",
            "--max-samples", "200"])
    return


@app.cell
def _(mo):
    mo.md("## 🧪 Step 3.5: Knowledge Distillation (Mobile Model)")
    return


@app.cell
def _(ROOT, mo):
    _runs = sorted((ROOT / "experiments").glob("*/final"),
                   key=lambda p: p.stat().st_mtime, reverse=True) \
            if (ROOT / "experiments").exists() else []
    _options = {str(p): str(p) for p in _runs} if _runs else {"No checkpoints found": ""}
    teacher_select = mo.ui.dropdown(options=_options, label="Teacher checkpoint (horizon-full)")
    mo.vstack([mo.md("Select the trained `horizon-full` checkpoint to distill from:"), teacher_select])
    return (teacher_select,)


@app.cell
def _(mo):
    run_distill_btn = mo.ui.run_button(label="▶ Run Distillation")
    run_distill_btn
    return (run_distill_btn,)


@app.cell
def _(mo, run_distill_btn, stream, teacher_select):
    mo.stop(not run_distill_btn.value)
    mo.stop(not teacher_select.value)
    stream(["python", "-m", "training.scripts.distill",
            "--teacher", teacher_select.value,
            "--config", "training/configs/mobile.yaml"])
    return


@app.cell
def _(mo):
    mo.md("## 📦 Step 4: Export Mobile Model to ONNX")
    return


@app.cell
def _(ROOT, mo):
    _mobile_runs = sorted((ROOT / "experiments").glob("mobile-*/final"),
                          key=lambda p: p.stat().st_mtime, reverse=True) \
                   if (ROOT / "experiments").exists() else []
    _options = {str(p): str(p) for p in _mobile_runs} if _mobile_runs else {"No mobile checkpoints found": ""}
    mobile_checkpoint_select = mo.ui.dropdown(options=_options, label="Mobile checkpoint to export")
    mo.vstack([mo.md("Select a trained mobile checkpoint to export:"), mobile_checkpoint_select])
    return (mobile_checkpoint_select,)


@app.cell
def _(mo):
    run_export_btn = mo.ui.run_button(label="▶ Export to ONNX")
    run_export_btn
    return (run_export_btn,)


@app.cell
def _(ROOT, mo, mobile_checkpoint_select, run_export_btn, stream):
    mo.stop(not run_export_btn.value)
    mo.stop(not mobile_checkpoint_select.value)
    stream(["python", "-m", "training.scripts.export_onnx",
            "--checkpoint", mobile_checkpoint_select.value,
            "--output", "models/mobile",
            "--quantize"])
    _sizes = {p.name: f"{p.stat().st_size / 1e6:.1f} MB"
              for p in (ROOT / "models/mobile").glob("*.onnx")} \
             if (ROOT / "models/mobile").exists() else {}
    if _sizes:
        mo.output.append(mo.md("\n**Output files:**\n" + "\n".join(f"- `{k}`: {v}" for k, v in _sizes.items())))
    return


@app.cell
def _(mo):
    mo.md("## 📋 Experiment History")
    return


@app.cell
def _(ROOT, mo):
    import yaml as _yaml

    _rows = []
    if (ROOT / "experiments").exists():
        for _run in sorted((ROOT / "experiments").glob("*/training_config.yaml"),
                           key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                _cfg = _yaml.safe_load(open(_run))
                _rows.append({
                    "Run": _run.parent.name,
                    "Steps": _cfg.get("training", {}).get("max_steps", "?"),
                    "LR": _cfg.get("training", {}).get("learning_rate", "?"),
                    "Batch": _cfg.get("training", {}).get("per_device_train_batch_size", "?"),
                    "LoRA rank": _cfg.get("lora", {}).get("rank", "?"),
                    "Seq len": _cfg.get("data", {}).get("max_seq_length", "?"),
                    "4-bit": _cfg.get("quantization", {}).get("load_in_4bit", "?"),
                })
            except Exception:
                pass

    if _rows:
        mo.ui.table(_rows, label="Past training runs")
    else:
        mo.callout(mo.md("No experiments yet. Run training to see history here."), kind="info")
    return


@app.cell
def _(mo):
    mo.md("## 📜 Log Viewer")
    return


@app.cell
def _(ROOT, mo):
    _logs_dir = ROOT / "logs"
    _logs = sorted(_logs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True) \
            if _logs_dir.exists() else []
    _options = {p.name: str(p) for p in _logs} if _logs else {"No logs yet": ""}
    log_select = mo.ui.dropdown(options=_options, label="Select log file")
    mo.vstack([mo.md("Browse persisted output from past runs:"), log_select])
    return (log_select,)


@app.cell
def _(ROOT, log_select, mo):
    from pathlib import Path as _Path
    if log_select.value and _Path(log_select.value).exists():
        _content = _Path(log_select.value).read_text()
        mo.vstack([
            mo.md(f"**{_Path(log_select.value).name}**"),
            mo.code(_content[-10000:], language="bash"),
        ])
    else:
        mo.callout(mo.md("No log selected or log file not found."), kind="info")
    return


if __name__ == "__main__":
    app.run()
