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
    import json
    import subprocess
    from pathlib import Path

    # Add project root to path
    ROOT = Path("..").resolve()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)

    from dotenv import load_dotenv
    load_dotenv()

    print(f"Project root: {ROOT}")
    print(f"HF_TOKEN set: {'yes' if os.getenv('HF_TOKEN') else 'NO — set in .env'}")
    print(f"BEDROCK_API_KEY set: {'yes' if os.getenv('BEDROCK_API_KEY') else 'NO — set in .env'}")
    return Path, ROOT, json, os, subprocess, sys


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
        hw_info = f"**GPU detected:** {gpu_name} ({gpu_mem:.1f} GB VRAM)"
        if gpu_mem >= 20:
            hw_info += "\n\n✅ High-VRAM GPU detected — use **L4 config** for full bfloat16 training (no 4-bit needed)."
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
            stats.append({"Category": cat, "Count": count, "Size (MB)": f"{size_mb:.1f}"})
            total += count
        else:
            stats.append({"Category": cat, "Count": 0, "Size (MB)": "—"})

    stats.append({"Category": "TOTAL", "Count": total, "Size (MB)": ""})

    mo.ui.table(stats, label=f"Raw dataset — {total:,} conversations")
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
def _(download_split, mo, run_download_btn, subprocess):
    mo.stop(not run_download_btn.value)

    _result = subprocess.run(
        ["python", "data/scripts/download_from_hub.py", "--split", download_split.value],
        capture_output=True, text=True
    )

    if _result.returncode == 0:
        mo.callout(mo.md(f"✅ Download complete\n```\n{_result.stdout[-2000:]}\n```"), kind="success")
    else:
        mo.callout(mo.md(f"❌ Download failed\n```\n{_result.stderr[-2000:]}\n```"), kind="danger")
    return


@app.cell
def _(mo):
    mo.md("## 🔄 Step 1: Preprocess Data")
    return


@app.cell
def _(mo):
    processed_dir = "data/processed"
    split_ratio = mo.ui.slider(0.8, 0.95, value=0.9, step=0.05, label="Train/eval split")
    seed = mo.ui.number(value=42, label="Random seed")
    mo.vstack([
        mo.md("Configure preprocessing parameters:"),
        split_ratio,
        seed,
    ])
    return processed_dir, seed, split_ratio


@app.cell
def _(mo, processed_dir, seed, split_ratio):
    from pathlib import Path as _Path

    _train = _Path(processed_dir) / "train.jsonl"
    _eval = _Path(processed_dir) / "eval.jsonl"

    _train_count = sum(1 for l in open(_train) if l.strip()) if _train.exists() else 0
    _eval_count = sum(1 for l in open(_eval) if l.strip()) if _eval.exists() else 0

    if _train_count > 0:
        _status = mo.callout(
            mo.md(f"✅ Preprocessed data found: **{_train_count:,} train** / **{_eval_count:,} eval**"),
            kind="success"
        )
    else:
        _status = mo.callout(mo.md("⚠️ No preprocessed data found. Run preprocessing below."), kind="warn")

    _status
    return


@app.cell
def _(mo):
    run_preprocess_btn = mo.ui.run_button(label="▶ Run Preprocessing")
    run_preprocess_btn
    return (run_preprocess_btn,)


@app.cell
def _(mo, processed_dir, run_preprocess_btn, seed, split_ratio, subprocess):
    mo.stop(not run_preprocess_btn.value)

    _result = subprocess.run(
        ["python", "-m", "training.scripts.preprocess",
         "--input", "data/raw",
         "--output", processed_dir,
         "--split", str(split_ratio.value),
         "--seed", str(int(seed.value))],
        capture_output=True, text=True
    )

    if _result.returncode == 0:
        mo.callout(mo.md(f"✅ Preprocessing complete\n```\n{_result.stdout}\n```"), kind="success")
    else:
        mo.callout(mo.md(f"❌ Error\n```\n{_result.stderr}\n```"), kind="danger")
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
            "Mobile distillation (MobileBERT)": "training/configs/mobile.yaml",
        },
        value="Quick (500 steps, for testing)",
        label="Training config",
    )
    resume_path = mo.ui.text(placeholder="experiments/run-xxx/checkpoints/step-500 (optional)", label="Resume from checkpoint")
    mo.vstack([config_choice, resume_path])
    return config_choice, resume_path


@app.cell
def _(mo):
    run_train_btn = mo.ui.run_button(label="▶ Start Training")
    run_train_btn
    return (run_train_btn,)


@app.cell
def _(config_choice, mo, resume_path, run_train_btn, subprocess):
    mo.stop(not run_train_btn.value)

    _cmd = ["python", "-m", "training.scripts.train", "--config", config_choice.value]
    if resume_path.value.strip():
        _cmd += ["--resume", resume_path.value.strip()]

    mo.callout(mo.md(f"Running: `{' '.join(_cmd)}`\n\n⏳ This may take a while — check your terminal for live progress."), kind="info")

    _result = subprocess.run(_cmd, capture_output=True, text=True)

    if _result.returncode == 0:
        mo.callout(mo.md(f"✅ Training complete\n```\n{_result.stdout[-3000:]}\n```"), kind="success")
    else:
        mo.callout(mo.md(f"❌ Training failed\n```\n{_result.stderr[-3000:]}\n```"), kind="danger")
    return


@app.cell
def _(mo):
    mo.md("## 📈 Step 3: Evaluate Model")
    return


@app.cell
def _(Path, mo):
    _runs = sorted(Path("experiments").glob("*/final"), key=lambda p: p.stat().st_mtime, reverse=True) if Path("experiments").exists() else []
    _options = {str(p): str(p) for p in _runs} if _runs else {"No checkpoints found": ""}

    checkpoint_select = mo.ui.dropdown(options=_options, label="Select checkpoint")
    mo.vstack([
        mo.md("Choose a trained checkpoint to evaluate:"),
        checkpoint_select,
    ])
    return (checkpoint_select,)


@app.cell
def _(mo):
    run_eval_btn = mo.ui.run_button(label="▶ Run Evaluation")
    run_eval_btn
    return (run_eval_btn,)


@app.cell
def _(checkpoint_select, mo, run_eval_btn, subprocess):
    mo.stop(not run_eval_btn.value)
    mo.stop(not checkpoint_select.value)

    _result = subprocess.run(
        ["python", "-m", "evaluation.metrics.evaluate",
         "--checkpoint", checkpoint_select.value,
         "--test-set", "data/evaluation/test.jsonl",
         "--max-samples", "200"],
        capture_output=True, text=True
    )

    if _result.returncode == 0:
        mo.callout(mo.md(f"✅ Evaluation complete\n```\n{_result.stdout}\n```"), kind="success")
    else:
        mo.callout(mo.md(f"❌ Evaluation failed\n```\n{_result.stderr}\n```"), kind="danger")
    return


@app.cell
def _(mo):
    mo.md("## 🧪 Step 3.5: Knowledge Distillation (Mobile Model)")
    return


@app.cell
def _(Path, mo):
    _runs = sorted(Path("experiments").glob("*/final"), key=lambda p: p.stat().st_mtime, reverse=True) if Path("experiments").exists() else []
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
def _(mo, run_distill_btn, subprocess, teacher_select):
    mo.stop(not run_distill_btn.value)
    mo.stop(not teacher_select.value)
    _result = subprocess.run(
        ["python", "-m", "training.scripts.distill",
         "--teacher", teacher_select.value,
         "--config", "training/configs/mobile.yaml"],
        capture_output=True, text=True
    )
    if _result.returncode == 0:
        mo.callout(mo.md(f"✅ Distillation complete\n```\n{_result.stdout[-3000:]}\n```"), kind="success")
    else:
        mo.callout(mo.md(f"❌ Distillation failed\n```\n{_result.stderr[-3000:]}\n```"), kind="danger")
    return


@app.cell
def _(mo):
    mo.md("## 📦 Step 4.5: Export Mobile Model to ONNX")
    return


@app.cell
def _(Path, mo):
    _mobile_runs = sorted(Path("experiments").glob("mobile-*/final"), key=lambda p: p.stat().st_mtime, reverse=True) if Path("experiments").exists() else []
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
def _(mo, mobile_checkpoint_select, run_export_btn, subprocess):
    mo.stop(not run_export_btn.value)
    mo.stop(not mobile_checkpoint_select.value)
    _result = subprocess.run(
        ["python", "-m", "training.scripts.export_onnx",
         "--checkpoint", mobile_checkpoint_select.value,
         "--output", "models/mobile",
         "--quantize"],
        capture_output=True, text=True
    )
    if _result.returncode == 0:
        from pathlib import Path as _Path
        _sizes = {p.name: f"{p.stat().st_size / 1e6:.1f} MB" for p in _Path("models/mobile").glob("*.onnx")} if _Path("models/mobile").exists() else {}
        _size_info = "\n".join(f"- `{k}`: {v}" for k, v in _sizes.items()) or "No ONNX files found"
        mo.callout(mo.md(f"✅ Export complete\n\n**Output files:**\n{_size_info}\n\n```\n{_result.stdout[-2000:]}\n```"), kind="success")
    else:
        mo.callout(mo.md(f"❌ Export failed\n```\n{_result.stderr[-3000:]}\n```"), kind="danger")
    return


@app.cell
def _(mo):
    mo.md("## 📋 Experiment History")
    return


@app.cell
def _(Path, json, mo):
    _rows = []
    for _run in sorted(Path("experiments").glob("*/training_config.yaml"), key=lambda p: p.stat().st_mtime, reverse=True) if Path("experiments").exists() else []:
        import yaml as _yaml
        try:
            _cfg = _yaml.safe_load(open(_run))
            _rows.append({
                "Run": _run.parent.name,
                "Steps": _cfg.get("training", {}).get("max_steps", "?"),
                "LR": _cfg.get("training", {}).get("learning_rate", "?"),
                "Batch": _cfg.get("training", {}).get("per_device_train_batch_size", "?"),
            })
        except Exception:
            pass

    if _rows:
        mo.ui.table(_rows, label="Past training runs")
    else:
        mo.callout(mo.md("No experiments yet. Run training to see history here."), kind="info")
    return


if __name__ == "__main__":
    app.run()
