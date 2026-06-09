# Project Horizon

**SafeCircle Risk Detection Model Training System**

Project Horizon trains custom AI models for privacy-preserving child safety risk detection. Fine-tuned from Llama 3.2 3B Instruct using QLoRA, the model detects seven risk categories (grooming, bullying, sexual content, isolation, personal info, platform migration, threats) and returns structured JSON — no free-text, no identity leakage.

A compact mobile variant (Gemma 3 1B) is also trained and exported to [LiteRT-LM](https://ai.google.dev/edge/litert-lm/overview) for on-device deployment on Android and iOS.

---

## Quick Start

```bash
# Install dependencies (requires uv)
uv sync

# Copy and fill in secrets
cp .env.example .env

# Generate dataset on SLURM (see Data Generation section)
sbatch slurm/generate.sbatch

# Preprocess raw data into training format
python -m training.scripts.preprocess --input data/raw --output data/processed

# Train on SLURM
sbatch slurm/train_h100.sbatch

# Evaluate a checkpoint  (--export must come BEFORE the script path)
sbatch --export=ALL,CHECKPOINT=experiments/h100-<timestamp>/checkpoint-25000 slurm/evaluate.sbatch
```

---

## Project Structure

```
horizon/
├── data/
│   ├── raw/                   # Per-category JSONL (500K generated)
│   ├── processed/             # train.jsonl + eval.jsonl (TRL messages format)
│   ├── generation/            # Generation pipeline (generators, prompts, scripts)
│   └── scripts/               # upload_to_hub.py, download_from_hub.py
├── training/
│   ├── configs/               # h100.yaml, l4.yaml, quick.yaml, mobile.yaml
│   ├── model/                 # loader.py (QLoRA), mobile.py (constants)
│   └── scripts/               # preprocess.py, train.py, export_litert.py
├── evaluation/                # evaluate.py, plot_results.py
├── slurm/                     # SLURM sbatch scripts + install_env.sh
├── scripts/                   # merge_lora.py, upload_to_hf.py, export_gguf.sh
├── quantization/              # GGUF export
├── inference/                 # API server, CLI
├── experiments/               # Training runs & checkpoints (git-ignored)
├── models/                    # Trained model artifacts (git-ignored)
└── notebooks/                 # horizon_training.py (Marimo dashboard)
```

---

## Environment Setup

```bash
cp .env.example .env
```

Fill in `.env`:

```
ANTHROPIC_API_KEY=...      # for Claude data generator
OPENAI_API_KEY=...         # for OpenAI data generator (optional)
BEDROCK_API_KEY=...        # for Amazon Bedrock generator (eu-west-3)
HF_TOKEN=...               # HuggingFace token with safecircleai org access
```

---

## Data Generation

The dataset is **500K conversations** across 8 categories, bilingual (60% English / 40% Spanish).

| Category | Count |
|---|---|
| grooming | 80 000 |
| bullying | 70 000 |
| sexual_content | 70 000 |
| isolation | 50 000 |
| personal_info | 50 000 |
| platform_migration | 30 000 |
| threats | 50 000 |
| benign | 100 000 |

### SLURM (ANTS cluster) — recommended

> **Note:** `--export` must always come **before** the script path in sbatch.

```bash
# Full 500K mixed EN/ES dataset (~6h on H100 with Qwen2.5-7B)
sbatch slurm/generate.sbatch

# Spanish only
sbatch --export=ALL,GEN_LANGUAGE=es slurm/generate.sbatch

# English only
sbatch --export=ALL,GEN_LANGUAGE=en slurm/generate.sbatch

# Monitor
squeue -u $USER
tail -f /slurm/home/$USER/output/<JOBID>/terminal.out
grep "Total Conversations" /slurm/home/$USER/output/<JOBID>/*.log
```

Generation is fully resumable — if the job hits the time limit, resubmit with the same command and it continues from where it stopped.

After generation, preprocess:

```bash
python -m training.scripts.preprocess \
    --input data/raw \
    --output data/processed \
    --adversarial-ratio 0.10
```

Output: 495 000 train + 50 000 eval examples (+ 45 000 adversarial hardening in train).

### Single-category (local or cloud)

```bash
# vLLM (local H100)
python -m data.generation.scripts.generate \
    --category grooming --count 8000 --generator vllm --language mixed --resume

# Bedrock / Claude / OpenAI
python -m data.generation.scripts.generate \
    --category bullying --count 1000 --generator bedrock
```

### Dataset on HuggingFace

```bash
# Upload raw + processed to safecircleai/horizon-training-data
python data/scripts/upload_to_hub.py

# Upload only processed splits (faster)
python data/scripts/upload_to_hub.py --split processed

# Download
make download-data
```

---

## Training

| Config | GPU | VRAM | Steps | Notes |
|---|---|---|---|---|
| `h100.yaml` | H100 NVL | 96 GB | 25 000 | bf16, rank-256 LoRA, no 4-bit |
| `l4.yaml` | L4 | 24 GB | 15 000 | bf16, rank-128 LoRA, no 4-bit |
| `quick.yaml` | Any | 8 GB+ | 500 | 4-bit QLoRA, smoke test |
| `mobile.yaml` | L4 | 24 GB | 8 000 | Gemma 3 1B, 4-bit, for LiteRT-LM |

### SLURM (ANTS cluster)

> **Note:** Always use `--export=ALL,VAR=value` (with `ALL,` prefix) so SLURM inherits the full environment. `--export` must come **before** the script path.

```bash
# H100 NVL (partition: gpuMax)
sbatch slurm/train_h100.sbatch

# L4 (partition: gpu)
sbatch slurm/train_l4.sbatch

# Gemma 3 1B mobile (partition: gpu, slurm-gpu08 L40s)
sbatch slurm/train_mobile.sbatch

# Resume from a checkpoint
sbatch --export=ALL,RESUME_CHECKPOINT=experiments/h100-<timestamp>/checkpoint-5000 \
    slurm/train_h100.sbatch

# Monitor
squeue -u $USER
tail -f /slurm/home/$USER/output/<JOBID>/training.log
```

Checkpoints are saved as `experiments/<run>/checkpoint-<step>/` and synced to persistent storage every 10 minutes and on job exit.

### Local

```bash
python -m training.scripts.train --config training/configs/h100.yaml

# Quick smoke test
make train-quick

# Resume
python -m training.scripts.train \
    --config training/configs/h100.yaml \
    --resume experiments/h100-<timestamp>/checkpoint-5000
```

---

## Evaluation

Checkpoints use the format `checkpoint-<step>` (e.g. `checkpoint-25000`). The final merged checkpoint is at `final/`.

```bash
# SLURM  (--export must come before the script path)
sbatch --export=ALL,CHECKPOINT=experiments/h100-<timestamp>/checkpoint-25000 \
    slurm/evaluate.sbatch

# Or evaluate the final checkpoint
sbatch --export=ALL,CHECKPOINT=experiments/h100-<timestamp>/final \
    slurm/evaluate.sbatch

# Local
python -m evaluation.metrics.evaluate \
    --checkpoint experiments/h100-<timestamp>/checkpoint-25000 \
    --test-set data/processed/eval.jsonl \
    --output evaluation/reports
```

Results are written to `evaluation/reports/latest/results.json`.

---

## Mobile Model (LiteRT-LM)

The mobile model is Gemma 3 1B fine-tuned with QLoRA, exported in two quantization variants targeting different device capabilities.

| Variant | Quantization | Size | Target |
|---|---|---|---|
| `mobile-standard` | INT8 | ~1.2 GB | 6 GB+ RAM phones (mid-range 2022+) |
| `mobile-lite` | INT4 | ~0.7 GB | 4 GB RAM phones (budget/older) |

### 1. Train

```bash
sbatch slurm/train_mobile.sbatch
# or locally:
python -m training.scripts.train --config training/configs/mobile.yaml
```

### 2. Install export dependencies (workstation only)

```bash
uv pip install ai-edge-torch litert-lm-builder
```

### 3. Export both variants

```bash
# Standard (INT8, 6GB+ phones) — merges LoRA first
python -m training.scripts.export_litert \
    --checkpoint experiments/mobile-<timestamp>/final \
    --output models/mobile-standard \
    --quantization int8

# Lite (INT4, 4GB phones) — reuses merged/ from standard export
python -m training.scripts.export_litert \
    --checkpoint experiments/mobile-<timestamp>/final \
    --output models/mobile-lite \
    --quantization int4 \
    --skip-merge \
    --merged-dir models/mobile-standard/merged
```

Output:

```
models/mobile-standard/
    merged/                            # shared merged HF checkpoint
    model.tflite                       # INT8 TFLite flatbuffer
    horizon-mobile-int8.litertlm       # for 6GB+ phones

models/mobile-lite/
    model.tflite                       # INT4 TFLite flatbuffer
    horizon-mobile-int4.litertlm       # for 4GB phones
```

### 4. Test locally

```bash
uvx litert-lm run models/mobile-standard/horizon-mobile-int8.litertlm \
    --prompt "Analyse this conversation for risks"

uvx litert-lm run models/mobile-lite/horizon-mobile-int4.litertlm \
    --prompt "Analyse this conversation for risks"
```

---

## Merge & Upload

```bash
# Merge LoRA into base model (required before GGUF export or HF upload)
python scripts/merge_lora.py \
    --checkpoint experiments/h100-<timestamp>/final \
    --output models/horizon-full-merged

# Upload merged model to HuggingFace
python scripts/upload_to_hf.py --model models/horizon-full-merged

# Export to GGUF (requires llama.cpp)
bash scripts/export_gguf.sh models/horizon-full-merged models/horizon-full-merged/gguf
```

---

## Model Output Format

```json
{
  "risk_level": "high",
  "categories": ["grooming", "personal_info"],
  "confidence": 0.91
}
```

Off-task or jailbreak queries always return:

```json
{"error": "I only analyze conversations for child safety risks."}
```

---

## Risk Categories

| Category | Description |
|---|---|
| `grooming` | Trust building, boundary testing, secrecy requests |
| `bullying` | Harassment, threats, cyberbullying |
| `sexual_content` | Explicit messages, inappropriate requests |
| `isolation` | Controlling behavior, network isolation |
| `personal_info` | Requests for identifying information |
| `platform_migration` | Moving to less monitored platforms |
| `threats` | Violent threats, dangerous challenges |

---

## Requirements

- Python 3.13
- [uv](https://github.com/astral-sh/uv) package manager
- CUDA-capable GPU (H100/L4 for full runs, any 8 GB+ GPU for quick runs)
- HuggingFace token with `safecircleai` org access
- `ai-edge-torch` + `litert-lm-builder` (only for mobile export, install separately)

---

## License

[SafeCircle Internal License — see LICENSE-SAFECIRCLE.md]

## Contact

https://safecircle.tech

---

**Status:** In Development · **Version:** 0.3.0 · **Updated:** 2026-06-03
