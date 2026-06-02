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

# Generate dataset (see Data Generation section)
bash data/generation/scripts/generate_bulk.sh

# Preprocess raw data into training format
python -m training.scripts.preprocess --input data/raw --output data/processed

# Train (local GPU)
python -m training.scripts.train --config training/configs/h100.yaml

# Evaluate a checkpoint
make evaluate CHECKPOINT=experiments/h100-<timestamp>/checkpoints/step-5000
```

---

## Project Structure

```
horizon/
├── data/
│   ├── raw/                   # Per-category JSONL (generated)
│   ├── processed/             # train.jsonl + eval.jsonl (TRL messages format)
│   ├── generation/            # Generation pipeline (generators, prompts, scripts)
│   └── scripts/               # upload_to_hub.py, download_from_hub.py
├── training/
│   ├── configs/               # h100.yaml, l4.yaml, quick.yaml, mobile.yaml
│   ├── model/                 # loader.py (QLoRA), mobile.py (constants)
│   └── scripts/               # preprocess.py, train.py, export_litert.py
├── evaluation/                # evaluate.py, plot_results.py
├── slurm/                     # SLURM sbatch scripts for ANTS cluster
├── lib/                       # install_env.sh (idempotent cluster env setup)
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

### Local vLLM (recommended for large runs)

```bash
# 1. Start vLLM server on the H100
vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.92 \
    --port 8000

# 2. Generate all 8 categories (~50K examples total)
COUNT=200000 CONCURRENCY=150 bash data/generation/scripts/generate_bulk.sh

# Resume a partial run
bash data/generation/scripts/generate_bulk.sh --resume
```

| Model | VRAM | ~tok/s | Quality | 50K ETA |
|---|---|---|---|---|
| `Qwen2.5-72B-Instruct-AWQ` | 40 GB | 4–5k | High | ~56h |
| `Qwen2.5-7B-Instruct` | 16 GB | 18k | Good | ~14h |

### Single-category generation

```bash
python -m data.generation.scripts.generate \
    --category grooming \
    --count 8000 \
    --generator bedrock \
    --resume
```

### Cloud generators

```bash
# Bedrock (default), Claude, or OpenAI
python -m data.generation.scripts.generate \
    --category bullying --count 1000 --generator bedrock
```

### Dataset on HuggingFace

The processed dataset is hosted privately at [`safecircleai/horizon-training-data`](https://huggingface.co/datasets/safecircleai/horizon-training-data).

```bash
# Download to data/processed/
make download-data

# Upload after generating locally
make upload-data
```

---

## Preprocessing

Converts raw JSONL → TRL `messages` format and injects adversarial hardening examples:

```bash
python -m training.scripts.preprocess \
    --input data/raw \
    --output data/processed \
    --adversarial-ratio 0.10
```

---

## Training

Three configs are available — pick based on your GPU:

| Config | GPU | VRAM | Steps | Notes |
|---|---|---|---|---|
| `h100.yaml` | H100 NVL | 96 GB | 25 000 | bf16, rank-256 LoRA, no 4-bit |
| `l4.yaml` | L4 | 24 GB | 15 000 | bf16, rank-128 LoRA, no 4-bit |
| `quick.yaml` | Any | 8 GB+ | 500 | 4-bit QLoRA, smoke test |
| `mobile.yaml` | L4 | 24 GB | 8 000 | Gemma 3 1B, 4-bit, for LiteRT-LM |

### Local

```bash
# Full run
python -m training.scripts.train --config training/configs/h100.yaml

# Quick smoke test
make train-quick

# Resume from checkpoint
python -m training.scripts.train \
    --config training/configs/h100.yaml \
    --resume experiments/h100-<timestamp>/checkpoints/step-5000
```

### SLURM (ANTS cluster)

Copy the project to your cluster home first:

```bash
rsync -av --exclude='.venv' --exclude='data/raw' . \
    $USER@cluster:/slurm/home/$USER/safecircle/horizon/
```

Then submit from `/slurm/home/$USER/safecircle/horizon`:

```bash
# H100 NVL — 48h limit
sbatch slurm/train_h100.sbatch

# L4 — 24h limit
sbatch slurm/train_l4.sbatch

# Gemma 3 1B mobile — 12h limit
sbatch slurm/train_mobile.sbatch

# Resume from a checkpoint
sbatch slurm/train_h100.sbatch \
    --export=RESUME_CHECKPOINT=experiments/h100-<timestamp>/checkpoints/step-5000

# Monitor
squeue -u $USER
tail -f /slurm/home/$USER/output/<JOBID>/terminal.out
```

Checkpoints are synced back to `experiments/` automatically when the job ends (including on timeout).

---

## Evaluation

```bash
make evaluate CHECKPOINT=experiments/h100-<timestamp>/checkpoints/step-5000

# Or directly
python -m evaluation.metrics.evaluate \
    --checkpoint experiments/h100-<timestamp>/checkpoints/step-5000 \
    --test-set data/processed/eval.jsonl \
    --output evaluation/reports
```

Results are written to `evaluation/reports/latest/results.json`.

---

## Mobile Model (LiteRT-LM)

The mobile model is Gemma 3 1B fine-tuned with QLoRA, then exported to a `.litertlm` container for on-device inference via [LiteRT-LM](https://ai.google.dev/edge/litert-lm/overview) on Android and iOS.

### 1. Train

```bash
# On cluster
sbatch slurm/train_mobile.sbatch

# Local
python -m training.scripts.train --config training/configs/mobile.yaml
```

### 2. Install export dependencies (workstation only)

```bash
uv pip install ai-edge-torch litert-lm-builder
```

### 3. Export to `.litertlm`

```bash
python -m training.scripts.export_litert \
    --checkpoint experiments/mobile-<timestamp>/final \
    --output models/mobile
```

Output:

```
models/mobile/
    merged/                   # merged HF checkpoint
    model.tflite              # INT8 quantised TFLite
    horizon-mobile.litertlm   # deployable container
```

### 4. Test locally

```bash
uvx litert-lm run models/mobile/horizon-mobile.litertlm \
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

**Status:** In Development · **Version:** 0.2.0 · **Updated:** 2026-06-02
