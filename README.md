# Project Horizon

**SafeCircle Risk Detection Model Training System**

Project Horizon trains custom AI models for privacy-preserving child safety risk detection. Fine-tuned from Llama 3.2 3B Instruct using QLoRA, the model detects seven risk categories (grooming, bullying, sexual content, isolation, personal info, platform migration, threats) and returns structured JSON — no free-text, no identity leakage.

## Quick Start

```bash
# Install dependencies (requires uv)
uv sync

# Generate dataset locally on H100 (see Data Generation section)
bash data/generation/scripts/generate_bulk.sh

# Preprocess raw data into training format
python -m training.scripts.preprocess --input data/raw --output data/processed

# Train
python -m training.scripts.train --config training/configs/h100.yaml

# Evaluate
make evaluate CHECKPOINT=experiments/latest/checkpoints/step-5000

# Quantize for deployment
make quantize MODEL=models/v1/full FORMAT=q4_k_m
```

## Project Structure

```
horizon/
├── data/
│   ├── raw/               # Per-category JSONL (generated)
│   ├── processed/         # train.jsonl + eval.jsonl (TRL messages format)
│   └── generation/        # Generation pipeline (generators, prompts, scripts)
├── training/
│   ├── configs/           # h100.yaml, l4.yaml, quick.yaml
│   ├── model/             # Model loading (QLoRA, 4-bit)
│   └── scripts/           # preprocess.py, train.py, distill.py
├── evaluation/            # Metrics, reports
├── quantization/          # GGUF / ONNX export
├── inference/             # API, CLI
├── experiments/           # Training runs & checkpoints
├── models/                # Trained model artifacts
└── scripts/               # merge_lora.py, upload_to_hf.py, export_gguf.sh
```

## Key Features

- **Privacy-First:** Synthetic data only, no real child messages
- **Identity-Hardened:** 4 500+ adversarial jailbreak examples baked into training; the model always responds with a locked JSON error to off-task queries
- **Correct Loss Masking:** TRL `SFTTrainer` with `assistant_only_loss=True` — loss computed only on assistant completions, not on system/user tokens
- **Local Data Generation:** vLLM generator for H100 (~4–18k tok/s), no cloud API costs for large runs
- **Production-Ready:** Quantized INT4 models for flexible deployment
- **Deployment-Agnostic:** Exports to GGUF, ONNX, vLLM, Ollama formats

## Data Generation

### Local vLLM (recommended for large runs)

Run everything on the H100 at zero API cost:

```bash
# 1. Start vLLM server
vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.92 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --port 8000

# 2. Generate ~100GB across all 8 categories (parallelised)
COUNT=200000 CONCURRENCY=150 bash data/generation/scripts/generate_bulk.sh

# Resume a partial run
bash data/generation/scripts/generate_bulk.sh --resume
```

**Model tradeoffs:**

| Model | VRAM | ~tok/s | Quality | 100GB ETA |
|---|---|---|---|---|
| `Qwen2.5-72B-Instruct-AWQ` | 40GB | 4–5k | High | ~56h |
| `Qwen2.5-7B-Instruct` | 16GB | 18k | Good | ~14h |

Switch model in `data/generation/config.yaml` → `model_vllm`.

### Single-category generation

```bash
python -m data.generation.scripts.generate \
    --category grooming \
    --count 50000 \
    --generator vllm \
    --output data/raw/grooming.jsonl \
    --resume
```

### Cloud generators (smaller runs)

```bash
# Claude / OpenAI / Amazon Bedrock
python -m data.generation.scripts.generate \
    --category bullying --count 1000 --generator bedrock
```

## Training

### Preprocess

Converts raw JSONL → TRL `messages` format and injects adversarial hardening examples:

```bash
python -m training.scripts.preprocess \
    --input data/raw \
    --output data/processed \
    --adversarial-ratio 0.10   # 10% jailbreak hardening examples
```

### Train (H100)

```bash
python -m training.scripts.train --config training/configs/h100.yaml
```

Configs available: `h100.yaml` (80GB, bf16), `l4.yaml` (24GB, QLoRA 4-bit), `quick.yaml` (smoke test).

## Dataset

Training data is stored as a private HuggingFace dataset at [`safecircleai/horizon-training-data`](https://huggingface.co/datasets/safecircleai/horizon-training-data).

**Access requires membership in the [safecircleai HuggingFace org](https://huggingface.co/safecircleai).**

```python
from datasets import load_dataset

ds = load_dataset("safecircleai/horizon-training-data", name="processed", token="hf_...")
train, eval = ds["train"], ds["eval"]
```

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- CUDA-capable GPU (H100 for full runs, L4/A10G for QLoRA, CPU for smoke tests)
- HuggingFace token with SafeCircle org access (dataset download/upload)
- vLLM installed on the training server (for local data generation)

## Model Output Format

```json
{
  "risk_level": "high",
  "categories": ["grooming", "personal_info"],
  "confidence": 0.91,
  "matched_terms": [],
  "reasoning": "Adult establishing secrecy while requesting personal media"
}
```

Off-task or jailbreak queries always return:

```json
{"error": "I only analyze conversations for child safety risks."}
```

## Risk Categories

1. **Grooming** — Trust building, boundary testing, secrecy requests
2. **Bullying** — Harassment, threats, cyberbullying
3. **Sexual Content** — Explicit messages, inappropriate requests
4. **Isolation/Control** — Controlling behavior, network isolation
5. **Personal Info** — Requests for identifying information
6. **Platform Migration** — Moving to less monitored platforms
7. **Threats/Violence** — Violent threats, dangerous challenges

## License

[To be determined — pending SafeCircle legal review]

## Contact

For questions about SafeCircle or Project Horizon, visit https://safecircle.tech

---

**Status:** In Development  
**Version:** 0.1.0 (Pre-release)  
**Last Updated:** 2026-05-16
