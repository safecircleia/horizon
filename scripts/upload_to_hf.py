#!/usr/bin/env python3
"""Upload all Horizon models to HuggingFace Hub with model cards.

Usage:
    python scripts/upload_to_hf.py --what full      # upload merged full model
    python scripts/upload_to_hf.py --what gguf      # upload GGUF variants
    python scripts/upload_to_hf.py --what mobile    # upload mobile ONNX
    python scripts/upload_to_hf.py --what all       # upload everything

Prerequisites:
    huggingface-cli login   (or set HF_TOKEN env var)
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ORG = "safecircleia"

FULL_REPO = f"{ORG}/horizon-full"
GGUF_REPO = f"{ORG}/horizon-full-gguf"
MOBILE_REPO = f"{ORG}/horizon-mobile"


# ── Model cards ──────────────────────────────────────────────────────────────

FULL_MODEL_CARD = """\
---
license: llama3.2
language:
- en
tags:
- child-safety
- content-moderation
- risk-detection
- llama
- fine-tuned
base_model: meta-llama/Llama-3.2-3B-Instruct
pipeline_tag: text-generation
---

# Horizon Full — SafeCircle Risk Detection Model

Horizon Full is a fine-tuned [Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) model trained to detect child safety risks in online conversations. It classifies conversations across 7 risk categories with severity levels.

## Model Details

| Property | Value |
|----------|-------|
| Base model | meta-llama/Llama-3.2-3B-Instruct |
| Fine-tuning method | QLoRA (rank 256, all projection layers) |
| Training data | 50K synthetic conversations (8 categories) |
| Input | Conversation text |
| Output | JSON: risk_level, categories, confidence, reasoning |

## Risk Categories

| Category | Description |
|----------|-------------|
| `grooming` | Predatory relationship-building patterns |
| `bullying` | Cyberbullying and harassment |
| `sexual_content` | Inappropriate sexual content or solicitation |
| `isolation` | Attempts to isolate child from support network |
| `personal_info` | Solicitation of personal/location information |
| `platform_migration` | Attempts to move to less-monitored platforms |
| `threats` | Explicit or implicit threats |
| `benign` | Normal, safe conversation |

## Severity Levels

`none` → `low` → `medium` → `high` → `critical`

## Evaluation Results (5,000 examples)

| Metric | Score |
|--------|-------|
| Binary F1 | 0.983 |
| False Positive Rate | 2.80% |
| False Negative Rate | 1.01% |
| Grooming F1 | 0.909 |
| Bullying F1 | 0.994 |
| Sexual Content F1 | 0.952 |
| Isolation F1 | 0.990 |
| Personal Info F1 | 0.935 |
| Platform Migration F1 | 0.998 |
| Threats F1 | 0.979 |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json

model = AutoModelForCausalLM.from_pretrained("safecircleia/horizon-full", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("safecircleia/horizon-full")

SYSTEM_PROMPT = "You are SafeCircle's risk detection model. Analyze conversations for child safety risks. Output JSON with: risk_level (none/low/medium/high/critical), categories (array), confidence (0-1), matched_terms (array), reasoning (brief)."

conversation = "Child: Hey, what are you doing later?\\nOther: Nothing much. Want to meet up? Don't tell your parents."

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": f"Analyze this conversation:\\n{conversation}"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)

generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
result = json.loads(generated.split("\\nassistant")[0].strip())
print(result)
# {'risk_level': 'high', 'categories': ['grooming'], 'confidence': 0.91, ...}
```

## Training Details

- **Hardware**: NVIDIA H100 80GB
- **Training time**: ~4 hours
- **Optimizer**: AdamW (fused)
- **Learning rate**: 1.5e-4 with cosine restarts
- **Batch size**: 32 (effective)
- **LoRA rank**: 256, alpha: 512

## Intended Use

This model is designed for child safety monitoring in chat platforms. It is intended to assist human moderators — not replace them. All flagged conversations should be reviewed by trained safety professionals.

## Limitations

- Trained on synthetic data; real-world performance may differ
- English-only
- May miss subtle, long-form grooming patterns
- Not designed for audio, video, or image content

## Citation

```bibtex
@misc{horizon2025,
  title={Horizon: Child Safety Risk Detection via Fine-tuned LLMs},
  author={SafeCircle},
  year={2025},
  url={https://huggingface.co/safecircleia/horizon-full}
}
```
"""

GGUF_MODEL_CARD = """\
---
license: llama3.2
language:
- en
tags:
- child-safety
- content-moderation
- risk-detection
- gguf
- llama
- quantized
base_model: safecircleia/horizon-full
---

# Horizon Full GGUF — SafeCircle Risk Detection Model

GGUF quantized variants of [safecircleia/horizon-full](https://huggingface.co/safecircleia/horizon-full) for use with [llama.cpp](https://github.com/ggerganov/llama.cpp), [Ollama](https://ollama.ai), and compatible runtimes.

## Available Files

| File | Quantization | Size | Use case |
|------|-------------|------|----------|
| `horizon-full-Q4_K_M.gguf` | Q4_K_M | ~2.0 GB | Best quality/size tradeoff — recommended |
| `horizon-full-Q5_K_M.gguf` | Q5_K_M | ~2.3 GB | Higher quality, slightly larger |
| `horizon-full-Q8_0.gguf` | Q8_0 | ~3.3 GB | Near-lossless, largest |
| `horizon-full-f16.gguf` | F16 | ~6.0 GB | Full precision reference |

## Usage with llama.cpp

```bash
./llama-cli \\
  -m horizon-full-Q4_K_M.gguf \\
  --system-prompt "You are SafeCircle's risk detection model. Analyze conversations for child safety risks. Output JSON with: risk_level (none/low/medium/high/critical), categories (array), confidence (0-1), matched_terms (array), reasoning (brief)." \\
  -p "Analyze this conversation:\\nChild: Hey want to meet up? Keep it secret from your parents.\\nOther: Sure, I know a quiet place." \\
  --temp 0 -n 256
```

## Usage with Ollama

```bash
ollama pull safecircleia/horizon-full-gguf
ollama run safecircleia/horizon-full-gguf
```

## Performance

See [safecircleia/horizon-full](https://huggingface.co/safecircleia/horizon-full) for full evaluation metrics.
Q4_K_M shows <1% degradation vs F16 on the risk detection benchmark.
"""

MOBILE_MODEL_CARD = """\
---
license: apache-2.0
language:
- en
tags:
- child-safety
- content-moderation
- onnx
- mobile
- on-device
- mobilebert
pipeline_tag: text-classification
base_model: google/mobilebert-uncased
---

# Horizon Mobile — On-Device Child Safety Classifier

Horizon Mobile is a lightweight binary risk classifier designed for on-device inference on iOS and Android. It is distilled from [safecircleia/horizon-full](https://huggingface.co/safecircleia/horizon-full) using knowledge distillation.

## Model Details

| Property | Value |
|----------|-------|
| Base model | google/mobilebert-uncased |
| Parameters | ~25M |
| Format | ONNX (float32) |
| Input | Conversation text (max 512 tokens) |
| Output | `safe` / `risk` label + confidence score |
| Inference time | ~25ms on CPU |

## Evaluation Results (5,000 examples)

| Metric | Score |
|--------|-------|
| F1 | 0.9562 |
| Precision | 0.9997 |
| Recall | 0.9163 |
| False Positive Rate | 0.10% (1/1034) |
| False Negative Rate | 8.37% (332/3966) |

## Architecture

MobileBERT encoder → pooled CLS representation → Linear(512, 2) → softmax

Binary classification: `safe` (0) vs `risk` (1)

## Intended Deployment Pattern

```
User message
     │
     ▼
Horizon Mobile (on-device, ~25ms)
     │
     ├── safe ──► No action
     │
     └── risk ──► Horizon Full API (category + severity)
                        │
                        ▼
                  Human moderator review
```

## Usage with ONNX Runtime

```python
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

session = ort.InferenceSession("horizon-mobile.onnx")
tokenizer = AutoTokenizer.from_pretrained("safecircleia/horizon-mobile")

conversation = "Child: Can we meet? Don't tell your parents."
enc = tokenizer(conversation, return_tensors="np", truncation=True, max_length=512, padding=False)

logits, = session.run(None, {
    "input_ids": enc["input_ids"].astype(np.int64),
    "attention_mask": enc["attention_mask"].astype(np.int64),
})

label = ["safe", "risk"][logits[0].argmax()]
confidence = float(np.exp(logits[0]) / np.exp(logits[0]).sum()).max()
print(f"{label} ({confidence:.2%})")
# risk (94.3%)
```

## Limitations

- Binary only — no category or severity classification (use Horizon Full for that)
- English only
- FNR of 8.37% means ~1 in 12 risk conversations are missed — always pair with Horizon Full
- Not suitable as a standalone safety system
"""


# ── Upload functions ──────────────────────────────────────────────────────────

def ensure_repo(api: HfApi, repo_id: str, repo_type: str = "model"):
    try:
        create_repo(repo_id, repo_type=repo_type, exist_ok=True, private=False)
        print(f"  Repo ready: {repo_id}")
    except Exception as e:
        print(f"  Warning: {e}")


def upload_full_model(api: HfApi, model_dir: str):
    print(f"\n==> Uploading full model to {FULL_REPO}")
    ensure_repo(api, FULL_REPO)

    api.upload_file(
        path_or_fileobj=FULL_MODEL_CARD.encode(),
        path_in_repo="README.md",
        repo_id=FULL_REPO,
        commit_message="Add model card",
    )
    print("  Uploading model files (this will take a few minutes)...")
    api.upload_folder(
        folder_path=model_dir,
        repo_id=FULL_REPO,
        commit_message="Upload merged Horizon Full model",
        ignore_patterns=["*.py", "*.sh"],
    )
    print(f"  Done: https://huggingface.co/{FULL_REPO}")


def upload_gguf_models(api: HfApi, gguf_dir: str):
    print(f"\n==> Uploading GGUF variants to {GGUF_REPO}")
    ensure_repo(api, GGUF_REPO)

    api.upload_file(
        path_or_fileobj=GGUF_MODEL_CARD.encode(),
        path_in_repo="README.md",
        repo_id=GGUF_REPO,
        commit_message="Add model card",
    )

    gguf_path = Path(gguf_dir)
    for gguf_file in sorted(gguf_path.glob("*.gguf")):
        size_gb = gguf_file.stat().st_size / 1e9
        print(f"  Uploading {gguf_file.name} ({size_gb:.1f} GB)...")
        api.upload_file(
            path_or_fileobj=str(gguf_file),
            path_in_repo=gguf_file.name,
            repo_id=GGUF_REPO,
            commit_message=f"Upload {gguf_file.name}",
        )
    print(f"  Done: https://huggingface.co/{GGUF_REPO}")


def upload_mobile_model(api: HfApi, mobile_dir: str):
    print(f"\n==> Uploading mobile ONNX model to {MOBILE_REPO}")
    ensure_repo(api, MOBILE_REPO)

    api.upload_file(
        path_or_fileobj=MOBILE_MODEL_CARD.encode(),
        path_in_repo="README.md",
        repo_id=MOBILE_REPO,
        commit_message="Add model card",
    )
    api.upload_folder(
        folder_path=mobile_dir,
        repo_id=MOBILE_REPO,
        commit_message="Upload Horizon Mobile ONNX model",
    )
    print(f"  Done: https://huggingface.co/{MOBILE_REPO}")


def main():
    parser = argparse.ArgumentParser(description="Upload Horizon models to HuggingFace")
    parser.add_argument("--what", choices=["full", "gguf", "mobile", "all"], default="all")
    parser.add_argument("--full-model-dir", default="models/horizon-full-merged")
    parser.add_argument("--gguf-dir", default="models/horizon-full-gguf")
    parser.add_argument("--mobile-dir", default="models/mobile")
    args = parser.parse_args()

    api = HfApi()

    try:
        user = api.whoami()
        print(f"Logged in as: {user['name']}")
    except Exception:
        print("ERROR: Not logged in. Run: huggingface-cli login")
        return

    if args.what in ("full", "all"):
        if not Path(args.full_model_dir).exists():
            print(f"Full model not found at {args.full_model_dir}")
            print("Run first: python scripts/merge_lora.py --checkpoint experiments/h100-20260514-134855/final")
        else:
            upload_full_model(api, args.full_model_dir)

    if args.what in ("gguf", "all"):
        if not list(Path(args.gguf_dir).glob("*.gguf")) if Path(args.gguf_dir).exists() else True:
            print(f"No GGUF files found in {args.gguf_dir}")
            print("Run first: bash scripts/export_gguf.sh")
        else:
            upload_gguf_models(api, args.gguf_dir)

    if args.what in ("mobile", "all"):
        if not Path(args.mobile_dir).exists():
            print(f"Mobile model not found at {args.mobile_dir}")
        else:
            upload_mobile_model(api, args.mobile_dir)

    print("\nAll uploads complete.")


if __name__ == "__main__":
    main()
