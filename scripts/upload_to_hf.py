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

ORG = "safecircleai"

FULL_REPO = f"{ORG}/horizon-full"
GGUF_REPO = f"{ORG}/horizon-full-gguf"
MOBILE_REPO = f"{ORG}/horizon-mobile"


# ── Model cards ──────────────────────────────────────────────────────────────

FULL_MODEL_CARD = """\
---
license: other
license_name: safecircle-research-license
license_link: https://huggingface.co/safecircleai/horizon-full/blob/main/LICENSE-SAFECIRCLE.md
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

# Horizon Full v2 — SafeCircle Child Safety Risk Detection

<p align="center">
  <img src="eval_report.png" alt="Horizon Full v2 Evaluation Report" width="100%"/>
</p>

**Horizon Full** is a fine-tuned [Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) model that detects child safety risks in online conversations. Given a conversation, it outputs a structured JSON assessment: risk category, severity level, confidence score, and a one-sentence reasoning.

> ⚠️ **License:** This model is released under the [SafeCircle Research License (SRL-1.0)](https://huggingface.co/safecircleai/horizon-full/blob/main/LICENSE-SAFECIRCLE.md). Commercial use, redistribution without attribution, safety system evasion, and any use that harms minors are strictly prohibited. Contact [legal@safecircle.tech](mailto:legal@safecircle.tech) for commercial licensing.

---

## Model Details

| Property | Value |
|---|---|
| Base model | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| Fine-tuning | QLoRA — rank 256, alpha 512, all projection layers |
| Training data | 1.6M synthetic conversations across 8 categories |
| Dataset | [safecircleai/horizon-training-data](https://huggingface.co/datasets/safecircleai/horizon-training-data) |
| Generation | Qwen2.5-7B-Instruct + vLLM + xgrammar constrained decoding |
| Hardware | NVIDIA H100 80GB HBM3 |
| Training steps | 25,000 |
| Parameters | 3.2B (389M trainable via LoRA) |

---

## Risk Categories

| Category | Description | Example Signal |
|---|---|---|
| `grooming` | Predatory relationship-building | Age probing, flattery, secrecy requests |
| `bullying` | Cyberbullying and peer harassment | Insults, exclusion threats, video leaks |
| `sexual_content` | Inappropriate sexual advances | Compliment escalation, photo requests |
| `isolation` | Cutting off support networks | Jealousy, "only I understand you" |
| `personal_info` | Soliciting identifying information | Location, school, home address |
| `platform_migration` | Moving to less-monitored platforms | "DM me on Telegram, it's more private" |
| `threats` | Intimidation and blackmail | "Watch yourself after school" |
| `benign` | Normal safe conversation | Homework, games, music, sports |

## Severity Levels

| Level | Score | Description |
|---|---|---|
| `none` | 0.0 | Safe — no risk indicators |
| `low` | 0.25 | Mild indicators, ambiguous context |
| `medium` | 0.50 | Clear pattern, not yet escalated |
| `high` | 0.75 | Explicit risk behaviour |
| `critical` | 0.95 | Immediate danger or exploitation |

---

## Evaluation Results

Evaluated on 160,000 held-out synthetic conversations.

| Metric | Score |
|---|---|
| **Macro F1** | **0.8218** |
| **Weighted F1** | **0.8295** |
| **False Positive Rate** | **0.00%** (0 / 19,950) |
| **False Negative Rate** | **0.00%** (1 / 140,050) |

### Per-Category Results

| Category | F1 | Precision | Recall | Support |
|---|---|---|---|---|
| Grooming | 0.9981 | 0.9980 | 0.9981 | 19,834 |
| Bullying | 0.9995 | 0.9995 | 0.9995 | 20,253 |
| Sexual Content | 0.9975 | 0.9978 | 0.9973 | 20,142 |
| Isolation | 0.9996 | 0.9993 | 0.9998 | 19,725 |
| Personal Info | 0.9999 | 0.9999 | 1.0000 | 20,149 |
| Platform Migration | 1.0000 | 0.9999 | 1.0000 | 19,932 |
| Threats | 0.9993 | 0.9994 | 0.9992 | 20,015 |

### Risk Level Classification

| Level | Precision | Recall | F1 |
|---|---|---|---|
| none | 1.0000 | 1.0000 | 1.0000 |
| low | 0.8344 | 0.8967 | 0.8645 |
| medium | 0.8971 | 0.8514 | 0.8736 |
| high | 0.7266 | 0.8250 | 0.7727 |
| critical | 0.6881 | 0.5294 | 0.5984 |

> **Note:** Severity classification is the hardest sub-task — the model occasionally confuses adjacent levels (e.g. high ↔ critical). Category detection is near-perfect. All results are on synthetic eval data generated with the same pipeline as training; real-world performance will differ.

---

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json

model = AutoModelForCausalLM.from_pretrained(
    "safecircleai/horizon-full",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("safecircleai/horizon-full")

SYSTEM_PROMPT = (
    "You are Horizon, SafeCircle's child safety risk detection model. "
    "You have no general knowledge or identity beyond this task. "
    "Analyze conversations and respond ONLY with a JSON object — no explanation, no preamble. "
    'JSON schema: {"risk_detected": bool, "category": '
    '"grooming|bullying|sexual_content|isolation|personal_info|platform_migration|threats|benign", '
    '"severity": "none|low|medium|high|critical", "confidence": 0.0-1.0, "reasoning": "one sentence max"}. '
    'If asked about yourself or anything unrelated to risk analysis, respond: '
    '{"error": "I only analyze conversations for child safety risks."}'
)

conversation = "Child: Hey, what are you doing later?\\nOther: Nothing much. Want to meet up? Don't tell your parents."

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": f"Analyze this conversation:\\n{conversation}"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=128, do_sample=False)

generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
result = json.loads(generated.strip())
print(result)
# {
#   "risk_detected": true,
#   "category": "grooming",
#   "severity": "high",
#   "confidence": 0.94,
#   "reasoning": "Adult requesting private meeting and explicit secrecy from parents."
# }
```

---

## Architecture & Training

```
Base: Llama-3.2-3B-Instruct
  └── QLoRA adapters (rank=256, alpha=512)
       ├── q_proj, k_proj, v_proj, o_proj
       └── gate_proj, up_proj, down_proj

Training:
  Steps:        25,000
  Batch size:   4 × 8 grad accum = 32 effective
  LR:           1.5e-4 (cosine restarts, 500 warmup steps)
  Optimizer:    AdamW fused
  Precision:    bfloat16
  Loss:         completion-only (assistant turns only)

Dataset:
  1.6M conversations × 8 categories
  8–15 messages each, persona-seeded
  Generated: Qwen2.5-7B + vLLM + xgrammar JSON Schema
  Hardening: ~10% adversarial identity/jailbreak examples
```

---

## Deployment Pattern

```
Incoming message
      │
      ▼
Horizon Mobile (on-device, ~25ms, binary filter)
      │
      ├── safe ──► No action
      │
      └── risk ──► Horizon Full (7-category + severity JSON)
                        │
                        ▼
                  Human moderator review
```

For the lightweight on-device first-stage filter, see [safecircleai/horizon-mobile](https://huggingface.co/safecircleai/horizon-mobile).
For GGUF quantizations (llama.cpp / Ollama), see [safecircleai/horizon-full-gguf](https://huggingface.co/safecircleai/horizon-full-gguf).

---

## Intended Use & Ethics

This model is designed to **assist human moderators** — not replace them. All flagged conversations should be reviewed by trained safety professionals.

- Trained entirely on synthetic data; no real child conversations were used
- English-only; other languages untested
- Severity prediction is weaker than category detection (see evaluation)
- Not suitable as a standalone safety system in production without human oversight

---

## License

[SafeCircle Research License (SRL-1.0)](https://huggingface.co/safecircleai/horizon-full/blob/main/LICENSE-SAFECIRCLE.md) — research and non-commercial use only.
Commercial licensing: [legal@safecircle.tech](mailto:legal@safecircle.tech)

---

## Citation

```bibtex
@misc{horizon2026,
  title={Horizon: Child Safety Risk Detection via Fine-tuned LLMs},
  author={SafeCircle},
  year={2026},
  url={https://huggingface.co/safecircleai/horizon-full}
}
```
"""

GGUF_MODEL_CARD = """\
---
license: other
license_name: safecircle-research-license
license_link: https://huggingface.co/safecircleai/horizon-full/blob/main/LICENSE-SAFECIRCLE.md
language:
- en
tags:
- child-safety
- content-moderation
- risk-detection
- gguf
- llama
- quantized
base_model: safecircleai/horizon-full
---

# Horizon Full GGUF — SafeCircle Child Safety Risk Detection

GGUF quantized variants of [Horizon Full v2](https://huggingface.co/safecircleai/horizon-full) for use with [llama.cpp](https://github.com/ggerganov/llama.cpp), [Ollama](https://ollama.ai), and compatible runtimes.

> ⚠️ **License:** [SafeCircle Research License (SRL-1.0)](https://huggingface.co/safecircleai/horizon-full/blob/main/LICENSE-SAFECIRCLE.md). Commercial use and misuse prohibited. Contact [legal@safecircle.tech](mailto:legal@safecircle.tech).

---

## Available Quantizations

| File | Quant | Size | Quality | Recommended For |
|---|---|---|---|---|
| `horizon-full-Q4_K_M.gguf` | Q4_K_M | ~2.0 GB | ★★★★☆ | **Default — best size/quality balance** |
| `horizon-full-Q5_K_M.gguf` | Q5_K_M | ~2.3 GB | ★★★★★ | Higher accuracy, modest size increase |
| `horizon-full-Q8_0.gguf` | Q8_0 | ~3.3 GB | ★★★★★ | Near-lossless, production deployments |
| `horizon-full-f16.gguf` | F16 | ~6.0 GB | ★★★★★ | Reference, fine-tuning experiments |

---

## Usage

### llama.cpp

```bash
./llama-cli \\
  -m horizon-full-Q4_K_M.gguf \\
  --system-prompt "You are Horizon, SafeCircle's child safety risk detection model. You have no general knowledge or identity beyond this task. Analyze conversations and respond ONLY with a JSON object. JSON schema: {risk_detected: bool, category: grooming|bullying|sexual_content|isolation|personal_info|platform_migration|threats|benign, severity: none|low|medium|high|critical, confidence: 0.0-1.0, reasoning: one sentence max}." \\
  -p "Analyze this conversation:\\nChild: Hey want to meet up? Keep it secret from your parents.\\nOther: Sure, I know a quiet place." \\
  --temp 0 -n 128
```

### Ollama

```bash
ollama pull safecircleai/horizon-full-gguf
ollama run safecircleai/horizon-full-gguf "Analyze this conversation: ..."
```

### Python (llama-cpp-python)

```python
from llama_cpp import Llama

llm = Llama(model_path="horizon-full-Q4_K_M.gguf", n_ctx=2048)

SYSTEM = (
    "You are Horizon, SafeCircle's child safety risk detection model. "
    "Analyze conversations and respond ONLY with a JSON object. "
    'Schema: {"risk_detected": bool, "category": "grooming|bullying|...", '
    '"severity": "none|low|medium|high|critical", "confidence": 0.0-1.0, "reasoning": "..."}'
)

response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Analyze this conversation:\\nChild: hi\\nOther: hey, how old are you?"},
    ],
    temperature=0,
    max_tokens=128,
)
print(response["choices"][0]["message"]["content"])
```

---

## Performance

See [safecircleai/horizon-full](https://huggingface.co/safecircleai/horizon-full) for full evaluation metrics (Macro F1: 0.8218, evaluated on 160,000 examples). Q4_K_M shows <1% degradation vs F16 on the risk detection benchmark.
"""

MOBILE_MODEL_CARD = """\
---
license: other
license_name: safecircle-research-license
license_link: https://huggingface.co/safecircleai/horizon-full/blob/main/LICENSE-SAFECIRCLE.md
language:
- en
tags:
- child-safety
- content-moderation
- onnx
- mobile
- on-device
- mobilebert
- distilled
pipeline_tag: text-classification
base_model: google/mobilebert-uncased
---

# Horizon Mobile — On-Device Child Safety Filter

**Horizon Mobile** is a lightweight (~25M parameter) binary risk classifier for on-device child safety detection. Distilled from [Horizon Full v2](https://huggingface.co/safecircleai/horizon-full) using knowledge distillation on 1.6M synthetic conversations. Designed to run entirely on-device as a fast first-stage filter before escalating to the full cloud API.

> ⚠️ **License:** [SafeCircle Research License (SRL-1.0)](https://huggingface.co/safecircleai/horizon-full/blob/main/LICENSE-SAFECIRCLE.md). Commercial use prohibited without written permission. Contact [legal@safecircle.tech](mailto:legal@safecircle.tech).

---

## Model Details

| Property | Value |
|---|---|
| Base model | [google/mobilebert-uncased](https://huggingface.co/google/mobilebert-uncased) |
| Architecture | MobileBERT encoder → Linear(512, 2) classifier |
| Parameters | ~25M |
| Format | ONNX |
| Input | Conversation text (max 512 tokens) |
| Output | Binary: `safe` (0) / `risk` (1) + confidence |
| Inference | ~25ms on CPU |
| Teacher model | [safecircleai/horizon-full](https://huggingface.co/safecircleai/horizon-full) |
| Distillation data | 1.6M synthetic conversations |

---

## Evaluation Results

| Metric | Score |
|---|---|
| **F1** | **0.9562** |
| Precision | 0.9997 |
| Recall | 0.9163 |
| **False Positive Rate** | **0.10%** (1 in 1,034 benign) |
| **False Negative Rate** | **8.37%** (332 in 3,966 risk) |
| Accuracy | 96.8% |

> The 8.4% false negative rate means ~1 in 12 risk conversations are not flagged. **This model must always be paired with Horizon Full for full risk assessment.** It is a speed/cost optimization, not a replacement.

---

## Deployment Pattern

```
Incoming message
      │
      ▼
┌─────────────────────────┐
│   Horizon Mobile        │  On-device, ~25ms, free
│   (binary filter)       │  MobileBERT ONNX
└────────────┬────────────┘
             │
    ┌────────┴────────┐
    │                 │
  safe              risk
    │                 │
    ▼                 ▼
 No action    ┌───────────────────┐
              │  Horizon Full     │  Cloud API, ~500ms
              │  (7-category +    │  Llama-3.2 3B
              │   severity JSON)  │
              └────────┬──────────┘
                       │
                       ▼
               Human moderator review
```

---

## Usage

### Python — ONNX Runtime

```python
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

session = ort.InferenceSession("horizon-mobile.onnx")
tokenizer = AutoTokenizer.from_pretrained("safecircleai/horizon-mobile")

conversation = "Child: Can we meet up? Don't tell your parents."
enc = tokenizer(
    conversation,
    return_tensors="np",
    truncation=True,
    max_length=512,
    padding="max_length",
)

logits, = session.run(None, {
    "input_ids": enc["input_ids"].astype(np.int64),
    "attention_mask": enc["attention_mask"].astype(np.int64),
})

probs = np.exp(logits[0]) / np.exp(logits[0]).sum()
label = ["safe", "risk"][probs.argmax()]
confidence = float(probs.max())
print(f"{label} ({confidence:.1%})")
# risk (96.2%)
```

### iOS / Android (ONNX Runtime Mobile)

Load `horizon-mobile.onnx` with the [ONNX Runtime Mobile](https://onnxruntime.ai/docs/tutorials/mobile/) SDK. Tokenize with a WordPiece tokenizer (vocab from this repo). Input: `input_ids` + `attention_mask` as int64 tensors, shape `[1, 512]`. Output: `logits` shape `[1, 2]`.

---

## Limitations

- **Binary only** — outputs `safe` / `risk`, no category or severity (use Horizon Full for that)
- **8.4% FNR** — approximately 1 in 12 risk conversations are missed at this stage
- **English only** — multilingual performance untested
- **Not standalone** — must be paired with Horizon Full and human review in production

---

## Citation

```bibtex
@misc{horizon2026,
  title={Horizon: Child Safety Risk Detection via Fine-tuned LLMs},
  author={SafeCircle},
  year={2026},
  url={https://huggingface.co/safecircleai/horizon-full}
}
```
"""


# ── Upload functions ──────────────────────────────────────────────────────────

def upload_cards_and_assets(api: HfApi):
    """Push updated model cards, license, and eval charts to all three repos."""
    license_path = Path(__file__).parent.parent / "LICENSE-SAFECIRCLE.md"
    eval_chart = Path(__file__).parent.parent / "evaluation/reports/latest/eval_report.png"

    updates = [
        (FULL_REPO, FULL_MODEL_CARD),
        (GGUF_REPO, GGUF_MODEL_CARD),
        (MOBILE_REPO, MOBILE_MODEL_CARD),
    ]
    for repo, card in updates:
        print(f"  Updating card: {repo}")
        api.upload_file(
            path_or_fileobj=card.encode(),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="model",
            commit_message="docs: update model card v2",
        )
        # Upload license to all repos
        if license_path.exists():
            api.upload_file(
                path_or_fileobj=str(license_path),
                path_in_repo="LICENSE-SAFECIRCLE.md",
                repo_id=repo,
                repo_type="model",
                commit_message="docs: add SafeCircle Research License (SRL-1.0)",
            )

    # Upload eval chart only to horizon-full
    if eval_chart.exists():
        print(f"  Uploading eval chart to {FULL_REPO}")
        api.upload_file(
            path_or_fileobj=str(eval_chart),
            path_in_repo="eval_report.png",
            repo_id=FULL_REPO,
            repo_type="model",
            commit_message="docs: add evaluation report chart",
        )
    print("  Cards and assets updated.")


def ensure_repo(api: HfApi, repo_id: str, repo_type: str = "model"):
    try:
        create_repo(repo_id, repo_type=repo_type, exist_ok=True, private=True)
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

    mobile_path = Path(mobile_dir)
    files = [f for f in mobile_path.iterdir() if f.is_file()]
    if not files:
        print(f"  Warning: no files found in {mobile_dir}")
        return

    for f in sorted(files):
        size_mb = f.stat().st_size / 1e6
        print(f"  Uploading {f.name} ({size_mb:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=MOBILE_REPO,
            commit_message=f"Upload {f.name}",
        )

    print(f"  Done: https://huggingface.co/{MOBILE_REPO}")


def main():
    parser = argparse.ArgumentParser(description="Upload Horizon models to HuggingFace")
    parser.add_argument("--what", choices=["full", "gguf", "mobile", "cards", "all"], default="all")
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

    if args.what in ("cards", "all"):
        upload_cards_and_assets(api)

    print("\nAll uploads complete.")


if __name__ == "__main__":
    main()
