#!/usr/bin/env python3
"""Upload all Horizon models to HuggingFace Hub with model cards.

Usage:
    python scripts/upload_to_hf.py --what full      # upload merged full model
    python scripts/upload_to_hf.py --what gguf      # upload GGUF variants
    python scripts/upload_to_hf.py --what mobile    # upload mobile ONNX
    python scripts/upload_to_hf.py --what edge-2b   # upload Gemma 4 E2B edge model
    python scripts/upload_to_hf.py --what edge-4b   # upload Gemma 4 E4B edge model
    python scripts/upload_to_hf.py --what all       # upload everything

Prerequisites:
    huggingface-cli login   (or set HF_TOKEN env var)
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ORG = "safecircleai"

FULL_REPO = f"{ORG}/horizon-full"
GGUF_REPO = f"{ORG}/horizon-full-gguf"
MOBILE_REPO = f"{ORG}/horizon-mobile"
EDGE_2B_REPO = f"{ORG}/horizon-edge-2b"
EDGE_4B_REPO = f"{ORG}/horizon-edge-4b"


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
- risk-detection
- gemma
- fine-tuned
- mobile
- on-device
- litert
- litert-lm
pipeline_tag: text-generation
base_model: google/gemma-3-1b-it
---

# Horizon Mobile — On-Device Child Safety Risk Detection

**Horizon Mobile** is a fine-tuned [Gemma 3 1B IT](https://huggingface.co/google/gemma-3-1b-it) model for on-device child safety risk detection, packaged in [LiteRT-LM](https://ai.google.dev/edge/litert-lm/overview) format for Android and iOS deployment.

It runs the same structured JSON risk assessment as [Horizon Full](https://huggingface.co/safecircleai/horizon-full) — directly on-device, with no cloud dependency.

> ⚠️ **License:** [SafeCircle Research License (SRL-1.0)](https://huggingface.co/safecircleai/horizon-full/blob/main/LICENSE-SAFECIRCLE.md). Commercial use prohibited without written permission. Contact [legal@safecircle.tech](mailto:legal@safecircle.tech).

---

## Available Variants

| File | Quantization | Size | Target |
|---|---|---|---|
| `horizon-mobile-int8_q8_ekv1280.litertlm` | INT8 dynamic | ~1.2 GB | 6 GB+ RAM phones (mid-range 2022+) |
| `horizon-mobile-int4_q4_block128_ekv1280.litertlm` | INT4 block-128 | ~507 MB | 4 GB RAM phones (budget/older) |

Both variants share the same fine-tuned weights; only quantization differs.

---

## Model Details

| Property | Value |
|---|---|
| Base model | [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) |
| Fine-tuning | QLoRA — rank 64, alpha 128, all projection layers |
| Training data | 500K synthetic conversations across 8 risk categories |
| Dataset | [safecircleai/horizon-training-data](https://huggingface.co/datasets/safecircleai/horizon-training-data) |
| Hardware | NVIDIA L40s (48 GB) |
| Training steps | 8,000 |
| Format | LiteRT-LM (.litertlm) |
| KV cache | 1,280 tokens |
| Prefill signatures | 8, 64, 128, 256, 512 tokens |

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
| `benign` | Safe, normal conversation |

---

## Output Format

```json
{
  "risk_detected": true,
  "category": "grooming",
  "severity": "high",
  "confidence": 0.91,
  "reasoning": "Adult requesting private meeting and explicit secrecy from parents."
}
```

---

## Usage

### Android / iOS — LiteRT-LM SDK

Load the `.litertlm` file with the [LiteRT-LM SDK](https://ai.google.dev/edge/litert-lm/overview). The model includes the tokenizer, system prompt, and chat template — no additional configuration required.

### Test locally (CLI)

```bash
# Install LiteRT-LM CLI
uvx litert-lm run horizon-mobile-int8_q8_ekv1280.litertlm \\
    --prompt "Analyze this conversation: Child: hey, want to meet up? Don't tell your parents."
```

---

## Deployment Pattern

```
Incoming message
      │
      ▼
Horizon Mobile (on-device, LiteRT-LM)
      │
      ├── safe ──► No action
      │
      └── risk ──► Horizon Full (cloud API, 7-category + severity)
                        │
                        ▼
                  Human moderator review
```

For the full server-side model, see [safecircleai/horizon-full](https://huggingface.co/safecircleai/horizon-full).

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


EDGE_2B_MODEL_CARD = """\
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
- gemma
- fine-tuned
- litert-lm
- on-device
- mobile
base_model: google/gemma-4-E2B-it
pipeline_tag: text-generation
---

# Horizon Edge 2B — On-Device Child Safety Risk Detection

**Horizon Edge 2B** is a fine-tuned [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) model packaged in [LiteRT-LM](https://developers.google.com/edge/litert-lm/overview) format for on-device child safety risk detection on Android, iOS, Desktop, IoT, and Web.

It runs the same structured JSON risk assessment as [Horizon Full](https://huggingface.co/safecircleai/horizon-full) — directly on-device, with no cloud dependency.

> ⚠️ **License:** [SafeCircle Research License (SRL-1.0)](https://huggingface.co/safecircleai/horizon-full/blob/main/LICENSE-SAFECIRCLE.md). Commercial use prohibited without written permission. Contact [legal@safecircle.tech](mailto:legal@safecircle.tech).

---

## Available Variants

| File | Target | Optimized For |
|---|---|---|
| `horizon-edge-e2b.litertlm` | General | CPU/GPU cross-platform |
| `horizon-edge-e2b_Google_Tensor_G5.litertlm` | Google Tensor G5 | Pixel 10 |
| `horizon-edge-e2b_intel_LNL.litertlm` | Intel Lunar Lake | Ultra 200V laptops |
| `horizon-edge-e2b_intel_PTL.litertlm` | Intel Panther Lake | Core Ultra 300 |
| `horizon-edge-e2b_qualcomm_qcs8275.litertlm` | Qualcomm QCS8275 | Dragonwing IQ8 (NPU) |
| `horizon-edge-e2b_qualcomm_sm8750.litertlm` | Qualcomm SM8750 | Snapdragon 8 Elite |
| `horizon-edge-e2b-web.litertlm` | Web | WebGPU browsers |

All variants share the same fine-tuned weights; they differ only in hardware-specific compilation.

---

## Model Details

| Property | Value |
|---|---|
| Base model | [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) |
| Fine-tuning | QLoRA — rank 64, alpha 128, all projection layers |
| Training data | 1.6M synthetic conversations across 8 categories |
| Dataset | [safecircleai/horizon-training-data](https://huggingface.co/datasets/safecircleai/horizon-training-data) |
| Hardware | NVIDIA H100 NVL (94 GB) |
| Training steps | 10,000 |
| Format | LiteRT-LM (.litertlm) |
| Model size | ~2.6 GB |

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
| `benign` | Safe, normal conversation |

---

## Usage

### LiteRT-LM CLI

```bash
uv tool install litert-lm

litert-lm run horizon-edge-e2b.litertlm \\
    --prompt "Analyze this conversation: Child: hey, want to meet up? Don't tell your parents."
```

### Android / iOS — LiteRT-LM SDK

Load the `.litertlm` file with the [LiteRT-LM SDK](https://developers.google.com/edge/litert-lm/overview). The model includes the tokenizer, system prompt, and chat template — no additional configuration required.

### Python

```python
from litert_lm import LiteRTLM

model = LiteRTLM("horizon-edge-e2b.litertlm")
result = model.generate(
    "Analyze this conversation:\\nChild: hey\\nOther: how old are you? where do you live?"
)
print(result)
```

---

## Deployment Pattern

```
Incoming message
      │
      ▼
Horizon Edge 2B (on-device, LiteRT-LM)
      │
      ├── safe ──► No action
      │
      └── risk ──► Horizon Full (cloud API, full severity assessment)
                        │
                        ▼
                  Human moderator review
```

For the full server-side model, see [safecircleai/horizon-full](https://huggingface.co/safecircleai/horizon-full).

---

## Citation

```bibtex
@misc{horizon2026,
  title={Horizon: Child Safety Risk Detection via Fine-tuned LLMs},
  author={SafeCircle},
  year={2026},
  url={https://huggingface.co/safecircleai/horizon-edge-2b}
}
```
"""

EDGE_4B_MODEL_CARD = """\
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
- gemma
- fine-tuned
- litert-lm
- on-device
- mobile
base_model: google/gemma-4-E4B-it
pipeline_tag: text-generation
---

# Horizon Edge 4B — On-Device Child Safety Risk Detection

**Horizon Edge 4B** is a fine-tuned [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it) model packaged in [LiteRT-LM](https://developers.google.com/edge/litert-lm/overview) format. It delivers higher accuracy than [Horizon Edge 2B](https://huggingface.co/safecircleai/horizon-edge-2b) at a moderate size increase (~3.7 GB), making it the recommended edge choice for devices with 6 GB+ RAM.

> ⚠️ **License:** [SafeCircle Research License (SRL-1.0)](https://huggingface.co/safecircleai/horizon-full/blob/main/LICENSE-SAFECIRCLE.md). Commercial use prohibited without written permission. Contact [legal@safecircle.tech](mailto:legal@safecircle.tech).

---

## Available Variants

| File | Target | Optimized For |
|---|---|---|
| `horizon-edge-e4b.litertlm` | General | CPU/GPU cross-platform |
| `horizon-edge-e4b_Google_Tensor_G5.litertlm` | Google Tensor G5 | Pixel 10 |
| `horizon-edge-e4b_intel_LNL.litertlm` | Intel Lunar Lake | Ultra 200V laptops |
| `horizon-edge-e4b_intel_PTL.litertlm` | Intel Panther Lake | Core Ultra 300 |
| `horizon-edge-e4b_qualcomm_qcs8275.litertlm` | Qualcomm QCS8275 | Dragonwing IQ8 (NPU) |
| `horizon-edge-e4b_qualcomm_sm8750.litertlm` | Qualcomm SM8750 | Snapdragon 8 Elite |
| `horizon-edge-e4b-web.litertlm` | Web | WebGPU browsers |

All variants share the same fine-tuned weights; they differ only in hardware-specific compilation.

---

## Model Details

| Property | Value |
|---|---|
| Base model | [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) |
| Fine-tuning | QLoRA — rank 128, alpha 256, all projection layers |
| Training data | 1.6M synthetic conversations across 8 categories |
| Dataset | [safecircleai/horizon-training-data](https://huggingface.co/datasets/safecircleai/horizon-training-data) |
| Hardware | NVIDIA H100 NVL (94 GB) |
| Training steps | 15,000 |
| Format | LiteRT-LM (.litertlm) |
| Model size | ~3.7 GB |

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
| `benign` | Safe, normal conversation |

---

## Usage

### LiteRT-LM CLI

```bash
uv tool install litert-lm

litert-lm run horizon-edge-e4b.litertlm \\
    --prompt "Analyze this conversation: Child: hey, want to meet up? Don't tell your parents."
```

### Android / iOS — LiteRT-LM SDK

Load the `.litertlm` file with the [LiteRT-LM SDK](https://developers.google.com/edge/litert-lm/overview). The model includes the tokenizer, system prompt, and chat template — no additional configuration required.

### Python

```python
from litert_lm import LiteRTLM

model = LiteRTLM("horizon-edge-e4b.litertlm")
result = model.generate(
    "Analyze this conversation:\\nChild: hey\\nOther: how old are you? where do you live?"
)
print(result)
```

---

## Deployment Pattern

```
Incoming message
      │
      ▼
Horizon Edge 4B (on-device, LiteRT-LM)
      │
      ├── safe ──► No action
      │
      └── risk ──► Horizon Full (cloud API, full severity assessment)
                        │
                        ▼
                  Human moderator review
```

For the full server-side model, see [safecircleai/horizon-full](https://huggingface.co/safecircleai/horizon-full).
For the smaller 2B variant, see [safecircleai/horizon-edge-2b](https://huggingface.co/safecircleai/horizon-edge-2b).

---

## Citation

```bibtex
@misc{horizon2026,
  title={Horizon: Child Safety Risk Detection via Fine-tuned LLMs},
  author={SafeCircle},
  year={2026},
  url={https://huggingface.co/safecircleai/horizon-edge-4b}
}
```
"""


# ── Upload functions ──────────────────────────────────────────────────────────


def upload_cards_and_assets(api: HfApi):
    """Push updated model cards, license, and eval charts to all three repos."""
    license_path = Path(__file__).parent.parent / "LICENSE-SAFECIRCLE.md"
    eval_chart = (
        Path(__file__).parent.parent / "evaluation/reports/latest/eval_report.png"
    )

    updates = [
        (FULL_REPO, FULL_MODEL_CARD),
        (GGUF_REPO, GGUF_MODEL_CARD),
        (MOBILE_REPO, MOBILE_MODEL_CARD),
        (EDGE_2B_REPO, EDGE_2B_MODEL_CARD),
        (EDGE_4B_REPO, EDGE_4B_MODEL_CARD),
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


def upload_mobile_model(api: HfApi, mobile_standard_dir: str, mobile_lite_dir: str):
    print(f"\n==> Uploading mobile LiteRT-LM models to {MOBILE_REPO}")
    ensure_repo(api, MOBILE_REPO)

    api.upload_file(
        path_or_fileobj=MOBILE_MODEL_CARD.encode(),
        path_in_repo="README.md",
        repo_id=MOBILE_REPO,
        commit_message="Add model card",
    )

    dirs = [
        (mobile_standard_dir, "INT8 standard"),
        (mobile_lite_dir, "INT4 lite"),
    ]
    for dir_path, label in dirs:
        p = Path(dir_path)
        if not p.exists():
            print(f"  Warning: {label} dir not found: {dir_path}")
            continue
        files = sorted(
            f for f in p.iterdir() if f.is_file() and f.suffix == ".litertlm"
        )
        if not files:
            print(f"  Warning: no .litertlm files found in {dir_path}")
            continue
        for f in files:
            size_mb = f.stat().st_size / 1e6
            print(f"  Uploading {f.name} ({size_mb:.0f} MB) [{label}]...")
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f.name,
                repo_id=MOBILE_REPO,
                commit_message=f"Upload {f.name}",
            )

    print(f"  Done: https://huggingface.co/{MOBILE_REPO}")


def upload_edge_model(api: HfApi, repo: str, card: str, model_dir: str, label: str):
    """Upload .litertlm variants for an edge model."""
    print(f"\n==> Uploading {label} to {repo}")
    ensure_repo(api, repo)

    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo,
        commit_message="Add model card",
    )

    model_path = Path(model_dir)
    litertlm_files = sorted(model_path.glob("*.litertlm"))
    if not litertlm_files:
        print(f"  WARNING: no .litertlm files found in {model_dir}")
        print("  Run first: sbatch --export=MODEL_SIZE=... slurm/export_edge.sbatch")
        return

    for f in litertlm_files:
        size_mb = f.stat().st_size / 1e6
        print(f"  Uploading {f.name} ({size_mb:.0f} MB)...")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=repo,
            commit_message=f"Upload {f.name}",
        )

    print(f"  Uploaded {len(litertlm_files)} variants.")
    print(f"  Done: https://huggingface.co/{repo}")


def main():
    parser = argparse.ArgumentParser(description="Upload Horizon models to HuggingFace")
    parser.add_argument(
        "--what",
        choices=["full", "gguf", "mobile", "edge-2b", "edge-4b", "cards", "all"],
        default="all",
    )
    parser.add_argument("--full-model-dir", default="models/horizon-full-merged")
    parser.add_argument("--gguf-dir", default="models/horizon-full-gguf")
    parser.add_argument("--mobile-standard-dir", default="models/mobile-standard")
    parser.add_argument("--mobile-lite-dir", default="models/mobile-lite")
    parser.add_argument("--edge-2b-dir", default="models/horizon-edge-2b-litert")
    parser.add_argument("--edge-4b-dir", default="models/horizon-edge-4b-litert")
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
            print(
                "Run first: python scripts/merge_lora.py --checkpoint experiments/h100-20260514-134855/final"
            )
        else:
            upload_full_model(api, args.full_model_dir)

    if args.what in ("gguf", "all"):
        if (
            not list(Path(args.gguf_dir).glob("*.gguf"))
            if Path(args.gguf_dir).exists()
            else True
        ):
            print(f"No GGUF files found in {args.gguf_dir}")
            print("Run first: bash scripts/export_gguf.sh")
        else:
            upload_gguf_models(api, args.gguf_dir)

    if args.what in ("mobile", "all"):
        upload_mobile_model(api, args.mobile_standard_dir, args.mobile_lite_dir)

    if args.what in ("edge-2b", "all"):
        if not Path(args.edge_2b_dir).exists():
            print(f"Edge 2B LiteRT-LM models not found at {args.edge_2b_dir}")
            print("Run first: sbatch --export=MODEL_SIZE=e2b slurm/export_edge.sbatch")
        else:
            upload_edge_model(
                api,
                EDGE_2B_REPO,
                EDGE_2B_MODEL_CARD,
                args.edge_2b_dir,
                "Horizon Edge 2B",
            )

    if args.what in ("edge-4b", "all"):
        if not Path(args.edge_4b_dir).exists():
            print(f"Edge 4B LiteRT-LM models not found at {args.edge_4b_dir}")
            print("Run first: sbatch --export=MODEL_SIZE=e4b slurm/export_edge.sbatch")
        else:
            upload_edge_model(
                api,
                EDGE_4B_REPO,
                EDGE_4B_MODEL_CARD,
                args.edge_4b_dir,
                "Horizon Edge 4B",
            )

    if args.what in ("cards", "all"):
        upload_cards_and_assets(api)

    print("\nAll uploads complete.")


if __name__ == "__main__":
    main()
