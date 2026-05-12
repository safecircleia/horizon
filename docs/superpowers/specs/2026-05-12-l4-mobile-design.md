# Horizon: L4 GPU Optimization + Mobile Model Design

**Date:** 2026-05-12
**Status:** Approved

## Overview

Two parallel improvements to the Horizon training system:

1. **L4 GPU optimization** — remove 6GB VRAM constraints, train `horizon-full` (Llama 3.2 3B) at full capacity on the NVIDIA L4 (24GB VRAM)
2. **`horizon-mobile`** — a tiny on-device classifier (~50MB) for iOS and Android, distilled from `horizon-full`, with explainability output

---

## Model Variants

### `horizon-full` (server)

- Base: `meta-llama/Llama-3.2-3B-Instruct`
- Training: QLoRA with full bfloat16 (no 4-bit), Flash Attention 2
- Output: structured JSON (category, severity, confidence, reasoning)
- Deployment: GPU server

### `horizon-mobile` (on-device)

- Base: `google/mobilebert-uncased` (25MB, 24M parameters)
- Training: knowledge distillation from `horizon-full` + hard label cross-entropy
- Output: `{ category, severity, confidence, attributions: [{token, score}] }`
- Deployment: ONNX Runtime Mobile on iOS and Android
- Target size: ~20–25MB (INT8 quantized), <100ms inference

---

## L4 Training Config (`training/configs/l4.yaml`)

Changes from `base.yaml`:

| Parameter | base.yaml | l4.yaml |
|---|---|---|
| `load_in_4bit` | true | false |
| `torch_dtype` | bfloat16 (w/ 4bit) | bfloat16 (full) |
| `attn_implementation` | default | flash_attention_2 |
| `per_device_train_batch_size` | 4 | 16 |
| `gradient_accumulation_steps` | 4 | 2 |
| `gradient_checkpointing` | false | false |
| `lora.rank` | 64 | 128 |
| `lora.alpha` | 128 | 256 |
| `max_seq_length` | 2048 | 4096 |
| `max_steps` | 10000 | 15000 |
| `optim` | default | adamw_torch_fused |

Effective batch size: 32 (16 × 2 accumulation steps).

### Loader change (`training/model/loader.py`)

New branch: when `load_in_4bit: false` and CUDA available, load model with:
- `torch_dtype=torch.bfloat16`
- `device_map="cuda:0"`
- `attn_implementation="flash_attention_2"`

---

## Mobile Model Architecture

MobileBERT encoder with two classification heads sharing the pooled output:

```
MobileBERT encoder
       │
  [CLS] pooled output
       │
  ┌────┴────┐
  │         │
category  severity
 head      head
(8-class) (5-class)
```

**Category classes:** grooming, bullying, sexual_content, isolation, personal_info, platform_migration, threats, benign

**Severity classes:** none, low, medium, high, critical

---

## Training Pipeline (`training/configs/mobile.yaml` + `training/scripts/distill.py`)

### Step 1: Soft label generation
- Load trained `horizon-full` checkpoint
- Run each training example through it
- Collect category probability distributions as soft labels (temperature T=4 for softer distributions)

### Step 2: Distillation training
- Loss: `0.7 × KL_divergence(soft_labels) + 0.3 × CrossEntropy(hard_labels)`
- Severity head trained on hard labels only (3B model doesn't output severity probabilities)
- Optimizer: AdamW, lr=2e-5, warmup 200 steps

### Mobile config parameters
- `max_seq_length: 256` (conversations truncated — mobile use case is message-by-message)
- `per_device_train_batch_size: 32`
- `max_steps: 5000`

---

## Explainability

Implemented via **Integrated Gradients** using the `captum` library.

- Computes token-level attribution scores against the predicted category
- Top-N tokens (default N=5) returned with normalized scores 0–1
- Exported as a second ONNX graph: `horizon-mobile-explain.onnx`
- The main inference graph (`horizon-mobile.onnx`) does NOT include attribution — kept separate to avoid inference overhead when explainability is not needed

---

## ONNX Export (`training/scripts/export_onnx.py`)

1. Export classifier graph with `torch.onnx.export`, dynamic sequence length axis
2. INT8 post-training quantization via `onnxruntime.quantization.quantize_dynamic`
3. Optionally export explainability graph (flag: `--with-explain`)
4. Output to `models/mobile/`:
   - `horizon-mobile.onnx` (~20–25MB)
   - `horizon-mobile-explain.onnx` (~20–25MB, optional)
   - `tokenizer/` — MobileBERT tokenizer files

---

## Notebook Updates (`notebooks/horizon_training.py`)

### Modified cells
- **Hardware detection** — when ≥20GB VRAM detected, show callout recommending `l4.yaml` and explaining 4-bit is no longer needed
- **Training config dropdown** — add `l4.yaml` and `mobile.yaml` options

### New sections (inserted in pipeline order)
- **Step 3.5: Knowledge Distillation** — teacher checkpoint selector, run button, distillation loss output
- **Step 4.5: ONNX Export** — mobile checkpoint selector, explainability toggle, run button, final model size display

---

## New Files

```
training/configs/l4.yaml
training/configs/mobile.yaml
training/scripts/distill.py
training/scripts/export_onnx.py
models/mobile/                  (output, gitignored)
```

## Modified Files

```
training/model/loader.py        — Flash Attention 2 branch
notebooks/horizon_training.py   — L4 config, distillation + export sections
```
