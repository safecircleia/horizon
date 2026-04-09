# Project Horizon - SafeCircle Risk Detection Model
## Design Document

**Version:** 1.0  
**Date:** 2026-04-09  
**Status:** Approved  

---

## Executive Summary

Project Horizon is SafeCircle's custom AI model training system for privacy-preserving child safety risk detection. The project trains a fine-tuned Llama 3.1 8B model on synthetic conversation data to detect seven categories of risk (grooming, bullying, sexual content, isolation, personal info requests, platform migration, threats) with structured JSON output suitable for flexible deployment.

**Key Objectives:**
- Train a deployment-agnostic risk detection model (not tied to any specific app)
- Achieve >0.85 overall F1 score with >0.90 F1 on critical categories
- Maintain privacy-first approach using only synthetic/adult-sourced data
- Deliver quantized model variants (INT4 ~2GB) for diverse deployment scenarios
- Complete production-ready v1.0 in 10 weeks with ~$400 budget

---

## 1. System Architecture

The system consists of five main components:

### 1.1 Data Generation Engine

**Purpose:** Generate synthetic conversation datasets that simulate risky interactions while maintaining privacy.

**Components:**
- **Prompt Templates:** Category-specific prompts for each risk type (grooming, bullying, etc.)
- **LLM Generators:** Claude/GPT-4 API integration for conversation synthesis
- **Quality Validators:** Automated checks + human review pipeline
- **Output Format:** JSONL with conversations + structured labels

**Key Features:**
- Severity level control (low/medium/high/critical)
- Conversation stage progression (early signals → escalation → explicit risk)
- False positive scenario generation (edge cases)
- Cultural/linguistic diversity
- Age-appropriate teen communication patterns

### 1.2 Training Pipeline

**Purpose:** Fine-tune base model on SafeCircle's risk detection task.

**Architecture:**
- **Base Model:** Llama 3.1 8B Instruct (permissive license, strong instruction-following)
- **Method:** QLoRA (4-bit quantized training for efficiency)
- **Framework:** HuggingFace Transformers + PEFT + bitsandbytes
- **Output Format:** Structured JSON (risk level, categories, confidence, reasoning)

**Training Configuration:**
```yaml
base_model: meta-llama/Llama-3.1-8B-Instruct
method: QLoRA
rank: 64
alpha: 128
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
dropout: 0.05
batch_size: 8 (per device)
gradient_accumulation: 4
learning_rate: 2e-4
max_steps: 10000
```

**Training Stages:**
1. **Stage 1 (Steps 0-3000):** Category recognition across all 7 risk types
2. **Stage 2 (Steps 3000-7000):** Severity calibration (none/low/medium/high/critical)
3. **Stage 3 (Steps 7000-10000):** False positive reduction on benign conversations

### 1.3 Evaluation Framework

**Purpose:** Validate model quality and track performance across dimensions.

**Metrics (Per-Category + Overall):**
- **Classification:** Precision, Recall, F1, Confusion Matrix, ROC-AUC
- **Severity:** Exact match accuracy, ±1 level accuracy, MAE
- **Operational:** False positive rate (<2% target), false negative rate (<5% target)

**Test Sets:**
- **Standard:** 20% held-out from generated data, stratified by category/severity
- **Adversarial:** Hand-crafted edge cases, code-switching, cultural variations
- **Real-World Proxy:** Adult roleplays, safety research examples (optional)

**Evaluation Protocol:**
1. Automated evaluation every 500 training steps
2. Human review at major checkpoints (100 samples per category)
3. Ablation studies on final model (by length, age, context)

**Success Criteria:**

| Metric | MVP | Production-Ready |
|--------|-----|------------------|
| Overall F1 | >0.85 | >0.92 |
| Critical Category F1 | >0.90 | >0.95 |
| False Positive Rate | <3% | <1.5% |
| False Negative Rate (Critical) | <5% | <2% |
| Inference Time (Quantized) | <500ms | <300ms |
| Model Size | <2.5GB | <2GB |

### 1.4 Quantization Pipeline

**Purpose:** Convert trained model to INT4 format for efficient deployment.

**Method:** GPTQ or llama.cpp quantization
- **Original FP16:** ~16GB
- **INT8:** ~8GB
- **INT4 (Q4_K_M):** ~4GB
- **INT4 Optimized:** ~2GB (target)

**Quality Validation:**
- Perplexity increase: <5% vs FP16
- F1 score drop: <2% vs FP16
- Per-category accuracy maintained

**Pipeline Steps:**
1. Merge LoRA weights into base model
2. Quantize to INT4 using llama.cpp (Q4_K_M format)
3. Validate quality retention on test set
4. Export to multiple formats (GGUF, ONNX, etc.)

### 1.5 Integration Layer

**Purpose:** Provide flexible deployment options for the trained model.

**Standardized API Contract:**

```json
// Input
{
  "conversation": [
    {"role": "sent", "content": "hey", "timestamp": 1234567890},
    {"role": "received", "content": "hi there", "timestamp": 1234567895}
  ],
  "context": {
    "child_age": 14,
    "platform": "instagram"
  }
}

// Output
{
  "risk_level": "medium",
  "categories": ["personal_info", "platform_move"],
  "confidence": 0.82,
  "matched_terms": ["snapchat", "add me"],
  "reasoning": "Attempt to move conversation to less monitored platform",
  "detection_layers": ["keywords", "context"],
  "message_risks": [
    {"index": 1, "risk": "low", "categories": []},
    {"index": 2, "risk": "medium", "categories": ["platform_move"]}
  ]
}
```

**Deployment Options:**
- **Cloud API:** vLLM/TGI for server-side inference
- **On-Device:** llama.cpp/MediaPipe for privacy-preserving local inference
- **Edge:** ONNX Runtime for IoT/embedded devices
- **Hybrid:** Mix of cloud and on-device based on requirements

---

## 2. Data Strategy (Privacy-First)

### 2.1 Dataset Composition

**Target:** 50,000 conversations across 7 risk categories + benign baseline

| Category | Training | Test | Priority |
|----------|----------|------|----------|
| Grooming | 8,000 | 1,000 | Critical |
| Bullying | 7,000 | 1,000 | High |
| Sexual Content | 7,000 | 1,000 | Critical |
| Isolation/Control | 5,000 | 500 | High |
| Personal Info Requests | 5,000 | 500 | Medium |
| Platform Migration | 3,000 | 500 | Medium |
| Threats/Violence | 5,000 | 500 | High |
| **Benign Conversations** | 10,000 | 2,000 | Critical |

### 2.2 Synthetic Generation Strategy

For each risk category, generate conversations with:

1. **Severity Levels:** Low/Medium/High/Critical variants of each risk
2. **Conversation Stages:** Early signals → escalation → explicit risk
3. **False Positive Scenarios:** Edge cases that appear risky but aren't
4. **Cultural/Linguistic Diversity:** Different communication styles, slang, abbreviations
5. **Age-Appropriate Context:** Teen language patterns, gaming culture, social media norms

**Example Prompt Template (Grooming):**

```
Generate a realistic chat conversation between a teen (13-16) and an adult 
pretending to be a peer. The adult is engaging in grooming behavior at 
[LOW/MEDIUM/HIGH] severity level. Include:
- Gradual trust building
- Boundary testing
- Secrecy requests
- [specific grooming tactic]

Make it realistic to how teens actually communicate (abbreviations, emojis, etc.)
```

### 2.3 Quality Validation

- **Human Review:** 10% sample validation by domain experts
- **Cross-Validation:** Compare with existing safety datasets (HateCheck, ToxicChat)
- **Diversity Metrics:** Vocabulary distribution, conversation length, pattern variety
- **Augmentation:** Adapt public datasets (with proper licensing) for child safety context

### 2.4 Privacy Guarantees

- **No Real Child Data:** Zero messages from actual children
- **Adult-Sourced Only:** Any non-synthetic examples from consenting adults
- **Synthetic First:** Primary reliance on LLM-generated conversations
- **Research Ethics:** Follow academic standards for sensitive content research

---

## 3. Training Methodology

### 3.1 Base Model Selection

**Llama 3.1 8B Instruct**

**Why Llama 3.1 8B:**
- Permissive license (Llama 3.1 Community License - commercial use allowed)
- Strong instruction-following capabilities
- Good balance of size vs. performance (8B manageable for training)
- Proven quantization to INT4 with minimal quality loss
- Active community and extensive tooling support

### 3.2 Fine-Tuning Approach

**QLoRA (Quantized Low-Rank Adaptation)**

Benefits:
- Train 8B model on single A100 40GB GPU
- 4-bit quantization during training reduces memory
- LoRA adapters are small (~100MB) and mergeable
- Maintains >99% of full fine-tuning quality

**LoRA Configuration:**
- Rank: 64 (controls adapter capacity)
- Alpha: 128 (scaling factor)
- Target modules: All attention + FFN layers
- Dropout: 0.05 (regularization)

### 3.3 Prompt Format

**SafeCircle System Prompt:**

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are SafeCircle's risk detection model. Analyze conversations for child safety risks.
Output JSON with: risk_level (none/low/medium/high/critical), categories (array), 
confidence (0-1), matched_terms (array), reasoning (brief).
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Analyze this conversation:
[CONVERSATION TEXT]
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
{
  "risk_level": "high",
  "categories": ["grooming", "personal_info"],
  "confidence": 0.87,
  "matched_terms": ["our secret", "don't tell", "send photo"],
  "reasoning": "Adult establishing secrecy while requesting personal media"
}
<|eot_id|>
```

### 3.4 Training Hardware & Timeline

**Recommended Setup:**
- GPU: A100 40GB (or 2x RTX 4090 48GB)
- Cloud Provider: RunPod, Lambda Labs, or Modal
- Training Time: 24-48 hours for 10K steps
- Cost: $1.50/hour × 48 hours = **~$72**

**Alternative (Budget):**
- GPU: RTX 4090 24GB (with gradient checkpointing)
- Training Time: 60-72 hours
- Cost: Lower hourly rate, longer duration

### 3.5 Early Stopping & Checkpointing

- Evaluate every 500 steps on validation set
- Track per-category F1 scores
- Save best checkpoint based on weighted F1
- Stop if validation loss plateaus for 1500 steps
- Keep top 3 checkpoints for ensembling experiments

---

## 4. Evaluation & Quality Assurance

### 4.1 Automated Metrics

**Per-Category Metrics:**
- Precision (minimize false positives)
- Recall (catch real risks)
- F1 Score (harmonic mean)
- Confusion Matrix (understand misclassifications)

**Overall Performance:**
- Macro F1 (unweighted average across categories)
- Weighted F1 (by category frequency)
- False Positive Rate on benign conversations
- False Negative Rate on critical risks

### 4.2 Human Evaluation

**Checkpoint Reviews:**
- Sample 100 predictions per category
- Expert labeling by child safety professionals
- Qualitative feedback on reasoning quality
- Identify systematic failure patterns

**Criteria:**
- Accuracy: Is the prediction correct?
- Reasoning: Is the explanation sound?
- Consistency: Similar cases get similar scores?
- Calibration: Confidence matches actual accuracy?

### 4.3 Error Analysis Framework

**Track and categorize failures:**

1. **Type I Errors (False Positives):**
   - Benign conversations flagged as risky
   - Impact: Erodes trust, alert fatigue
   - Analysis: What triggered the false alarm?

2. **Type II Errors (False Negatives):**
   - Risky conversations missed
   - Impact: Safety risk, potential harm
   - Analysis: What signals were missed?

3. **Severity Errors:**
   - Wrong risk level assigned (e.g., critical labeled as low)
   - Impact: Under/over-reaction to threats
   - Analysis: Calibration issues

4. **Category Errors:**
   - Wrong categories identified
   - Impact: Misunderstood threat type
   - Analysis: Category confusion patterns

### 4.4 Ablation Studies

Test model performance across dimensions:

- **Conversation Length:** Short (1-3 msgs) vs Long (20+ msgs)
- **Age Context:** Different age groups mentioned
- **Platform Context:** Different social media platforms
- **Language Style:** Formal vs slang vs code-switching
- **Temporal:** First message vs later in conversation

---

## 5. Deployment Artifacts

### 5.1 Model Outputs

**Full Precision:**
```
models/v1.0/full/
├── model.safetensors      # Model weights
├── config.json            # Model configuration
├── tokenizer.model        # Tokenizer
├── generation_config.json # Generation settings
└── adapter_config.json    # LoRA config (if not merged)
```

**Quantized Variants:**
```
models/v1.0/quantized/
├── safecircle-q8.gguf          # INT8 (~8GB)
├── safecircle-q4_k_m.gguf      # INT4 balanced (~4GB)
├── safecircle-q4_k_s.gguf      # INT4 small (~3.5GB)
└── safecircle-q4_0.gguf        # INT4 optimized (~2GB)
```

**API-Ready Formats:**
```
models/v1.0/deployment/
├── vllm/          # vLLM engine format
├── tgi/           # Text Generation Inference format
├── onnx/          # ONNX Runtime format
└── ollama/        # Ollama Modelfile
```

**Metadata:**
```
models/v1.0/metadata/
├── model_card.md              # Capabilities, limitations, usage
├── evaluation_results.json    # Benchmark scores
├── training_config.yaml       # Training hyperparameters
└── quantization_report.md     # Quality retention analysis
```

### 5.2 Inference Examples

**Python (vLLM):**
```python
from vllm import LLM, SamplingParams

llm = LLM(model="models/v1.0/full")
sampling_params = SamplingParams(temperature=0.1, max_tokens=512)

prompt = format_safecircle_prompt(conversation)
output = llm.generate([prompt], sampling_params)[0]
risk_assessment = json.loads(output.outputs[0].text)
```

**CLI (llama.cpp):**
```bash
./llama-cli \
  --model models/v1.0/quantized/safecircle-q4_k_m.gguf \
  --prompt-file conversation.txt \
  --temp 0.1 \
  --n-predict 512
```

**REST API (FastAPI + vLLM):**
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d @conversation.json
```

---

## 6. Project Structure

### 6.1 Repository Layout

```
horizon/
├── data/
│   ├── generation/           # Synthetic data generation
│   │   ├── prompts/         # Category-specific templates
│   │   ├── generators/      # LLM conversation generators
│   │   └── validators/      # Quality validation
│   ├── raw/                 # Generated conversations (JSONL)
│   ├── processed/           # Formatted training data
│   └── evaluation/          # Test sets & benchmarks
│
├── training/
│   ├── configs/             # Training configs (YAML)
│   ├── scripts/             # Training entry points
│   ├── models/              # Model definitions
│   └── utils/               # Training utilities
│
├── evaluation/
│   ├── metrics/             # Metric implementations
│   ├── analysis/            # Error analysis tools
│   └── reports/             # Generated reports
│
├── quantization/
│   ├── scripts/             # Quantization pipelines
│   └── validation/          # Post-quant validation
│
├── inference/
│   ├── api/                 # REST API
│   ├── cli/                 # CLI tool
│   └── examples/            # Usage examples
│
├── experiments/             # Training runs & logs
├── models/                  # Trained artifacts
├── notebooks/               # Analysis notebooks
├── tests/                   # Unit & integration tests
├── docs/                    # Documentation
│
├── requirements.txt
├── pyproject.toml
├── README.md
└── Makefile
```

### 6.2 Technology Stack

**Core Training:**
- Python 3.10+
- PyTorch 2.3+
- Transformers 4.40+
- PEFT 0.10+ (LoRA)
- bitsandbytes 0.43+ (quantization)
- accelerate 0.29+ (distributed training)

**Data Generation:**
- anthropic (Claude API)
- openai (GPT-4 API)
- datasets (HuggingFace)
- jsonlines

**Evaluation:**
- scikit-learn (metrics)
- wandb (experiment tracking)
- matplotlib/seaborn (visualization)
- jupyter (analysis)

**Quantization:**
- llama.cpp (GGUF)
- GPTQ
- optimum (HF tools)

**Inference:**
- vllm (fast serving)
- fastapi (REST API)
- ollama (local testing)

**Development:**
- black (formatting)
- ruff (linting)
- pytest (testing)
- pre-commit (git hooks)

### 6.3 Key Commands

```makefile
# Data generation
make generate-data CATEGORY=grooming COUNT=1000
make validate-data

# Training
make train CONFIG=configs/base.yaml
make train-quick  # Fast iteration

# Evaluation
make evaluate CHECKPOINT=checkpoints/step-5000
make analyze-errors

# Quantization
make quantize MODEL=models/v1/full FORMAT=q4_k_m
make validate-quantized

# Inference
make serve MODEL=models/v1/quantized/safecircle-q4.gguf
make inference-cli CONVERSATION=examples/test.json

# Testing
make test
make test-inference
```

---

## 7. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Goals:**
- Set up development environment
- Build data generation pipeline
- Generate initial dataset (10K samples)

**Deliverables:**
- Repository structure (horizon/)
- Data generation scripts working
- 10K synthetic conversations across categories
- Quality validation framework
- Dataset statistics report

**Success Criteria:**
- Generate 1K conversations per category
- Human review shows >90% quality
- Balanced severity distribution

### Phase 2: Training Infrastructure (Weeks 3-4)

**Goals:**
- Set up training pipeline
- Configure experiment tracking
- Run initial baseline training

**Deliverables:**
- Training scripts with QLoRA
- W&B/TensorBoard integration
- Baseline model (Llama 3.1 8B + 10K samples)
- Evaluation pipeline
- Initial benchmark results

**Success Criteria:**
- Training runs end-to-end
- Evaluation pipeline works
- Baseline achieves >0.70 F1 (proves pipeline)

### Phase 3: Dataset Expansion & Training v1 (Weeks 5-7)

**Goals:**
- Scale to 50K conversations
- Train production candidate
- Comprehensive evaluation

**Deliverables:**
- Full 50K dataset with augmentation
- Trained model v1.0
- Complete evaluation report
- Error analysis & failure patterns
- Model card documentation

**Success Criteria:**
- Overall F1: >0.85
- Critical categories F1: >0.90
- False positive rate: <3%
- Pass human evaluation (100 samples/category)

### Phase 4: Quantization & Optimization (Week 8)

**Goals:**
- Quantize to INT4/INT8
- Validate quality retention
- Optimize for inference

**Deliverables:**
- Quantized variants (Q8, Q4_K_M, Q4 optimized)
- Quantization quality report
- Inference benchmarks
- Deployment-ready artifacts

**Success Criteria:**
- Q4 model <2GB
- F1 drop <2% vs full precision
- Inference <500ms on target hardware

### Phase 5: API & Documentation (Week 9)

**Goals:**
- Build inference API
- Complete documentation
- Prepare for integration

**Deliverables:**
- REST API (FastAPI + vLLM)
- CLI inference tool
- Integration examples
- Complete documentation
- Model release package

**Success Criteria:**
- API handles 100 req/sec
- Documentation covers all use cases
- Clear integration path

### Phase 6: Iteration & Refinement (Week 10+)

**Goals:**
- Address failure modes
- Improve edge cases
- Continuous improvement

**Activities:**
- Analyze production feedback
- Generate targeted training data
- Retrain with expanded dataset
- Release v1.1, v1.2, etc.

---

## 8. Resource Requirements

### 8.1 Compute Budget

| Item | Cost |
|------|------|
| Data generation (Claude/GPT-4 API) | $200 |
| Training (A100 GPU, 48 hours) | $150 |
| Evaluation & testing | $50 |
| **Total** | **~$400** |

### 8.2 Human Resources

- **ML Engineer:** Full-time for 10 weeks (primary developer)
- **Domain Expert:** Part-time (review samples, validate categories)
- **DevOps (optional):** Infrastructure setup

### 8.3 Tools & Services

- **Experiment Tracking:** W&B or TensorBoard ($0-50/month)
- **Cloud GPU:** RunPod, Lambda Labs, or Modal (pay-per-use)
- **Code Hosting:** GitHub/GitLab (free tier sufficient)
- **API Access:** Claude API, GPT-4 API (pay-per-token)

---

## 9. Risk Mitigation

### 9.1 Technical Risks

**Risk:** Training doesn't converge to target metrics
- **Mitigation:** Start with proven baseline (Llama 3.1), use established QLoRA recipe
- **Fallback:** Increase dataset size, adjust hyperparameters, try alternative base models

**Risk:** Quantization degrades quality too much
- **Mitigation:** Test multiple quantization methods (GPTQ, llama.cpp), validate at each step
- **Fallback:** Use INT8 instead of INT4, or deploy full precision for critical use cases

**Risk:** Model generates biased or harmful outputs
- **Mitigation:** Diverse synthetic data, adversarial testing, human review loops
- **Fallback:** Post-processing filters, confidence thresholds, human-in-the-loop

### 9.2 Data Risks

**Risk:** Synthetic data doesn't reflect real-world patterns
- **Mitigation:** Consult child safety experts, validate against research literature
- **Fallback:** Incorporate adult-sourced examples, iterate with domain feedback

**Risk:** Dataset lacks diversity (cultural, linguistic)
- **Mitigation:** Explicitly generate diverse scenarios, multiple language styles
- **Fallback:** Post-launch data collection with proper consent/ethics

### 9.3 Timeline Risks

**Risk:** Training takes longer than expected
- **Mitigation:** Buffer time in schedule, use faster GPUs if needed
- **Fallback:** Release MVP with 30K dataset, iterate to 50K in v1.1

**Risk:** API costs exceed budget
- **Mitigation:** Optimize prompts, use cheaper models for augmentation
- **Fallback:** Generate fewer samples, focus on critical categories first

---

## 10. Success Metrics

### 10.1 Technical Metrics

**Model Quality:**
- ✅ Overall F1: >0.85
- ✅ Critical category F1: >0.90
- ✅ False positive rate: <3%
- ✅ Quantized model: <2GB, <2% F1 drop

**Performance:**
- ✅ Inference latency: <500ms (quantized)
- ✅ API throughput: >100 req/sec
- ✅ Model loads: <5 seconds cold start

### 10.2 Project Metrics

**Deliverables:**
- ✅ 50K synthetic conversation dataset
- ✅ Trained model v1.0 (full + quantized)
- ✅ Complete evaluation report
- ✅ REST API + CLI tools
- ✅ Documentation & integration examples

**Timeline:**
- ✅ v1.0 release: 10 weeks from project start
- ✅ Budget: <$500 total spend

### 10.3 Long-Term Metrics

**Model Evolution:**
- v1.1: Improved edge case handling (Week 12)
- v1.5: Expanded to 100K dataset (Week 16)
- v2.0: Multilingual support (Week 20)

**Community:**
- Open-source release (pending legal review)
- Academic collaborations for dataset validation
- Industry adoption as child safety standard

---

## 11. Appendix

### 11.1 Risk Categories Defined

1. **Grooming:** Adult building trust to exploit child (secrecy, boundary testing, manipulation)
2. **Bullying:** Harassment, threats, insults, exclusion, cyberbullying
3. **Sexual Content:** Explicit sexual messages, requests for photos/videos, inappropriate advances
4. **Isolation/Control:** Attempts to isolate child from support network, controlling behavior
5. **Personal Info Requests:** Requests for address, school, location, passwords, identifying info
6. **Platform Migration:** Attempts to move conversation to less monitored platforms
7. **Threats/Violence:** Violent threats, self-harm encouragement, dangerous challenges

### 11.2 Ethical Considerations

- **No Real Child Data:** Project uses only synthetic + adult-sourced examples
- **Privacy by Design:** Model trained to detect risk, not surveil content
- **Transparency:** Clear documentation of capabilities and limitations
- **Human Oversight:** Model is decision-support tool, not autonomous enforcement
- **Bias Monitoring:** Regular audits for demographic bias in predictions

### 11.3 References

- **Llama 3.1 Model Card:** https://ai.meta.com/llama/
- **QLoRA Paper:** Dettmers et al., 2023 (https://arxiv.org/abs/2305.14314)
- **Child Safety Research:** NCMEC, IWF, Thorn
- **Quantization Methods:** llama.cpp, GPTQ
- **Safety Datasets:** HateCheck, ToxicChat (adapted)

---

## Changelog

**v1.0 (2026-04-09):**
- Initial design document
- Architecture, data strategy, training methodology defined
- 10-week roadmap with resource requirements
- Success criteria and risk mitigation strategies

---

**Document Status:** ✅ Approved  
**Next Step:** Write implementation plan
