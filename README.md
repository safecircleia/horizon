# Project Horizon

**SafeCircle Risk Detection Model Training System**

Project Horizon trains custom AI models for privacy-preserving child safety risk detection. Starting from Llama 3.1 8B, the system fine-tunes on synthetic conversation data to detect seven risk categories (grooming, bullying, sexual content, isolation, personal info, platform migration, threats) with structured JSON output.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic data (1000 conversations)
make generate-data CATEGORY=grooming COUNT=1000

# Train model
make train CONFIG=configs/base.yaml

# Evaluate
make evaluate CHECKPOINT=experiments/latest/checkpoints/step-5000

# Quantize for deployment
make quantize MODEL=models/v1/full FORMAT=q4_k_m

# Serve API
make serve MODEL=models/v1/quantized/horizon-q4.gguf
```

## Project Structure

```
horizon/
├── data/              # Datasets (synthetic generation, raw, processed)
├── training/          # Training scripts, configs, model definitions
├── evaluation/        # Metrics, analysis, reports
├── quantization/      # Model quantization pipelines
├── inference/         # API, CLI, examples
├── experiments/       # Training runs & logs
├── models/            # Trained model artifacts
├── notebooks/         # Jupyter notebooks
├── tests/             # Unit & integration tests
└── docs/              # Documentation
```

## Key Features

- **Privacy-First:** Synthetic data only, no real child messages
- **Production-Ready:** Quantized INT4 models (~2GB) for flexible deployment
- **High Accuracy:** Target >0.85 F1 overall, >0.90 F1 on critical categories
- **Deployment-Agnostic:** Exports to GGUF, ONNX, vLLM, Ollama formats
- **Comprehensive Evaluation:** Per-category metrics, ablation studies, error analysis

## Documentation

- [Design Document](docs/superpowers/specs/2026-04-09-horizon-design.md) - Complete system architecture and methodology
- [Training Guide](docs/training.md) - Step-by-step training instructions (coming soon)
- [API Reference](docs/api.md) - REST API and CLI documentation (coming soon)
- [Model Card](models/v1.0/metadata/model_card.md) - Model capabilities and limitations (after training)

## Requirements

- Python 3.10+
- CUDA-capable GPU (A100 40GB recommended, RTX 4090 24GB minimum)
- 100GB disk space (datasets + models)
- Claude/GPT-4 API access (for data generation)

## Timeline

- **Weeks 1-2:** Foundation (data generation pipeline, 10K dataset)
- **Weeks 3-4:** Training infrastructure (baseline model)
- **Weeks 5-7:** Full training (50K dataset, v1.0 model)
- **Week 8:** Quantization & optimization
- **Week 9:** API & documentation
- **Week 10+:** Iteration & refinement

## Budget

- Data generation: $200 (API calls)
- Training: $150 (GPU compute)
- Testing: $50
- **Total: ~$400**

## Risk Categories

1. **Grooming** - Trust building, boundary testing, secrecy
2. **Bullying** - Harassment, threats, cyberbullying
3. **Sexual Content** - Explicit messages, inappropriate requests
4. **Isolation/Control** - Controlling behavior, network isolation
5. **Personal Info** - Requests for identifying information
6. **Platform Migration** - Moving to less monitored platforms
7. **Threats/Violence** - Violent threats, dangerous challenges

## Model Output Format

```json
{
  "risk_level": "high",
  "categories": ["grooming", "personal_info"],
  "confidence": 0.87,
  "matched_terms": ["our secret", "don't tell", "send photo"],
  "reasoning": "Adult establishing secrecy while requesting personal media",
  "message_risks": [
    {"index": 0, "risk": "low", "categories": []},
    {"index": 1, "risk": "high", "categories": ["grooming", "personal_info"]}
  ]
}
```

## License

[To be determined - pending SafeCircle legal review]

## Contributing

Project Horizon is currently in active development. Contribution guidelines will be published after v1.0 release.

## Contact

For questions about SafeCircle or Project Horizon, visit https://safecircle.tech

---

**Status:** In Development  
**Version:** 0.1.0 (Pre-release)  
**Last Updated:** 2026-04-09
