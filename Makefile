.PHONY: help install install-dev clean test format lint generate-data generate-all validate-data stats clean-data generate-test train evaluate quantize serve

# Default target
help:
	@echo "Project Horizon - SafeCircle Risk Detection Model"
	@echo ""
	@echo "Available commands:"
	@echo "  make install          Install production dependencies"
	@echo "  make install-dev      Install development dependencies"
	@echo "  make clean            Clean generated files and caches"
	@echo ""
	@echo "Data Generation:"
	@echo "  make generate-data CATEGORY=<category> COUNT=<count>"
	@echo "                        Generate synthetic conversations"
	@echo "  make generate-all     Generate complete 50K dataset"
	@echo "  make generate-test    Generate 100 benign samples for testing"
	@echo "  make validate-data    Validate generated dataset quality"
	@echo "  make stats            Show dataset statistics and validation"
	@echo "  make clean-data       Remove all generated JSONL files"
	@echo ""
	@echo "Training:"
	@echo "  make train CONFIG=<config>  Train model with specified config"
	@echo "  make train-quick      Quick training for iteration"
	@echo "  make resume CHECKPOINT=<path>  Resume training from checkpoint"
	@echo ""
	@echo "Evaluation:"
	@echo "  make evaluate CHECKPOINT=<path>  Evaluate model checkpoint"
	@echo "  make analyze-errors   Run error analysis on predictions"
	@echo ""
	@echo "Quantization:"
	@echo "  make quantize MODEL=<path> FORMAT=<format>"
	@echo "                        Quantize model (formats: q8, q4_k_m, q4_0)"
	@echo "  make validate-quantized  Validate quantized model quality"
	@echo ""
	@echo "Inference:"
	@echo "  make serve MODEL=<path>  Start inference API server"
	@echo "  make inference-cli CONVERSATION=<path>  Run CLI inference"
	@echo ""
	@echo "Development:"
	@echo "  make test             Run test suite"
	@echo "  make format           Format code with black"
	@echo "  make lint             Lint code with ruff"
	@echo "  make notebook         Start Jupyter Lab"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,api,tracking,viz]"
	pre-commit install

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .ruff_cache/

# Data Generation
generate-data:
	@if [ -z "$(CATEGORY)" ]; then echo "Error: CATEGORY required. Use: make generate-data CATEGORY=grooming COUNT=1000"; exit 1; fi
	@if [ -z "$(COUNT)" ]; then echo "Error: COUNT required. Use: make generate-data CATEGORY=grooming COUNT=1000"; exit 1; fi
	@PYTHON=python3; if [ -f .venv/bin/python ]; then PYTHON=.venv/bin/python; fi; \
	$$PYTHON -m data.generation.scripts.generate --category $(CATEGORY) --count $(COUNT) --generator $(or $(GENERATOR),bedrock) $(RESUME)

generate-all:
	@echo "Generating complete 50K dataset (resumable)..."
	make generate-data CATEGORY=grooming COUNT=8000 RESUME=--resume
	make generate-data CATEGORY=bullying COUNT=7000 RESUME=--resume
	make generate-data CATEGORY=sexual_content COUNT=7000 RESUME=--resume
	make generate-data CATEGORY=isolation COUNT=5000 RESUME=--resume
	make generate-data CATEGORY=personal_info COUNT=5000 RESUME=--resume
	make generate-data CATEGORY=platform_migration COUNT=3000 RESUME=--resume
	make generate-data CATEGORY=threats COUNT=5000 RESUME=--resume
	make generate-data CATEGORY=benign COUNT=10000 RESUME=--resume

validate-data:
	@if [ -f .venv/bin/python ]; then .venv/bin/python -m data.generation.validators.main validate data/raw/; else python3 -m data.generation.validators.main validate data/raw/; fi

stats:
	@if [ -f .venv/bin/python ]; then .venv/bin/python -m data.generation.scripts.stats; else python3 -m data.generation.scripts.stats; fi

clean-data:
	@echo "Removing generated JSONL files..."
	find data/raw -name "*.jsonl" -delete
	find data/processed -name "*.jsonl" -delete 2>/dev/null || true
	@echo "Done!"

generate-test:
	@echo "Generating 100 benign samples for testing..."
	@PYTHON=python3; if [ -f .venv/bin/python ]; then PYTHON=.venv/bin/python; fi; \
	$$PYTHON -m data.generation.scripts.generate --category benign --count 100 --generator $(or $(GENERATOR),bedrock) --output data/raw/test_benign.jsonl

# Training
preprocess:
	@PYTHON=python3; if [ -f .venv/bin/python ]; then PYTHON=.venv/bin/python; fi; \
	$$PYTHON -m training.scripts.preprocess --input data/raw --output data/processed

train:
	@if [ -z "$(CONFIG)" ]; then echo "Error: CONFIG required. Use: make train CONFIG=training/configs/base.yaml"; exit 1; fi
	@PYTHON=python3; if [ -f .venv/bin/python ]; then PYTHON=.venv/bin/python; fi; \
	$$PYTHON -m training.scripts.train --config $(CONFIG)

train-quick:
	@PYTHON=python3; if [ -f .venv/bin/python ]; then PYTHON=.venv/bin/python; fi; \
	$$PYTHON -m training.scripts.train --config training/configs/quick.yaml

resume:
	@if [ -z "$(CHECKPOINT)" ]; then echo "Error: CHECKPOINT required. Use: make resume CHECKPOINT=experiments/run-foo/checkpoints/step-500"; exit 1; fi
	@PYTHON=python3; if [ -f .venv/bin/python ]; then PYTHON=.venv/bin/python; fi; \
	$$PYTHON -m training.scripts.train --config training/configs/base.yaml --resume $(CHECKPOINT)

# Evaluation
evaluate:
	@if [ -z "$(CHECKPOINT)" ]; then echo "Error: CHECKPOINT required. Use: make evaluate CHECKPOINT=experiments/run-1/checkpoints/step-5000"; exit 1; fi
	python -m evaluation.metrics.evaluate --checkpoint $(CHECKPOINT) --test-set data/evaluation/test.jsonl

analyze-errors:
	python -m evaluation.analysis.error_analysis --predictions evaluation/reports/latest/predictions.jsonl

# Quantization
quantize:
	@if [ -z "$(MODEL)" ]; then echo "Error: MODEL required. Use: make quantize MODEL=models/v1/full FORMAT=q4_k_m"; exit 1; fi
	@if [ -z "$(FORMAT)" ]; then echo "Error: FORMAT required. Use: make quantize MODEL=models/v1/full FORMAT=q4_k_m"; exit 1; fi
	python -m quantization.scripts.quantize --model $(MODEL) --format $(FORMAT)

validate-quantized:
	python -m quantization.validation.validate --original $(ORIGINAL) --quantized $(QUANTIZED)

# Inference
serve:
	@if [ -z "$(MODEL)" ]; then echo "Error: MODEL required. Use: make serve MODEL=models/v1/quantized/horizon-q4.gguf"; exit 1; fi
	python -m inference.api.serve --model $(MODEL) --port 8000

inference-cli:
	@if [ -z "$(CONVERSATION)" ]; then echo "Error: CONVERSATION required. Use: make inference-cli CONVERSATION=examples/test.json"; exit 1; fi
	python -m inference.cli.main --model $(MODEL) --conversation $(CONVERSATION)

# Development
test:
	pytest tests/ -v

test-coverage:
	pytest tests/ --cov=. --cov-report=html --cov-report=term

format:
	black .
	ruff check --fix .

lint:
	ruff check .
	black --check .

notebook:
	jupyter lab notebooks/

# Docker (future)
docker-build:
	docker build -t horizon:latest .

docker-run:
	docker run -it --gpus all -p 8000:8000 horizon:latest
