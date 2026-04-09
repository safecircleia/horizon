.PHONY: help install install-dev clean test format lint generate-data train evaluate quantize serve

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
	@echo "  make validate-data    Validate generated dataset quality"
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
	python -m data.generation.generators.main generate $(CATEGORY) --count $(COUNT)

generate-all:
	@echo "Generating complete 50K dataset..."
	make generate-data CATEGORY=grooming COUNT=8000
	make generate-data CATEGORY=bullying COUNT=7000
	make generate-data CATEGORY=sexual COUNT=7000
	make generate-data CATEGORY=isolation COUNT=5000
	make generate-data CATEGORY=personal_info COUNT=5000
	make generate-data CATEGORY=platform_move COUNT=3000
	make generate-data CATEGORY=threats COUNT=5000
	make generate-data CATEGORY=benign COUNT=10000

validate-data:
	python -m data.generation.validators.main validate data/raw/

# Training
train:
	@if [ -z "$(CONFIG)" ]; then echo "Error: CONFIG required. Use: make train CONFIG=configs/base.yaml"; exit 1; fi
	python -m training.scripts.train --config $(CONFIG)

train-quick:
	python -m training.scripts.train --config configs/quick.yaml

resume:
	@if [ -z "$(CHECKPOINT)" ]; then echo "Error: CHECKPOINT required. Use: make resume CHECKPOINT=experiments/run-1/checkpoints/step-5000"; exit 1; fi
	python -m training.scripts.train --resume $(CHECKPOINT)

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
