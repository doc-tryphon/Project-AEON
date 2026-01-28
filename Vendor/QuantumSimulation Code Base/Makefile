# Quantum Simulation Project Makefile
.PHONY: help install install-dev test format lint clean demo benchmark docs

# Default target
help:
	@echo "Available commands:"
	@echo "  install       Install production dependencies"
	@echo "  install-dev   Install development dependencies"
	@echo "  test          Run all tests"
	@echo "  test-unit     Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  format        Format code with black"
	@echo "  lint          Lint code with flake8"
	@echo "  clean         Clean cache and temporary files"
	@echo "  demo          Run quick demo"
	@echo "  benchmark     Run performance benchmark"
	@echo "  docs          Generate documentation"
	@echo "  setup         Full development environment setup"

# Installation targets
install:
	python scripts/setup/install.py

install-dev:
	python scripts/setup/install.py --dev

# Testing targets
test:
	python -m pytest tests/ -v

test-unit:
	python -m pytest tests/unit/ -v

test-integration:
	python -m pytest tests/integration/ -v

test-coverage:
	python -m pytest tests/ --cov=src --cov-report=html

# Code quality targets
format:
	python -m black src/ tests/ main.py scripts/

lint:
	python -m flake8 src/ tests/ main.py scripts/

typecheck:
	python -m mypy src/

# Cleaning targets
clean:
	@echo "Cleaning cache and temporary files..."
	@python -c "import shutil, os; [shutil.rmtree(p, ignore_errors=True) for p in ['__pycache__', 'cache', 'temp', '.pytest_cache', 'htmlcov', '.coverage']]"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleanup complete!"

# Simulation targets
demo:
	python main.py --mode demo

interactive:
	python main.py --mode interactive

benchmark:
	python main.py --mode benchmark --steps 500

# Development setup
setup: install-dev
	@echo "Setting up pre-commit hooks..."
	@python -m pre_commit install 2>/dev/null || echo "Pre-commit not available"
	@echo "Development environment ready!"

# Analysis targets
analyze:
	python tools/codebase_analyzer.py

count:
	python scripts/analysis/count_code.py

# Documentation targets
docs:
	@echo "Documentation generation not yet implemented"

# Performance targets
profile:
	python -m cProfile -o profile.stats main.py --mode demo
	@echo "Profile saved to profile.stats"

# Verification targets
verify:
	python scripts/setup/verify_installation.py

# Git workflow helpers
pre-commit: format lint test
	@echo "Pre-commit checks passed!"

# Docker targets (if needed later)
docker-build:
	@echo "Docker support not yet implemented"

docker-run:
	@echo "Docker support not yet implemented"