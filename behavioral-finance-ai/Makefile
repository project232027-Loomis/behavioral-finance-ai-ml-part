# ==============================================================================
# Behavioral Finance AI — Makefile
# ==============================================================================

.PHONY: help install generate-data train-ise train-ssi train-all api test lint clean

PYTHON := $(shell which python3 2>/dev/null || which python 2>/dev/null || echo python3)
SRC    := src
DATA   := data/synthetic/daily_summary.csv

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install all dependencies
	pip install -r requirements.txt

# ── Data Generation ──────────────────────────────────────────────────────────

generate-data:  ## Generate synthetic persona transaction data
	$(PYTHON) -m src.personas.faker_generator

# ── Model Training ────────────────────────────────────────────────────────────

train-lstm:  ## Train ISE LSTM model on synthetic transaction data
	$(PYTHON) -m src.ise.lstm_model

train-prophet:  ## Fit FB-Prophet balance forecaster for all personas
	$(PYTHON) -m src.ise.prophet_model

train-ssi:  ## Train SSI XGBoost exit-signal classifier
	$(PYTHON) -m src.ssi.xgboost_model

train-ise: generate-data train-lstm train-prophet  ## Full ISE training pipeline
	@echo "✅ ISE training complete"

train-all: train-ise train-ssi  ## Train all models end-to-end
	@echo "✅ All models trained"

# ── API ───────────────────────────────────────────────────────────────────────

api:  ## Start FastAPI server (dev mode with hot reload)
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

api-prod:  ## Start FastAPI server (production mode)
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2

# ── Testing ───────────────────────────────────────────────────────────────────

test:  ## Run full test suite
	pytest tests/ -v --tb=short

test-cov:  ## Run tests with coverage report
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# ── Simulation ────────────────────────────────────────────────────────────────

simulate:  ## Run full ISE simulation for all personas
	$(PYTHON) -m src.ise.ise_engine

report:  ## Generate training & evaluation visual report (HTML + plots)
	$(PYTHON) -m src.reporting.generate_report
	@echo "Report saved to reports/summary/model_card.html"

# ── Code Quality ──────────────────────────────────────────────────────────────

lint:  ## Run ruff linter
	ruff check src/ api/ tests/

format:  ## Run black formatter
	black src/ api/ tests/

type-check:  ## Run mypy type checker
	mypy src/ api/ --ignore-missing-imports

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:  ## Remove generated files and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	@echo "✅ Cleaned"
