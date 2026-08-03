param(
    [Parameter(Mandatory=$false)]
    [string]$Command = "help"
)

# Detect a working Python executable (avoids the broken Windows Store stub for python3)
$PYTHON = $null
foreach ($candidate in @("py", "python", "python3")) {
    try {
        $ver = & $candidate --version 2>&1
        if ($LASTEXITCODE -eq 0 -and $ver -match "Python 3") {
            $PYTHON = $candidate
            break
        }
    } catch {}
}
if (-not $PYTHON) {
    Write-Host "ERROR: Python 3.10+ not found. Install from https://python.org/downloads" -ForegroundColor Red
    exit 1
}

Write-Host "Using: $PYTHON" -ForegroundColor Cyan

switch ($Command) {

    "help" {
        Write-Host ""
        Write-Host "Behavioral Finance AI - Command Runner" -ForegroundColor Green
        Write-Host "Usage: .\run.ps1 <command>" -ForegroundColor Green
        Write-Host ""
        Write-Host "  install         Install all dependencies"
        Write-Host "  generate-data   Generate synthetic persona transaction data"
        Write-Host "  train-lstm      Train ISE LSTM model"
        Write-Host "  train-prophet   Fit FB-Prophet balance forecaster"
        Write-Host "  train-ssi       Train SSI XGBoost exit-signal classifier"
        Write-Host "  train-ise       Full ISE pipeline (generate + lstm + prophet)"
        Write-Host "  train-all       Train all models end-to-end"
        Write-Host "  api             Start FastAPI server at http://localhost:8000/docs"
        Write-Host "  test            Run full test suite"
        Write-Host "  test-cov        Run tests with coverage report"
        Write-Host "  simulate        Run full ISE simulation for all personas"
        Write-Host "  lint            Run ruff linter"
        Write-Host "  format          Run black formatter"
        Write-Host "  clean           Remove __pycache__ and .pytest_cache"
        Write-Host ""
    }

    "install" {
        Write-Host "Installing dependencies..." -ForegroundColor Yellow
        pip install -r requirements.txt
    }

    "generate-data" {
        Write-Host "Generating synthetic persona transaction data..." -ForegroundColor Yellow
        & $PYTHON -m src.personas.faker_generator
    }

    "train-lstm" {
        Write-Host "Training ISE LSTM model..." -ForegroundColor Yellow
        & $PYTHON -m src.ise.lstm_model
    }

    "train-prophet" {
        Write-Host "Fitting FB-Prophet models..." -ForegroundColor Yellow
        & $PYTHON -m src.ise.prophet_model
    }

    "train-ssi" {
        Write-Host "Training SSI XGBoost exit-signal model..." -ForegroundColor Yellow
        & $PYTHON -m src.ssi.xgboost_model
    }

    "train-ise" {
        Write-Host "Running full ISE training pipeline..." -ForegroundColor Yellow
        & $PYTHON -m src.personas.faker_generator
        & $PYTHON -m src.ise.lstm_model
        & $PYTHON -m src.ise.prophet_model
        Write-Host "ISE training complete." -ForegroundColor Green
    }

    "train-all" {
        Write-Host "Training all models..." -ForegroundColor Yellow
        & $PYTHON -m src.personas.faker_generator
        & $PYTHON -m src.ise.lstm_model
        & $PYTHON -m src.ise.prophet_model
        & $PYTHON -m src.ssi.xgboost_model
        Write-Host "All models trained." -ForegroundColor Green
    }

    "api" {
        Write-Host "Starting FastAPI at http://localhost:8000/docs ..." -ForegroundColor Yellow
        uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    }

    "api-prod" {
        Write-Host "Starting FastAPI in production mode..." -ForegroundColor Yellow
        uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
    }

    "test" {
        Write-Host "Running test suite..." -ForegroundColor Yellow
        & $PYTHON -m pytest tests/ -v --tb=short
    }

    "test-cov" {
        Write-Host "Running tests with coverage..." -ForegroundColor Yellow
        & $PYTHON -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
        Write-Host "Coverage report saved to htmlcov/index.html" -ForegroundColor Cyan
    }

    "simulate" {
        Write-Host "Running full ISE simulation..." -ForegroundColor Yellow
        & $PYTHON -m src.ise.ise_engine
    }

    "lint" {
        Write-Host "Running ruff linter..." -ForegroundColor Yellow
        & $PYTHON -m ruff check src/ api/ tests/
    }

    "format" {
        Write-Host "Running black formatter..." -ForegroundColor Yellow
        & $PYTHON -m black src/ api/ tests/
    }

    "report" {
        Write-Host "Generating training & evaluation report..." -ForegroundColor Yellow
        & $PYTHON -m src.reporting.generate_report
        Write-Host "Report ready: reports/summary/model_card.html" -ForegroundColor Green
    }

    "clean" {
        Write-Host "Cleaning cache files..." -ForegroundColor Yellow
        Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Recurse -Directory -Filter ".pytest_cache" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue
        Write-Host "Clean complete." -ForegroundColor Green
    }

    default {
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Write-Host "Run .\run.ps1 help to see available commands." -ForegroundColor Yellow
    }
}
