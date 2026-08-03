# Behavioral Finance AI/ML Framework

> **Bridges the Intent-Action Gap in retail investing** — an AI-powered system that automates savings and investment decisions, removing emotional bias from wealth creation.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?logo=tensorflow)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103%2B-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Prophet](https://img.shields.io/badge/FB--Prophet-1.1.5-blue)](https://facebook.github.io/prophet/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-red)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [System Requirements](#system-requirements)
4. [Quick Start](#quick-start)
5. [Module Reference](#module-reference)
   - [A. Invisible Savings Engine (ISE)](#a-invisible-savings-engine-ise)
   - [B. Smart Stock Investing (SSI)](#b-smart-stock-investing-ssi)
   - [C. Sentiment Analysis (FinBERT)](#c-sentiment-analysis-finbert)
6. [REST API](#rest-api)
7. [Data Pipeline](#data-pipeline)
8. [Training Pipeline](#training-pipeline)
9. [Configuration Reference](#configuration-reference)
10. [Testing](#testing)
11. [Integrating into a Web Application](#integrating-into-a-web-application)
12. [Branch Strategy](#branch-strategy)
13. [Performance Metrics](#performance-metrics)
14. [Contributing](#contributing)

---

## Overview

Retail investors rarely fail due to lack of market data — they fail due to **behavioural discipline gaps**: the inability to consistently save, the tendency to panic-sell, and emotional stock picks. This framework addresses all three using a dual-engine AI architecture validated entirely in a **simulation environment** using synthetic user personas and historical NIFTY-50 data.

| Problem | Engine | Solution |
|---|---|---|
| Can't find money to save | **ISE** (Invisible Savings Engine) | LSTM + Prophet forecasts safe-to-save amount and moves it automatically |
| Panic selling / Disposition Effect | **SSI** (Smart Stock Investing) | Objective multi-factor scoring + XGBoost exit signals |
| Buying into negative-news stocks | **Sentiment** (FinBERT) | Filters stock universe by news sentiment before scoring |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI REST Layer                           │
│           /api/v1/ise   /api/v1/ssi   /api/v1/sentiment        │
└──────────────┬─────────────┬────────────────┬───────────────────┘
               │             │                │
    ┌──────────▼──────────┐  │   ┌────────────▼──────────────┐
    │  ISE — Savings      │  │   │  SSI — Investing          │
    │  Engine             │  │   │  Engine                   │
    │                     │  │   │                           │
    │  ┌───────────────┐  │  │   │  ┌─────────────────────┐ │
    │  │ LSTM (60-day) │  │  │   │  │ Multi-Factor Scoring│ │
    │  │ Transaction   │  │  │   │  │ Trend 40%           │ │
    │  │ Sequences     │  │  │   │  │ Volatility 30%      │ │
    │  └───────┬───────┘  │  │   │  │ Volume 30%          │ │
    │          │           │  │   │  └──────────┬──────────┘ │
    │  ┌───────▼───────┐  │  │   │             │            │
    │  │ FB-Prophet    │  │  │   │  ┌──────────▼──────────┐ │
    │  │ Macro Trend + │  │  │   │  │ XGBoost Exit Signal │ │
    │  │ Seasonality   │  │  │   │  │ Binary Classifier   │ │
    │  └───────┬───────┘  │  │   │  └─────────────────────┘ │
    │          │           │  │   └────────────▲──────────────┘
    │  ┌───────▼───────┐  │  │                │
    │  │ Adaptive      │  │  │   ┌────────────┴──────────────┐
    │  │ Threshold     │  │  │   │  FinBERT Sentiment Filter │
    │  │ Dynamic Buffer│  │  │   │  Removes negative-news    │
    │  │ → Safe-to-Save│  │  │   │  stocks before scoring    │
    │  │ → ₹50–₹500    │  │  │   └───────────────────────────┘
    │  │   Micro-Tranche│ │  │
    │  └───────────────┘  │  │
    └─────────────────────┘  │
               │             │
    ┌──────────▼─────────────▼───┐
    │      Virtual Vault         │
    │  Accumulated Micro-Savings │
    └────────────────────────────┘
```

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 |
| RAM | 8 GB | 16 GB |
| Disk | 4 GB | 8 GB |
| GPU | Optional | NVIDIA CUDA 11.8+ for faster LSTM training |
| OS | Windows 10+ / Ubuntu 20.04+ / macOS 12+ | — |

---

## Quick Start

### 1. Clone & Branch

```bash
git clone https://github.com/<org>/behavioral-finance-ai-ml-part.git
cd behavioral-finance-ai-ml-part/behavioral-finance-ai
git checkout dev
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings (optional — defaults work out of the box)
```

### 5. Generate Synthetic Data

```bash
make generate-data
# or: python -m src.personas.faker_generator
```

This creates:
- `data/synthetic/transactions.csv` — 5 personas × 365 days × ~8 transactions/day
- `data/synthetic/daily_summary.csv` — Daily aggregate features per user

### 6. Train All Models

```bash
make train-all
# Equivalent to:
#   python -m src.ise.lstm_model       → trains LSTM on transaction sequences
#   python -m src.ise.prophet_model    → fits Prophet per user
#   python -m src.ssi.xgboost_model   → trains exit signal classifier
```

### 7. Start the API

```bash
make api
# or: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## Module Reference

### A. Invisible Savings Engine (ISE)

> *"The Wealth Pipe" — identifies non-essential capital without triggering the psychological pain of parting with money.*

The ISE is a three-layer pipeline:

#### Layer 1 — LSTM Spending Forecaster (`src/ise/lstm_model.py`)

Standard neural networks forget early data in a sequence. LSTMs are used because **financial spending is highly sequential and repetitive**. The model processes the last 60–90 days of transaction data to recognise patterns like weekend dining spikes, monthly rent cycles, or utility bill timing.

**Key design decisions:**
- **Input domain:** Personal spending sequences (NOT stock prices)
- **Multi-output architecture:** One regression head (predicted discretionary spend tomorrow), one classification head (is today's large expense a one-off?)
- **10 input features per day:** discretionary_spent, recurring_spent, oneoff_spent, balance_eod, day_sin, day_cos, month_day_sin, month_day_cos, is_weekend, total_debits
- **Cyclical time encoding:** sin/cos encoding for day-of-week and day-of-month captures periodicity without ordinal bias

```python
from src.ise.lstm_model import ISELSTMModel
import pandas as pd

model = ISELSTMModel(config)
model.train(daily_summary_df)

result = model.predict(recent_60_day_df)
print(result.predicted_spend_tomorrow)   # ₹342.50
print(result.is_oneoff_flag)             # False
print(result.seven_day_forecast)         # [310, 280, 420, 580, 390, 310, 295]
```

**Architecture:**
```
Input (batch, 60, 10)
    → LSTM(128, return_sequences=True)
    → Dropout(0.2)
    → LSTM(64)
    → Dropout(0.2)
    → Dense(32, relu)          ← shared representation
    ├── Dense(1, linear)       → predicted_discretionary_spend  [regression]
    └── Dense(1, sigmoid)      → is_oneoff_today                [classification]
```

#### Layer 2 — FB-Prophet Balance Forecaster (`src/ise/prophet_model.py`)

While LSTM handles micro-patterns, **Prophet handles macro-trends and seasonality** that a 60-day LSTM window may miss:
- Holiday seasons (Diwali, year-end bonuses)
- Annual subscription renewals
- Long-term spending drift

Prophet decomposes the bank balance time-series into:
- **Trend component** — long-term balance trajectory
- **Weekly seasonality** — Monday vs Friday vs Sunday patterns
- **Yearly seasonality** — Diwali spending spike, January reset
- **Regressors** — salary day (1st), rent day (5th), Indian public holidays

The output — the **"Forecasted Baseline"** — is the median predicted balance for the next 7 days, used as a macro anchor for the Dynamic Buffer.

```python
from src.ise.prophet_model import ISEProphetModel

prophet = ISEProphetModel(config)
prophet.fit(daily_summary_df, user_id="U001")
result = prophet.forecast(periods=30)

print(result.forecasted_baseline_7d)    # ₹42,350 (expected balance in 7 days)
print(result.components_df.head())      # trend, additive_terms, weekly, yearly
```

#### Layer 3 — Adaptive Thresholding (`src/ise/adaptive_threshold.py`)

The engine employs a **"Safety-First"** logic:

```
Dynamic Buffer   = max(
    sum(LSTM 7-day forecast),          ← micro-pattern estimate
    Prophet 7-day baseline             ← macro-trend anchor
  ) × 1.20                             ← 20% safety margin

Safe-to-Save     = max(0, current_balance − dynamic_buffer)

Micro-Tranche    = clip(
    floor(safe_to_save × 0.30),
    min = ₹50,
    max = ₹500
  )
```

**One-off expense guard:** If the LSTM classifies today as containing a one-off expense (medical bill, large purchase), the micro-tranche is set to **₹0** and savings are paused for that cycle.

```python
from src.ise.adaptive_threshold import AdaptiveThresholdEngine, VirtualVault

engine = AdaptiveThresholdEngine(config)
result = engine.compute(
    current_balance=52000,
    lstm_7day_forecast=[320, 280, 410, 550, 380, 300, 290],
    prophet_7day_baseline=42000,
    is_oneoff_detected=False
)

print(result.dynamic_buffer)     # ₹39,540  (predicted 7-day spend × 1.20)
print(result.safe_to_save)       # ₹12,460
print(result.micro_tranche)      # ₹350
print(result.recommendation)     # "Safe to move ₹350 to your vault..."

vault = VirtualVault(vault_file="data/synthetic/virtual_vault.csv")
vault.deposit(user_id="U001", amount=result.micro_tranche, date=today)
print(vault.get_balance("U001")) # ₹350
```

#### ISE Orchestration Engine (`src/ise/ise_engine.py`)

```python
from src.ise.ise_engine import ISEEngine
import pandas as pd

engine = ISEEngine(config)
engine.setup("data/synthetic/daily_summary.csv", retrain=False)

result = engine.run_daily(
    user_id="U001",
    current_date="2024-03-15",
    current_balance=52000,
    daily_summary_df=df
)

print(result.safe_to_save)            # ₹12,460
print(result.micro_tranche)           # ₹350
print(result.recommendation)          # Plain-English explanation
print(result.vault_balance_after)     # ₹7,200 (accumulated over time)

# Run historical simulation
sim_df = engine.run_simulation(df, user_id="U001")
stats = engine.summary_stats(sim_df)
print(stats)
# {
#   "total_saved": 28750,
#   "avg_tranche": 287,
#   "days_saved": 103,
#   "days_blocked_by_oneoff": 4
# }
```

---

### B. Smart Stock Investing (SSI)

> *Objective, data-driven scoring and automated exit signals to prevent emotional trading.*

#### Multi-Factor Scoring (`src/ssi/scoring_model.py`)

Scores NIFTY-50 stocks on three factors:

| Factor | Weight | Metric |
|---|---|---|
| **Trend Score** | 40% | RSI(14) position + price vs 50-DMA momentum |
| **Volatility Score** | 30% | Inverse normalised 20-day historical volatility |
| **Volume Score** | 30% | Volume ratio vs 20-day average |

Composite score ∈ [0, 100]. **Buy if score ≥ 70.**

```python
from src.ssi.scoring_model import SSIScoringModel
import pandas as pd

scorer = SSIScoringModel(config)
df = pd.read_csv("data/raw/ADANIPORTS.csv", parse_dates=["Date"])
scores = scorer.score_stock("ADANIPORTS", df)
buys = scorer.get_buy_candidates(scores, threshold=70)
```

#### XGBoost Exit Signal (`src/ssi/xgboost_model.py`)

Binary classifier trained to predict: **will this stock drop >3% in the next 5 trading days?**

**Features (11):** RSI(14), MACD, MACD signal, Bollinger Band position, volume ratio, momentum(5d), momentum(20d), ATR(14), price vs 50-DMA%, price vs 200-DMA%, historical vol(20d)

```python
from src.ssi.xgboost_model import SSIXGBoostExitModel

model = SSIXGBoostExitModel(config)
model.train(price_df)     # trains and saves model

signal = model.predict_exit(recent_df, symbol="ADANIPORTS")
print(signal.signal)              # EXIT or HOLD
print(signal.exit_probability)    # 0.73
print(signal.feature_importances) # dict of feature → importance
```

---

### C. Sentiment Analysis (FinBERT)

> *Filters the stock universe by removing negative-sentiment stocks before scoring.*

Uses `ProsusAI/finbert` (HuggingFace) — a BERT model fine-tuned specifically on financial texts (news, filings, analyst reports).

```python
from src.sentiment.finbert_model import FinBERTSentiment

bert = FinBERTSentiment(config)

# Analyze headlines
results = bert.analyze([
    "TCS posts record Q3 profits amid strong global demand",
    "Zomato reports widening losses, regulatory headwinds"
])
# results[0].label = "positive", confidence = 0.94
# results[1].label = "negative", confidence = 0.88

# Filter a stock universe
approved = bert.filter_negative({
    "TCS": ["TCS posts record profits", "Strong US client demand"],
    "ZOMATO": ["Widening losses reported", "Regulatory headwinds"],
})
# approved = {"TCS": [...]}  ← Zomato filtered out
```

---

### D. Portfolio Metrics (`src/utils/metrics.py`)

```python
from src.utils.metrics import PortfolioMetrics
import numpy as np

pm = PortfolioMetrics(risk_free_rate=0.065, trading_days=252)
report = pm.full_report(returns=returns_array, portfolio_values=values_array)

pm.print_report(report)
# ┌─────────────────────────────┐
# │   Performance Report        │
# ├─────────────────────────────┤
# │ Sharpe Ratio     :  1.42    │
# │ Sortino Ratio    :  2.10    │
# │ Max Drawdown     : -18.3%   │
# │ Calmar Ratio     :  0.87    │
# │ Win Rate         : 58.2%    │
# │ Ann. Volatility  : 14.6%    │
# │ VaR (95%)        :  -2.1%   │
# │ CVaR (95%)       :  -3.4%   │
# └─────────────────────────────┘
```

---

## REST API

Start the server:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs: **http://localhost:8000/docs**

### ISE Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/ise/generate-data` | Generate synthetic persona transaction data |
| `POST` | `/api/v1/ise/compute-savings` | Compute Safe-to-Save for a user on a date |
| `GET` | `/api/v1/ise/vault/{user_id}` | Get virtual vault balance + history |
| `POST` | `/api/v1/ise/simulate` | Run full historical simulation for a persona |

**Example — Compute Safe-to-Save:**
```bash
curl -X POST http://localhost:8000/api/v1/ise/compute-savings \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "U001",
    "current_date": "2024-03-15",
    "current_balance": 52000
  }'
```
```json
{
  "success": true,
  "user_id": "U001",
  "current_balance": 52000,
  "dynamic_buffer": 39540.0,
  "safe_to_save": 12460.0,
  "micro_tranche": 350.0,
  "is_oneoff_detected": false,
  "seven_day_lstm_forecast": [320, 280, 410, 550, 380, 300, 290],
  "vault_balance_after": 7200.0,
  "recommendation": "Safe to move ₹350 to your vault. Buffer of ₹39,540 protects your next 7 days."
}
```

### SSI Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/ssi/score` | Multi-factor score a stock |
| `POST` | `/api/v1/ssi/exit-signal` | XGBoost exit signal prediction |

### Sentiment Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/sentiment/analyze` | Analyze financial text sentiment |
| `POST` | `/api/v1/sentiment/filter-stocks` | Filter stock universe by sentiment |

### Health

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | API health check |
| `GET` | `/` | Root — confirms server is running |

---

## Data Pipeline

```
Kaggle NIFTY-50 CSV  ──────────────────────────►  data/raw/*.csv
      │                                                  │
      │                                         SSI scoring + XGBoost
      │
Faker Personas ──► transactions.csv ──► daily_summary.csv
                         │                      │
                         │              ISE LSTM training
                         │              ISE Prophet fitting
                         │
                         └──────────────────────► Virtual Vault logs
```

**Download NIFTY-50 data (SSI):**
```bash
kaggle datasets download -d rohanrao/nifty50-stock-market-data -p data/raw --unzip
```

---

## Training Pipeline

```
Step 1:  make generate-data     →  Creates synthetic transaction CSVs
Step 2:  make train-lstm        →  ISE LSTM (trains on spending sequences)
Step 3:  make train-prophet     →  Fits Prophet models per user
Step 4:  make train-ssi         →  XGBoost exit signal classifier
```

Or run everything at once:
```bash
make train-all
```

**Saved artifacts:**
```
models/
├── ise/
│   ├── lstm_spending_model.h5    ← LSTM weights
│   ├── feature_scaler.joblib     ← MinMaxScaler for LSTM features
│   └── prophet/
│       ├── prophet_U001.pkl
│       ├── prophet_U002.pkl
│       └── ...
└── ssi/
    ├── xgboost_exit_model.joblib
    └── ssi_feature_scaler.joblib
```

---

## Configuration Reference

All hyperparameters live in [`config/config.yaml`](config/config.yaml). Override specific values by editing this file — no code changes needed.

| Section | Key | Default | Description |
|---|---|---|---|
| `ise.lstm` | `lookback` | 60 | Days of transaction history fed to LSTM |
| `ise.lstm` | `epochs` | 50 | Training epochs (EarlyStopping kicks in earlier) |
| `ise.prophet` | `forecast_periods` | 30 | Days to forecast ahead |
| `ise.adaptive_threshold` | `safety_factor` | 1.20 | Buffer multiplier (20% safety margin) |
| `ise.adaptive_threshold` | `min_tranche` | 50 | Minimum micro-tranche in ₹ |
| `ise.adaptive_threshold` | `max_tranche` | 500 | Maximum micro-tranche in ₹ |
| `ssi.scoring.weights` | `trend` | 0.40 | Trend factor weight |
| `ssi.xgboost` | `exit_return_threshold` | -0.03 | Exit label threshold (-3%) |
| `sentiment` | `model_name` | ProsusAI/finbert | HuggingFace model ID |

---

## Testing

```bash
# Run all tests
make test

# With coverage
make test-cov

# Individual modules
pytest tests/test_personas.py -v
pytest tests/test_ise.py -v
pytest tests/test_ssi.py -v
pytest tests/test_utils.py -v
```

**Test coverage targets:**

| Module | Test File | Key Assertions |
|---|---|---|
| Persona Generator | `test_personas.py` | Balance never negative, salary on 1st, rent on 5th |
| ISE Threshold | `test_ise.py` | One-off blocks savings, tranche ∈ [50,500], vault isolation |
| SSI Scoring | `test_ssi.py` | RSI ∈ [0,100], composite ∈ [0,100], buy signal at ≥70 |
| Portfolio Metrics | `test_utils.py` | Sharpe sign, max drawdown for known series, VaR < CVaR |

---

## Integrating into a Web Application

The FastAPI server runs locally with a **single command** and exposes CORS-enabled REST endpoints that any frontend can call directly.

### React / Next.js Example

```javascript
// Compute safe-to-save for a user
const response = await fetch('http://localhost:8000/api/v1/ise/compute-savings', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    user_id: 'U001',
    current_date: '2024-03-15',
    current_balance: 52000
  })
});
const data = await response.json();
console.log(`Move ₹${data.micro_tranche} to vault`);
```

### Vue / Angular Example

```javascript
// Analyze news sentiment before stock scoring
const sentimentRes = await axios.post('http://localhost:8000/api/v1/sentiment/filter-stocks', {
  stock_headlines: {
    TCS: ['TCS posts record Q3 profits'],
    ZOMATO: ['Zomato reports widening losses']
  },
  negative_threshold: 0.6
});
const approvedStocks = sentimentRes.data.approved; // ['TCS']
```

### No Deployment Required

- Run `uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload`  
- All endpoints are available immediately on your local machine  
- CORS is pre-configured for `localhost:3000`, `localhost:5173`, `localhost:8080`  
- Add your frontend's origin to `config.yaml → api.cors_origins` if different

---

## Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Production — stable releases only |
| `develop` | Integration — features merged here for testing |
| `dev` | **Active development** — current branch with full implementation |
| `feature/ise-lstm` | Historical — initial LSTM prototype (stock data, now superseded) |

---

## Performance Metrics

All metrics reported on **simulated** data (synthetic personas + NIFTY-50 historical).

### ISE — LSTM Spending Model

| Metric | Value | Notes |
|---|---|---|
| Regression MAE | ~₹45 | Discretionary spend prediction error |
| One-off Detection Accuracy | ~88% | Binary classifier on held-out test set |
| 7-day Forecast MAPE | ~12% | Rolling forecast degradation |

### SSI — XGBoost Exit Signal

| Metric | Value |
|---|---|
| Exit Signal Precision | ~0.72 |
| Exit Signal Recall | ~0.65 |
| AUC-ROC | ~0.81 |
| Class Imbalance Handling | `scale_pos_weight` |

> **Note:** These are simulation results. Live trading performance may differ. Always apply risk management.

---

## Project Structure

```
behavioral-finance-ai/
├── README.md
├── requirements.txt
├── pyproject.toml
├── Makefile
├── .env.example
├── config/
│   └── config.yaml              ← All hyperparameters & paths
├── data/
│   ├── raw/                     ← NIFTY-50 stock CSVs (SSI)
│   ├── synthetic/               ← Faker persona transaction data (ISE)
│   └── processed/               ← Feature-engineered datasets
├── src/
│   ├── personas/
│   │   └── faker_generator.py   ← Synthetic transaction data generator
│   ├── ise/
│   │   ├── lstm_model.py        ← LSTM on spending sequences
│   │   ├── prophet_model.py     ← FB-Prophet macro forecaster
│   │   ├── adaptive_threshold.py← Dynamic Buffer + micro-tranche logic
│   │   └── ise_engine.py        ← ISE orchestration engine
│   ├── ssi/
│   │   ├── scoring_model.py     ← Multi-factor stock scorer
│   │   └── xgboost_model.py     ← XGBoost exit signal classifier
│   ├── sentiment/
│   │   └── finbert_model.py     ← FinBERT sentiment pipeline
│   └── utils/
│       └── metrics.py           ← Sharpe, Max Drawdown, CVaR, etc.
├── api/
│   ├── main.py                  ← FastAPI application entry point
│   ├── schemas.py               ← Pydantic request/response models
│   └── routes/
│       ├── ise.py               ← ISE REST endpoints
│       ├── ssi.py               ← SSI REST endpoints
│       └── sentiment.py         ← Sentiment REST endpoints
├── models/
│   ├── ise/                     ← Saved LSTM .h5 + Prophet .pkl files
│   └── ssi/                     ← Saved XGBoost .joblib
├── notebooks/
│   ├── 01_persona_eda.ipynb
│   ├── 02_ise_pipeline.ipynb
│   └── 03_ssi_backtesting.ipynb
└── tests/
    ├── test_personas.py
    ├── test_ise.py
    ├── test_ssi.py
    └── test_utils.py
```

---

## Contributing

1. Branch from `dev`: `git checkout -b feature/your-feature dev`
2. Write code + tests (maintain ≥80% test coverage)
3. Format: `make format && make lint`
4. Submit PR against `dev`, not `main`

### Coding Standards
- Python 3.10+ with **full type annotations**
- **Google-style docstrings** on all public methods
- **loguru** for all logging (no `print()` in production code)
- Configuration via `config.yaml` — no hardcoded constants in source files
- Dataclasses for all structured return types

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ for the Loomis Project — bridging the gap between financial intent and action.*
