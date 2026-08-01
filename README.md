# AI-Driven Behavioral Finance Framework

An AI-powered simulation framework that bridges the **Intent-Action Gap** in retail investing by automating savings and investment decisions — removing emotional bias from wealth creation.

##  Overview

Retail investors rarely fail due to lack of market data — they fail due to lack of **behavioral discipline**. This project addresses that gap using a dual-engine AI architecture:

- **Invisible Savings Engine (ISE)** — Predicts safe-to-save amounts from spending patterns and automates micro-savings, removing the willpower required to save manually.
- **Smart Stock Investing (SSI)** — Uses objective, data-driven scoring and automated exit signals to prevent emotional trading decisions like panic selling and the disposition effect.

The system is validated entirely in a **simulation environment** using synthetic user personas and historical market data — no live banking or brokerage integration.

##  Objectives

1. Automate the transition from income to savings, bridging the Intent-Action Gap
2. Build an ISE that predicts safe-to-save amounts using cash-flow forecasting
3. Mitigate the Disposition Effect and Panic Selling via automated exit signals
4. Use sentiment analysis to filter market noise and improve stock selection


##  Project Structure

```
behavioral-finance-ai/
├── docs/            # Project documentation & reports
├── data/            # Raw, synthetic, and processed datasets
├── src/
│   ├── ise/         # Invisible Savings Engine
│   ├── ssi/         # Smart Stock Investing
│   ├── sentiment/   # FinBERT pipeline
│   ├── personas/    # Faker persona generator
│   └── utils/       # Metrics: Sharpe Ratio, Max Drawdown
├── models/          # Trained model artifacts
├── notebooks/       # Experiment notebooks
└── tests/           # Unit tests
```

## ⚙️ Getting Started

### Prerequisites
- Python 3.10+
- PostgreSQL 14+
- pip / virtualenv

### Installation

```bash
git clone https://github.com/<org>/behavioral-finance-ai.git
cd behavioral-finance-ai

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file:

```
DATABASE_URL=postgresql://user:password@localhost:5432/behavioral_finance
KAGGLE_DATASET_PATH=data/raw/
```

### Generate Synthetic Personas

```bash
python src/personas/faker_generator.py
```

### Train Models

```bash
# Invisible Savings Engine
python src/ise/lstm_model.py
python src/ise/prophet_model.py

# Smart Stock Investing
python src/ssi/scoring_model.py
python src/ssi/xgboost_model.py

# Sentiment Analysis
python src/sentiment/finbert_model.py
```
