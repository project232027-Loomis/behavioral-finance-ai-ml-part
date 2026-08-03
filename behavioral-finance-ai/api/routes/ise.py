"""
ISE API Routes
===============
Exposes the Invisible Savings Engine over REST.

Endpoints:
  POST /api/v1/ise/generate-data      → Generate synthetic persona data
  POST /api/v1/ise/compute-savings    → Compute Safe-to-Save for a user
  GET  /api/v1/ise/vault/{user_id}    → Get virtual vault balance & history
  POST /api/v1/ise/simulate           → Run full historical simulation for a persona
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import yaml
import pandas as pd
from fastapi import APIRouter, HTTPException
from loguru import logger

from api.schemas import (
    GenerateDataRequest, GenerateDataResponse,
    ComputeSavingsRequest, ComputeSavingsResponse,
    VaultBalanceResponse,
    SimulationRequest, SimulationResponse,
)

router = APIRouter()

# ── Load config once ──────────────────────────────────────────────────────────

def _load_config() -> dict:
    with open(ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)

CONFIG = _load_config()

# ── Lazy-loaded engine (avoids importing heavy TF on startup) ─────────────────

_engine = None

def _get_engine():
    global _engine
    if _engine is None:
        from src.ise.ise_engine import ISEEngine
        _engine = ISEEngine(CONFIG)
    return _engine


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/generate-data", response_model=GenerateDataResponse, summary="Generate synthetic persona transaction data")
def generate_data(request: GenerateDataRequest):
    """
    Generates 1 year of synthetic daily transactions for all 5 user personas.
    Saves to:
      - data/synthetic/transactions.csv
      - data/synthetic/daily_summary.csv

    **Must be run before training models.**
    """
    try:
        from src.personas.faker_generator import PersonaGenerator
        cfg = dict(CONFIG.get("personas", {}))
        cfg["num_days"] = request.num_days
        cfg["start_date"] = request.start_date
        cfg["random_seed"] = request.random_seed

        gen = PersonaGenerator(cfg)
        df_tx, df_daily = gen.run(output_dir=str(ROOT / "data" / "synthetic"))

        return GenerateDataResponse(
            success=True,
            message="Synthetic transaction data generated successfully.",
            num_transactions=len(df_tx),
            num_daily_rows=len(df_daily),
            output_files=[
                str(ROOT / "data" / "synthetic" / "transactions.csv"),
                str(ROOT / "data" / "synthetic" / "daily_summary.csv"),
            ],
        )
    except Exception as e:
        logger.exception("generate-data failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compute-savings", response_model=ComputeSavingsResponse, summary="Compute Safe-to-Save amount")
def compute_savings(request: ComputeSavingsRequest):
    """
    Core ISE endpoint. Given a user ID, date, and current balance:
    1. Runs LSTM to forecast next 7 days of discretionary spending
    2. Runs Prophet to get macro balance baseline
    3. Applies Adaptive Thresholding to compute Dynamic Buffer
    4. Returns Safe-to-Save amount and micro-tranche to move to virtual vault

    **Requires models to be trained first** (`make train-ise`).
    """
    try:
        daily_summary_path = ROOT / "data" / "synthetic" / "daily_summary.csv"
        if not daily_summary_path.exists():
            raise HTTPException(
                status_code=400,
                detail="daily_summary.csv not found. Run POST /ise/generate-data first."
            )

        df = pd.read_csv(daily_summary_path, parse_dates=["date"])
        engine = _get_engine()
        engine.setup(str(daily_summary_path), retrain=False)

        result = engine.run_daily(
            user_id=request.user_id,
            current_date=request.current_date,
            current_balance=request.current_balance,
            daily_summary_df=df,
        )

        return ComputeSavingsResponse(
            success=True,
            user_id=result.user_id,
            date=str(result.date),
            current_balance=result.current_balance,
            dynamic_buffer=round(result.dynamic_buffer, 2),
            safe_to_save=round(result.safe_to_save, 2),
            micro_tranche=round(result.micro_tranche, 2),
            is_safe_to_save=result.is_safe_to_save,
            is_oneoff_detected=result.is_oneoff_detected,
            predicted_spend_tomorrow=round(result.predicted_spend_tomorrow, 2),
            prophet_baseline_7d=round(result.prophet_baseline_7d, 2),
            seven_day_lstm_forecast=[round(x, 2) for x in result.seven_day_lstm_forecast],
            vault_balance_after=round(result.vault_balance_after, 2),
            recommendation=result.recommendation,
            confidence_score=round(result.confidence_score, 4),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("compute-savings failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vault/{user_id}", response_model=VaultBalanceResponse, summary="Get virtual vault balance")
def get_vault(user_id: str):
    """
    Returns the accumulated virtual vault (savings) balance for a user,
    along with full deposit history.
    """
    try:
        from src.ise.adaptive_threshold import VirtualVault
        vault_path = str(ROOT / CONFIG["ise"]["adaptive_threshold"]["vault_file"])
        vault = VirtualVault(vault_file=vault_path)
        balance = vault.get_balance(user_id)
        history_df = vault.get_history(user_id)
        return VaultBalanceResponse(
            user_id=user_id,
            vault_balance=round(balance, 2),
            total_deposits=len(history_df),
            history=history_df.to_dict(orient="records"),
        )
    except Exception as e:
        logger.exception("vault lookup failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulate", response_model=SimulationResponse, summary="Run full historical ISE simulation")
def simulate(request: SimulationRequest):
    """
    Runs the ISE engine over the full available history for a persona.
    Useful for backtesting how much would have been saved over time.
    Returns aggregated statistics: total saved, average tranche, days blocked.
    """
    try:
        daily_summary_path = ROOT / "data" / "synthetic" / "daily_summary.csv"
        if not daily_summary_path.exists():
            raise HTTPException(status_code=400, detail="daily_summary.csv not found.")

        df = pd.read_csv(daily_summary_path, parse_dates=["date"])
        engine = _get_engine()
        engine.setup(str(daily_summary_path), retrain=False)

        sim_df = engine.run_simulation(df, user_id=request.user_id)
        stats = engine.summary_stats(sim_df)

        return SimulationResponse(
            success=True,
            user_id=request.user_id,
            simulation_rows=len(sim_df),
            summary=stats,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("simulation failed")
        raise HTTPException(status_code=500, detail=str(e))
