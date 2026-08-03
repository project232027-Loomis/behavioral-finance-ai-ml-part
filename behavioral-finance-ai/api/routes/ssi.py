"""
SSI API Routes
===============
Exposes the Smart Stock Investing module over REST.

Endpoints:
  POST /api/v1/ssi/score         → Multi-factor score a stock
  POST /api/v1/ssi/exit-signal   → XGBoost exit signal for a stock
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
    ScoreRequest, ScoreResponse, StockScoreResult,
    ExitSignalRequest, ExitSignalResponse,
)

router = APIRouter()


def _load_config() -> dict:
    with open(ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


CONFIG = _load_config()

_scorer = None
_exit_model = None


def _get_scorer():
    global _scorer
    if _scorer is None:
        from src.ssi.scoring_model import SSIScoringModel
        _scorer = SSIScoringModel(CONFIG)
    return _scorer


def _get_exit_model():
    global _exit_model
    if _exit_model is None:
        from src.ssi.xgboost_model import SSIXGBoostExitModel
        _exit_model = SSIXGBoostExitModel(CONFIG)
        model_path = ROOT / CONFIG["ssi"]["xgboost"]["model_path"]
        if model_path.exists():
            _exit_model.load(str(model_path))
    return _exit_model


def _load_price_df(symbol: str, csv_path: str | None) -> pd.DataFrame:
    """Load OHLCV price data from the given CSV path or default raw data dir."""
    if csv_path:
        path = Path(csv_path)
    else:
        path = ROOT / "data" / "raw" / f"{symbol}.csv"
        if not path.exists():
            path = ROOT / "data" / "ise-data" / f"{symbol}.csv"

    if not path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"CSV not found for symbol '{symbol}'. Provide csv_path or place file in data/raw/{symbol}.csv",
        )
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════

@router.post("/score", response_model=ScoreResponse, summary="Multi-factor score a stock")
def score_stock(request: ScoreRequest):
    """
    Computes Trend (40%), Volatility (30%), and Volume (30%) scores.
    Returns composite score and BUY / HOLD / SELL signal for the latest trading date.

    **Signal interpretation:**
    - composite_score >= 70 → **BUY**
    - 40 < composite_score < 70 → **HOLD**
    - composite_score <= 40 → **SELL**
    """
    try:
        df = _load_price_df(request.symbol, request.csv_path)
        scorer = _get_scorer()
        scores_df = scorer.score_stock(request.symbol, df)
        latest = scores_df.tail(1).to_dict(orient="records")

        buy_candidates = [
            request.symbol
            for r in latest
            if r.get("signal") == "BUY"
        ]

        score_items = [
            StockScoreResult(
                symbol=r["symbol"],
                date=str(r["date"]),
                trend_score=round(r["trend_score"], 2),
                volatility_score=round(r["volatility_score"], 2),
                volume_score=round(r["volume_score"], 2),
                composite_score=round(r["composite_score"], 2),
                signal=r["signal"],
                rsi_14=round(r["rsi_14"], 2),
                price_vs_50dma_pct=round(r["price_vs_50dma_pct"], 4),
                volume_ratio=round(r["volume_ratio"], 4),
                hist_volatility_20d=round(r["hist_volatility_20d"], 6),
            )
            for r in latest
        ]

        return ScoreResponse(success=True, scores=score_items, buy_candidates=buy_candidates)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("score endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/exit-signal", response_model=ExitSignalResponse, summary="XGBoost exit signal prediction")
def exit_signal(request: ExitSignalRequest):
    """
    Uses the trained XGBoost model to predict whether to **EXIT (sell)** or **HOLD**
    a position based on technical indicators.

    **Requires model training:** `make train-ssi`
    """
    try:
        df = _load_price_df(request.symbol, request.csv_path)
        model = _get_exit_model()
        signal = model.predict_exit(df, symbol=request.symbol)

        recommendation = (
            f"⚠️ EXIT signal for {request.symbol} — exit probability {signal.exit_probability:.1%}. Consider closing position."
            if signal.signal == "EXIT"
            else f"✅ HOLD signal for {request.symbol} — exit probability {signal.exit_probability:.1%}. No action needed."
        )

        return ExitSignalResponse(
            success=True,
            symbol=signal.symbol,
            date=str(signal.date),
            signal=signal.signal,
            exit_probability=round(signal.exit_probability, 4),
            confidence=round(signal.confidence, 4),
            feature_importances=signal.feature_importances,
            recommendation=recommendation,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("exit-signal endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))
