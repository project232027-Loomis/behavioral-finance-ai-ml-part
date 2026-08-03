"""
API Pydantic Schemas
=====================
Request and response models for all API endpoints.
Strict typing ensures safe serialization for any web client.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


# ══════════════════════════════════════════════════════════════════════════════
# SHARED
# ══════════════════════════════════════════════════════════════════════════════

class SuccessResponse(BaseModel):
    success: bool = True
    message: str = "OK"


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# ISE — Invisible Savings Engine
# ══════════════════════════════════════════════════════════════════════════════

class GenerateDataRequest(BaseModel):
    """Request to (re-)generate synthetic persona transaction data."""
    num_days: int = Field(default=365, ge=30, le=730, description="Days of history to generate")
    start_date: str = Field(default="2023-01-01", description="ISO date string YYYY-MM-DD")
    random_seed: int = Field(default=42, description="Seed for reproducibility")


class GenerateDataResponse(BaseModel):
    success: bool
    message: str
    num_transactions: int
    num_daily_rows: int
    output_files: list[str]


class ComputeSavingsRequest(BaseModel):
    """Request to compute Safe-to-Save amount for a user on a given date."""
    user_id: str = Field(..., description="Persona user ID, e.g. 'U001'")
    current_date: str = Field(..., description="Date to compute savings for (YYYY-MM-DD)")
    current_balance: float = Field(..., gt=0, description="User's current bank balance in ₹")

    @field_validator("current_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError("current_date must be in YYYY-MM-DD format")
        return v


class ComputeSavingsResponse(BaseModel):
    success: bool
    user_id: str
    date: str
    current_balance: float
    dynamic_buffer: float = Field(..., description="Minimum ₹ to keep for next 7 days")
    safe_to_save: float = Field(..., description="Amount safely movable to vault")
    micro_tranche: float = Field(..., description="₹50–₹500 slice moved this cycle")
    is_safe_to_save: bool
    is_oneoff_detected: bool = Field(..., description="True if unusual large expense detected — savings paused")
    predicted_spend_tomorrow: float
    prophet_baseline_7d: float
    seven_day_lstm_forecast: list[float]
    vault_balance_after: float
    recommendation: str
    confidence_score: float


class VaultBalanceResponse(BaseModel):
    user_id: str
    vault_balance: float
    total_deposits: int
    history: list[dict[str, Any]]


class SimulationRequest(BaseModel):
    """Run a full historical simulation for a persona."""
    user_id: str = Field(..., description="Persona user ID, e.g. 'U001'")


class SimulationResponse(BaseModel):
    success: bool
    user_id: str
    simulation_rows: int
    summary: dict[str, Any]


# ══════════════════════════════════════════════════════════════════════════════
# SSI — Smart Stock Investing
# ══════════════════════════════════════════════════════════════════════════════

class ScoreRequest(BaseModel):
    """Score a stock using the multi-factor model. Provide CSV path or raw OHLCV data."""
    symbol: str = Field(..., description="NSE stock symbol, e.g. 'ADANIPORTS'")
    csv_path: Optional[str] = Field(
        default=None,
        description="Path to stock CSV file (Date, Open, High, Low, Close, Volume columns)"
    )


class StockScoreResult(BaseModel):
    symbol: str
    date: str
    trend_score: float
    volatility_score: float
    volume_score: float
    composite_score: float
    signal: str = Field(..., description="BUY | HOLD | SELL")
    rsi_14: float
    price_vs_50dma_pct: float
    volume_ratio: float
    hist_volatility_20d: float


class ScoreResponse(BaseModel):
    success: bool
    scores: list[StockScoreResult]
    buy_candidates: list[str]


class ExitSignalRequest(BaseModel):
    symbol: str = Field(..., description="NSE stock symbol")
    csv_path: Optional[str] = Field(default=None, description="Path to stock CSV")


class ExitSignalResponse(BaseModel):
    success: bool
    symbol: str
    date: str
    signal: str = Field(..., description="EXIT | HOLD")
    exit_probability: float
    confidence: float
    feature_importances: dict[str, float]
    recommendation: str


# ══════════════════════════════════════════════════════════════════════════════
# SENTIMENT — FinBERT
# ══════════════════════════════════════════════════════════════════════════════

class SentimentRequest(BaseModel):
    """Analyze sentiment of financial text(s)."""
    texts: list[str] = Field(..., min_length=1, description="List of financial headlines/texts to analyze")

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: list[str]) -> list[str]:
        if any(not t.strip() for t in v):
            raise ValueError("texts must not contain empty strings")
        return [t.strip() for t in v]


class SentimentResultItem(BaseModel):
    text: str
    label: str = Field(..., description="positive | negative | neutral")
    positive_score: float
    negative_score: float
    neutral_score: float
    confidence: float


class SentimentResponse(BaseModel):
    success: bool
    results: list[SentimentResultItem]
    aggregate: dict[str, Any]


class StockSentimentRequest(BaseModel):
    """Analyze and filter a universe of stocks based on their news headlines."""
    stock_headlines: dict[str, list[str]] = Field(
        ...,
        description="Dict mapping symbol -> list of recent news headlines"
    )
    negative_threshold: float = Field(
        default=0.60,
        ge=0.0, le=1.0,
        description="Stocks with negative_score above this are filtered out"
    )


class StockSentimentResponse(BaseModel):
    success: bool
    universe_size: int
    filtered_out: list[str]
    approved: list[str]
    scores: list[dict[str, Any]]
