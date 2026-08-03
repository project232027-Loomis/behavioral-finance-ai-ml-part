"""
Sentiment API Routes
=====================
Exposes FinBERT financial sentiment analysis over REST.

Endpoints:
  POST /api/v1/sentiment/analyze          → Analyze text sentiment
  POST /api/v1/sentiment/filter-stocks    → Filter stocks by news sentiment
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import yaml
from fastapi import APIRouter, HTTPException
from loguru import logger

from api.schemas import (
    SentimentRequest, SentimentResponse, SentimentResultItem,
    StockSentimentRequest, StockSentimentResponse,
)

router = APIRouter()


def _load_config() -> dict:
    with open(ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


CONFIG = _load_config()

_bert = None


def _get_bert():
    global _bert
    if _bert is None:
        from src.sentiment.finbert_model import FinBERTSentiment
        _bert = FinBERTSentiment(CONFIG)
    return _bert


# ══════════════════════════════════════════════════════════════════════════════

@router.post("/analyze", response_model=SentimentResponse, summary="Analyze financial text sentiment")
def analyze_sentiment(request: SentimentRequest):
    """
    Runs FinBERT (ProsusAI/finbert) on the provided financial texts.
    Returns per-text sentiment scores (positive/negative/neutral) and
    an aggregate summary.

    **Note:** First call downloads the FinBERT model (~400MB). Subsequent calls are fast.
    """
    try:
        bert = _get_bert()
        results = bert.analyze(request.texts)
        aggregate = bert.aggregate_sentiment(results)

        items = [
            SentimentResultItem(
                text=r.text,
                label=r.label,
                positive_score=round(r.positive_score, 4),
                negative_score=round(r.negative_score, 4),
                neutral_score=round(r.neutral_score, 4),
                confidence=round(r.confidence, 4),
            )
            for r in results
        ]

        return SentimentResponse(success=True, results=items, aggregate=aggregate)

    except Exception as e:
        logger.exception("sentiment analyze failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/filter-stocks", response_model=StockSentimentResponse, summary="Filter stocks by news sentiment")
def filter_stocks(request: StockSentimentRequest):
    """
    Analyzes news headlines for each stock symbol and filters out stocks
    with predominantly negative sentiment (negative_score > threshold).

    **Use case:** Pre-filter your SSI stock universe before scoring.
    Pass the approved list to `/ssi/score` to avoid buying into negative-momentum stocks.

    Example request body:
    ```json
    {
      "stock_headlines": {
        "TCS": ["TCS posts record Q3 profits", "Strong demand from US clients"],
        "ZOMATO": ["Zomato reports widening losses", "Regulatory headwinds ahead"]
      },
      "negative_threshold": 0.6
    }
    ```
    """
    try:
        bert = _get_bert()
        scores_df = bert.score_stock_universe(request.stock_headlines)
        approved_df = bert.filter_negative(
            request.stock_headlines,
            threshold=request.negative_threshold
        )

        all_symbols = list(request.stock_headlines.keys())
        approved = list(approved_df.keys())
        filtered_out = [s for s in all_symbols if s not in approved]

        return StockSentimentResponse(
            success=True,
            universe_size=len(all_symbols),
            filtered_out=filtered_out,
            approved=approved,
            scores=scores_df.to_dict(orient="records"),
        )

    except Exception as e:
        logger.exception("filter-stocks failed")
        raise HTTPException(status_code=500, detail=str(e))
