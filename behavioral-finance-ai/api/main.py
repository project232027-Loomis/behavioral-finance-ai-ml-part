"""
Behavioral Finance AI — FastAPI Application
============================================
Plug-and-play REST API that wraps all ML modules (ISE, SSI, Sentiment).
Any web application can call these endpoints without any complex deployment.

Start the server:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Interactive docs:
    http://localhost:8000/docs        (Swagger UI)
    http://localhost:8000/redoc       (ReDoc)

All endpoints are prefixed with /api/v1
"""

import os
import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from api.routes import ise, ssi, sentiment


# ── Load config ───────────────────────────────────────────────────────────────

def load_config() -> dict:
    config_path = ROOT / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
API_CFG = CONFIG.get("api", {})

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=API_CFG.get("title", "Behavioral Finance AI API"),
    description=API_CFG.get("description", ""),
    version=API_CFG.get("version", "1.0.0"),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Configured to accept requests from any web frontend (React, Vue, Angular, etc.)

app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CFG.get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────

PREFIX = API_CFG.get("prefix", "/api/v1")

app.include_router(ise.router,       prefix=f"{PREFIX}/ise",       tags=["ISE — Invisible Savings Engine"])
app.include_router(ssi.router,       prefix=f"{PREFIX}/ssi",       tags=["SSI — Smart Stock Investing"])
app.include_router(sentiment.router, prefix=f"{PREFIX}/sentiment",  tags=["Sentiment — FinBERT Analysis"])


# ── Health & Root ─────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Root endpoint — confirms API is running."""
    return {
        "message": "Behavioral Finance AI API is running",
        "docs": "/docs",
        "version": API_CFG.get("version", "1.0.0"),
    }


@app.get(f"{PREFIX}/health", tags=["Health"])
def health_check():
    """
    Health check endpoint.
    Returns: API status, version, and available modules.
    """
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "version": API_CFG.get("version", "1.0.0"),
            "modules": {
                "ise": "Invisible Savings Engine (LSTM + Prophet + Adaptive Threshold)",
                "ssi": "Smart Stock Investing (Multi-Factor Scoring + XGBoost Exit Signals)",
                "sentiment": "Financial Sentiment Analysis (FinBERT)",
            },
        },
    )


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    logger.info("Behavioral Finance AI API starting up...")
    logger.info(f"Docs available at http://{API_CFG.get('host','0.0.0.0')}:{API_CFG.get('port',8000)}/docs")


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Behavioral Finance AI API shutting down.")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=API_CFG.get("host", "0.0.0.0"),
        port=API_CFG.get("port", 8000),
        reload=True,
        log_level="info",
    )
