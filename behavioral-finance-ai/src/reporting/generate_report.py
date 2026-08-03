"""
Model Reporting & Visualization Generator
==========================================
Generates all training/testing visuals and a professional HTML summary
report that can be shared with clients after model training.

Output structure:
  reports/
  ├── ise/
  │   ├── lstm/
  │   │   ├── loss_curve.png
  │   │   ├── actual_vs_predicted.png
  │   │   ├── zoomed_recent_90d.png
  │   │   ├── residuals.png
  │   │   ├── seven_day_forecast_sample.png
  │   │   └── oneoff_confusion_matrix.png
  │   └── prophet/
  │       ├── U001_forecast.png
  │       ├── U001_components.png
  │       └── ... (one per persona)
  ├── ssi/
  │   ├── feature_importance.png
  │   ├── roc_curve.png
  │   ├── confusion_matrix.png
  │   └── precision_recall.png
  └── summary/
      ├── model_card.html      <- shareable client report
      └── metrics_summary.json <- raw metrics for programmatic use

Usage:
    python -m src.reporting.generate_report
    # or via Makefile / run.ps1:
    make report   /   .\\run.ps1 report
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import yaml
from loguru import logger

# ── Style ──────────────────────────────────────────────────────────────────────

PALETTE = {
    "primary":    "#2563EB",   # blue
    "secondary":  "#7C3AED",   # purple
    "success":    "#059669",   # green
    "warning":    "#D97706",   # amber
    "danger":     "#DC2626",   # red
    "neutral":    "#6B7280",   # gray
    "bg":         "#F8FAFC",
    "card":       "#FFFFFF",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["card"],
    "axes.edgecolor":    "#E2E8F0",
    "axes.labelcolor":   "#1E293B",
    "axes.titlecolor":   "#0F172A",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.color":       "#64748B",
    "ytick.color":       "#64748B",
    "grid.color":        "#E2E8F0",
    "grid.linewidth":    0.8,
    "lines.linewidth":   2.0,
    "font.family":       "DejaVu Sans",
    "legend.framealpha": 0.9,
    "legend.fontsize":   9,
})

sns.set_palette([PALETTE["primary"], PALETTE["danger"], PALETTE["success"],
                 PALETTE["secondary"], PALETTE["warning"]])


# ══════════════════════════════════════════════════════════════════════════════
# ISE — LSTM Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_lstm_loss(history: dict, out_dir: Path) -> Path:
    """Training vs validation loss curve."""
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(history["loss"]) + 1)
    ax.plot(epochs, history["loss"],     color=PALETTE["primary"],  label="Training Loss",   linewidth=2)
    ax.plot(epochs, history["val_loss"], color=PALETTE["danger"],   label="Validation Loss", linewidth=2, linestyle="--")
    ax.fill_between(epochs, history["loss"], history["val_loss"],
                    alpha=0.08, color=PALETTE["primary"])
    ax.set_title("LSTM — Training vs Validation Loss (MSE)", fontweight="bold", pad=15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE, scaled)")
    ax.legend()
    ax.grid(True, alpha=0.5)
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    ax.axvline(best_epoch, color=PALETTE["success"], linestyle=":", linewidth=1.5,
               label=f"Best epoch ({best_epoch})")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "loss_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def plot_actual_vs_predicted(dates, y_true, y_pred, out_dir: Path) -> Path:
    """Full test-set actual vs predicted discretionary spend."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, y_true, color=PALETTE["neutral"],  label="Actual Spend",    linewidth=1.5, alpha=0.9)
    ax.plot(dates, y_pred, color=PALETTE["primary"],  label="Predicted Spend", linewidth=1.5, alpha=0.85, linestyle="--")
    ax.fill_between(dates, y_true, y_pred, alpha=0.07, color=PALETTE["danger"])
    ax.set_title("ISE LSTM — Actual vs Predicted Discretionary Spend (Test Set)", fontweight="bold", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Spend (INR)")
    ax.legend()
    ax.grid(True, alpha=0.5)
    plt.xticks(rotation=30)
    fig.tight_layout()
    path = out_dir / "actual_vs_predicted.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def plot_zoomed_recent(dates, y_true, y_pred, out_dir: Path, n: int = 90) -> Path:
    """Zoomed last N days of test set."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates[-n:], y_true[-n:], color=PALETTE["neutral"], label="Actual",
            marker="o", markersize=3, linewidth=1.5)
    ax.plot(dates[-n:], y_pred[-n:], color=PALETTE["primary"], label="Predicted",
            marker="x", markersize=4, linewidth=1.5, linestyle="--")
    ax.set_title(f"ISE LSTM — Last {n} Days (Zoomed View)", fontweight="bold", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Discretionary Spend (INR)")
    ax.legend()
    ax.grid(True, alpha=0.5)
    plt.xticks(rotation=30)
    fig.tight_layout()
    path = out_dir / "zoomed_recent_90d.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def plot_residuals(y_true, y_pred, out_dir: Path) -> Path:
    """Residuals over time + distribution."""
    residuals = np.array(y_true) - np.array(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(residuals, color=PALETTE["secondary"], linewidth=1, alpha=0.8)
    axes[0].axhline(0, color=PALETTE["danger"], linestyle="--", linewidth=1.5)
    axes[0].fill_between(range(len(residuals)), residuals, 0,
                         where=(residuals > 0), alpha=0.15, color=PALETTE["success"])
    axes[0].fill_between(range(len(residuals)), residuals, 0,
                         where=(residuals < 0), alpha=0.15, color=PALETTE["danger"])
    axes[0].set_title("Residuals Over Time (Actual - Predicted)", fontweight="bold")
    axes[0].set_xlabel("Test Sample Index")
    axes[0].set_ylabel("Residual (INR)")
    axes[0].grid(True, alpha=0.5)

    axes[1].hist(residuals, bins=40, color=PALETTE["primary"], edgecolor="white", alpha=0.85)
    axes[1].axvline(np.mean(residuals), color=PALETTE["danger"], linestyle="--",
                    linewidth=1.5, label=f"Mean: {np.mean(residuals):.1f}")
    axes[1].set_title("Residual Distribution", fontweight="bold")
    axes[1].set_xlabel("Residual (INR)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    axes[1].grid(True, alpha=0.5)

    fig.suptitle("ISE LSTM — Error Analysis", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = out_dir / "residuals.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def plot_oneoff_confusion(cm: list, out_dir: Path) -> Path:
    """One-off expense detection confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_array = np.array(cm)
    sns.heatmap(cm_array, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Predicted: Normal", "Predicted: One-Off"],
                yticklabels=["Actual: Normal", "Actual: One-Off"],
                linewidths=1, linecolor="#E2E8F0", cbar=False,
                annot_kws={"size": 13, "weight": "bold"})
    ax.set_title("One-Off Expense Detector\nConfusion Matrix", fontweight="bold", pad=12)
    fig.tight_layout()
    path = out_dir / "oneoff_confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def plot_seven_day_forecast_sample(forecast: list[float], actual: list[float],
                                   user_id: str, out_dir: Path) -> Path:
    """Sample 7-day forward spending forecast vs actual."""
    days = [f"Day {i+1}" for i in range(len(forecast))]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(days))
    width = 0.35
    bars1 = ax.bar(x - width/2, actual,   width, label="Actual",    color=PALETTE["neutral"],   alpha=0.85)
    bars2 = ax.bar(x + width/2, forecast, width, label="Predicted", color=PALETTE["primary"],   alpha=0.85)
    for bar in bars1: ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                              f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8, color=PALETTE["neutral"])
    for bar in bars2: ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                              f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8, color=PALETTE["primary"])
    ax.set_title(f"7-Day Discretionary Spend Forecast Sample ({user_id})", fontweight="bold", pad=12)
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("Spend (INR)")
    ax.set_xticks(x)
    ax.set_xticklabels(days)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.5)
    fig.tight_layout()
    path = out_dir / f"seven_day_forecast_{user_id}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# ISE — Prophet Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_prophet_forecast(forecast_df: pd.DataFrame, actual_df: pd.DataFrame,
                          user_id: str, out_dir: Path) -> Path:
    """Balance forecast with confidence intervals."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Actuals
    ax.plot(actual_df["ds"], actual_df["y"], color=PALETTE["neutral"],
            label="Actual Balance", linewidth=1.5, alpha=0.9)

    # Forecast
    future = forecast_df[forecast_df["ds"] > actual_df["ds"].max()]
    ax.plot(forecast_df["ds"], forecast_df["yhat"],
            color=PALETTE["primary"], label="Forecast", linewidth=2, linestyle="--")
    ax.fill_between(forecast_df["ds"], forecast_df["yhat_lower"], forecast_df["yhat_upper"],
                    alpha=0.15, color=PALETTE["primary"], label="Confidence Interval")
    ax.axvline(actual_df["ds"].max(), color=PALETTE["warning"], linestyle=":",
               linewidth=2, label="Forecast Start")

    ax.set_title(f"FB-Prophet — Bank Balance Forecast ({user_id})", fontweight="bold", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Balance (INR)")
    ax.legend()
    ax.grid(True, alpha=0.5)
    plt.xticks(rotation=30)
    fig.tight_layout()
    path = out_dir / f"{user_id}_forecast.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def plot_prophet_components(components_df: pd.DataFrame, user_id: str, out_dir: Path) -> Path:
    """Trend + weekly + yearly seasonality decomposition."""
    cols = ["trend", "weekly", "yearly"]
    available = [c for c in cols if c in components_df.columns]
    fig, axes = plt.subplots(len(available), 1, figsize=(14, 4 * len(available)))
    if len(available) == 1:
        axes = [axes]

    titles = {"trend": "Long-Term Balance Trend", "weekly": "Weekly Spending Pattern",
              "yearly": "Yearly Seasonality (Annual Cycle)"}
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["success"]]

    for ax, col, color in zip(axes, available, colors):
        ax.plot(components_df["ds"], components_df[col], color=color, linewidth=2)
        ax.fill_between(components_df["ds"], 0, components_df[col],
                        alpha=0.1, color=color)
        ax.set_title(titles.get(col, col), fontweight="bold")
        ax.set_ylabel("Component Value (INR)")
        ax.grid(True, alpha=0.5)
        ax.axhline(0, color="#94A3B8", linestyle="-", linewidth=0.8)

    fig.suptitle(f"FB-Prophet — Seasonality Decomposition ({user_id})",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.xticks(rotation=30)
    fig.tight_layout()
    path = out_dir / f"{user_id}_components.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# SSI Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(importances: dict, out_dir: Path) -> Path:
    """XGBoost feature importance bar chart."""
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    features, values = zip(*sorted_items)
    colors = [PALETTE["primary"] if v >= sorted(values)[-3] else PALETTE["neutral"] for v in values]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(features, values, color=colors, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9, color="#374151")
    ax.set_title("SSI XGBoost — Feature Importance\n(Exit Signal Classifier)", fontweight="bold", pad=15)
    ax.set_xlabel("Importance Score")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.5)
    fig.tight_layout()
    path = out_dir / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def plot_roc_curve(fpr, tpr, auc_score: float, out_dir: Path) -> Path:
    """ROC curve for XGBoost exit signal classifier."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=PALETTE["primary"], linewidth=2.5,
            label=f"ROC Curve (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], color=PALETTE["neutral"], linewidth=1.5,
            linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE["primary"])
    ax.set_title("SSI XGBoost — ROC Curve\n(Exit Signal Classifier)", fontweight="bold", pad=15)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.5)
    fig.tight_layout()
    path = out_dir / "roc_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def plot_confusion_matrix_ssi(cm: list, out_dir: Path) -> Path:
    """Confusion matrix for SSI exit signal."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_array = np.array(cm)
    # Percentage annotations
    cm_pct = cm_array.astype(float) / cm_array.sum(axis=1, keepdims=True) * 100
    labels = np.array([[f"{v}\n({p:.1f}%)" for v, p in zip(row_v, row_p)]
                        for row_v, row_p in zip(cm_array, cm_pct)])
    sns.heatmap(cm_array, annot=labels, fmt="", cmap="Blues", ax=ax,
                xticklabels=["Predicted: HOLD", "Predicted: EXIT"],
                yticklabels=["Actual: HOLD", "Actual: EXIT"],
                linewidths=1, linecolor="#E2E8F0", cbar=False,
                annot_kws={"size": 11, "weight": "bold"})
    ax.set_title("SSI XGBoost — Exit Signal\nConfusion Matrix", fontweight="bold", pad=12)
    fig.tight_layout()
    path = out_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def plot_precision_recall(precision, recall, avg_precision: float, out_dir: Path) -> Path:
    """Precision-Recall curve."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color=PALETTE["secondary"], linewidth=2.5,
            label=f"PR Curve (AP = {avg_precision:.3f})")
    ax.fill_between(recall, precision, alpha=0.08, color=PALETTE["secondary"])
    ax.set_title("SSI XGBoost — Precision-Recall Curve\n(Exit Signal Classifier)", fontweight="bold", pad=15)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(True, alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    fig.tight_layout()
    path = out_dir / "precision_recall.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def plot_scoring_distribution(scores_df: pd.DataFrame, out_dir: Path) -> Path:
    """Distribution of composite scores across all stocks."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(scores_df["composite_score"], bins=30, color=PALETTE["primary"],
            edgecolor="white", alpha=0.85)
    ax.axvline(70, color=PALETTE["success"],  linestyle="--", linewidth=2, label="Buy Threshold (70)")
    ax.axvline(40, color=PALETTE["danger"],   linestyle="--", linewidth=2, label="Sell Threshold (40)")
    ax.set_title("SSI — Composite Score Distribution Across Stocks", fontweight="bold", pad=15)
    ax.set_xlabel("Composite Score (0-100)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.5)
    fig.tight_layout()
    path = out_dir / "score_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Simulation Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_vault_growth(sim_df: pd.DataFrame, user_id: str, out_dir: Path) -> Path:
    """Cumulative vault savings growth over simulation period."""
    user_df = sim_df[sim_df["user_id"] == user_id].copy().sort_values("date")
    user_df["cumulative_saved"] = user_df["micro_tranche"].cumsum()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].fill_between(user_df["date"], user_df["cumulative_saved"],
                         alpha=0.3, color=PALETTE["success"])
    axes[0].plot(user_df["date"], user_df["cumulative_saved"],
                 color=PALETTE["success"], linewidth=2)
    axes[0].set_title(f"Cumulative Vault Savings — {user_id}", fontweight="bold")
    axes[0].set_ylabel("Total Saved (INR)")
    axes[0].grid(True, alpha=0.5)

    axes[1].bar(user_df["date"], user_df["micro_tranche"],
                color=np.where(user_df["micro_tranche"] > 0, PALETTE["primary"], PALETTE["neutral"]),
                alpha=0.7, width=1)
    axes[1].set_title("Daily Micro-Tranche Deposited", fontweight="bold")
    axes[1].set_ylabel("Tranche Amount (INR)")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, axis="y", alpha=0.5)
    plt.xticks(rotation=30)

    fig.tight_layout()
    path = out_dir / f"vault_growth_{user_id}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# HTML Summary Report
# ══════════════════════════════════════════════════════════════════════════════

def generate_html_report(metrics: dict, image_paths: list[Path],
                         out_dir: Path, generated_at: str) -> Path:
    """Generate a professional shareable HTML report for clients."""
    import base64

    def img_to_b64(path: Path) -> str:
        if path.exists():
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        return ""

    def metric_card(label: str, value: str, unit: str = "", color: str = "#2563EB") -> str:
        return f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color}">{value}<span class="metric-unit">{unit}</span></div>
        </div>"""

    # Build image gallery HTML
    gallery_html = ""
    section_map = {
        "loss_curve":              ("ISE — LSTM Training", "Training vs Validation Loss"),
        "actual_vs_predicted":     ("ISE — LSTM Accuracy", "Actual vs Predicted Spend (Test Set)"),
        "zoomed_recent_90d":       ("ISE — LSTM Close-up", "Last 90 Days Detailed View"),
        "residuals":               ("ISE — LSTM Error Analysis", "Residuals Distribution"),
        "oneoff_confusion_matrix": ("ISE — One-Off Detector", "Confusion Matrix"),
        "seven_day_forecast":      ("ISE — 7-Day Forecast", "Sample Forward Forecast"),
        "U001_forecast":           ("Prophet — Balance Forecast", "U001 Balance Projection"),
        "U001_components":         ("Prophet — Decomposition", "U001 Trend & Seasonality"),
        "feature_importance":      ("SSI — XGBoost", "Feature Importance"),
        "roc_curve":               ("SSI — XGBoost Performance", "ROC Curve"),
        "confusion_matrix":        ("SSI — XGBoost Performance", "Exit Signal Confusion Matrix"),
        "precision_recall":        ("SSI — XGBoost Performance", "Precision-Recall Curve"),
        "score_distribution":      ("SSI — Scoring Model", "Composite Score Distribution"),
        "vault_growth":            ("ISE — Simulation", "Vault Savings Growth"),
    }

    for img_path in image_paths:
        stem = img_path.stem
        key = next((k for k in section_map if stem.startswith(k) or stem == k), stem)
        section, caption = section_map.get(key, ("Visualization", stem))
        b64 = img_to_b64(img_path)
        if b64:
            gallery_html += f"""
            <div class="plot-card">
                <div class="plot-section-tag">{section}</div>
                <img src="data:image/png;base64,{b64}" alt="{caption}" loading="lazy"/>
                <div class="plot-caption">{caption}</div>
            </div>"""

    # Metrics cards
    lstm_m  = metrics.get("lstm", {})
    ssi_m   = metrics.get("ssi", {})
    sim_m   = metrics.get("simulation", {})

    cards_html = ""
    if lstm_m:
        cards_html += metric_card("LSTM — MAE",    f'{lstm_m.get("mae", 0):.1f}',   " INR")
        cards_html += metric_card("LSTM — MAPE",   f'{lstm_m.get("mape", 0):.1f}',  "%")
        cards_html += metric_card("LSTM — R²",     f'{lstm_m.get("r2", 0):.3f}',    "",      PALETTE["success"])
        cards_html += metric_card("One-Off Acc.",  f'{lstm_m.get("oneoff_acc", 0)*100:.1f}', "%", PALETTE["secondary"])
    if ssi_m:
        cards_html += metric_card("SSI — AUC-ROC", f'{ssi_m.get("auc", 0):.3f}',    "", PALETTE["primary"])
        cards_html += metric_card("Exit Precision", f'{ssi_m.get("precision", 0)*100:.1f}', "%")
        cards_html += metric_card("Exit Recall",    f'{ssi_m.get("recall", 0)*100:.1f}',    "%")
    if sim_m:
        cards_html += metric_card("Total Saved",  f'Rs {sim_m.get("total_saved", 0):,.0f}',  "", PALETTE["success"])
        cards_html += metric_card("Avg Tranche",  f'Rs {sim_m.get("avg_tranche", 0):.0f}',   "")
        cards_html += metric_card("Days Saved",   f'{sim_m.get("days_saved", 0)}',           " days", PALETTE["secondary"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Behavioral Finance AI — Model Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #F1F5F9; color: #1E293B; }}
  .header {{ background: linear-gradient(135deg, #1E3A8A 0%, #7C3AED 100%);
             color: white; padding: 48px 40px; text-align: center; }}
  .header h1 {{ font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px; }}
  .header p  {{ font-size: 1rem; opacity: 0.85; margin-top: 8px; }}
  .badge {{ display: inline-block; background: rgba(255,255,255,0.2);
            border-radius: 20px; padding: 4px 14px; font-size: 0.8rem;
            margin-top: 12px; letter-spacing: 0.5px; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 32px 24px; }}
  .section-title {{ font-size: 1.3rem; font-weight: 700; color: #0F172A;
                    margin: 40px 0 20px; padding-left: 12px;
                    border-left: 4px solid #2563EB; }}
  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
                   gap: 16px; margin-bottom: 40px; }}
  .metric-card {{ background: white; border-radius: 12px; padding: 20px;
                  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
                  border-top: 3px solid #2563EB; }}
  .metric-label {{ font-size: 0.75rem; color: #64748B; font-weight: 600;
                   text-transform: uppercase; letter-spacing: 0.5px; }}
  .metric-value {{ font-size: 1.8rem; font-weight: 800; color: #2563EB; margin-top: 6px; }}
  .metric-unit  {{ font-size: 0.9rem; font-weight: 400; color: #64748B; margin-left: 2px; }}
  .plots-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(580px, 1fr));
                 gap: 24px; }}
  .plot-card {{ background: white; border-radius: 16px; overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08); transition: transform 0.2s;}}
  .plot-card:hover {{ transform: translateY(-3px); box-shadow: 0 8px 20px rgba(0,0,0,0.12); }}
  .plot-section-tag {{ background: #EFF6FF; color: #1D4ED8; font-size: 0.72rem;
                       font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px;
                       padding: 8px 16px; }}
  .plot-card img {{ width: 100%; display: block; }}
  .plot-caption {{ padding: 12px 16px; font-size: 0.85rem; color: #475569;
                   font-weight: 500; border-top: 1px solid #F1F5F9; }}
  .footer {{ text-align: center; padding: 40px; color: #94A3B8; font-size: 0.85rem; }}
  .info-bar {{ background: white; border-radius: 12px; padding: 16px 24px;
               display: flex; gap: 32px; flex-wrap: wrap; margin-bottom: 32px;
               box-shadow: 0 1px 3px rgba(0,0,0,0.06); font-size: 0.85rem; }}
  .info-item span {{ color: #64748B; }} .info-item strong {{ color: #1E293B; }}
</style>
</head>
<body>
<div class="header">
  <h1>Behavioral Finance AI — Model Report</h1>
  <p>Invisible Savings Engine (ISE) &nbsp;|&nbsp; Smart Stock Investing (SSI) &nbsp;|&nbsp; FinBERT Sentiment</p>
  <div class="badge">Generated: {generated_at}</div>
</div>
<div class="container">
  <div class="info-bar">
    <div class="info-item"><span>Framework: </span><strong>TensorFlow / XGBoost / FB-Prophet / FinBERT</strong></div>
    <div class="info-item"><span>Data: </span><strong>Synthetic Personas (5 users, 365 days each)</strong></div>
    <div class="info-item"><span>Branch: </span><strong>dev</strong></div>
    <div class="info-item"><span>Environment: </span><strong>Simulation (no live data)</strong></div>
  </div>

  <div class="section-title">Key Performance Metrics</div>
  <div class="metrics-grid">{cards_html}</div>

  <div class="section-title">Training & Evaluation Visualizations</div>
  <div class="plots-grid">{gallery_html}</div>
</div>
<div class="footer">
  Behavioral Finance AI &mdash; Loomis Project &nbsp;&bull;&nbsp;
  Simulation results only. Not financial advice.
</div>
</body>
</html>"""

    path = out_dir / "model_card.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.success(f"HTML report saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_reports(config: dict) -> None:
    """
    Main entry point. Loads trained models and data, generates all plots
    and the HTML report. Safe to call even if some models are not yet trained.
    """
    reports_dir   = ROOT / "reports"
    lstm_dir      = reports_dir / "ise" / "lstm"
    prophet_dir   = reports_dir / "ise" / "prophet"
    ssi_dir       = reports_dir / "ssi"
    summary_dir   = reports_dir / "summary"

    for d in [lstm_dir, prophet_dir, ssi_dir, summary_dir]:
        d.mkdir(parents=True, exist_ok=True)

    all_images: list[Path] = []
    all_metrics: dict = {}
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    daily_path = ROOT / "data" / "synthetic" / "daily_summary.csv"

    # ── ISE LSTM ──────────────────────────────────────────────────────────────
    lstm_model_path = ROOT / config["ise"]["lstm"]["model_path"]
    if lstm_model_path.exists() and daily_path.exists():
        logger.info("Generating ISE LSTM plots...")
        try:
            from src.ise.lstm_model import ISELSTMModel
            lstm = ISELSTMModel(config)
            lstm.load(str(lstm_model_path))
            df = pd.read_csv(daily_path, parse_dates=["date"])
            eval_result = lstm.evaluate(df)

            # Use eval_result data for plots
            y_true = np.array(eval_result.get("y_true", []))
            y_pred = np.array(eval_result.get("y_pred", []))
            dates  = eval_result.get("dates", np.arange(len(y_true)))
            history = eval_result.get("history", {"loss": [], "val_loss": []})
            cm      = eval_result.get("confusion_matrix", [[0,0],[0,0]])

            if len(y_true) > 0:
                all_images.append(plot_lstm_loss(history, lstm_dir))
                all_images.append(plot_actual_vs_predicted(dates, y_true, y_pred, lstm_dir))
                if len(y_true) > 90:
                    all_images.append(plot_zoomed_recent(dates, y_true, y_pred, lstm_dir))
                all_images.append(plot_residuals(y_true, y_pred, lstm_dir))
                all_images.append(plot_oneoff_confusion(cm, lstm_dir))

            all_metrics["lstm"] = {
                "mae":       eval_result.get("mae", 0),
                "mape":      eval_result.get("mape", 0),
                "r2":        eval_result.get("r2", 0),
                "oneoff_acc": eval_result.get("oneoff_accuracy", 0),
            }
        except Exception as e:
            logger.warning(f"LSTM plots skipped: {e}")
    else:
        logger.warning("LSTM model or data not found — skipping LSTM plots. Run: make train-lstm")

    # ── ISE Prophet ───────────────────────────────────────────────────────────
    prophet_model_dir = ROOT / config["ise"]["prophet"].get("model_dir", "models/ise/prophet")
    if prophet_model_dir.exists() and daily_path.exists():
        logger.info("Generating ISE Prophet plots...")
        try:
            from src.ise.prophet_model import ISEProphetModel
            df = pd.read_csv(daily_path, parse_dates=["date"])
            user_ids = df["user_id"].unique()[:3]  # first 3 users
            for uid in user_ids:
                pkl = prophet_model_dir / f"prophet_{uid}.pkl"
                if pkl.exists():
                    prophet = ISEProphetModel(config)
                    prophet.load(str(pkl))
                    result = prophet.forecast(periods=30)
                    user_df = df[df["user_id"] == uid][["date", "balance_eod"]].rename(
                        columns={"date": "ds", "balance_eod": "y"})
                    all_images.append(plot_prophet_forecast(result.forecast_df, user_df, uid, prophet_dir))
                    if result.components_df is not None:
                        all_images.append(plot_prophet_components(result.components_df, uid, prophet_dir))
        except Exception as e:
            logger.warning(f"Prophet plots skipped: {e}")
    else:
        logger.warning("Prophet models not found — skipping. Run: make train-prophet")

    # ── ISE Simulation ────────────────────────────────────────────────────────
    if daily_path.exists():
        try:
            sim_cache = ROOT / "reports" / "summary" / "simulation_cache.csv"
            if sim_cache.exists():
                sim_df = pd.read_csv(sim_cache, parse_dates=["date"])
            else:
                from src.ise.ise_engine import ISEEngine
                engine = ISEEngine(config)
                engine.setup(str(daily_path), retrain=False)
                df = pd.read_csv(daily_path, parse_dates=["date"])
                sim_df = engine.run_simulation(df, user_id="U001")
                sim_df.to_csv(sim_cache, index=False)

            if len(sim_df) > 0:
                all_images.append(plot_vault_growth(sim_df, "U001", reports_dir / "ise" / "lstm"))
                stats = {
                    "total_saved": float(sim_df["micro_tranche"].sum()),
                    "avg_tranche": float(sim_df[sim_df["micro_tranche"] > 0]["micro_tranche"].mean()),
                    "days_saved":  int((sim_df["micro_tranche"] > 0).sum()),
                }
                all_metrics["simulation"] = stats
        except Exception as e:
            logger.warning(f"Simulation plots skipped: {e}")

    # ── SSI XGBoost ───────────────────────────────────────────────────────────
    ssi_model_path = ROOT / config["ssi"]["xgboost"]["model_path"]
    if ssi_model_path.exists():
        logger.info("Generating SSI plots...")
        try:
            from src.ssi.xgboost_model import SSIXGBoostExitModel
            model = SSIXGBoostExitModel(config)
            model.load(str(ssi_model_path))
            eval_r = model.evaluate(pd.read_csv(
                ROOT / "data" / "ise-data" / "ADANIPORTS.csv", parse_dates=["Date"]))
            fi = eval_r.get("feature_importances", {})
            if fi:
                all_images.append(plot_feature_importance(fi, ssi_dir))
            fpr, tpr = eval_r.get("fpr", []), eval_r.get("tpr", [])
            auc = eval_r.get("auc", 0)
            if len(fpr) > 0:
                all_images.append(plot_roc_curve(fpr, tpr, auc, ssi_dir))
            cm = eval_r.get("confusion_matrix", [[0,0],[0,0]])
            all_images.append(plot_confusion_matrix_ssi(cm, ssi_dir))
            prec, rec = eval_r.get("precision_curve", []), eval_r.get("recall_curve", [])
            ap = eval_r.get("average_precision", 0)
            if len(prec) > 0:
                all_images.append(plot_precision_recall(prec, rec, ap, ssi_dir))
            all_metrics["ssi"] = {
                "auc": auc, "precision": eval_r.get("precision", 0),
                "recall": eval_r.get("recall", 0), "f1": eval_r.get("f1", 0),
            }
        except Exception as e:
            logger.warning(f"SSI plots skipped: {e}")
    else:
        logger.warning("SSI model not found — skipping. Run: make train-ssi")

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    metrics_path = summary_dir / "metrics_summary.json"
    with open(metrics_path, "w") as f:
        json.dump({**all_metrics, "generated_at": generated_at}, f, indent=2)
    logger.info(f"Metrics JSON saved: {metrics_path}")

    # ── HTML Report ───────────────────────────────────────────────────────────
    valid_images = [p for p in all_images if p and p.exists()]
    html_path = generate_html_report(all_metrics, valid_images, summary_dir, generated_at)

    logger.success(f"\n{'='*60}")
    logger.success(f"  REPORT COMPLETE")
    logger.success(f"  Plots   : {len(valid_images)} images saved in reports/")
    logger.success(f"  HTML    : {html_path}")
    logger.success(f"  Metrics : {metrics_path}")
    logger.success(f"{'='*60}")
    print(f"\nOpen your report: {html_path}")


if __name__ == "__main__":
    with open(ROOT / "config" / "config.yaml") as f:
        config = yaml.safe_load(f)
    generate_all_reports(config)
