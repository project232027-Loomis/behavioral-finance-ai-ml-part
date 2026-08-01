"""
Invisible Savings Engine (ISE) - LSTM Model
Predicts next-day Close price from historical stock time-series data.
Includes evaluation metrics and visualizations.
Branch: feature/ise-lstm
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

DATA_PATH = r"E:\client-project\behavioral-finance-ai-ml-part\behavioral-finance-ai\data\ise-data\ADANIPORTS.csv"
MODEL_OUT = "models/ise/lstm_savings_model.h5"
PLOTS_DIR = "models/ise/plots"
LOOKBACK = 60


# 1. Load data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")
    df = df.dropna(subset=["Close"])
    return df


# 2. Prepare sequences (60-day lookback window)
def create_sequences(data, lookback=LOOKBACK):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


# 3. Build LSTM model
def build_model(lookback=LOOKBACK):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# 4. Metrics
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape,
        "R2 Score": r2
    }

    print("\n===== Evaluation Metrics (on test set, original price scale) =====")
    for k, v in metrics.items():
        print(f"{k:12s}: {v:.4f}")
    print("=====================================================================\n")

    return metrics


# 5. Visualizations
def plot_loss_curve(history, out_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("LSTM Training vs Validation Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE, scaled)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_actual_vs_predicted(dates, y_true, y_pred, out_dir):
    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_true, label="Actual Close Price", color="black", linewidth=1.5)
    plt.plot(dates, y_pred, label="Predicted Close Price", color="red", linewidth=1.2, alpha=0.8)
    plt.title("Actual vs Predicted Close Price (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = os.path.join(out_dir, "actual_vs_predicted.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_zoomed_recent(dates, y_true, y_pred, out_dir, n=90):
    plt.figure(figsize=(12, 5))
    plt.plot(dates[-n:], y_true[-n:], label="Actual", color="black", marker="o", markersize=3)
    plt.plot(dates[-n:], y_pred[-n:], label="Predicted", color="red", marker="x", markersize=3)
    plt.title(f"Actual vs Predicted — Last {n} Days (Zoomed)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = os.path.join(out_dir, "zoomed_recent.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_residuals(y_true, y_pred, out_dir):
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(residuals, color="purple", linewidth=1)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Residuals Over Time (Actual - Predicted)")
    axes[0].set_xlabel("Test Sample Index")
    axes[0].set_ylabel("Residual (Price)")
    axes[0].grid(alpha=0.3)

    axes[1].hist(residuals, bins=40, color="teal", edgecolor="black")
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual (Price)")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "residuals.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# 6. Train
def train():
    # --- GPU check ---
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU detected: {[g.name for g in gpus]}")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    else:
        print("No GPU detected — training will run on CPU.")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs("models/ise", exist_ok=True)

    df = load_data()
    prices = df[["Close"]].values
    dates = df["Date"].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    X, y = create_sequences(scaled, lookback=LOOKBACK)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Align dates with sequences (dates correspond to the y target, offset by LOOKBACK)
    seq_dates = dates[LOOKBACK:]

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dates_test = seq_dates[split:]

    model = build_model()
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop]
    )

    final_val_loss = min(history.history["val_loss"])
    print(f"\nBest validation loss (scaled MSE): {final_val_loss:.6f}")

    # --- Predictions on test set ---
    y_pred_scaled = model.predict(X_test)

    # Inverse-transform back to real price scale
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_actual = scaler.inverse_transform(y_pred_scaled).flatten()

    # --- Metrics ---
    compute_metrics(y_test_actual, y_pred_actual)

    # --- Visualizations ---
    plot_loss_curve(history, PLOTS_DIR)
    plot_actual_vs_predicted(dates_test, y_test_actual, y_pred_actual, PLOTS_DIR)
    plot_zoomed_recent(dates_test, y_test_actual, y_pred_actual, PLOTS_DIR, n=90)
    plot_residuals(y_test_actual, y_pred_actual, PLOTS_DIR)

    # --- Save model ---
    model.save(MODEL_OUT)
    print(f"\nModel saved to {MODEL_OUT}")
    print(f"All plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    train()