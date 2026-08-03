# ==============================================================================
# Approach 2: Model Persistence (Offline Training & Model Serialization)
# ==============================================================================
# Trains the LSTM and Prophet models once and serializes their weights and 
# artifacts to the models/ise directory for fast online inference.

import os
import sys
import pandas as pd
import numpy as np
import joblib
import torch
from prophet import Prophet
from prophet.serialize import model_to_json

# Dynamically resolve src path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # src/ise
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # src
MODEL_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", "models", "ise"))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

os.makedirs(MODEL_DIR, exist_ok=True)

from personas.data_generator import (
    generate_user_cashflow,
    preprocess_and_feature_engineering
)
from ise.lstm_model import train_lstm_model


def train_and_save():
    print("--- 1. Generating & Preprocessing Historical Cash Flow Data ---")
    df = generate_user_cashflow(user_id="trainer_01", days=365)
    df = preprocess_and_feature_engineering(df)

    # --------------------------------------------------
    # 2. Train & Save LSTM Model + Scaler
    # --------------------------------------------------
    print("\n--- 2. Training PyTorch LSTM Model ---")
    lstm_model, scaler = train_lstm_model(df['balance'], seq_length=14, epochs=50)

    lstm_path = os.path.join(MODEL_DIR, "lstm_savings_model.pt")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")

    torch.save(lstm_model.state_dict(), lstm_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved LSTM Model to: {lstm_path}")
    print(f"Saved Scaler to: {scaler_path}")

    # --------------------------------------------------
    # 3. Train & Save FB Prophet Model
    # --------------------------------------------------
    print("\n--- 3. Training FB Prophet Model ---")
    prophet_df = df[['date', 'balance']].rename(columns={'date': 'ds', 'balance': 'y'})
    prophet_model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    prophet_model.fit(prophet_df)

    prophet_path = os.path.join(MODEL_DIR, "prophet_model.json")
    with open(prophet_path, 'w') as f:
        f.write(model_to_json(prophet_model))
    print(f"Saved Prophet Model to: {prophet_path}")

    print("\n✅ All models successfully trained and stored in models/ise/!")


if __name__ == "__main__":
    train_and_save()