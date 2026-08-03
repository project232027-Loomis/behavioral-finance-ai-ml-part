import os
import sys
import pandas as pd
import numpy as np
import joblib
import torch
from prophet.serialize import model_from_json

# ------------------------------------------------------------------------------
# Dynamic Path Resolution
# ------------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # src/ise
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # src
MODEL_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", "models", "ise"))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from personas.data_generator import (
    generate_user_cashflow,
    preprocess_and_feature_engineering
)
from ise.lstm_model import LSTMModel
from ise.savings_engine import calculate_micro_savings
from ise.db_storage import init_db, log_savings_transaction


def predict_from_saved_models(df_recent, forecast_days=7, lstm_weight=0.5):
    # 1. Load Prophet Model
    prophet_path = os.path.join(MODEL_DIR, "prophet_model.json")
    with open(prophet_path, 'r') as f:
        prophet_model = model_from_json(f.read())

    future = prophet_model.make_future_dataframe(periods=forecast_days)
    prophet_preds = prophet_model.predict(future)['yhat'].tail(forecast_days).values

    # 2. Load Scaler & LSTM Model
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    scaler = joblib.load(scaler_path)

    lstm_path = os.path.join(MODEL_DIR, "lstm_savings_model.pt")
    lstm_model = LSTMModel(input_size=1, hidden_layer_size=50, output_size=1)
    lstm_model.load_state_dict(torch.load(lstm_path))
    lstm_model.eval()

    last_14_days = df_recent['balance'].tail(14).values
    curr_seq = scaler.transform(last_14_days.reshape(-1, 1))

    lstm_preds = []
    for _ in range(forecast_days):
        with torch.no_grad():
            x_tensor = torch.FloatTensor(curr_seq).unsqueeze(0)
            next_scaled = lstm_model(x_tensor).numpy()[0, 0]
            lstm_preds.append(next_scaled)
            curr_seq = np.append(curr_seq[1:], [[next_scaled]], axis=0)

    lstm_preds = scaler.inverse_transform(np.array(lstm_preds).reshape(-1, 1)).flatten()

    # 3. Combine Hybrid Predictions
    hybrid_forecast = (lstm_weight * lstm_preds) + ((1 - lstm_weight) * prophet_preds)
    return hybrid_forecast


def run_pipeline():
    print("--- 1. Fetching Recent User Cashflow Data ---")
    df = generate_user_cashflow(user_id="test_user_02", days=30)
    df = preprocess_and_feature_engineering(df)

    print("\n--- 2. Loading Saved Models & Generating 7-Day Forecast ---")
    predicted_balances = predict_from_saved_models(df, forecast_days=7)
    print("Predicted Balances for Next 7 Days:", np.round(predicted_balances, 2))

    print("\n--- 3. Computing Survival Buffer & Safe Micro-Savings ---")
    avg_daily_expense = df['expense'].tail(14).mean()
    expected_expenses_7d = [avg_daily_expense] * 7

    savings_result = calculate_micro_savings(predicted_balances, expected_expenses_7d)
    print("Survival Buffer:", savings_result['survival_buffer'])
    print("Safe Surplus:", savings_result['surplus'])
    print("Micro-Savings Recommendation:", savings_result['transfer_amount'])

    print("\n--- 4. Logging Execution Record to Docker PostgreSQL ---")
    try:
        init_db()
        log_savings_transaction(
            user_id="test_user_02",
            survival_buffer=savings_result['survival_buffer'],
            surplus=savings_result['surplus'],
            transfer_amount=savings_result['transfer_amount']
        )
        print("Successfully logged record into PostgreSQL Docker instance!")
    except Exception as e:
        print("Database error:", e)


# IMPORTANT: This block actually triggers execution when running the script directly
if __name__ == "__main__":
    run_pipeline()