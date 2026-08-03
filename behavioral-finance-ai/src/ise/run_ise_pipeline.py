#approach 1: In-Memory / Dynamic Execution


#   In memory/dynamic execution( the first approach )
    # it is on the fly ,continuus training or end to end pipline.
    # the script treains the models from screath in memory evry time it is runs, generated forecast and discards the odels once executin completes
    
    


import os
import sys

# Get the current file directory (src/ise) and add the src directory to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # src/ise
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # src
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pandas as pd
import numpy as np

# Direct imports from modules under src/
from personas.data_generator import (
    generate_user_cashflow,
    preprocess_and_feature_engineering
)
from ise.hybrid_forecaster import generate_hybrid_forecast
from ise.savings_engine import calculate_micro_savings
from ise.db_storage import init_db, log_savings_transaction


def run_test():
    print("--- 1. Generating & Preprocessing Cash Flow Data ---")
    df = generate_user_cashflow(user_id="test_user_01", days=180)
    df = preprocess_and_feature_engineering(df)
    print(f"Generated {len(df)} days of financial records.")

    print("\n--- 2. Generating 7-Day Hybrid Balance Forecast (LSTM + Prophet) ---")
    predicted_balances = generate_hybrid_forecast(df, forecast_days=7, lstm_weight=0.5)
    print("Predicted Balances for Next 7 Days:", np.round(predicted_balances, 2))

    print("\n--- 3. Estimating 7-Day Expected Expenses ---")
    avg_daily_expense = df['expense'].tail(14).mean()
    expected_expenses_7d = [avg_daily_expense] * 7
    print("Expected 7-Day Expenses:", np.round(expected_expenses_7d, 2))

    print("\n--- 4. Computing Survival Buffer & Safe-to-Save Surplus ---")
    savings_result = calculate_micro_savings(predicted_balances, expected_expenses_7d)
    print("Survival Buffer:", savings_result['survival_buffer'])
    print("Surplus Amount:", savings_result['surplus'])
    print("Micro-Savings Transfer Recommendation:", savings_result['transfer_amount'])

    print("\n--- 5. Logging to PostgreSQL (Docker) ---")
    try:
        init_db()
        log_savings_transaction(
            user_id="test_user_01",
            survival_buffer=savings_result['survival_buffer'],
            surplus=savings_result['surplus'],
            transfer_amount=savings_result['transfer_amount']
        )
        print("Successfully logged record into PostgreSQL Docker instance!")
    except Exception as e:
        print("Database error:", e)


if __name__ == "__main__":
    run_test()