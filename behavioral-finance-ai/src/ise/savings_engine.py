import os
import sys
import numpy as np

# Dynamically resolve path to src directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # src/ise
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # src
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def calculate_micro_savings(predicted_balances, expected_expenses_7d, safety_margin_pct=0.15):
    """
    Calculates the 7-day survival buffer and determines the safe automated micro-savings transfer amount.
    """
    # 1. Total expected expenses over 7 days + safety margin
    total_expected_expense = np.sum(expected_expenses_7d)
    survival_buffer = total_expected_expense * (1 + safety_margin_pct)

    # 2. Minimum projected balance over the 7 days
    min_projected_balance = np.min(predicted_balances)

    # 3. Safe-to-save surplus calculation
    surplus = min_projected_balance - survival_buffer

    # 4. Micro-savings transfer rule (capped between ₹50 and ₹500)
    if surplus <= 0:
        recommended_transfer = 0.0
    else:
        calculated_amount = surplus * 0.10
        recommended_transfer = max(50.0, min(500.0, calculated_amount))

    return {
        "survival_buffer": round(float(survival_buffer), 2),
        "surplus": round(float(surplus), 2),
        "transfer_amount": round(float(recommended_transfer), 2)
    }