# step 1: Data generation with Faker & preprocessing

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()
Faker.seed(42)
np.random.seed(42)

def generate_user_cashflow(user_id="user_01", days=365, start_date="2025-01-01"):
    """
    Generates realistic daily spending and income patterns for a financial persona.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    dates = [start + timedelta(days=i) for i in range(days)]
    
    records = []
    balance = 50000.0  # Starting balance
    
    for dt in dates:
        # Salary on 1st of every month
        income = 60000.0 if dt.day == 1 else 0.0
        
        # Fixed monthly expenses (rent/bills on 5th)
        fixed_expense = 15000.0 if dt.day == 5 else 0.0
        
        # Variable daily spending (food, transit, leisure)
        daily_expense = np.random.exponential(scale=300) if dt.weekday() < 5 else np.random.exponential(scale=600)
        
        total_expense = fixed_expense + daily_expense
        balance = balance + income - total_expense
        
        records.append({
            "user_id": user_id,
            "date": dt.strftime("%Y-%m-%d"),
            "income": round(income, 2),
            "expense": round(total_expense, 2),
            "balance": round(balance, 2),
            "day_of_week": dt.strftime("%A"),
            "is_weekend": 1 if dt.weekday() >= 5 else 0
        })
        
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    return df

def preprocess_and_feature_engineering(df):
    """
    Cleans data and adds spending trends, rolling averages, and lag features.
    """
    df = df.ffill()
    df['rolling_expense_7d'] = df['expense'].rolling(window=7, min_periods=1).mean()
    df['rolling_balance_7d'] = df['balance'].rolling(window=7, min_periods=1).mean()
    df['expense_lag_1'] = df['expense'].shift(1).fillna(0)
    return df

if __name__ == "__main__":
    df = generate_user_cashflow()
    df = preprocess_and_feature_engineering(df)
    df.to_csv("user_cashflow.csv", index=False)
    print("Dataset generated and saved to user_cashflow.csv")