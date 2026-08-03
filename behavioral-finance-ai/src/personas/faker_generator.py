import os
import uuid
import random
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger

"""
Synthetic Financial Persona Generator
=======================================
Generates realistic 1-year daily transaction histories for 5 Indian user personas.
Used to train the ISE LSTM model on personal spending sequences.

Output files:
  - data/synthetic/transactions.csv     : individual transactions
  - data/synthetic/daily_summary.csv    : aggregated daily features per user

Usage:
  python -m src.personas.faker_generator
  # or
  from src.personas.faker_generator import PersonaGenerator
  gen = PersonaGenerator(config)
  df_tx, df_daily = gen.run()
"""

@dataclass
class Persona:
    user_id: str
    name: str
    monthly_salary: float
    rent: float
    utility_avg: float
    dining_daily: float
    dining_weekend_multiplier: float
    grocery_daily: float
    transport_daily: float
    entertainment_monthly: float
    subscription_monthly: float
    initial_balance: float
    medical_probability: float
    subscription_day: int = 15

# CATEGORY_TYPE_MAP
CATEGORY_TYPE_MAP = {
    'SALARY': 'income',
    'RENT': 'recurring',
    'UTILITIES': 'recurring',
    'GROCERIES': 'discretionary',
    'DINING': 'discretionary',
    'TRANSPORT': 'discretionary',
    'ENTERTAINMENT': 'discretionary',
    'MEDICAL': 'one_off',
    'SUBSCRIPTION': 'recurring'
}

# 5 hardcoded Indian personas
PERSONAS = [
    Persona("U001", "Arjun Sharma", 65000, 15000, 3000, 400, 2.5, 300, 150, 4000, 1000, 25000, 0.003, 12),
    Persona("U002", "Priya Mehta", 95000, 25000, 4500, 600, 3.0, 400, 250, 6000, 1500, 40000, 0.002, 10),
    Persona("U003", "Ravi Kumar", 40000, 10000, 2000, 200, 2.0, 250, 100, 2000, 500, 15000, 0.004, 18),
    Persona("U004", "Sneha Patel", 120000, 30000, 6000, 800, 2.5, 600, 400, 8000, 2500, 80000, 0.002, 5),
    Persona("U005", "Karan Singh", 55000, 12000, 2500, 300, 2.0, 350, 150, 3000, 800, 20000, 0.005, 20)
]

def generate_transactions(persona: Persona, start_date: datetime, num_days: int) -> pd.DataFrame:
    records = []
    current_date = start_date
    balance = persona.initial_balance

    for day in range(num_days):
        is_weekend = current_date.weekday() >= 5
        day_of_month = current_date.day
        date_str = current_date.strftime("%Y-%m-%d")

        def add_tx(category, amount, is_credit=False, desc=""):
            nonlocal balance
            amount = round(amount, 2)
            if not is_credit:
                if balance < amount:
                    return # No overdraft
                balance -= amount
                tx_type = "debit"
                signed_amount = -amount
                direction = "out"
            else:
                balance += amount
                tx_type = "credit"
                signed_amount = amount
                direction = "in"

            records.append({
                "date": date_str,
                "user_id": persona.user_id,
                "persona_name": persona.name,
                "transaction_id": str(uuid.uuid4()),
                "category": category,
                "transaction_type": CATEGORY_TYPE_MAP[category],
                "direction": direction,
                "amount": amount,
                "signed_amount": signed_amount,
                "description": desc or category.title(),
                "balance_after": balance
            })

        # 1st of month: Salary
        if day_of_month == 1:
            add_tx("SALARY", persona.monthly_salary, is_credit=True, desc="Monthly Salary")
        
        # 5th of month: Rent
        if day_of_month == 5:
            add_tx("RENT", persona.rent, desc="Monthly Rent")
        
        # 10th-15th: Utilities (randomly on one of these days)
        if 10 <= day_of_month <= 15:
            if random.random() < 0.2 or day_of_month == 15: # Ensuring it happens by 15th
                variation = random.uniform(0.7, 1.3)
                add_tx("UTILITIES", persona.utility_avg * variation, desc="Utility Bills")
        
        # Subscription on fixed day
        if day_of_month == persona.subscription_day:
            add_tx("SUBSCRIPTION", persona.subscription_monthly, desc="Subscription Renewals")

        # Groceries: Daily (70% probability on weekends, 30% weekdays)
        grocery_prob = 0.7 if is_weekend else 0.3
        if random.random() < grocery_prob:
            variation = random.uniform(0.8, 1.5)
            add_tx("GROCERIES", persona.grocery_daily * variation)

        # Dining
        if random.random() < 0.6: # 40% probability to skip dining
            mult = persona.dining_weekend_multiplier if is_weekend else 1.0
            variation = random.uniform(0.5, 1.2)
            add_tx("DINING", persona.dining_daily * mult * variation)

        # Transport
        trans_prob = 0.4 if is_weekend else 0.85
        if random.random() < trans_prob:
            variation = random.uniform(0.8, 1.2)
            add_tx("TRANSPORT", persona.transport_daily * variation)

        # Entertainment
        if is_weekend and random.random() < 0.35:
            variation = random.uniform(0.5, 1.5)
            add_tx("ENTERTAINMENT", (persona.entertainment_monthly / 8) * variation)

        # Medical (One-off)
        if random.random() < persona.medical_probability:
            med_amount = random.uniform(2000, 25000)
            add_tx("MEDICAL", med_amount, desc="Medical Emergency")

        current_date += timedelta(days=1)

    return pd.DataFrame(records)

def build_daily_summary(df_tx: pd.DataFrame) -> pd.DataFrame:
    df_tx['date'] = pd.to_datetime(df_tx['date'])
    
    # Calculate components
    daily_groups = df_tx.groupby(['date', 'user_id'])
    
    summaries = []
    for (date, uid), group in daily_groups:
        discretionary_spent = group[group['transaction_type'] == 'discretionary']['amount'].sum()
        recurring_spent = group[group['transaction_type'] == 'recurring']['amount'].sum()
        oneoff_spent = group[group['transaction_type'] == 'one_off']['amount'].sum()
        income = group[group['transaction_type'] == 'income']['amount'].sum()
        total_debits = discretionary_spent + recurring_spent + oneoff_spent
        
        # EOD balance is the last balance_after for that day
        balance_eod = group['balance_after'].iloc[-1]
        
        is_weekend = int(date.weekday() >= 5)
        day_of_week = date.weekday()
        day_of_month = date.day
        has_oneoff_today = int(oneoff_spent > 0)
        
        # Cyclical encoding
        day_sin = math.sin(2 * math.pi * day_of_week / 7.0)
        day_cos = math.cos(2 * math.pi * day_of_week / 7.0)
        days_in_month = (date.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        month_day_sin = math.sin(2 * math.pi * day_of_month / days_in_month.day)
        month_day_cos = math.cos(2 * math.pi * day_of_month / days_in_month.day)
        
        summaries.append({
            'date': date,
            'user_id': uid,
            'discretionary_spent': discretionary_spent,
            'recurring_spent': recurring_spent,
            'oneoff_spent': oneoff_spent,
            'income': income,
            'total_debits': total_debits,
            'balance_eod': balance_eod,
            'is_weekend': is_weekend,
            'day_of_week': day_of_week,
            'day_of_month': day_of_month,
            'has_oneoff_today': has_oneoff_today,
            'day_sin': day_sin,
            'day_cos': day_cos,
            'month_day_sin': month_day_sin,
            'month_day_cos': month_day_cos
        })
        
    df_daily = pd.DataFrame(summaries)
    df_daily = df_daily.sort_values(by=['user_id', 'date']).reset_index(drop=True)
    return df_daily

class PersonaGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.num_days = config.get("num_days", 365)
        self.start_date = datetime.strptime(config.get("start_date", "2023-01-01"), "%Y-%m-%d")
        
    def run(self, output_dir: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Starting Persona Data Generation...")
        all_tx = []
        for persona in PERSONAS:
            logger.info(f"Generating data for {persona.name} ({persona.user_id})")
            df_tx = generate_transactions(persona, self.start_date, self.num_days)
            all_tx.append(df_tx)
            
        final_tx_df = pd.concat(all_tx, ignore_index=True)
        logger.info("Building Daily Summaries...")
        final_daily_df = build_daily_summary(final_tx_df)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            tx_path = os.path.join(output_dir, "transactions.csv")
            daily_path = os.path.join(output_dir, "daily_summary.csv")
            final_tx_df.to_csv(tx_path, index=False)
            final_daily_df.to_csv(daily_path, index=False)
            logger.info(f"Saved transactions to {tx_path}")
            logger.info(f"Saved daily summaries to {daily_path}")
            
        return final_tx_df, final_daily_df

if __name__ == '__main__':
    config = {
        "num_days": 365,
        "start_date": "2023-01-01"
    }
    gen = PersonaGenerator(config)
    
    # Ensure directory exists
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    out_dir = os.path.join(base_dir, "data", "synthetic")
    
    gen.run(output_dir=out_dir)
