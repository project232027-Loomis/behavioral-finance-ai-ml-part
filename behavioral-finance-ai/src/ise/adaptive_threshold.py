import os
import pandas as pd
from datetime import date
from dataclasses import dataclass
from loguru import logger

"""
ISE — Adaptive Thresholding Engine
====================================
Implements the 'Safety-First' logic:

  Dynamic Buffer   = max(LSTM 7-day forecast sum, Prophet 7-day baseline) × safety_factor
  Safe-to-Save     = max(0, current_balance − dynamic_buffer)
  Micro-Tranche    = clip(floor(safe_to_save × tranche_ratio), min=₹50, max=₹500)

The micro-tranche is moved to the VirtualVault — the simulated investment account.

Usage:
  from src.ise.adaptive_threshold import AdaptiveThresholdEngine, VirtualVault
  engine = AdaptiveThresholdEngine(config)
  result = engine.compute(current_balance=50000, lstm_7day=[300,350,280,...], prophet_baseline=42000)
  vault = VirtualVault()
  vault.deposit(user_id='U001', amount=result.micro_tranche, date=today)
"""

@dataclass
class ThresholdResult:
    dynamic_buffer: float
    safe_to_save: float
    micro_tranche: int
    is_safe_to_save: bool
    recommendation: str
    breakdown: dict

class AdaptiveThresholdEngine:
    def __init__(self, config: dict):
        self.safety_factor = config.get("safety_factor", 1.20)
        self.min_tranche = config.get("min_tranche", 50)
        self.max_tranche = config.get("max_tranche", 500)
        self.tranche_ratio = config.get("tranche_ratio", 0.30)
        self.buffer_days = config.get("buffer_days", 7)

    def compute(self, current_balance: float, lstm_7day_forecast: list[float], prophet_7day_baseline: float, is_oneoff_detected: bool = False) -> ThresholdResult:
        if is_oneoff_detected:
            return ThresholdResult(
                dynamic_buffer=0.0,
                safe_to_save=0.0,
                micro_tranche=0,
                is_safe_to_save=False,
                recommendation="Detected potential one-off expense today. Pausing micro-savings to ensure liquidity.",
                breakdown={"reason": "oneoff_detected"}
            )
            
        lstm_sum = sum(lstm_7day_forecast)
        base_val = max(lstm_sum, prophet_7day_baseline)
        
        dynamic_buffer = base_val * self.safety_factor
        safe_to_save = max(0.0, current_balance - dynamic_buffer)
        
        if safe_to_save > self.min_tranche:
            raw_tranche = int(safe_to_save * self.tranche_ratio)
            micro_tranche = max(self.min_tranche, min(raw_tranche, self.max_tranche))
            is_safe = True
            rec = f"Safe to save. Depositing ₹{micro_tranche}."
        else:
            micro_tranche = 0
            is_safe = False
            rec = "Balance too close to required buffer. Skipping deposit today."
            
        breakdown = {
            "current_balance": current_balance,
            "lstm_7day_sum": lstm_sum,
            "prophet_7day_baseline": prophet_7day_baseline,
            "dynamic_buffer": dynamic_buffer,
            "safe_to_save": safe_to_save
        }
        
        return ThresholdResult(
            dynamic_buffer=dynamic_buffer,
            safe_to_save=safe_to_save,
            micro_tranche=micro_tranche,
            is_safe_to_save=is_safe,
            recommendation=rec,
            breakdown=breakdown
        )

    def explain(self, result: ThresholdResult) -> str:
        return result.recommendation


@dataclass
class VaultTransaction:
    user_id: str
    date: date
    amount: float
    vault_balance_after: float
    source: str

class VirtualVault:
    def __init__(self, vault_file: str = 'data/synthetic/virtual_vault.csv'):
        # Ensure path is absolute if it's relative
        if not os.path.isabs(vault_file):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self.vault_file = os.path.join(base_dir, vault_file)
        else:
            self.vault_file = vault_file
            
        self.history = []
        if os.path.exists(self.vault_file):
            df = pd.read_csv(self.vault_file)
            df['date'] = pd.to_datetime(df['date']).dt.date
            self.history = [VaultTransaction(**row) for row in df.to_dict('records')]
        else:
            os.makedirs(os.path.dirname(self.vault_file), exist_ok=True)

    def deposit(self, user_id: str, amount: float, date_val: date, source: str = 'ISE') -> VaultTransaction:
        current_balance = self.get_balance(user_id)
        new_balance = current_balance + amount
        
        tx = VaultTransaction(
            user_id=user_id,
            date=date_val,
            amount=amount,
            vault_balance_after=new_balance,
            source=source
        )
        self.history.append(tx)
        self._save()
        return tx

    def get_balance(self, user_id: str) -> float:
        user_tx = [tx for tx in self.history if tx.user_id == user_id]
        if not user_tx:
            return 0.0
        # sort by date to get the latest
        user_tx.sort(key=lambda x: x.date)
        return user_tx[-1].vault_balance_after

    def get_history(self, user_id: str) -> pd.DataFrame:
        user_tx = [tx for tx in self.history if tx.user_id == user_id]
        return pd.DataFrame([tx.__dict__ for tx in user_tx])

    def total_saved_all_users(self) -> dict:
        totals = {}
        for tx in self.history:
            totals[tx.user_id] = tx.vault_balance_after # This assumes chronological insert
        
        # A more robust way:
        df = pd.DataFrame([tx.__dict__ for tx in self.history])
        if df.empty:
            return {}
        
        latest_balances = df.groupby('user_id').last()['vault_balance_after'].to_dict()
        return latest_balances
        
    def _save(self):
        df = pd.DataFrame([tx.__dict__ for tx in self.history])
        df.to_csv(self.vault_file, index=False)
