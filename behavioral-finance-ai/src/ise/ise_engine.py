import os
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from loguru import logger

from src.ise.lstm_model import ISELSTMModel
from src.ise.prophet_model import ISEProphetModel
from src.ise.adaptive_threshold import AdaptiveThresholdEngine, VirtualVault

"""
ISE — Orchestration Engine
============================
Coordinates LSTM + Prophet + Adaptive Threshold into a single daily run.

Usage:
  from src.ise.ise_engine import ISEEngine
  engine = ISEEngine(config)
  engine.setup()  # loads models
  result = engine.run_daily(user_id='U001', current_date='2024-01-15', current_balance=52000, daily_summary_df=df)
  print(result.recommendation)
  print(f'Moving ₹{result.micro_tranche} to vault')
"""

@dataclass
class ISEResult:
    user_id: str
    date: str
    current_balance: float
    dynamic_buffer: float
    safe_to_save: float
    micro_tranche: int
    is_safe_to_save: bool
    is_oneoff_detected: bool
    predicted_spend_tomorrow: float
    prophet_baseline_7d: float
    seven_day_lstm_forecast: list[float]
    vault_balance_after: float
    recommendation: str
    confidence_score: float

class ISEEngine:
    def __init__(self, config: dict):
        self.config = config
        self.lstm_model = ISELSTMModel(config.get("lstm_config", {}))
        self.prophet_model = ISEProphetModel(config.get("prophet_config", {}))
        self.threshold_engine = AdaptiveThresholdEngine(config.get("threshold_config", {}))
        
        vault_file = config.get("vault_file", "data/synthetic/virtual_vault.csv")
        self.vault = VirtualVault(vault_file)
        
        # In a real app, Prophet needs to be loaded per user, or cached. 
        # Here we'll cache them in memory.
        self.prophet_models = {}

    def setup(self, daily_summary_path: str, retrain: bool = False):
        logger.info("Setting up ISE Engine...")
        if not os.path.exists(daily_summary_path):
            raise FileNotFoundError(f"Data file not found: {daily_summary_path}")
            
        df = pd.read_csv(daily_summary_path)
        
        if retrain:
            logger.info("Retraining LSTM...")
            self.lstm_model.train(df)
            
            logger.info("Retraining Prophet for all users...")
            for uid in df['user_id'].unique():
                pm = ISEProphetModel(self.config.get("prophet_config", {}))
                pm.fit(df, user_id=uid)
                self.prophet_models[uid] = pm
                
                # Save it
                model_dir = os.path.dirname(self.lstm_model.model_path)
                pm.save(os.path.join(model_dir, f"prophet_{uid}.pkl"))
        else:
            logger.info("Loading LSTM...")
            self.lstm_model.load()
            
            logger.info("Loading Prophet models...")
            for uid in df['user_id'].unique():
                pm = ISEProphetModel(self.config.get("prophet_config", {}))
                model_dir = os.path.dirname(self.lstm_model.model_path)
                p_path = os.path.join(model_dir, f"prophet_{uid}.pkl")
                if os.path.exists(p_path):
                    pm.load(p_path)
                    self.prophet_models[uid] = pm
                else:
                    logger.warning(f"Prophet model for {uid} not found. Will train now.")
                    pm.fit(df, user_id=uid)
                    pm.save(p_path)
                    self.prophet_models[uid] = pm

    def run_daily(self, user_id: str, current_date: str, current_balance: float, daily_summary_df: pd.DataFrame) -> ISEResult:
        # Get history up to current_date
        user_df = daily_summary_df[daily_summary_df['user_id'] == user_id].copy()
        user_df['date'] = pd.to_datetime(user_df['date'])
        target_date = pd.to_datetime(current_date)
        history_df = user_df[user_df['date'] <= target_date]
        
        # LSTM prediction
        lstm_result = self.lstm_model.predict(history_df)
        
        # Prophet baseline
        pm = self.prophet_models.get(user_id)
        if pm is None:
            raise ValueError(f"Prophet model not found for user {user_id}")
            
        # Normally Prophet is predicting from its last date. We assume its last date is close to current_date.
        prophet_baseline = pm.get_7day_baseline()
        
        # Adaptive Threshold
        threshold_result = self.threshold_engine.compute(
            current_balance=current_balance,
            lstm_7day_forecast=lstm_result.seven_day_forecast,
            prophet_7day_baseline=prophet_baseline,
            is_oneoff_detected=lstm_result.is_oneoff_flag
        )
        
        # Vault deposit
        vault_balance = self.vault.get_balance(user_id)
        if threshold_result.micro_tranche > 0:
            tx = self.vault.deposit(user_id, threshold_result.micro_tranche, target_date.date())
            vault_balance = tx.vault_balance_after
            
        return ISEResult(
            user_id=user_id,
            date=current_date,
            current_balance=current_balance,
            dynamic_buffer=threshold_result.dynamic_buffer,
            safe_to_save=threshold_result.safe_to_save,
            micro_tranche=threshold_result.micro_tranche,
            is_safe_to_save=threshold_result.is_safe_to_save,
            is_oneoff_detected=lstm_result.is_oneoff_flag,
            predicted_spend_tomorrow=lstm_result.predicted_spend_tomorrow,
            prophet_baseline_7d=prophet_baseline,
            seven_day_lstm_forecast=lstm_result.seven_day_forecast,
            vault_balance_after=vault_balance,
            recommendation=threshold_result.recommendation,
            confidence_score=1.0 - lstm_result.oneoff_probability # Simple confidence proxy
        )

    def run_simulation(self, daily_summary_df: pd.DataFrame, user_id: str) -> pd.DataFrame:
        user_df = daily_summary_df[daily_summary_df['user_id'] == user_id].copy()
        user_df['date'] = pd.to_datetime(user_df['date'])
        user_df = user_df.sort_values('date')
        
        lookback = self.lstm_model.lookback
        dates = user_df['date'].dt.strftime("%Y-%m-%d").tolist()
        
        if len(dates) <= lookback:
            logger.warning("Not enough data to run simulation.")
            return pd.DataFrame()
            
        results = []
        for i in range(lookback, len(dates)):
            curr_date = dates[i]
            # using balance_eod as current_balance for simulation
            curr_bal = user_df.iloc[i]['balance_eod']
            
            res = self.run_daily(user_id, curr_date, curr_bal, daily_summary_df)
            results.append(res.__dict__)
            
        return pd.DataFrame(results)

    def summary_stats(self, simulation_df: pd.DataFrame) -> dict:
        if simulation_df.empty:
            return {}
        
        total_saved = simulation_df['micro_tranche'].sum()
        avg_tranche = simulation_df[simulation_df['micro_tranche'] > 0]['micro_tranche'].mean()
        days_saved = (simulation_df['micro_tranche'] > 0).sum()
        days_blocked = simulation_df['is_oneoff_detected'].sum()
        
        return {
            "total_saved": float(total_saved),
            "avg_tranche": float(avg_tranche) if not pd.isna(avg_tranche) else 0.0,
            "days_saved": int(days_saved),
            "days_blocked_by_oneoff": int(days_blocked),
            "total_days": len(simulation_df)
        }

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(base_dir, "data", "synthetic", "daily_summary.csv")
    
    config = {
        "lstm_config": {
            "lookback": 60,
            "model_path": os.path.join(base_dir, "models", "ise_lstm.keras"),
            "scaler_path": os.path.join(base_dir, "models", "ise_lstm_scaler.pkl")
        },
        "prophet_config": {},
        "threshold_config": {
            "safety_factor": 1.20,
            "min_tranche": 50,
            "max_tranche": 500,
            "tranche_ratio": 0.30
        },
        "vault_file": os.path.join(base_dir, "data", "synthetic", "virtual_vault.csv")
    }
    
    engine = ISEEngine(config)
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        engine.setup(data_path, retrain=False) # Assumes trained models exist
        
        logger.info("Running simulation for U001...")
        sim_df = engine.run_simulation(df, user_id='U001')
        stats = engine.summary_stats(sim_df)
        print("Simulation Stats for U001:", stats)
    else:
        print("Data file not found. Generate data and train models first.")
