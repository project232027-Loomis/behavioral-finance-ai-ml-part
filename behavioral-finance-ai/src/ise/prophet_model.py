import os
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass
from loguru import logger
from prophet import Prophet
import matplotlib.pyplot as plt

"""
ISE — FB-Prophet Balance Forecaster
=====================================
Handles macro-trends and seasonality that LSTM misses:
  - Holiday seasons (Diwali, year-end bonuses)
  - Annual subscription renewals  
  - Monthly payroll cycles
  - Long-term spending drift

Decomposes balance time-series into trend + weekly + yearly seasonality.
Provides the 'Forecasted Baseline' for the Adaptive Threshold engine.

Usage:
  from src.ise.prophet_model import ISEProphetModel
  prophet = ISEProphetModel(config)
  prophet.fit(daily_summary_df, user_id='U001')
  forecast = prophet.forecast(periods=30)
  baseline = prophet.get_7day_baseline()
"""

# Indian Holidays (2023-2024 subset)
INDIAN_HOLIDAYS_DATA = [
    ('2023-01-26', 'Republic Day'),
    ('2023-03-08', 'Holi'),
    ('2023-04-07', 'Good Friday'),
    ('2023-04-14', 'Ambedkar Jayanti'),
    ('2023-05-01', 'Labour Day'),
    ('2023-08-15', 'Independence Day'),
    ('2023-10-02', 'Gandhi Jayanti'),
    ('2023-10-24', 'Dussehra'),
    ('2023-11-12', 'Diwali'),
    ('2023-12-25', 'Christmas'),
    ('2024-01-01', 'New Year'),
    ('2024-01-26', 'Republic Day'),
    ('2024-03-25', 'Holi'),
    ('2024-03-29', 'Good Friday'),
    ('2024-04-14', 'Ambedkar Jayanti'),
    ('2024-05-01', 'Labour Day'),
    ('2024-08-15', 'Independence Day'),
    ('2024-10-02', 'Gandhi Jayanti'),
    ('2024-10-12', 'Dussehra'),
    ('2024-10-31', 'Diwali'),
    ('2024-12-25', 'Christmas')
]
INDIAN_HOLIDAYS = pd.DataFrame(INDIAN_HOLIDAYS_DATA, columns=['ds', 'holiday'])

@dataclass
class ProphetForecastResult:
    user_id: str
    forecast_df: pd.DataFrame
    trend: np.ndarray
    weekly_seasonality: np.ndarray
    yearly_seasonality: np.ndarray
    forecasted_baseline_7d: float
    confidence_lower: np.ndarray
    confidence_upper: np.ndarray
    components_df: pd.DataFrame

class ISEProphetModel:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.user_id = None
        self.last_date = None
        self.last_forecast = None

    def fit(self, df: pd.DataFrame, user_id: str):
        self.user_id = user_id
        user_df = df[df['user_id'] == user_id].copy()
        user_df['date'] = pd.to_datetime(user_df['date'])
        user_df = user_df.sort_values('date')
        
        self.last_date = user_df['date'].max()
        
        prophet_df = pd.DataFrame({
            'ds': user_df['date'],
            'y': user_df['balance_eod']
        })
        
        # Add regressors
        prophet_df['is_salary_day'] = (prophet_df['ds'].dt.day == 1).astype(int)
        prophet_df['is_rent_day'] = (prophet_df['ds'].dt.day == 5).astype(int)
        
        self.model = Prophet(holidays=INDIAN_HOLIDAYS, 
                             yearly_seasonality=True, 
                             weekly_seasonality=True, 
                             daily_seasonality=False)
        self.model.add_regressor('is_salary_day')
        self.model.add_regressor('is_rent_day')
        
        logger.info(f"Fitting Prophet model for user {user_id}...")
        self.model.fit(prophet_df)

    def forecast(self, periods: int = 30) -> ProphetForecastResult:
        if self.model is None:
            raise ValueError("Model must be fitted before forecasting.")
            
        future = self.model.make_future_dataframe(periods=periods)
        future['is_salary_day'] = (future['ds'].dt.day == 1).astype(int)
        future['is_rent_day'] = (future['ds'].dt.day == 5).astype(int)
        
        forecast = self.model.predict(future)
        self.last_forecast = forecast
        
        # future part only
        future_only = forecast[forecast['ds'] > self.last_date].head(periods)
        
        # 7-day baseline
        baseline_7d = float(future_only.head(7)['yhat'].median())
        
        return ProphetForecastResult(
            user_id=self.user_id,
            forecast_df=future_only[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
            trend=future_only['trend'].values,
            weekly_seasonality=future_only['weekly'].values,
            yearly_seasonality=future_only['yearly'].values,
            forecasted_baseline_7d=baseline_7d,
            confidence_lower=future_only['yhat_lower'].values,
            confidence_upper=future_only['yhat_upper'].values,
            components_df=forecast
        )

    def get_7day_baseline(self) -> float:
        if self.last_forecast is None:
            self.forecast(periods=7)
        future_only = self.last_forecast[self.last_forecast['ds'] > self.last_date]
        return float(future_only.head(7)['yhat'].median())

    def save(self, path: str):
        if self.model is None:
            raise ValueError("No model to save.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'user_id': self.user_id, 'last_date': self.last_date}, f)
        logger.info(f"Saved Prophet model to {path}")

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.user_id = data['user_id']
            self.last_date = data['last_date']
        logger.info(f"Loaded Prophet model from {path}")

    def plot_components(self, result: ProphetForecastResult, out_dir: str):
        if self.model is None:
            return
        os.makedirs(out_dir, exist_ok=True)
        fig = self.model.plot_components(result.components_df)
        out_path = os.path.join(out_dir, f"prophet_components_{self.user_id}.png")
        fig.savefig(out_path)
        plt.close(fig)
        logger.info(f"Saved components plot to {out_path}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(base_dir, "data", "synthetic", "daily_summary.csv")
    out_dir = os.path.join(base_dir, "models")
    
    config = {}
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        prophet = ISEProphetModel(config)
        prophet.fit(df, user_id='U001')
        forecast = prophet.forecast(periods=30)
        print(f"7-day baseline for U001: {forecast.forecasted_baseline_7d}")
        prophet.save(os.path.join(out_dir, "prophet_U001.pkl"))
    else:
        print(f"Data file not found: {data_path}")
