#approach 1: In-Memory / Dynamic Execution


#Uses FB Prophet to capture weekly/monthly seasonality and long-term trends.

import pandas as pd
from prophet import Prophet

def train_prophet_forecast(df, periods=7):
    """
    Fits FB Prophet on daily account balance and forecasts next N days.
    """
    prophet_df = df[['date', 'balance']].rename(columns={'date': 'ds', 'balance': 'y'})
    
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    model.fit(prophet_df)
    
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
