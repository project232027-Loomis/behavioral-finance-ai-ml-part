
#approach 1: In-Memory / Dynamic Execution

#Combines predictions from LSTM and Prophet to produce an optimized 7-day balance forecast.

import numpy as np
import pandas as pd
from lstm_model import train_lstm_model
from prophet_model import train_prophet_forecast

def generate_hybrid_forecast(df, forecast_days=7, lstm_weight=0.5):
    # 1. Prophet Predictions
    prophet_pred = train_prophet_forecast(df, periods=forecast_days)['yhat'].values
    
    # 2. LSTM Predictions
    lstm_model, scaler = train_lstm_model(df['balance'])
    lstm_model.eval()
    
    last_seq = df['balance'].tail(14).values
    lstm_preds = []
    
    curr_seq = scaler.transform(last_seq.reshape(-1, 1))
    for _ in range(forecast_days):
        with torch.no_grad():
            x_tensor = torch.FloatTensor(curr_seq).unsqueeze(0)
            next_scaled = lstm_model(x_tensor).numpy()[0, 0]
            lstm_preds.append(next_scaled)
            curr_seq = np.append(curr_seq[1:], [[next_scaled]], axis=0)
            
    lstm_preds = scaler.inverse_transform(np.array(lstm_preds).reshape(-1, 1)).flatten()
    
    # 3. Hybrid Ensemble (Weighted Average)
    hybrid_balance_forecast = (lstm_weight * lstm_preds) + ((1 - lstm_weight) * prophet_pred)
    
    return hybrid_balance_forecast

