import os
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from loguru import logger
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

"""
ISE — LSTM Spending Forecast Model
====================================
Trained on synthetic daily transaction summaries (NOT stock prices).
Predicts next-day discretionary spending and detects one-off expense patterns.

Architecture:
  Input  : (batch, lookback, num_features)  where num_features=10
  LSTM-1 : 128 units, return_sequences=True
  Dropout: 0.2
  LSTM-2 : 64 units
  Dropout: 0.2
  Dense  : 32, relu (shared)
  Head-1 : Dense(1, linear)  -> predicted_discretionary_spend  [regression]
  Head-2 : Dense(1, sigmoid) -> is_oneoff_today                [classification]

Usage:
  from src.ise.lstm_model import ISELSTMModel
  model = ISELSTMModel(config)
  model.train(daily_summary_df)
  result = model.predict(recent_60_day_df)  # returns ISELSTMResult
"""

FEATURES = [
    'discretionary_spent', 'recurring_spent', 'oneoff_spent', 'balance_eod',
    'day_sin', 'day_cos', 'month_day_sin', 'month_day_cos', 'is_weekend', 'total_debits'
]

@dataclass
class ISELSTMResult:
    predicted_spend_tomorrow: float
    is_oneoff_flag: bool
    oneoff_probability: float
    seven_day_forecast: list[float]

class ISELSTMModel:
    def __init__(self, config: dict):
        self.config = config
        self.lookback = config.get("lookback", 60)
        self.epochs = config.get("epochs", 50)
        self.batch_size = config.get("batch_size", 32)
        self.patience = config.get("patience", 10)
        self.model_path = config.get("model_path", "models/ise_lstm.keras")
        self.scaler_path = config.get("scaler_path", "models/ise_lstm_scaler.pkl")
        self.model = None
        self.scaler = MinMaxScaler()
        
    def _build_model(self) -> Model:
        inputs = Input(shape=(self.lookback, len(FEATURES)))
        x = LSTM(128, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        x = LSTM(64)(x)
        x = Dropout(0.2)(x)
        shared_dense = Dense(32, activation='relu')(x)
        
        out_spend = Dense(1, activation='linear', name='spend_pred')(shared_dense)
        out_oneoff = Dense(1, activation='sigmoid', name='oneoff_pred')(shared_dense)
        
        model = Model(inputs=inputs, outputs=[out_spend, out_oneoff])
        model.compile(
            optimizer='adam',
            loss={'spend_pred': 'mse', 'oneoff_pred': 'binary_crossentropy'},
            metrics={'spend_pred': 'mae', 'oneoff_pred': 'accuracy'}
        )
        return model

    def _prepare_sequences(self, df: pd.DataFrame, is_training: bool = True):
        # Sort values just in case
        df = df.sort_values(by=['user_id', 'date']).reset_index(drop=True)
        
        if is_training:
            scaled_data = self.scaler.fit_transform(df[FEATURES])
        else:
            scaled_data = self.scaler.transform(df[FEATURES])
            
        X, y_spend, y_oneoff = [], [], []
        
        # Iterate per user
        for uid in df['user_id'].unique():
            user_idx = df.index[df['user_id'] == uid].tolist()
            user_scaled = scaled_data[user_idx]
            user_df = df.iloc[user_idx].reset_index(drop=True)
            
            for i in range(len(user_scaled) - self.lookback):
                X.append(user_scaled[i : i + self.lookback])
                # Target is the next day's discretionary spend & oneoff
                y_spend.append(user_df.loc[i + self.lookback, 'discretionary_spent'])
                y_oneoff.append(user_df.loc[i + self.lookback, 'has_oneoff_today'])
                
        return np.array(X), np.array(y_spend), np.array(y_oneoff)

    def train(self, df: pd.DataFrame) -> dict:
        logger.info("Preparing sequences for training...")
        X, y_spend, y_oneoff = self._prepare_sequences(df, is_training=True)
        
        self.model = self._build_model()
        
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=self.patience, restore_best_weights=True
        )
        
        logger.info("Training LSTM model...")
        history = self.model.fit(
            X, {'spend_pred': y_spend, 'oneoff_pred': y_oneoff},
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        logger.info(f"Model saved to {self.model_path}")
        
        metrics = {
            'mse': history.history['loss'][-1],
            'val_mse': history.history['val_loss'][-1],
            'spend_mae': history.history['spend_pred_mae'][-1],
            'oneoff_accuracy': history.history['oneoff_pred_accuracy'][-1]
        }
        return metrics

    def load(self, model_path: str = None, scaler_path: str = None):
        mp = model_path or self.model_path
        sp = scaler_path or self.scaler_path
        
        if not os.path.exists(mp) or not os.path.exists(sp):
            raise FileNotFoundError("Model or scaler file not found. Train the model first.")
            
        self.model = tf.keras.models.load_model(mp)
        self.scaler = joblib.load(sp)
        logger.info(f"Loaded model from {mp}")

    def predict(self, recent_df: pd.DataFrame) -> ISELSTMResult:
        """Expects recent_df to have at least self.lookback rows for a single user."""
        if len(recent_df) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} days of history, got {len(recent_df)}")
            
        df = recent_df.tail(self.lookback).copy()
        scaled_data = self.scaler.transform(df[FEATURES])
        
        # Prepare input
        X = np.array([scaled_data])
        
        # Predict next day
        pred_spend, pred_oneoff = self.model.predict(X, verbose=0)
        
        spend_val = float(max(0, pred_spend[0][0]))
        oneoff_prob = float(pred_oneoff[0][0])
        is_oneoff = bool(oneoff_prob > 0.5)
        
        # 7-day forecast
        seven_day_fcst = self._seven_day_forecast(scaled_data)
        
        return ISELSTMResult(
            predicted_spend_tomorrow=spend_val,
            is_oneoff_flag=is_oneoff,
            oneoff_probability=oneoff_prob,
            seven_day_forecast=seven_day_fcst
        )
        
    def _seven_day_forecast(self, scaled_sequence: np.ndarray) -> list[float]:
        # scaled_sequence shape: (lookback, features)
        current_seq = np.copy(scaled_sequence)
        forecast = []
        spend_idx = FEATURES.index('discretionary_spent')
        
        for _ in range(7):
            X = np.array([current_seq])
            p_spend, p_oneoff = self.model.predict(X, verbose=0)
            
            pred_s = float(max(0, p_spend[0][0]))
            forecast.append(pred_s)
            
            # create next step (dummy logic for features, main focus is spend)
            next_step = np.copy(current_seq[-1])
            # We must inverse transform to put the new spend, then transform again? 
            # Or just hack it by replacing the scaled spend feature. 
            # For simplicity, we just inject the scaled prediction.
            
            # Create a dummy row to scale the predicted spend properly
            dummy_row = np.zeros(len(FEATURES))
            dummy_row[spend_idx] = pred_s
            dummy_scaled = self.scaler.transform([dummy_row])[0]
            
            next_step[spend_idx] = dummy_scaled[spend_idx]
            
            current_seq = np.vstack([current_seq[1:], next_step])
            
        return forecast

    def evaluate(self, df: pd.DataFrame) -> dict:
        X, y_spend, y_oneoff = self._prepare_sequences(df, is_training=False)
        results = self.model.evaluate(X, {'spend_pred': y_spend, 'oneoff_pred': y_oneoff}, verbose=0)
        return {
            'loss': results[0],
            'spend_loss': results[1],
            'oneoff_loss': results[2],
            'spend_mae': results[3],
            'oneoff_accuracy': results[4]
        }

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(base_dir, "data", "synthetic", "daily_summary.csv")
    
    config = {
        "lookback": 60,
        "epochs": 10,  # lower for quick demo
        "batch_size": 32,
        "model_path": os.path.join(base_dir, "models", "ise_lstm.keras"),
        "scaler_path": os.path.join(base_dir, "models", "ise_lstm_scaler.pkl")
    }
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        model = ISELSTMModel(config)
        metrics = model.train(df)
        print("Training Metrics:", metrics)
    else:
        print(f"Data file not found: {data_path}. Run faker_generator.py first.")