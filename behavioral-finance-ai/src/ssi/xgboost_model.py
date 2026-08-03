import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from dataclasses import dataclass, field
import joblib
from loguru import logger

FEATURES = [
    'rsi_14', 'macd', 'macd_signal', 'bb_position', 'volume_ratio',
    'momentum_5d', 'momentum_20d', 'atr_14', 'price_vs_50dma_pct',
    'price_vs_200dma_pct', 'volatility_20d'
]

@dataclass
class ExitSignal:
    symbol: str
    date: str
    exit_probability: float
    signal: str
    confidence: float
    feature_importances: dict = field(default_factory=dict)

class SSIXGBoostExitModel:
    def __init__(self, config: dict):
        self.n_estimators = config.get('n_estimators', 300)
        self.max_depth = config.get('max_depth', 5)
        self.learning_rate = config.get('learning_rate', 0.05)
        self.exit_threshold = config.get('exit_threshold', -0.03)
        self.exit_window = config.get('exit_window', 5)
        self.model_path = config.get('model_path', 'xgb_exit_model.pkl')
        self.model = None
        self.scaler = None
        logger.info(f"Initialized SSIXGBoostExitModel with exit_threshold {self.exit_threshold}")

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        sma = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['bb_position'] = (df['Close'] - (sma - 2*std)) / (4*std)
        
        # Volume
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        # Momentum
        df['momentum_5d'] = df['Close'].pct_change(periods=5)
        df['momentum_20d'] = df['Close'].pct_change(periods=20)
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(window=14).mean()
        
        # DMA
        df['price_vs_50dma_pct'] = (df['Close'] - df['Close'].rolling(window=50).mean()) / df['Close'].rolling(window=50).mean()
        df['price_vs_200dma_pct'] = (df['Close'] - df['Close'].rolling(window=200).mean()) / df['Close'].rolling(window=200).mean()
        
        # Volatility
        df['volatility_20d'] = df['Close'].pct_change().rolling(window=20).std()
        
        return df

    def _create_labels(self, df: pd.DataFrame) -> pd.Series:
        forward_return = df['Close'].pct_change(periods=self.exit_window).shift(-self.exit_window)
        return (forward_return < self.exit_threshold).astype(int)

    def train(self, price_df: pd.DataFrame) -> dict:
        logger.info("Training XGBoost Exit Model")
        
        df = self._compute_features(price_df)
        df['label'] = self._create_labels(df)
        df = df.dropna()
        
        X = df[FEATURES]
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1.0
        
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        preds = self.model.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True)
        
        joblib.dump(self.model, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
        
        return report

    def load(self, path: str = None):
        load_path = path or self.model_path
        self.model = joblib.load(load_path)
        logger.info(f"Model loaded from {load_path}")

    def predict_exit(self, recent_df: pd.DataFrame, symbol: str) -> ExitSignal:
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        
        df = self._compute_features(recent_df).dropna(subset=FEATURES)
        if len(df) == 0:
            raise ValueError("Not enough data to compute features.")
            
        latest = df.iloc[-1]
        x = latest[FEATURES].to_frame().T
        
        prob = self.model.predict_proba(x)[0][1]
        signal = 'EXIT' if prob > 0.5 else 'HOLD'
        
        importances = dict(zip(FEATURES, self.model.feature_importances_))
        
        return ExitSignal(
            symbol=symbol,
            date=str(latest['Date']) if 'Date' in latest else "",
            exit_probability=float(prob),
            signal=signal,
            confidence=float(abs(prob - 0.5) * 2),
            feature_importances=importances
        )

    def batch_predict(self, price_df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for symbol, group in price_df.groupby('Symbol'):
            try:
                res = self.predict_exit(group, symbol)
                results.append({
                    'Symbol': res.symbol,
                    'Date': res.date,
                    'Exit_Probability': res.exit_probability,
                    'Signal': res.signal
                })
            except Exception as e:
                logger.warning(f"Failed to predict for {symbol}: {e}")
        return pd.DataFrame(results)

    def evaluate(self, price_df: pd.DataFrame) -> dict:
        if self.model is None:
            raise ValueError("Model not loaded.")
        df = self._compute_features(price_df)
        df['label'] = self._create_labels(df)
        df = df.dropna()
        
        preds = self.model.predict(df[FEATURES])
        return classification_report(df['label'], preds, output_dict=True)

if __name__ == '__main__':
    logger.info("Demo XGBoost Model")
    dates = pd.date_range('2022-01-01', periods=500)
    data = pd.DataFrame({
        'Date': dates,
        'Symbol': 'ADANIPORTS',
        'Close': np.cumsum(np.random.randn(500) + 0.1) + 100,
        'High': np.cumsum(np.random.randn(500) + 0.1) + 105,
        'Low': np.cumsum(np.random.randn(500) + 0.1) + 95,
        'Volume': np.random.randint(1000, 5000, 500)
    })
    
    model = SSIXGBoostExitModel({'model_path': 'temp_model.pkl'})
    report = model.train(data)
    print("Eval Report:", report)
