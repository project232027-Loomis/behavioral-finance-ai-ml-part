import pandas as pd
import numpy as np
from dataclasses import dataclass
from loguru import logger

@dataclass
class StockScore:
    symbol: str
    date: str
    trend_score: float
    volatility_score: float
    volume_score: float
    composite_score: float
    signal: str
    rsi_14: float
    price_vs_50dma_pct: float
    volume_ratio: float
    hist_volatility_20d: float

class SSIScoringModel:
    def __init__(self, config: dict):
        self.weights = config.get('weights', {'trend': 0.40, 'volatility': 0.30, 'volume': 0.30})
        self.buy_threshold = config.get('buy_threshold', 70)
        logger.info(f"Initialized SSIScoringModel with weights {self.weights} and threshold {self.buy_threshold}")

    def compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def compute_trend_score(self, df: pd.DataFrame) -> pd.Series:
        rsi = self.compute_rsi(df['Close'])
        ma50 = df['Close'].rolling(window=50).mean()
        price_vs_ma = ((df['Close'] - ma50) / ma50) * 100
        
        rsi_score = rsi.clip(30, 70).apply(lambda x: (x - 30) / 40 * 100)
        ma_score = price_vs_ma.clip(-10, 10).apply(lambda x: (x + 10) / 20 * 100)
        
        return (rsi_score * 0.5 + ma_score * 0.5).fillna(50)

    def compute_volatility_score(self, df: pd.DataFrame) -> pd.Series:
        returns = df['Close'].pct_change()
        volatility = returns.rolling(window=20).std()
        
        vol_min = volatility.rolling(window=252, min_periods=20).min()
        vol_max = volatility.rolling(window=252, min_periods=20).max()
        
        vol_score = 100 - ((volatility - vol_min) / (vol_max - vol_min) * 100)
        return vol_score.fillna(50)

    def compute_volume_score(self, df: pd.DataFrame) -> pd.Series:
        vol_ma = df['Volume'].rolling(window=20).mean()
        vol_ratio = df['Volume'] / vol_ma
        
        vol_score = vol_ratio.clip(0.5, 2.0).apply(lambda x: (x - 0.5) / 1.5 * 100)
        return vol_score.fillna(50)

    def score_stock(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            trend = self.compute_trend_score(df)
            vol = self.compute_volatility_score(df)
            volum = self.compute_volume_score(df)
            
            comp_score = (trend * self.weights['trend'] + 
                          vol * self.weights['volatility'] + 
                          volum * self.weights['volume'])
            
            df['Symbol'] = symbol
            df['trend_score'] = trend
            df['volatility_score'] = vol
            df['volume_score'] = volum
            df['composite_score'] = comp_score
            df['signal'] = np.where(df['composite_score'] >= self.buy_threshold, 'BUY', 'HOLD')
            
            df['rsi_14'] = self.compute_rsi(df['Close'])
            ma50 = df['Close'].rolling(window=50).mean()
            df['price_vs_50dma_pct'] = ((df['Close'] - ma50) / ma50) * 100
            df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
            df['hist_volatility_20d'] = df['Close'].pct_change().rolling(window=20).std()
            
            return df
        except Exception as e:
            logger.error(f"Error scoring stock {symbol}: {e}")
            return pd.DataFrame()

    def score_all(self, price_df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Scoring all stocks in dataframe of size {len(price_df)}")
        results = []
        for symbol, group in price_df.groupby('Symbol'):
            res = self.score_stock(symbol, group)
            results.append(res)
            
        if not results:
            return pd.DataFrame()
            
        combined = pd.concat(results).sort_values(by=['Date', 'composite_score'], ascending=[True, False])
        return combined

    def get_buy_candidates(self, scores_df: pd.DataFrame, threshold: float = 70) -> pd.DataFrame:
        return scores_df[scores_df['signal'] == 'BUY']

    def latest_scores(self, price_df: pd.DataFrame) -> pd.DataFrame:
        scores = self.score_all(price_df)
        latest_date = scores['Date'].max()
        return scores[scores['Date'] == latest_date]

if __name__ == '__main__':
    # Demo code
    logger.info("Running demo scoring...")
    dates = pd.date_range('2023-01-01', periods=300)
    data = pd.DataFrame({
        'Date': dates,
        'Symbol': 'ADANIPORTS',
        'Close': np.linspace(100, 200, 300) + np.random.randn(300) * 5,
        'Volume': np.random.randint(1000, 5000, 300)
    })
    
    scorer = SSIScoringModel({})
    scores = scorer.score_all(data)
    buys = scorer.get_buy_candidates(scores)
    print("Top candidates:")
    print(buys.tail())
