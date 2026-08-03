import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ssi.scoring_model import SSIScoringModel
from src.ssi.xgboost_model import SSIXGBoostExitModel

@pytest.fixture
def sample_data():
    dates = pd.date_range('2022-01-01', periods=300)
    np.random.seed(42)
    df = pd.DataFrame({
        'Date': dates,
        'Symbol': 'TEST',
        'Close': np.cumsum(np.random.randn(300)) + 100,
        'High': np.cumsum(np.random.randn(300)) + 105,
        'Low': np.cumsum(np.random.randn(300)) + 95,
        'Volume': np.random.randint(1000, 5000, 300)
    })
    return df

def test_rsi_range(sample_data):
    model = SSIScoringModel({})
    rsi = model.compute_rsi(sample_data['Close'])
    assert (rsi.dropna() >= 0).all()
    assert (rsi.dropna() <= 100).all()

def test_composite_score_range(sample_data):
    model = SSIScoringModel({})
    scores = model.score_stock('TEST', sample_data)
    assert (scores['composite_score'].dropna() >= 0).all()
    assert (scores['composite_score'].dropna() <= 100).all()

def test_buy_threshold(sample_data):
    model = SSIScoringModel({'buy_threshold': 70})
    scores = model.score_stock('TEST', sample_data)
    buys = model.get_buy_candidates(scores, threshold=70)
    if not buys.empty:
        assert (buys['composite_score'] >= 70).all()

def test_score_columns_present(sample_data):
    model = SSIScoringModel({})
    scores = model.score_stock('TEST', sample_data)
    expected_cols = [
        'Symbol', 'trend_score', 'volatility_score', 'volume_score', 
        'composite_score', 'signal', 'rsi_14', 'price_vs_50dma_pct', 
        'volume_ratio', 'hist_volatility_20d'
    ]
    for col in expected_cols:
        assert col in scores.columns

def test_features_no_nan_after_warmup(sample_data):
    xgb = SSIXGBoostExitModel({'model_path': 'dummy'})
    features = xgb._compute_features(sample_data)
    # After 200 rows (max window for 200dma), there should be no NaNs
    assert not features.iloc[200:].isnull().any().any()

def test_exit_signal_label(sample_data, tmp_path):
    model_path = tmp_path / "test_model.pkl"
    xgb = SSIXGBoostExitModel({'model_path': str(model_path)})
    xgb.train(sample_data)
    
    signal = xgb.predict_exit(sample_data, 'TEST')
    assert signal.signal in ['EXIT', 'HOLD']

def test_exit_probability_range(sample_data, tmp_path):
    model_path = tmp_path / "test_model.pkl"
    xgb = SSIXGBoostExitModel({'model_path': str(model_path)})
    xgb.train(sample_data)
    
    signal = xgb.predict_exit(sample_data, 'TEST')
    assert 0.0 <= signal.exit_probability <= 1.0
