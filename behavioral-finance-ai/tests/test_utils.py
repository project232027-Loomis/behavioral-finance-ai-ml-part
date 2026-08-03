import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio, win_rate,
    portfolio_volatility, information_ratio, value_at_risk, cvar,
    PortfolioMetrics, PerformanceReport
)

@pytest.fixture
def pos_returns():
    return np.array([0.01, 0.02, 0.015, 0.01, 0.03])

@pytest.fixture
def neg_returns():
    return np.array([-0.01, -0.02, -0.015, -0.01, -0.03])

@pytest.fixture
def flat_portfolio():
    return np.array([100, 101, 102, 103, 105])

@pytest.fixture
def crash_portfolio():
    return np.array([100, 50, 25, 10])

def test_sharpe_positive_returns(pos_returns):
    # risk_free_rate = 0 to easily check > 0
    assert sharpe_ratio(pos_returns, risk_free_rate=0) > 0

def test_sharpe_negative_returns(neg_returns):
    assert sharpe_ratio(neg_returns, risk_free_rate=0) < 0

def test_max_drawdown_flat(flat_portfolio):
    assert max_drawdown(flat_portfolio) == 0.0

def test_max_drawdown_crash(crash_portfolio):
    # 100 -> 10 is a 90% drawdown
    assert max_drawdown(crash_portfolio) == -0.90

def test_sortino_ignores_upside(pos_returns):
    # If all returns are positive and mar=0, downside is empty, expected inf
    assert np.isinf(sortino_ratio(pos_returns, mar=0.0))

def test_win_rate_all_positive(pos_returns):
    assert win_rate(pos_returns) == 1.0

def test_win_rate_all_negative(neg_returns):
    assert win_rate(neg_returns) == 0.0

def test_var_ordering(neg_returns):
    # VaR and CVaR on negative returns
    var = value_at_risk(neg_returns, 0.95)
    cvar_val = cvar(neg_returns, 0.95)
    assert cvar_val <= var

@pytest.mark.parametrize("returns,portfolio,bm", [
    (np.random.randn(100)/100, np.cumprod(1 + np.random.randn(100)/100)*100, np.random.randn(100)/100)
])
def test_full_report_fields(returns, portfolio, bm):
    pm = PortfolioMetrics()
    report = pm.full_report(returns, portfolio, bm)
    
    assert isinstance(report, PerformanceReport)
    assert hasattr(report, 'sharpe_ratio')
    assert hasattr(report, 'sortino_ratio')
    assert hasattr(report, 'max_drawdown')
    assert hasattr(report, 'calmar_ratio')
    assert hasattr(report, 'win_rate')
    assert hasattr(report, 'portfolio_volatility')
    assert hasattr(report, 'information_ratio')
    assert hasattr(report, 'value_at_risk')
    assert hasattr(report, 'cvar')
