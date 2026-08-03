import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Union

def sharpe_ratio(returns: Union[np.ndarray, pd.Series], risk_free_rate: float = 0.065, periods: int = 252) -> float:
    returns = np.asarray(returns)
    if len(returns) == 0:
        return 0.0
    excess = returns - (risk_free_rate / periods)
    std = np.std(returns, ddof=1)
    if std == 0:
        return 0.0
    return np.mean(excess) / std * np.sqrt(periods)

def sortino_ratio(returns: Union[np.ndarray, pd.Series], mar: float = 0.0, periods: int = 252) -> float:
    returns = np.asarray(returns)
    if len(returns) == 0:
        return 0.0
    downside = returns[returns < mar]
    if len(downside) == 0:
        return np.inf
    down_std = np.std(downside, ddof=1) if len(downside) > 1 else np.std(downside)
    if down_std == 0:
        return np.inf
    return np.mean(returns - mar) / down_std * np.sqrt(periods)

def max_drawdown(portfolio_values: Union[np.ndarray, pd.Series]) -> float:
    pv = np.asarray(portfolio_values)
    if len(pv) == 0:
        return 0.0
    peaks = np.maximum.accumulate(pv)
    drawdowns = (pv - peaks) / peaks
    return np.min(drawdowns)

def calmar_ratio(returns: Union[np.ndarray, pd.Series], portfolio_values: Union[np.ndarray, pd.Series], periods: int = 252) -> float:
    returns = np.asarray(returns)
    if len(returns) == 0:
        return 0.0
    mdd = max_drawdown(portfolio_values)
    if mdd == 0:
        return np.inf
    ann_ret = np.mean(returns) * periods
    return ann_ret / abs(mdd)

def win_rate(returns: Union[np.ndarray, pd.Series]) -> float:
    returns = np.asarray(returns)
    if len(returns) == 0:
        return 0.0
    return np.sum(returns > 0) / len(returns)

def portfolio_volatility(returns: Union[np.ndarray, pd.Series], annualise: bool = True, periods: int = 252) -> float:
    returns = np.asarray(returns)
    if len(returns) == 0:
        return 0.0
    vol = np.std(returns, ddof=1)
    return vol * np.sqrt(periods) if annualise else vol

def information_ratio(returns: Union[np.ndarray, pd.Series], benchmark_returns: Union[np.ndarray, pd.Series], periods: int = 252) -> float:
    returns = np.asarray(returns)
    bm = np.asarray(benchmark_returns)
    if len(returns) == 0 or len(bm) == 0 or len(returns) != len(bm):
        return 0.0
    diff = returns - bm
    std = np.std(diff, ddof=1)
    if std == 0:
        return 0.0
    return np.mean(diff) / std * np.sqrt(periods)

def value_at_risk(returns: Union[np.ndarray, pd.Series], confidence: float = 0.95) -> float:
    returns = np.asarray(returns)
    if len(returns) == 0:
        return 0.0
    return np.percentile(returns, (1 - confidence) * 100)

def cvar(returns: Union[np.ndarray, pd.Series], confidence: float = 0.95) -> float:
    returns = np.asarray(returns)
    if len(returns) == 0:
        return 0.0
    var = value_at_risk(returns, confidence)
    tail = returns[returns <= var]
    if len(tail) == 0:
        return var
    return np.mean(tail)

@dataclass
class PerformanceReport:
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    portfolio_volatility: float
    information_ratio: float
    value_at_risk: float
    cvar: float

class PortfolioMetrics:
    def __init__(self, risk_free_rate: float = 0.065, trading_days: int = 252):
        self.rf = risk_free_rate
        self.td = trading_days

    def full_report(self, returns: np.ndarray, portfolio_values: np.ndarray, benchmark_returns: np.ndarray = None) -> PerformanceReport:
        ir = 0.0
        if benchmark_returns is not None:
            ir = information_ratio(returns, benchmark_returns, self.td)
            
        return PerformanceReport(
            sharpe_ratio=sharpe_ratio(returns, self.rf, self.td),
            sortino_ratio=sortino_ratio(returns, 0.0, self.td),
            max_drawdown=max_drawdown(portfolio_values),
            calmar_ratio=calmar_ratio(returns, portfolio_values, self.td),
            win_rate=win_rate(returns),
            portfolio_volatility=portfolio_volatility(returns, True, self.td),
            information_ratio=ir,
            value_at_risk=value_at_risk(returns),
            cvar=cvar(returns)
        )

    def print_report(self, report: PerformanceReport):
        print("-" * 30)
        print("PORTFOLIO PERFORMANCE REPORT")
        print("-" * 30)
        for k, v in asdict(report).items():
            print(f"{k.replace('_', ' ').title():<20}: {v:.4f}")
        print("-" * 30)

    def to_dict(self, report: PerformanceReport) -> dict:
        return asdict(report)
