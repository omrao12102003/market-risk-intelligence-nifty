import numpy as np
import pandas as pd


def compute_equity_curve(returns):
    """
    Converts returns to cumulative equity curve.
    """
    return (1 + returns).cumprod()


def compute_sharpe_ratio(returns):
    """
    Computes annualized Sharpe ratio.
    """
    mean_ret = returns.mean() * 252
    vol = returns.std() * np.sqrt(252)
    return mean_ret / vol if vol != 0 else np.nan


def compute_max_drawdown(equity_curve):
    """
    Computes maximum drawdown.
    """
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return drawdown.min()
