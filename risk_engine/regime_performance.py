import numpy as np
import pandas as pd


def compute_regime_performance(
    df,
    regime_col="risk_signal",
    return_col="returns"
):
    """
    Computes performance statistics for each regime.
    """

    results = []

    for regime, group in df.groupby(regime_col):

        avg_daily_ret = group[return_col].mean()
        ann_return = avg_daily_ret * 252

        ann_vol = group[return_col].std() * np.sqrt(252)

        sharpe = (
            ann_return / ann_vol
            if ann_vol != 0 else np.nan
        )

        hit_ratio = (group[return_col] > 0).mean()

        cum_ret = (1 + group[return_col]).cumprod()
        rolling_max = cum_ret.cummax()
        drawdown = (cum_ret - rolling_max) / rolling_max
        max_dd = drawdown.min()

        results.append({
            "risk_signal": regime,
            "avg_daily_return": avg_daily_ret,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "hit_ratio": hit_ratio,
            "num_days": len(group)
        })

    return pd.DataFrame(results)
