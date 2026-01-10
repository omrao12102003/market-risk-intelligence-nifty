import numpy as np
import pandas as pd


def summarize_hmm_states(
    df,
    state_col="hmm_state",
    return_col="returns"
):
    """
    Computes financial statistics for each HMM state.
    """

    results = []

    for state, group in df.groupby(state_col):

        avg_daily_ret = group[return_col].mean()
        ann_return = avg_daily_ret * 252
        ann_vol = group[return_col].std() * np.sqrt(252)

        sharpe = (
            ann_return / ann_vol
            if ann_vol != 0 else np.nan
        )

        cum_ret = (1 + group[return_col]).cumprod()
        rolling_max = cum_ret.cummax()
        drawdown = (cum_ret - rolling_max) / rolling_max
        max_dd = drawdown.min()

        results.append({
            "hmm_state": state,
            "avg_daily_return": avg_daily_ret,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "num_days": len(group)
        })

    return pd.DataFrame(results)
