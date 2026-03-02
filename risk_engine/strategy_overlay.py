import numpy as np
import pandas as pd


def compute_strategy_returns(df, exposure_col, return_col="returns"):
    """
    Computes strategy returns given exposure.
    """
    strat_returns = df[exposure_col] * df[return_col]
    return strat_returns


def rule_based_exposure(df):
    """
    Exposure mapping for rule-based regimes.
    """
    exposure_map = {
        "RISK-ON": 1.0,
        "NEUTRAL": 0.5,
        "RISK-OFF": 0.0
    }

    return df["risk_signal"].map(exposure_map)


def hmm_based_exposure(df, hmm_summary):
    """
    Exposure mapping for HMM regimes based on Sharpe ranking.
    """

    ranked_states = (
        hmm_summary
        .sort_values("sharpe_ratio", ascending=False)
        ["hmm_state"]
        .tolist()
    )

    exposure_map = {
        ranked_states[0]: 1.0,
        ranked_states[1]: 0.5,
        ranked_states[2]: 0.0
    }

    return df["hmm_state"].map(exposure_map)
