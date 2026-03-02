# risk_engine/regime_detection.py

import numpy as np
import pandas as pd


def add_risk_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
    - volatility state
    - risk regime
    - risk signal
    """

    # --- Volatility thresholds (adaptive) ---
    vol_low = df['volatility_20d'].quantile(0.33)
    vol_high = df['volatility_20d'].quantile(0.66)

    # --- Volatility state ---
    def volatility_state(vol):
        if vol < vol_low:
            return "Low Volatility"
        elif vol < vol_high:
            return "Medium Volatility"
        else:
            return "High Volatility"

    df['vol_state'] = df['volatility_20d'].apply(volatility_state)

    # --- Risk regime ---
    def risk_regime(row):
        if row['vol_state'] == "Low Volatility" and row['drawdown'] > -0.05:
            return "Low Risk"
        elif row['vol_state'] == "High Volatility" and row['drawdown'] < -0.15:
            return "High Risk"
        else:
            return "Medium Risk"

    df['risk_regime'] = df.apply(risk_regime, axis=1)

    # --- Risk signal ---
    def risk_signal(regime):
        if regime == "Low Risk":
            return "RISK-ON"
        elif regime == "High Risk":
            return "RISK-OFF"
        else:
            return "NEUTRAL"

    df['risk_signal'] = df['risk_regime'].apply(risk_signal)

    return df
