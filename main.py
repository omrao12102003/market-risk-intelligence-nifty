# ===============================
# Market Risk Intelligence
# Day 03 – Regime Detection
# ===============================

import os
import numpy as np
import pandas as pd

from risk_engine.regime_detection import add_risk_regimes


print(">>> NEW MAIN.PY RUNNING <<<")

# ---------- LOAD DATA ----------
df = pd.read_csv("data/nifty50_daily.csv")

df.columns = df.columns.str.strip()

df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)


# ---------- RETURNS ----------
df['returns'] = df['close'].pct_change()


# ---------- VOLATILITY ----------
df['volatility_20d'] = (
    df['returns']
    .rolling(20)
    .std()
    * np.sqrt(252)
)


# ---------- DRAWDOWN ----------
df['cum_returns'] = (1 + df['returns']).cumprod()
df['rolling_max'] = df['cum_returns'].cummax()
df['drawdown'] = (df['cum_returns'] - df['rolling_max']) / df['rolling_max']


# ---------- CLEAN ----------
df = df.dropna()


# ---------- RISK REGIME ENGINE ----------
df = add_risk_regimes(df)


# ---------- OUTPUT ----------
final_df = df[
    [
        'close',
        'returns',
        'volatility_20d',
        'drawdown',
        'vol_state',
        'risk_regime',
        'risk_signal'
    ]
]

os.makedirs("results", exist_ok=True)
final_df.to_csv("results/nifty50_day03_risk_regimes.csv")

print("✅ CSV GENERATED SUCCESSFULLY")
