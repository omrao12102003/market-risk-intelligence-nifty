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




# ===============================
# Day 04A – Regime Persistence & Transition
# ===============================

from risk_engine.regime_analysis import (
    compute_regime_persistence,
    compute_transition_matrix
)

print(">>> DAY 04 RUNNING <<<")

# ---------- REGIME PERSISTENCE ----------
persistence_df = compute_regime_persistence(df)
persistence_df.to_csv(
    "results/nifty50_day04_regime_persistence.csv",
    index=False
)

# ---------- TRANSITION MATRIX ----------
transition_matrix = compute_transition_matrix(df)
transition_matrix.to_csv(
    "results/nifty50_day04_transition_matrix.csv"
)

print("✅ DAY 04 CSVs GENERATED SUCCESSFULLY")




# ===============================
# Day 04B – Regime Performance
# ===============================

from risk_engine.regime_performance import compute_regime_performance

print(">>> DAY 04B RUNNING <<<")

regime_perf_df = compute_regime_performance(df)

regime_perf_df.to_csv(
    "results/nifty50_day04b_regime_performance.csv",
    index=False
)

print("✅ DAY 04B CSV GENERATED SUCCESSFULLY")




# ===============================
# Day 05 – HMM Regime Detection
# ===============================

from risk_engine.hmm_regime import fit_hmm_regimes

print(">>> DAY 05 RUNNING (HMM) <<<")

hmm_features = [
    "returns",
    "volatility_20d",
    "drawdown"
]

df_hmm, hmm_model = fit_hmm_regimes(
    df,
    feature_cols=hmm_features,
    n_states=3
)

df_hmm.to_csv(
    "results/nifty50_day05_hmm_regimes.csv"
)

print("✅ DAY 05 HMM CSV GENERATED SUCCESSFULLY")




# ===============================
# Day 06 – HMM Regime Interpretation
# ===============================

from risk_engine.hmm_analysis import summarize_hmm_states

print(">>> DAY 06 RUNNING <<<")

hmm_summary_df = summarize_hmm_states(df_hmm)

hmm_summary_df.to_csv(
    "results/nifty50_day06_hmm_state_summary.csv",
    index=False
)

print("✅ DAY 06 HMM SUMMARY GENERATED SUCCESSFULLY")
