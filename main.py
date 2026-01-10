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



# ===============================
# Day 07 – Regime-Aware Strategy
# ===============================

from risk_engine.strategy_overlay import (
    compute_strategy_returns,
    rule_based_exposure,
    hmm_based_exposure
)

print(">>> DAY 07 RUNNING <<<")

# Buy & Hold
df["bh_returns"] = df["returns"]

# Rule-based strategy
df["rule_exposure"] = rule_based_exposure(df)
df["rule_strategy_returns"] = compute_strategy_returns(
    df, "rule_exposure"
)

# HMM-based strategy
df["hmm_exposure"] = hmm_based_exposure(
    df_hmm,
    hmm_summary_df
)
df["hmm_strategy_returns"] = compute_strategy_returns(
    df, "hmm_exposure"
)

# Save results
strategy_cols = [
    "returns",
    "bh_returns",
    "rule_strategy_returns",
    "hmm_strategy_returns"
]

df[strategy_cols].to_csv(
    "results/nifty50_day07_strategy_returns.csv"
)

print("✅ DAY 07 STRATEGY CSV GENERATED SUCCESSFULLY")
# ===============================
# Day 08 – Strategy Performance
# ===============================

from risk_engine.performance_metrics import (
    compute_equity_curve,
    compute_sharpe_ratio,
    compute_max_drawdown
)

print(">>> DAY 08 RUNNING <<<")

performance_summary = []

strategies = {
    "Buy_and_Hold": df["bh_returns"],
    "Rule_Based": df["rule_strategy_returns"],
    "HMM_Based": df["hmm_strategy_returns"]
}

for name, returns in strategies.items():

    equity = compute_equity_curve(returns)

    performance_summary.append({
        "strategy": name,
        "sharpe_ratio": compute_sharpe_ratio(returns),
        "max_drawdown": compute_max_drawdown(equity)
    })

performance_df = pd.DataFrame(performance_summary)

performance_df.to_csv(
    "results/nifty50_day08_strategy_performance.csv",
    index=False
)

print("✅ DAY 08 PERFORMANCE SUMMARY GENERATED SUCCESSFULLY")
