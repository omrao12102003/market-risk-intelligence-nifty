import pandas as pd


def compute_regime_persistence(df, regime_col="risk_signal"):
    df = df.copy()

    df["regime_change"] = df[regime_col] != df[regime_col].shift(1)
    df["regime_block"] = df["regime_change"].cumsum()

    persistence = (
        df.groupby(["regime_block", regime_col])
        .size()
        .reset_index(name="duration_days")
    )

    return persistence


def compute_transition_matrix(df, regime_col="risk_signal"):
    transitions = pd.crosstab(
        df[regime_col].shift(1),
        df[regime_col],
        normalize="index"
    )

    return transitions.fillna(0)
