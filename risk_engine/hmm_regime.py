import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def fit_hmm_regimes(
    df,
    feature_cols,
    n_states=3,
    random_state=42
):
    """
    Fits a Gaussian HMM and assigns latent market regimes.
    """

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        random_state=random_state
    )

    X = df[feature_cols].values
    model.fit(X)

    hidden_states = model.predict(X)

    df_hmm = df.copy()
    df_hmm["hmm_state"] = hidden_states

    return df_hmm, model
