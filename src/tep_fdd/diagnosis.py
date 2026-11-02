"""Supervised fault diagnosis (which of the 21 faults is happening?).

Raw TEP snapshots are noisy and several faults only show up as changes in
variance (random-variation and valve-sticking faults). We therefore build
features over a short sliding window inside each run: the current value, the
rolling mean and the rolling standard deviation of every variable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def window_features(X: np.ndarray, groups: np.ndarray, window: int = 10) -> np.ndarray:
    """Current values + rolling mean/std computed within each run (no leakage across runs)."""
    df = pd.DataFrame(X)
    g = pd.Series(groups, name="g")
    roll = df.groupby(g.values).rolling(window, min_periods=1)
    mean = roll.mean().reset_index(level=0, drop=True).sort_index()
    std = roll.std().reset_index(level=0, drop=True).sort_index().fillna(0.0)
    return np.hstack([X, mean.to_numpy(), std.to_numpy()])


def make_models(seed: int = 42) -> dict[str, object]:
    return {
        "Logistic regression (raw)": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)
        ),
        "LightGBM (window features)": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            random_state=seed,
            verbose=-1,
        ),
    }
