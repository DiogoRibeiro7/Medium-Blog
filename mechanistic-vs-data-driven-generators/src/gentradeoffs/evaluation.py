"""Yardsticks for synthetic tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from .world import COLUMNS


def discriminator_auc(real: pd.DataFrame, synth: pd.DataFrame,
                      seed: int = 0, folds: int = 5) -> float:
    """Out-of-fold ROC AUC of a classifier telling real rows from synthetic.

    0.5 means the classifier cannot tell them apart; 1.0 means every
    synthetic row is recognisable. This is the propensity-score utility
    measure of Snoke et al. (2018) with a boosted classifier.
    """
    X = np.vstack([real[COLUMNS].to_numpy(float),
                   synth[COLUMNS].to_numpy(float)])
    y = np.r_[np.zeros(len(real)), np.ones(len(synth))]
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True,
                                  random_state=seed).split(X, y):
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                             random_state=seed)
        clf.fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, p))


def day_level(table: pd.DataFrame, by: str = "day") -> pd.DataFrame:
    """Offered-weighted service level and abandonment for each day."""
    off = table["offered"]
    g = table.assign(w_sl=table["sl"] * off,
                     w_ab=table["abandon_rate"] * off).groupby(by)
    agg = g[["offered", "w_sl", "w_ab"]].sum()
    return pd.DataFrame({"sl": agg["w_sl"] / agg["offered"],
                         "abandon": agg["w_ab"] / agg["offered"]})


def pooled(table: pd.DataFrame) -> dict[str, float]:
    off = table["offered"].to_numpy(float)
    return {"sl": float(np.sum(table["sl"] * off) / off.sum()),
            "abandon": float(np.sum(table["abandon_rate"] * off) / off.sum())}


def distance_to_closest_record(train: pd.DataFrame, other: pd.DataFrame,
                               scale: np.ndarray) -> np.ndarray:
    """Euclidean distance from each row of ``other`` to its nearest
    training row, after dividing every column by ``scale``."""
    A = train[COLUMNS].to_numpy(float) / scale
    B = other[COLUMNS].to_numpy(float) / scale
    nn = NearestNeighbors(n_neighbors=1).fit(A)
    return nn.kneighbors(B, return_distance=True)[0][:, 0]
