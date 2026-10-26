"""The models. Deliberately ordinary: the experiment is about what surrounds them."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from .features import Feature

OTHER = "__other__"
MIN_LEVEL_COUNT = 30
MAX_LEVELS = 200


@dataclass
class Encoder:
    """Maps raw columns to model inputs using only what was seen in training."""

    features: list[Feature]
    levels: dict[str, list[str]] = field(default_factory=dict)

    @property
    def categorical(self) -> list[str]:
        return [f.name for f in self.features if f.kind == "categorical"]

    @property
    def numeric(self) -> list[str]:
        return [f.name for f in self.features if f.kind == "numeric"]

    def fit(self, df: pd.DataFrame) -> "Encoder":
        for col in self.categorical:
            counts = df[col].astype(str).value_counts()
            keep = counts[counts >= MIN_LEVEL_COUNT].index[:MAX_LEVELS]
            self.levels[col] = sorted(keep) + [OTHER]
        return self

    def frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self.numeric:
            out[col] = df[col].astype(float)
        for col in self.categorical:
            values = df[col].astype(str)
            values = values.where(values.isin(self.levels[col]), OTHER)
            out[col] = pd.Categorical(values, categories=self.levels[col])
        return out

    def matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Dense design matrix for the logistic model."""
        parts = []
        for col in self.numeric:
            x = df[col].astype(float).to_numpy()
            if col in ("lead_time", "days_in_waiting_list", "previous_cancellations",
                       "previous_bookings_not_canceled", "booking_changes"):
                x = np.log1p(np.clip(x, 0, None))
            if col == "adr":
                x = np.clip(x, 0, 500) / 100.0
            parts.append(x[:, None])
        frame = self.frame(df)
        for col in self.categorical:
            codes = frame[col].cat.codes.to_numpy()
            onehot = np.zeros((len(df), len(self.levels[col]) - 1))
            mask = codes < len(self.levels[col]) - 1
            onehot[np.flatnonzero(mask), codes[mask]] = 1.0
            parts.append(onehot)
        return np.hstack(parts)


class BaseRate:
    """No model: every visible booking gets the historical rate."""

    name = "base_rate"

    def __init__(self, features):
        self.features = features

    def fit(self, df, y):
        self.rate = float(np.mean(y))
        return self

    def predict_proba(self, df):
        return np.full(len(df), self.rate)


class Logistic:
    name = "logistic"

    def __init__(self, features):
        self.encoder = Encoder(features)

    def fit(self, df, y):
        self.encoder.fit(df)
        X = self.encoder.matrix(df)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-9
        self.model = LogisticRegression(C=1.0, max_iter=5000)
        self.model.fit((X - self.mu) / self.sd, y)
        return self

    def predict_proba(self, df):
        X = (self.encoder.matrix(df) - self.mu) / self.sd
        return self.model.predict_proba(X)[:, 1]


class Boosting:
    name = "boosting"

    def __init__(self, features, class_weight=None, **params):
        self.encoder = Encoder(features)
        self.class_weight = class_weight
        self.params = dict(random_state=0)
        self.params.update(params)

    def fit(self, df, y):
        self.encoder.fit(df)
        self.model = HistGradientBoostingClassifier(
            categorical_features="from_dtype", class_weight=self.class_weight, **self.params
        )
        self.model.fit(self.encoder.frame(df), y)
        return self

    def predict_proba(self, df):
        return self.model.predict_proba(self.encoder.frame(df))[:, 1]


TUNING_GRID = [
    dict(learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=leaf, l2_regularization=l2,
         max_iter=600, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30)
    for lr in (0.03, 0.1)
    for leaves in (15, 63)
    for leaf in (20, 200)
    for l2 in (0.0, 1.0)
]


def tune_boosting(features, df: pd.DataFrame, y: np.ndarray) -> dict:
    """Pick boosting parameters on the last 20% of the window, ordered by arrival."""
    order = np.argsort(df["arrival"].to_numpy(), kind="stable")
    cut = int(0.8 * len(order))
    fit_idx, val_idx = order[:cut], order[cut:]
    best, best_loss = None, np.inf
    for params in TUNING_GRID:
        model = Boosting(features, **params).fit(df.iloc[fit_idx], y[fit_idx])
        p = np.clip(model.predict_proba(df.iloc[val_idx]), 1e-6, 1 - 1e-6)
        loss = log_loss(y[val_idx], p, labels=[0, 1])
        if loss < best_loss:
            best, best_loss = params, loss
    return best
