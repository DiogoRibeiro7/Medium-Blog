"""Shared data, simulation and model definitions.

Data (public, fetched by fetch.py):
  * UCI Adult (census income), 48,842 rows, via the jbrownlee/Datasets mirror.
  * California housing (1990 census), 20,640 block groups, via ageron/handson-ml2.

Simulation with a known Bayes error: X ~ N(0, I_10), of which the first five
coordinates carry signal; the clean label is a fixed nonlinear rule of those
five; each label is then flipped independently with probability eta. Because
the rule is deterministic and the flips are independent of X, the Bayes error
is exactly eta (eta = 0.5 is pure noise).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
RES = HERE / "results"
FIG = HERE / "figures"

ADULT_COLS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
              "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
              "hours_per_week", "native_country", "income"]
ADULT_CAT = ["workclass", "education", "marital_status", "occupation", "relationship",
             "race", "sex", "native_country"]
ADULT_NUM = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]


def clean_rule(x: np.ndarray) -> np.ndarray:
    """Deterministic nonlinear rule on the first five coordinates."""
    s = x[:, 0] * x[:, 1] + np.sin(2 * x[:, 2]) + 0.5 * x[:, 3] ** 2 - 0.5 - 0.7 * x[:, 4]
    return (s > 0).astype(int)


def simulate(n: int, eta: float, rng: np.random.Generator, d: int = 10):
    x = rng.standard_normal((n, d))
    y_clean = clean_rule(x)
    flip = rng.random(n) < eta
    y = np.where(flip, 1 - y_clean, y_clean)
    return x, y


def load_adult(drop: tuple[str, ...] = ()) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(RAW / "adult-all.csv", header=None, names=ADULT_COLS, skipinitialspace=True)
    y = (df.pop("income").str.strip().str.startswith(">50K")).astype(int).to_numpy()
    df = df.drop(columns=list(drop))
    return df, y


def encode_adult(df: pd.DataFrame) -> np.ndarray:
    cats = [c for c in ADULT_CAT if c in df.columns]
    nums = [c for c in ADULT_NUM if c in df.columns]
    parts = [df[nums].astype(float).to_numpy()]
    if cats:
        parts.append(pd.get_dummies(df[cats].astype(str), dtype=float).to_numpy())
    return np.column_stack(parts)


def standardize(x_train: np.ndarray, *others: np.ndarray):
    mu, sd = x_train.mean(0), x_train.std(0)
    sd[sd == 0] = 1.0
    out = [(x_train - mu) / sd] + [(o - mu) / sd for o in others]
    return out if others else out[0]


def load_housing() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(RAW / "housing.csv")
    df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
    df = pd.get_dummies(df, columns=["ocean_proximity"], dtype=float)
    y = np.log(df.pop("median_house_value").to_numpy())
    return df.to_numpy(dtype=float), y


def memorizers(seed: int = 0) -> dict:
    """Four models set up to memorise: no depth limits, no penalties, long training."""
    return {
        "Decision tree (grown to purity)": DecisionTreeClassifier(random_state=seed),
        "Random forest (unlimited depth)": RandomForestClassifier(
            n_estimators=300, max_features="sqrt", n_jobs=2, random_state=seed),
        "XGBoost (deep, no penalties)": XGBClassifier(
            n_estimators=1500, max_depth=12, learning_rate=0.3, reg_lambda=0.0, reg_alpha=0.0,
            min_child_weight=0.0, gamma=0.0, subsample=1.0, tree_method="hist", max_bin=1024,
            n_jobs=2, random_state=seed, verbosity=0),
        # trained epoch by epoch until it fits every label (exp1_memorization.mlp_memorise)
        "Neural network (2x256, no weight decay)": MLPClassifier(
            hidden_layer_sizes=(256, 256), alpha=0.0, learning_rate_init=1e-3, batch_size=200,
            random_state=seed),
    }
