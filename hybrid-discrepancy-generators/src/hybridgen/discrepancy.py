"""Learned discrepancy: what reality adds to Erlang A's expected outcomes.

Following Kennedy and O'Hagan (2001), observations are written as

    y = eta(x) + delta(x) + noise,

where eta is the simulator's expected output at the calibrated parameters
and delta is a function learned from the residuals. Three learners are used
for delta because what a hybrid does outside the data is decided by how its
discrepancy model extrapolates:

* gradient boosting: piecewise constant, so it carries the correction found
  at the edge of the data unchanged beyond it;
* a Gaussian process with zero prior mean: the correction fades back to zero,
  and the hybrid back to Erlang A, away from the data;
* a ridge regression: the correction continues the trend it found.

A discrepancy-free data-driven regression of the outcomes on the same inputs
is included for comparison.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .eta import TARGETS

INTERVAL_SECONDS = 1800.0
GP_MAX_ROWS = 400


def features(frame: pd.DataFrame, service_mean: float) -> pd.DataFrame:
    """Inputs available to every learner for a row of the table."""
    load = frame["offered"] * service_mean / (INTERVAL_SECONDS
                                              * frame["agents"])
    return pd.DataFrame({
        "interval": frame["interval"].to_numpy(float),
        "dow": frame["dow"].to_numpy(float),
        "campaign": frame["campaign"].to_numpy(float),
        "offered": frame["offered"].to_numpy(float),
        "agents": frame["agents"].to_numpy(float),
        "load": load.to_numpy(float),
    }, index=frame.index)


def _weights(frame: pd.DataFrame) -> np.ndarray:
    return frame["offered"].to_numpy(float)


class _Learner:
    """Regression of one target on a feature matrix."""

    name = "base"

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray,
            rng: np.random.Generator) -> "_Learner":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Boosting(_Learner):
    name = "boosting"

    def fit(self, X, y, w, rng):
        self.model = HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=30, l2_regularization=1.0,
            random_state=int(rng.integers(2 ** 31)))
        self.model.fit(X, y, sample_weight=w)
        return self

    def predict(self, X):
        return self.model.predict(X)


class GaussianProcess(_Learner):
    """Zero-mean GP with an ARD squared-exponential kernel and white noise.

    Fitted on at most ``GP_MAX_ROWS`` rows drawn with probability
    proportional to offered calls, since exact GP regression is cubic in
    the number of rows. Kernel hyperparameters are set by maximising the
    marginal likelihood from one starting point.
    """

    name = "Gaussian process"

    def fit(self, X, y, w, rng):
        n = X.shape[0]
        if n > GP_MAX_ROWS:
            idx = rng.choice(n, GP_MAX_ROWS, replace=False, p=w / w.sum())
            X, y = X[idx], y[idx]
        self.scaler = StandardScaler().fit(X)
        kernel = (ConstantKernel(np.var(y) + 1e-6, (1e-8, 10.0))
                  * RBF(np.ones(X.shape[1]), (1e-2, 1e3))
                  + WhiteKernel(np.var(y) / 2 + 1e-8, (1e-10, 10.0)))
        self.model = GaussianProcessRegressor(
            kernel, normalize_y=False, n_restarts_optimizer=0,
            random_state=int(rng.integers(2 ** 31)))
        with warnings.catch_warnings():
            # Length scales hitting their bounds are expected for inputs
            # the residuals do not depend on.
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model.fit(self.scaler.transform(X), y)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))


class Linear(_Learner):
    name = "linear"

    def fit(self, X, y, w, rng):
        self.scaler = StandardScaler().fit(X)
        self.model = Ridge(alpha=1.0).fit(self.scaler.transform(X), y,
                                          sample_weight=w)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))


LEARNERS = {"boosting": Boosting, "Gaussian process": GaussianProcess,
            "linear": Linear}

# Features each learner sees for the discrepancy of target t: the row's
# inputs, plus Erlang A's expectation for t. The GP and ridge get a compact
# set; boosting gets the calendar too.
GP_FEATURES = ["load", "offered", "agents"]
LINEAR_FEATURES = ["load", "offered", "agents"]
BOOST_FEATURES = ["interval", "dow", "campaign", "offered", "agents", "load"]


def _columns(learner: str, target: str) -> list[str]:
    base = {"boosting": BOOST_FEATURES, "Gaussian process": GP_FEATURES,
            "linear": LINEAR_FEATURES}[learner]
    return base + [f"eta_{target}"]


EPS = 0.005


def link(t: str, v: np.ndarray) -> np.ndarray:
    """Logit for the two shares, log for the mean wait (in seconds + 1)."""
    v = np.asarray(v, float)
    if t == "awt":
        return np.log1p(np.maximum(v, 0.0))
    c = np.clip(v, EPS, 1 - EPS)
    return np.log(c / (1 - c))


def unlink(t: str, z: np.ndarray) -> np.ndarray:
    if t == "awt":
        return np.expm1(z)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class Discrepancy:
    """delta(x) for the three outcomes, learned with one kind of learner.

    ``scale`` is ``"additive"`` (delta is added to the outputs, as in
    Kennedy and O'Hagan) or ``"link"`` (delta is added on the logit scale
    for the shares and the log scale for the wait, so corrected values stay
    inside their ranges).
    """

    learner: str
    service_mean: float
    scale: str = "additive"
    models: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.learner + (" (link scale)" if self.scale == "link"
                               else "")

    def fit(self, table: pd.DataFrame, eta: pd.DataFrame,
            rng: np.random.Generator) -> "Discrepancy":
        data = features(table, self.service_mean).join(eta)
        ok = table[list(TARGETS)].notna().all(axis=1).to_numpy()
        w = _weights(table)[ok]
        for t in TARGETS:
            X = data[_columns(self.learner, t)].to_numpy(float)[ok]
            y = table[t].to_numpy(float)
            e = eta[f"eta_{t}"].to_numpy(float)
            resid = (y - e) if self.scale == "additive" else \
                (link(t, y) - link(t, e))
            self.models[t] = LEARNERS[self.learner]().fit(X, resid[ok], w,
                                                          rng)
        return self

    def predict(self, frame: pd.DataFrame, eta: pd.DataFrame) -> pd.DataFrame:
        data = features(frame, self.service_mean).join(eta)
        return pd.DataFrame(
            {t: self.models[t].predict(
                data[_columns(self.learner, t)].to_numpy(float))
             for t in TARGETS}, index=frame.index)


@dataclass
class DirectRegression:
    """Outcomes regressed on the inputs with boosting, no mechanism."""

    service_mean: float
    models: dict = field(default_factory=dict)

    def fit(self, table: pd.DataFrame,
            rng: np.random.Generator) -> "DirectRegression":
        X = features(table, self.service_mean)[BOOST_FEATURES].to_numpy(float)
        ok = table[list(TARGETS)].notna().all(axis=1).to_numpy()
        w = _weights(table)[ok]
        for t in TARGETS:
            self.models[t] = Boosting().fit(X[ok], table[t].to_numpy()[ok], w,
                                            rng)
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = features(frame, self.service_mean)[BOOST_FEATURES].to_numpy(float)
        return pd.DataFrame({t: self.models[t].predict(X) for t in TARGETS},
                            index=frame.index)


def apply(eta: pd.DataFrame, delta: pd.DataFrame, scale: str = "additive",
          keep_boundaries: bool = False) -> tuple[pd.DataFrame,
                                                  dict[str, float]]:
    """eta + delta, made feasible, and how often the raw result was not.

    Impossible means a share outside [0, 1], a negative wait, or a service
    level above one minus the abandonment share (calls answered within the
    threshold cannot outnumber calls answered). On the link scale the first
    two cannot happen. With ``keep_boundaries`` (used when correcting a
    simulated draw rather than an expectation), values exactly at a
    boundary -- everyone answered, nobody lost, nobody waited -- are left
    there, because a correction to an expectation says nothing about them.
    """
    cols = {}
    for t in TARGETS:
        base = eta[f"eta_{t}"].to_numpy(float)
        d = delta[t].to_numpy(float)
        if scale == "additive":
            v = base + d
        else:
            v = unlink(t, link(t, base) + d)
        if keep_boundaries:
            edge = (base == 0) | ((base == 1) & (t != "awt"))
            v = np.where(edge, base, v)
        cols[t] = v
    raw = pd.DataFrame(cols, index=eta.index)
    bad = ((raw["sl"] < 0) | (raw["sl"] > 1) | (raw["abandon_rate"] < 0)
           | (raw["abandon_rate"] > 1) | (raw["awt"] < 0)
           | (raw["sl"] > 1 - raw["abandon_rate"] + 1e-12))
    out = raw.copy()
    out["sl"] = out["sl"].clip(0, 1)
    out["abandon_rate"] = out["abandon_rate"].clip(0, 1)
    out["awt"] = out["awt"].clip(lower=0)
    out["sl"] = np.minimum(out["sl"], 1 - out["abandon_rate"])
    return out, {"share_impossible": float(bad.mean())}
