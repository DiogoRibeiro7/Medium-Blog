"""Rare-event classification tools written from scratch (NumPy/SciPy).

  logistic            logistic regression by Newton-Raphson, with optional case
                      weights, an optional ridge penalty (intercept not
                      penalised) and an optional offset
  oversample          random oversampling of the minority class (duplicates)
  undersample         random undersampling of the majority class
  smote               SMOTE (Chawla et al. 2002): synthetic minority points on
                      the segment between a minority point and one of its k
                      nearest minority neighbours; same random draws as
                      imbalanced-learn's SMOTE given a seed
  balanced_weights    the class weights n / (2 n_c) ("balanced")
  prior_correct       moves predicted probabilities from the class balance a
                      model was trained on back to the population prevalence
                      (King and Zeng 2001; Elkan 2001)
  auc, brier, logloss, calibration (intercept and slope; Cox 1958),
  expected_cost, cost_threshold, net_benefit (Vickers and Elkin 2006)
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, log_expit
from scipy.stats import rankdata


# ------------------------------------------------------------------ logistic regression
class Logit:
    def __init__(self, coef, cov, loglik, converged):
        self.coef, self.cov, self.loglik, self.converged = coef, cov, loglik, converged

    def se(self):
        return np.sqrt(np.diag(self.cov))

    def linear(self, X, offset=None):
        e = np.asarray(X, float) @ self.coef
        return e if offset is None else e + offset

    def predict(self, X, offset=None):
        return expit(self.linear(X, offset))


def logistic(X, y, w=None, ridge=0.0, offset=None, maxit=100, tol=1e-10):
    """Weighted, optionally ridge-penalised logistic regression (Newton-Raphson).
    The first column of X is taken as the intercept and is not penalised."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, k = X.shape
    w = np.ones(n) if w is None else np.asarray(w, float)
    off = np.zeros(n) if offset is None else np.asarray(offset, float)
    P = np.eye(k) * ridge
    P[0, 0] = 0.0
    b = np.zeros(k)
    p0 = np.clip(np.sum(w * y) / np.sum(w), 1e-12, 1 - 1e-12)
    b[0] = np.log(p0 / (1 - p0)) - np.mean(off)
    converged = False

    def obj(b):
        e = X @ b + off
        return np.sum(w * (y * log_expit(e) + (1 - y) * log_expit(-e))) - 0.5 * b @ P @ b

    f = obj(b)
    for _ in range(maxit):
        e = X @ b + off
        p = expit(e)
        g = X.T @ (w * (y - p)) - P @ b
        H = (X.T * (w * p * (1 - p))) @ X + P
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        t = 1.0
        while t > 1e-8:
            fn = obj(b + t * step)
            if fn >= f - 1e-12:
                break
            t /= 2
        b = b + t * step
        f_old, f = f, obj(b)
        if np.max(np.abs(t * step)) < tol or abs(f - f_old) < 1e-13 * (1 + abs(f)):
            converged = True
            break
    p = expit(X @ b + off)
    H = (X.T * (w * p * (1 - p))) @ X + P
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.full((k, k), np.nan)
    return Logit(b, cov, float(obj(b) + 0.5 * b @ P @ b), converged)


# ------------------------------------------------------------------ rebalancing
def oversample(X, y, rng):
    """Duplicate randomly chosen minority cases until both classes are equal."""
    i1, i0 = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    extra = rng.choice(i1, size=len(i0) - len(i1), replace=True)
    idx = np.concatenate([i0, i1, extra])
    return X[idx], y[idx]


def undersample(X, y, rng):
    """Keep all minority cases and an equal number of randomly chosen majority cases."""
    i1, i0 = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    keep = rng.choice(i0, size=len(i1), replace=False)
    idx = np.concatenate([keep, i1])
    return X[idx], y[idx]


def _knn_within(Xc, k):
    """Indices of the k nearest neighbours of each row of Xc among the other rows
    (Euclidean distance; ties broken by index, as in scikit-learn's brute force)."""
    sq = np.sum(Xc ** 2, axis=1)
    D = sq[:, None] + sq[None, :] - 2 * Xc @ Xc.T
    np.fill_diagonal(D, -np.inf)
    order = np.argsort(D, axis=1, kind="stable")
    return order[:, 1:k + 1]


def smote(X, y, rng_state, k=5, cols=None):
    """SMOTE until both classes are equal. rng_state is a numpy RandomState (the
    generator imbalanced-learn uses); cols restricts the distance to some columns
    (default: all), the interpolation is always on all columns."""
    X = np.asarray(X, float)
    i1, i0 = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    Xc = X[i1]
    nn = _knn_within(Xc if cols is None else Xc[:, cols], k)
    m = len(i0) - len(i1)
    s = rng_state.randint(low=0, high=nn.size, size=m)
    steps = rng_state.uniform(size=m)[:, None]
    rows, cc = np.floor_divide(s, nn.shape[1]), np.mod(s, nn.shape[1])
    Xn = Xc[rows] + steps * (Xc[nn[rows, cc]] - Xc[rows])
    return np.vstack([X, Xn]), np.concatenate([y, np.ones(m, dtype=y.dtype)])


def balanced_weights(y):
    n, n1 = len(y), int(np.sum(y))
    return np.where(y == 1, n / (2 * n1), n / (2 * (n - n1)))


def prior_correct(p, train_rate, true_rate):
    """Probabilities from a model trained where the event rate was train_rate,
    moved to a population where it is true_rate: the odds are multiplied by
    [true_rate / (1 - true_rate)] / [train_rate / (1 - train_rate)]."""
    r = (true_rate / (1 - true_rate)) / (train_rate / (1 - train_rate))
    o = p / (1 - p) * r
    return o / (1 + o)


# ------------------------------------------------------------------ metrics
def auc(y, s):
    """Area under the ROC curve (Mann-Whitney, ties counted as one half)."""
    y = np.asarray(y)
    r = rankdata(s)
    n1 = int(np.sum(y == 1))
    n0 = len(y) - n1
    return float((np.sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier(y, p):
    return float(np.mean((np.asarray(p) - y) ** 2))


def logloss(y, p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration(y, p, eps=1e-12):
    """Calibration intercept (calibration-in-the-large: the intercept of a
    logistic regression of y with logit(p) as offset; 0 is ideal) and slope (the
    coefficient on logit(p) in a logistic regression of y; 1 is ideal)."""
    lp = np.log(np.clip(p, eps, 1 - eps)) - np.log(np.clip(1 - p, eps, 1 - eps))
    one = np.ones((len(y), 1))
    a = logistic(one, y, offset=lp).coef[0]
    s = logistic(np.column_stack([one, lp]), y).coef[1]
    return float(a), float(s)


def confusion(y, p, t):
    pred = p >= t
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return tp, fp, fn, tn


def cost_threshold(c_fp, c_fn):
    """Threshold on a calibrated probability that minimises expected cost:
    flag when p * c_fn > (1 - p) * c_fp."""
    return c_fp / (c_fp + c_fn)


def expected_cost(y, p, t, c_fp, c_fn):
    """Average cost per case of flagging p >= t."""
    tp, fp, fn, tn = confusion(y, p, t)
    return (c_fp * fp + c_fn * fn) / len(y)


def net_benefit(y, p, t):
    """Net benefit of flagging p >= t, in true positives per case (Vickers and Elkin 2006)."""
    tp, fp, fn, tn = confusion(y, p, t)
    return (tp - fp * t / (1 - t)) / len(y)
