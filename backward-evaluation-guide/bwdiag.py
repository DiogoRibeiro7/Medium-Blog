"""bwdiag: read the train-period x test-period matrix of a forecasting model.

Cut a series into consecutive blocks, fit the model on each block and score it, one
step ahead, on every other block, earlier blocks included. Cell (i, j) of the matrix is
the mean log loss of block i's model on block j, minus that of block j's own model
(estimated out of fold inside block j). The excess is then split into three parts:

    level      the loss that a constant shift of the forecasts, fitted on block j, removes
    spread     the loss that a constant rescaling of the predictive distribution removes
    structure  what is left: predictable variation of block j that block i's model misses

``level + spread + structure`` equals the excess. The repairs are applied in that order:
first the best constant shift, then the best constant rescaling of the shifted forecasts,
so that a pure bias is read as level and not partly as a too-narrow distribution. Both
are fitted on the scored block itself, so they are an upper bound on what a
recalibration could do in real time; with about a hundred rows per block they remove a
few hundredths of a nat even when nothing has changed, and ``structure`` is then
slightly negative, because block j's own model is estimated out of fold.

The module computes and summarises; it does not choose windows or models. The articles
it accompanies found that rules which act on the matrix automatically lose to pooling
and to the ordinary forward backtest. What the matrix is good for is reading: where a
series changed, what kind of change it was, and which models carry what they learn from
one period into another.

Conventions
-----------
* A *row* is one forecast target ``z[r]`` with features ``X[r]``, rows in time order.
* A *model* has ``fit(X, z)`` returning itself and ``predict(X)`` returning
  ``(mean, scale)``; its predictive distribution is Gaussian, or Student-t if the model
  has a finite ``df`` attribute. Models that are not row-based (state-space filters, for
  example) can be used through :func:`matrix_from_forecasts`.
* Losses are log losses (negative log predictive density) in nats; lower is better.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

__all__ = [
    "nlpd", "lag_rows", "blocks_from_end", "LinearModel", "RegressorModel", "Repair", "repair",
    "Matrix", "matrix_from_forecasts", "transfer_matrix", "kfold_losses", "monitor",
]

LOG_2PI = float(np.log(2.0 * np.pi))
MIN_ROWS = 10


# ------------------------------------------------------------------------- losses


def nlpd(z, mean, scale, df: float = np.inf) -> np.ndarray:
    """Per-row negative log predictive density of a Gaussian or Student-t forecast."""
    w = (np.asarray(z, float) - mean) / scale
    if np.isinf(df):
        return 0.5 * LOG_2PI + np.log(scale) + 0.5 * w**2
    c = gammaln((df + 1.0) / 2.0) - gammaln(df / 2.0) - 0.5 * np.log(df * np.pi)
    return np.log(scale) - c + 0.5 * (df + 1.0) * np.log1p(w**2 / df)


def _ml_scale(r: np.ndarray, df: float, iters: int = 200) -> float:
    """Maximum-likelihood scale of residuals ``r`` with location fixed at 0."""
    r = np.asarray(r, float)
    if np.isinf(df):
        return float(np.sqrt(np.mean(r**2))) or 1e-8
    s = float(np.median(np.abs(r)) / 0.6745) or 1e-8
    for _ in range(iters):
        w = (df + 1.0) / (df + (r / s) ** 2)
        s_new = float(np.sqrt(np.sum(w * r**2) / len(r))) or 1e-8
        if abs(s_new - s) < 1e-10 * s:
            return s_new
        s = s_new
    return s


# ------------------------------------------------------------------------- rows


def lag_rows(y: np.ndarray, lags: int | Iterable[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Autoregressive rows: ``X[r] = (y[t-l] for l in lags)``, ``z[r] = y[t]``.

    Returns ``X``, ``z`` and ``t``, the position in ``y`` of each row's target.
    """
    y = np.asarray(y, float)
    lags = list(range(1, lags + 1)) if isinstance(lags, int) else list(lags)
    p = max(lags)
    t = np.arange(p, len(y))
    X = np.column_stack([y[t - lag] for lag in lags])
    return X, y[t], t


def blocks_from_end(n_rows: int, block_len: int) -> list[tuple[int, int]]:
    """Consecutive ``[a, b)`` blocks of ``block_len`` rows aligned to the end of the sample.

    The most recent block is always complete; leftover rows at the start are not used.
    """
    k = n_rows // block_len
    if k < 2:
        raise ValueError("need at least two blocks")
    off = n_rows - k * block_len
    return [(off + i * block_len, off + (i + 1) * block_len) for i in range(k)]


# ------------------------------------------------------------------------- models


class LinearModel:
    """Linear regression with Gaussian (``df=inf``) or Student-t errors.

    Student-t errors are fitted by maximum likelihood with iteratively reweighted least
    squares (Lange, Little and Taylor, 1989), so extreme rows move neither the
    coefficients nor the scale much. ``intercept=False`` if ``X`` already has a constant.
    """

    def __init__(self, df: float = np.inf, intercept: bool = True, iters: int = 100) -> None:
        self.df, self.intercept, self.iters = float(df), intercept, iters

    def _A(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        return np.hstack([np.ones((len(X), 1)), X]) if self.intercept else X

    def fit(self, X: np.ndarray, z: np.ndarray) -> "LinearModel":
        A, z = self._A(X), np.asarray(z, float)
        beta, *_ = np.linalg.lstsq(A, z, rcond=None)
        r = z - A @ beta
        if np.isinf(self.df):
            self.beta, self.scale = beta, _ml_scale(r, self.df)
            return self
        s = _ml_scale(r, self.df)
        for _ in range(self.iters):
            w = (self.df + 1.0) / (self.df + (r / s) ** 2)
            sw = np.sqrt(w)
            new, *_ = np.linalg.lstsq(A * sw[:, None], z * sw, rcond=None)
            r = z - A @ new
            s_new = float(np.sqrt(np.sum(w * r**2) / len(z))) or 1e-8
            done = np.max(np.abs(new - beta)) < 1e-8 and abs(s_new - s) < 1e-8 * s
            beta, s = new, s_new
            if done:
                break
        self.beta, self.scale = beta, s
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = self._A(X) @ self.beta
        return mean, np.full(len(mean), self.scale)


class RegressorModel:
    """Any point regressor with ``fit``/``predict`` (scikit-learn style), turned into a
    predictive distribution whose scale comes from out-of-fold residuals.

    In-sample residuals of flexible regressors are far too small, so the training rows
    are split into ``folds`` contiguous folds, each predicted by a model fitted on the
    others; the scale is estimated from those residuals (maximum likelihood for the
    Student-t). The final point model is fitted on all training rows.
    """

    def __init__(self, make: Callable[[], object], df: float = np.inf, folds: int = 3) -> None:
        self.make, self.df, self.folds = make, float(df), folds

    def fit(self, X: np.ndarray, z: np.ndarray) -> "RegressorModel":
        parts = np.array_split(np.arange(len(z)), self.folds)
        res = []
        for k in range(self.folds):
            tr = np.concatenate([parts[i] for i in range(self.folds) if i != k])
            m = self.make().fit(X[tr], z[tr])
            res.append(z[parts[k]] - m.predict(X[parts[k]]))
        self.scale = _ml_scale(np.concatenate(res), self.df)
        self.model = self.make().fit(X, z)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.asarray(self.model.predict(X), float)
        return mean, np.full(len(mean), self.scale)


# ------------------------------------------------------------------------- repairs


@dataclass(frozen=True)
class Repair:
    """Mean log loss of a set of forecasts, and how much two recalibrations remove.

    ``raw - repaired == level + spread``: ``level`` is what the best constant shift of
    the location removes, ``spread`` what the best constant rescaling then removes.
    ``shift`` and ``factor`` are the fitted shift and scale multiplier.
    """

    raw: float
    level: float
    spread: float
    repaired: float
    shift: float
    factor: float
    n: int


def _fit(z, mean, scale, df, free_shift: bool, free_scale: bool) -> tuple[float, float, float]:
    """Minimise mean NLPD over a constant shift ``c`` and/or log scale factor ``g``."""
    e = z - mean
    if np.isinf(df):  # closed form
        w = 1.0 / scale**2
        c = float(np.sum(w * e) / np.sum(w)) if free_shift else 0.0
        g = 0.5 * float(np.log(np.mean(((e - c) / scale) ** 2))) if free_scale else 0.0
        return float(np.mean(nlpd(z, mean + c, scale * np.exp(g), df))), c, g

    def f(v):
        c = v[0] if free_shift else 0.0
        g = v[-1] if free_scale else 0.0
        k = np.exp(g)
        u = (e - c) / (scale * k)
        q = df + u**2
        val = float(np.mean(np.log(scale * k) + 0.5 * (df + 1.0) * np.log1p(u**2 / df)))
        grad = []
        if free_shift:
            grad.append(float(np.mean(-(df + 1.0) * u / (q * scale * k))))
        if free_scale:
            grad.append(float(np.mean(1.0 - (df + 1.0) * u**2 / q)))
        return val, np.array(grad)

    x0 = []
    if free_shift:
        x0.append(float(np.median(e)))
    if free_scale:
        x0.append(float(np.log(np.median(np.abs(e - (x0[0] if free_shift else 0.0)) / scale) / 0.7267 + 1e-12)))
    r = minimize(f, np.array(x0), jac=True, method="L-BFGS-B")
    c = float(r.x[0]) if free_shift else 0.0
    g = float(r.x[-1]) if free_scale else 0.0
    return float(np.mean(nlpd(z, mean + c, scale * np.exp(g), df))), c, g


def repair(z, mean, scale, df: float = np.inf) -> Repair:
    """How much of the log loss of forecasts ``(mean, scale)`` a constant shift and a
    constant rescaling, each fitted on these very outcomes, would remove."""
    z, mean = np.asarray(z, float), np.asarray(mean, float)
    scale = np.broadcast_to(np.asarray(scale, float), z.shape)
    ok = np.isfinite(z) & np.isfinite(mean) & np.isfinite(scale)
    z, mean, scale = z[ok], mean[ok], scale[ok]
    raw = float(np.mean(nlpd(z, mean, scale, df)))
    l_shift, c1, _ = _fit(z, mean, scale, df, True, False)
    l_both, c, g = _fit(z, mean, scale, df, True, True)
    # the optimiser is local for the Student-t: keep every fit at least as good as the
    # simpler one it nests, so that neither part is negative by accident
    if l_shift > raw:
        l_shift, c1 = raw, 0.0
    if l_both > l_shift:
        l_both, c, g = l_shift, c1, 0.0
    return Repair(raw, raw - l_shift, l_shift - l_both, l_both, c, float(np.exp(g)), int(len(z)))


# ------------------------------------------------------------------------- the matrix


@dataclass
class Matrix:
    """Block i's model (rows) scored on block j (columns).

    ``loss[i, j]``      mean log loss of block i's model on the scored rows of block j
    ``reference[i, j]`` mean log loss of block j's own model on the same rows
    ``level``, ``spread``  the two repairs (see :func:`repair`); ``repaired = loss - level - spread``
    The diagonal holds block j's own model (``loss == reference``, all parts 0).
    """

    blocks: list[tuple[int, int]]
    labels: list[str]
    loss: np.ndarray
    reference: np.ndarray
    level: np.ndarray
    spread: np.ndarray
    rows: np.ndarray
    df: float = np.inf
    meta: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.blocks)

    @property
    def excess(self) -> np.ndarray:
        return self.loss - self.reference

    @property
    def repaired(self) -> np.ndarray:
        return self.loss - self.level - self.spread

    @property
    def structure(self) -> np.ndarray:
        return self.repaired - self.reference

    def part(self, name: str) -> np.ndarray:
        return {"excess": self.excess, "level": self.level, "spread": self.spread,
                "structure": self.structure}[name]

    def row(self, i: int = -1) -> list[dict]:
        """Block i's model on every other block: excess and its three parts."""
        i = i % self.k
        return [dict(block=j, label=self.labels[j], excess=self.excess[i, j], level=self.level[i, j],
                     spread=self.spread[i, j], structure=self.structure[i, j], rows=int(self.rows[i, j]))
                for j in range(self.k) if j != i and np.isfinite(self.loss[i, j])]

    def column(self, j: int) -> list[dict]:
        """Every other block's model on block j."""
        j = j % self.k
        return [dict(block=i, label=self.labels[i], excess=self.excess[i, j], level=self.level[i, j],
                     spread=self.spread[i, j], structure=self.structure[i, j], rows=int(self.rows[i, j]))
                for i in range(self.k) if i != j and np.isfinite(self.loss[i, j])]

    def _sym(self) -> np.ndarray:
        E = np.nan_to_num(self.excess, nan=0.0)
        return 0.5 * (E + E.T)

    def regimes(self, n: int) -> tuple[list[tuple[int, int]], float]:
        """Best split of the blocks into ``n`` contiguous regimes: the split that
        minimises the summed symmetric excess between blocks of the same regime.
        Returns the regimes as inclusive ``(first, last)`` block indices and the cost."""
        S, k = self._sym(), self.k
        if not 1 <= n <= k:
            raise ValueError("n must be between 1 and the number of blocks")
        cost = np.zeros((k, k))
        for a in range(k):
            for b in range(a + 1, k):
                cost[a, b] = cost[a, b - 1] + S[a:b, b].sum()
        best = np.full((n + 1, k + 1), np.inf)
        arg = np.zeros((n + 1, k + 1), dtype=int)
        best[0, 0] = 0.0
        for m in range(1, n + 1):
            for e in range(1, k + 1):
                for s in range(m - 1, e):
                    v = best[m - 1, s] + cost[s, e - 1]
                    if v < best[m, e]:
                        best[m, e], arg[m, e] = v, s
        out, e = [], k
        for m in range(n, 0, -1):
            s = arg[m, e]
            out.append((int(s), int(e - 1)))
            e = s
        return out[::-1], float(best[n, k])

    def regime_costs(self, max_n: int | None = None) -> np.ndarray:
        """Cost of the best split into 1, 2, ... regimes; look for where it stops falling."""
        return np.array([self.regimes(n)[1] for n in range(1, (max_n or self.k) + 1)])

    def describe(self, i: int = -1, digits: int = 3) -> str:
        """Plain-text reading of row ``i`` (by default today's model on the past)."""
        i = i % self.k
        lines = [f"model of block {self.labels[i]} on the other blocks "
                 f"(excess log loss = level + spread + structure, nats per row):"]
        for r in self.row(i):
            lines.append(f"  {r['label']:>14}: {r['excess']:+.{digits}f} = {r['level']:.{digits}f} + "
                         f"{r['spread']:.{digits}f} + {r['structure']:+.{digits}f}   ({r['rows']} rows)")
        return "\n".join(lines)


def matrix_from_forecasts(
    z: np.ndarray,
    blocks: Sequence[tuple[int, int]],
    forecast: Callable[[int], tuple[np.ndarray, np.ndarray]],
    own_losses: Callable[[int], np.ndarray],
    df: float = np.inf,
    gap: int = 0,
    labels: Sequence[str] | None = None,
) -> Matrix:
    """Build the matrix from any forecasting procedure.

    ``forecast(i)`` returns ``(mean, scale)`` over all rows from the model fitted on block
    i (NaN where it cannot forecast). ``own_losses(j)`` returns the per-row log losses on
    block j of block j's own model, estimated out of sample (for example out of fold).
    Rows within ``gap`` rows of the training block are not scored.
    """
    z = np.asarray(z, float)
    k, n = len(blocks), len(z)
    shape = (k, k)
    loss, ref, lev, spr = (np.full(shape, np.nan) for _ in range(4))
    rows = np.zeros(shape, dtype=int)
    own = {}
    for j, (a, b) in enumerate(blocks):
        own[j] = np.asarray(own_losses(j), float)
        loss[j, j] = ref[j, j] = float(np.nanmean(own[j]))
        lev[j, j] = spr[j, j] = 0.0
        rows[j, j] = b - a
    idx = np.arange(n)
    for i, (a, b) in enumerate(blocks):
        mean, scale = forecast(i)
        mean = np.asarray(mean, float)
        scale = np.broadcast_to(np.asarray(scale, float), (n,))
        for j, (c, d) in enumerate(blocks):
            if j == i:
                continue
            r = idx[c:d]
            r = r[(r < a - gap) | (r >= b + gap)]
            r = r[np.isfinite(mean[r]) & np.isfinite(own[j][r - c])]
            if len(r) < MIN_ROWS:
                continue
            rep = repair(z[r], mean[r], scale[r], df)
            loss[i, j], lev[i, j], spr[i, j] = rep.raw, rep.level, rep.spread
            ref[i, j] = float(np.mean(own[j][r - c]))
            rows[i, j] = len(r)
    return Matrix(list(blocks), list(labels) if labels is not None else [str(i) for i in range(k)],
                  loss, ref, lev, spr, rows, df)


def kfold_losses(X, z, factory, folds: int = 5, rng: np.random.Generator | None = None) -> np.ndarray:
    """Out-of-fold log losses of ``factory()`` inside one block (random folds).

    For autoregressive rows with uncorrelated errors, random K-fold estimates the
    out-of-sample loss without bias (Bergmeir, Hyndman and Koo, 2018); for other
    models it is an approximation.
    """
    rng = rng or np.random.default_rng(0)
    fold = rng.permutation(len(z)) % folds
    out = np.empty(len(z))
    for f in range(folds):
        tr, te = fold != f, fold == f
        m = factory().fit(X[tr], z[tr])
        mean, scale = m.predict(X[te])
        out[te] = nlpd(z[te], mean, scale, getattr(m, "df", np.inf))
    return out


def transfer_matrix(
    X: np.ndarray,
    z: np.ndarray,
    factory: Callable[[], object],
    block_len: int,
    gap: int = 0,
    folds: int = 5,
    seed: int = 0,
    labels: Sequence[str] | None = None,
) -> Matrix:
    """The matrix for a row-based model: fit ``factory()`` on each block, score it on
    every other block, and benchmark each block against its own out-of-fold model.

    ``gap`` should be at least the longest lag, so that no scored row's features overlap
    the training targets.
    """
    X, z = np.asarray(X, float), np.asarray(z, float)
    blocks = blocks_from_end(len(z), block_len)
    rng = np.random.default_rng(seed)
    df = getattr(factory(), "df", np.inf)

    def forecast(i):
        a, b = blocks[i]
        return factory().fit(X[a:b], z[a:b]).predict(X)

    def own(j):
        a, b = blocks[j]
        return kfold_losses(X[a:b], z[a:b], factory, folds, rng)

    return matrix_from_forecasts(z, blocks, forecast, own, df, gap, labels)


# ------------------------------------------------------------------------- monitoring


def monitor(
    X: np.ndarray,
    z: np.ndarray,
    factory: Callable[[], object],
    window: int,
    step: int,
    gap: int = 0,
    folds: int = 5,
    seed: int = 0,
    start: int | None = None,
) -> list[dict]:
    """Today's model against the window before it, repeated through time.

    At each origin ``o`` (every ``step`` rows from ``start``, by default the first origin
    with two full windows and the gap behind it), a model fitted on ``[o - window, o)``
    is scored on the previous window ``[o - 2*window - gap, o - window - gap)`` and
    compared with that window's own out-of-fold model. Uses only data before ``o``, so
    the readings are available in real time.
    """
    X, z = np.asarray(X, float), np.asarray(z, float)
    rng = np.random.default_rng(seed)
    first = 2 * window + gap if start is None else start
    out = []
    for o in range(first, len(z) + 1, step):
        a, b = o - 2 * window - gap, o - window - gap
        m = factory().fit(X[o - window:o], z[o - window:o])
        mean, scale = m.predict(X[a:b])
        rep = repair(z[a:b], mean, scale, getattr(m, "df", np.inf))
        ref = float(np.mean(kfold_losses(X[a:b], z[a:b], factory, folds, rng)))
        out.append(dict(origin=o, excess=rep.raw - ref, level=rep.level, spread=rep.spread,
                        structure=rep.repaired - ref, shift=rep.shift, factor=rep.factor))
    return out
