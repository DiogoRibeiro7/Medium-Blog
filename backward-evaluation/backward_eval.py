"""Train-period by test-period evaluation of one-step forecasts.

The usual backtest fits a model on one stretch of a series and scores it on a later
stretch. This module scores every fitted model on every other stretch, earlier ones
included, and returns the full matrix. It also implements the rule studied in the
article: fit a model on the most recent block, walk backwards block by block, and stop
at the first block where that model is significantly worse than a model fitted inside
the block itself.

Conventions
-----------
* A *row* is one forecast target ``z[r]`` with its feature vector ``X[r]``.
  ``embed_ar`` builds rows for an autoregression; any other feature builder works as
  long as the rows are in time order.
* A *model* is any object with ``fit(X, z)`` returning the fitted object and
  ``predict(X)`` returning ``(mean, scale)`` of its predictive distribution. The
  distribution is Gaussian unless the model has a finite ``df`` attribute, in which case
  it is Student-t with that many degrees of freedom. ``ARGaussian`` is the main model in
  the article, ``ARStudentT`` the heavy-tailed check, and ``ResidualGaussian`` wraps any
  scikit-learn style regressor.
* Losses are per row and lower is better: negative log predictive density (``nlpd``,
  the logarithmic score with the sign flipped), squared error (``sqerr``), Gaussian
  CRPS (``crps``) and a 0/1 indicator of the outcome falling in the central 90%
  interval (``cover90``, higher is closer to nominal when below 0.9).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

LOG_2PI = float(np.log(2.0 * np.pi))
Z90 = float(norm.ppf(0.95))
SCORES = ("nlpd", "sqerr", "crps", "cover90")


# --------------------------------------------------------------------------- rows


def embed_ar(y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows for an AR(p) with intercept.

    Returns ``X`` with columns ``[1, y[t-1], ..., y[t-p]]``, the targets ``z = y[t]``
    and the time index ``t`` of each row, for ``t = p, ..., len(y) - 1``.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n <= p:
        raise ValueError("series shorter than the lag order")
    X = np.ones((n - p, p + 1))
    for k in range(1, p + 1):
        X[:, k] = y[p - k : n - k]
    return X, y[p:].copy(), np.arange(p, n)


# ------------------------------------------------------------------------- models


class Model(Protocol):
    def fit(self, X: np.ndarray, z: np.ndarray) -> "Model": ...

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


class ARGaussian:
    """Least squares on the rows, Gaussian predictive with the residual variance.

    (``df = inf`` marks the predictive distribution as Gaussian.)

    The predictive variance is ``RSS / (n - k)``. Parameter uncertainty is ignored, as
    it is in most production forecasting code.
    """

    df = np.inf

    def __init__(self) -> None:
        self.beta: np.ndarray | None = None
        self.sd: float | None = None

    def fit(self, X: np.ndarray, z: np.ndarray) -> "ARGaussian":
        beta, *_ = np.linalg.lstsq(X, z, rcond=None)
        resid = z - X @ beta
        dof = max(len(z) - X.shape[1], 1)
        self.beta = beta
        self.sd = float(np.sqrt(resid @ resid / dof))
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.beta is not None and self.sd is not None, "fit first"
        mean = X @ self.beta
        return mean, np.full(len(mean), self.sd)


class ARStudentT:
    """Linear model with Student-t errors (``df`` fixed), fitted by maximum likelihood.

    EM / iteratively reweighted least squares: weights ``(df + 1) / (df + r^2 / s^2)``,
    starting from least squares with a MAD scale. Large residuals get small weights,
    so a few extreme months move neither the coefficients nor the scale much.
    """

    def __init__(self, df: float = 5.0, iters: int = 100, tol: float = 1e-8) -> None:
        self.df = float(df)
        self.iters = iters
        self.tol = tol
        self.beta: np.ndarray | None = None
        self.sd: float | None = None  # the t scale parameter

    def fit(self, X: np.ndarray, z: np.ndarray) -> "ARStudentT":
        beta, *_ = np.linalg.lstsq(X, z, rcond=None)
        r = z - X @ beta
        s = float(np.median(np.abs(r - np.median(r))) / student_t.ppf(0.75, self.df)) or 1e-6
        for _ in range(self.iters):
            w = (self.df + 1.0) / (self.df + (r / s) ** 2)
            sw = np.sqrt(w)
            new, *_ = np.linalg.lstsq(X * sw[:, None], z * sw, rcond=None)
            r = z - X @ new
            s_new = float(np.sqrt(np.sum(w * r**2) / len(z)))
            done = np.max(np.abs(new - beta)) < self.tol and abs(s_new - s) < self.tol
            beta, s = new, s_new
            if done:
                break
        self.beta, self.sd = beta, s
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.beta is not None and self.sd is not None, "fit first"
        mean = X @ self.beta
        return mean, np.full(len(mean), self.sd)


class ResidualGaussian:
    """Wrap a point regressor: Gaussian predictive with sd from out-of-fold residuals.

    ``make`` builds a fresh regressor with ``fit``/``predict`` (scikit-learn style).
    The residual sd is estimated by ``folds``-fold cross-fitting inside the training
    rows, so a flexible regressor that interpolates its training data does not get an
    sd of zero.
    """

    df = np.inf

    def __init__(self, make: Callable[[], object], folds: int = 5, seed: int = 0) -> None:
        self.make = make
        self.folds = folds
        self.seed = seed
        self.reg: object | None = None
        self.sd: float | None = None

    def fit(self, X: np.ndarray, z: np.ndarray) -> "ResidualGaussian":
        rng = np.random.default_rng(self.seed)
        fold = rng.permutation(len(z)) % self.folds
        oof = np.empty(len(z))
        for f in range(self.folds):
            tr, te = fold != f, fold == f
            reg = self.make()
            reg.fit(X[tr], z[tr])
            oof[te] = reg.predict(X[te])
        self.sd = float(np.sqrt(np.mean((z - oof) ** 2)))
        self.reg = self.make()
        self.reg.fit(X, z)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.reg is not None and self.sd is not None, "fit first"
        mean = np.asarray(self.reg.predict(X), dtype=float)
        return mean, np.full(len(mean), self.sd)


# ------------------------------------------------------------------------- scores


def row_losses(z: np.ndarray, mean: np.ndarray, sd: np.ndarray, df: float = np.inf) -> dict[str, np.ndarray]:
    """Per-row losses of predictive distributions with location ``mean`` and scale ``sd``.

    Gaussian if ``df`` is infinite, otherwise Student-t with ``df`` degrees of freedom
    (CRPS is only computed for the Gaussian case and is NaN otherwise).
    """
    e = z - mean
    w = e / sd
    if np.isinf(df):
        nlpd = 0.5 * LOG_2PI + np.log(sd) + 0.5 * w**2
        crps = sd * (w * (2.0 * norm.cdf(w) - 1.0) + 2.0 * norm.pdf(w) - 1.0 / np.sqrt(np.pi))
        q = Z90
    else:
        nlpd = np.log(sd) - student_t.logpdf(w, df)
        crps = np.full(len(w), np.nan)
        q = float(student_t.ppf(0.95, df))
    return {
        "nlpd": nlpd,
        "sqerr": e**2,
        "crps": crps,
        "cover90": (np.abs(w) <= q).astype(float),
    }


def mean_and_se(d: np.ndarray, hac_lags: int | None = None) -> tuple[float, float]:
    """Mean of ``d`` and its standard error, Newey-West if ``hac_lags`` > 0."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return float("nan"), float("nan")
    m = float(d.mean())
    u = d - m
    lrv = float(u @ u) / n
    L = int(hac_lags) if hac_lags is not None else int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    for lag in range(1, min(L, n - 1) + 1):
        w = 1.0 - lag / (L + 1.0)
        lrv += 2.0 * w * float(u[lag:] @ u[:-lag]) / n
    return m, float(np.sqrt(max(lrv, 1e-300) / n))


def native_losses(
    X: np.ndarray,
    z: np.ndarray,
    model_factory: Callable[[], Model],
    folds: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Out-of-fold losses of models fitted inside one block (random K-fold).

    Random K-fold inside a block is the benchmark "a model from the block's own time".
    For purely autoregressive models with uncorrelated errors it estimates the
    out-of-sample loss without the bias that worries people about K-fold on time
    series (Bergmeir, Hyndman and Koo, 2018).
    """
    fold = rng.permutation(len(z)) % folds
    out = {s: np.empty(len(z)) for s in SCORES}
    for f in range(folds):
        tr, te = fold != f, fold == f
        m = model_factory().fit(X[tr], z[tr])
        mean, sd = m.predict(X[te])
        for s, v in row_losses(z[te], mean, sd, getattr(m, "df", np.inf)).items():
            out[s][te] = v
    return out


# --------------------------------------------------------------- transfer matrix


def make_blocks(n_rows: int, block_len: int) -> list[tuple[int, int]]:
    """Consecutive blocks of ``block_len`` rows aligned to the END of the sample.

    The most recent block is always complete; leftover rows at the start are dropped.
    """
    k = n_rows // block_len
    if k < 2:
        raise ValueError("need at least two blocks")
    off = n_rows - k * block_len
    return [(off + i * block_len, off + (i + 1) * block_len) for i in range(k)]


@dataclass
class TransferResult:
    """Losses of every block's model on every block.

    ``loss[s][i, r]`` is the loss of the model fitted on block ``i`` at row ``r``
    (NaN where the row is not scored: inside block ``i`` itself, or within ``gap`` rows
    of it). ``native[s][r]`` is the out-of-fold loss of the model fitted inside the
    row's own block.
    """

    blocks: list[tuple[int, int]]
    loss: dict[str, np.ndarray]
    native: dict[str, np.ndarray]
    labels: list[str] = field(default_factory=list)

    @property
    def k(self) -> int:
        return len(self.blocks)

    def mean_matrix(self, score: str = "nlpd", keep: np.ndarray | None = None) -> np.ndarray:
        """``M[i, j]``: mean loss of block i's model on block j; diagonal = native.

        ``keep`` is an optional boolean mask over rows: rows where it is False are left
        out of every mean (for example, to see how much a handful of extreme months
        drive a column).
        """
        M = np.full((self.k, self.k), np.nan)
        loss = self.loss[score]
        nat = self.native[score]
        if keep is not None:
            loss = np.where(keep[None, :], loss, np.nan)
            nat = np.where(keep, nat, np.nan)
        for j, (a, b) in enumerate(self.blocks):
            for i in range(self.k):
                M[i, j] = np.nanmean(loss[i, a:b]) if i != j else np.nanmean(nat[a:b])
        return M

    def excess(self, score: str = "nlpd", ratio: bool | None = None, keep: np.ndarray | None = None) -> np.ndarray:
        """Loss of block i's model on block j relative to the native model on block j.

        Differences for ``nlpd``/``crps``/``cover90``; ratios for ``sqerr`` unless
        ``ratio`` says otherwise. The diagonal is 0 (or 1 for ratios).
        """
        M = self.mean_matrix(score, keep)
        nat = np.diag(M).copy()
        if ratio is None:
            ratio = score == "sqerr"
        E = M / nat[None, :] if ratio else M - nat[None, :]
        np.fill_diagonal(E, 1.0 if ratio else 0.0)
        return E

    def asymmetry(self, score: str = "nlpd") -> np.ndarray:
        """``E - E.T`` for the difference version of ``excess``: positive entry (i, j)
        means block i's model does worse on block j than block j's model does on i."""
        E = self.excess(score, ratio=False)
        return E - E.T

    def paired(self, i: int, j: int, score: str = "nlpd", keep: np.ndarray | None = None) -> tuple[float, float]:
        """Mean and Newey-West standard error of (model i - native) on block j."""
        a, b = self.blocks[j]
        d = self.loss[score][i, a:b] - self.native[score][a:b]
        if keep is not None:
            d = np.where(keep[a:b], d, np.nan)
        return mean_and_se(d)


def transfer_matrix(
    X: np.ndarray,
    z: np.ndarray,
    model_factory: Callable[[], Model],
    block_len: int,
    gap: int = 0,
    folds: int = 5,
    seed: int = 0,
    labels: list[str] | None = None,
) -> TransferResult:
    """Fit ``model_factory()`` on each block and score it on every row of every
    other block, skipping rows within ``gap`` rows of the training block.

    ``gap`` should be at least the lag order (so no test row's features overlap the
    training targets) plus whatever memory the series has beyond the model's lags.
    """
    rng = np.random.default_rng(seed)
    n = len(z)
    blocks = make_blocks(n, block_len)
    k = len(blocks)
    loss = {s: np.full((k, n), np.nan) for s in SCORES}
    native = {s: np.full(n, np.nan) for s in SCORES}
    rows = np.arange(n)

    for j, (a, b) in enumerate(blocks):
        nat = native_losses(X[a:b], z[a:b], model_factory, folds, rng)
        for s in SCORES:
            native[s][a:b] = nat[s]

    for i, (a, b) in enumerate(blocks):
        m = model_factory().fit(X[a:b], z[a:b])
        keep = (rows < a - gap) | (rows >= b + gap)
        mean, sd = m.predict(X[keep])
        for s, v in row_losses(z[keep], mean, sd, getattr(m, "df", np.inf)).items():
            loss[s][i, keep] = v
        # rows outside every block (leading remainder) are not part of the matrix
    lead = blocks[0][0]
    for s in SCORES:
        loss[s][:, :lead] = np.nan
    return TransferResult(blocks=blocks, loss=loss, native=native, labels=labels or [])


# ------------------------------------------------------------ backward window rule


@dataclass
class WindowChoice:
    rows: int  # number of most recent rows to train on
    stopped_at: int | None  # index (1 = block just before the anchor) where it stopped
    stats: list[tuple[float, float]]  # (mean excess nlpd, z statistic) per block examined


def backward_window(
    X: np.ndarray,
    z: np.ndarray,
    origin: int,
    model_factory: Callable[[], Model],
    block_len: int,
    max_back: int,
    z_crit: float,
    gap: int = 0,
    folds: int = 5,
    rng: np.random.Generator | None = None,
    score: str = "nlpd",
) -> WindowChoice:
    """How far back does a model of the most recent block still work?

    Rows ``[origin - block_len, origin)`` form the anchor block. For ``j = 1, 2, ...``
    the block ``[origin - (j + 1) * block_len, origin - j * block_len)`` is scored by
    the anchor model and by out-of-fold models fitted inside it. The walk stops at the
    first block where the anchor is worse with a one-sided z statistic above
    ``z_crit``; the training window is then the anchor plus the blocks that passed.
    If no block fails (or the series runs out) the window is everything before
    ``origin``.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    a0 = origin - block_len
    if a0 < 0:
        return WindowChoice(rows=origin, stopped_at=None, stats=[])
    anchor = model_factory().fit(X[a0:origin], z[a0:origin])
    stats: list[tuple[float, float]] = []
    for j in range(1, max_back + 1):
        lo, hi = origin - (j + 1) * block_len, origin - j * block_len
        if lo < 0:
            break
        hi_scored = min(hi, a0 - gap)
        if hi_scored - lo < max(10, block_len // 3):
            continue
        nat = native_losses(X[lo:hi], z[lo:hi], model_factory, folds, rng)
        mean, sd = anchor.predict(X[lo:hi_scored])
        d = row_losses(z[lo:hi_scored], mean, sd, getattr(anchor, "df", np.inf))[score] - nat[score][: hi_scored - lo]
        m, se = mean_and_se(d)
        zstat = m / se if se > 0 else 0.0
        stats.append((m, zstat))
        if zstat > z_crit:
            return WindowChoice(rows=j * block_len, stopped_at=j, stats=stats)
    return WindowChoice(rows=origin, stopped_at=None, stats=stats)


# --------------------------------------------------- reverse-ordered CUSUM of squares


def reverse_cusumsq_window(
    X: np.ndarray, z: np.ndarray, origin: int, min_rows: int, max_rows: int, crit: float = 1.358
) -> int:
    """A reverse-ordered CUSUM-of-squares window, in the spirit of Pesaran and
    Timmermann (2002).

    Rows before ``origin`` are taken newest first. Recursive residuals are computed by
    fitting least squares on the ``min_rows`` newest rows and predicting each older row
    in turn from all rows newer than it. With ``S_m`` the cumulative share of squared
    recursive residuals, a break is declared if ``max |S_m - m/M| * sqrt(M/2)`` exceeds
    ``crit`` (1.358 is the 5% point of the supremum of a Brownian bridge, the
    asymptotic Brown-Durbin-Evans bound). The window then ends at the most recent row
    where the deviation peaks; without a break it is ``max_rows`` (or all rows).

    This is our implementation of the idea, not a replication of their exact procedure.
    """
    lo = max(0, origin - max_rows)
    Xr = X[lo:origin][::-1]
    zr = z[lo:origin][::-1]
    n, k = Xr.shape
    if n <= min_rows + 10:
        return origin - lo
    Ainv = np.linalg.inv(Xr[:min_rows].T @ Xr[:min_rows])
    bvec = Xr[:min_rows].T @ zr[:min_rows]
    w = np.empty(n - min_rows)
    for m in range(min_rows, n):
        beta = Ainv @ bvec
        x = Xr[m]
        Ax = Ainv @ x
        f = 1.0 + x @ Ax
        w[m - min_rows] = (zr[m] - x @ beta) / np.sqrt(f)
        Ainv -= np.outer(Ax, Ax) / f  # Sherman-Morrison update for the added row
        bvec += x * zr[m]
    s = np.cumsum(w**2) / np.sum(w**2)
    M = len(w)
    dev = np.abs(s - np.arange(1, M + 1) / M)
    if dev.max() * np.sqrt(M / 2.0) > crit:
        m_star = int(np.argmax(dev))
        return min_rows + m_star + 1
    return origin - lo


# ------------------------------------------------------- population (no-noise) maps


def ar1_moments(mu: float, phi: float, sigma: float) -> np.ndarray:
    """``E[x x']`` for ``x = (1, y_{t-1})`` under a stationary Gaussian AR(1)."""
    v = sigma**2 / (1.0 - phi**2)
    return np.array([[1.0, mu], [mu, mu**2 + v]])


def population_excess_nlpd(
    beta_i: np.ndarray, sigma_i: float, beta_j: np.ndarray, sigma_j: float, Sxx_j: np.ndarray
) -> float:
    """Expected excess log loss of the Gaussian linear model (beta_i, sigma_i) on a
    regime with true (beta_j, sigma_j) and regressor moments ``Sxx_j``, over the true
    model of that regime:

        1/2 log(s_i^2/s_j^2) + (s_j^2 + d' Sxx_j d) / (2 s_i^2) - 1/2,  d = b_i - b_j.
    """
    d = np.asarray(beta_i) - np.asarray(beta_j)
    q = float(d @ Sxx_j @ d)
    return 0.5 * np.log(sigma_i**2 / sigma_j**2) + (sigma_j**2 + q) / (2.0 * sigma_i**2) - 0.5


def variance_ratio_excess(r: float) -> float:
    """Excess log loss when the predictive variance is right in mean but the test
    variance is ``r`` times the training variance: ``(r - 1 - log r) / 2``."""
    return 0.5 * (r - 1.0 - np.log(r))
