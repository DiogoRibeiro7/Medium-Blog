"""Linear models with absorbed fixed effects and the inference of METHODS.md, section 4.

Fixed effects are partialled out exactly by projection on their dummies (Frisch-Waugh-
Lovell). Inference uses two-way clustering by country and year (Cameron, Gelbach and
Miller 2011) and a wild cluster bootstrap over countries that imposes the null, with
the details fixed in the change log of 2026-10-24.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Array = NDArray[np.float64]
BOOTSTRAP_DRAWS = 9999
BOOTSTRAP_SEED = 20261101
BATCH = 500


def dummies(codes: Sequence[str]) -> Array:
    """One column per distinct code (no reference level)."""
    levels = sorted(set(codes))
    index = {c: i for i, c in enumerate(levels)}
    out = np.zeros((len(codes), len(levels)))
    out[np.arange(len(codes)), [index[c] for c in codes]] = 1.0
    return out


def singleton_mask(groups: Sequence[Sequence[str]]) -> NDArray[np.bool_]:
    """Observations to keep: drop those alone in any grouping, until none remain."""
    n = len(groups[0])
    keep = np.ones(n, dtype=bool)
    while True:
        drop = np.zeros(n, dtype=bool)
        for codes in groups:
            kept = [c for c, k in zip(codes, keep, strict=True) if k]
            counts: dict[str, int] = {}
            for c in kept:
                counts[c] = counts.get(c, 0) + 1
            drop |= np.array([k and counts[c] == 1 for c, k in zip(codes, keep, strict=True)])
        if not drop.any():
            return keep
        keep &= ~drop


def partial_out(controls: Array, columns: Array) -> Array:
    """Residuals of each column of ``columns`` on ``controls`` (least squares)."""
    if controls.shape[1] == 0:
        return columns
    coef, *_ = np.linalg.lstsq(controls, columns, rcond=None)
    return np.asarray(columns - controls @ coef, dtype=np.float64)


@dataclass(frozen=True)
class Fit:
    """OLS on partialled-out data, with the clusters needed for inference."""

    names: tuple[str, ...]
    beta: Array
    x: Array
    y: Array
    resid: Array
    country: NDArray[np.str_]
    year: NDArray[np.str_]

    @property
    def n(self) -> int:
        return int(self.y.size)

    @property
    def k(self) -> int:
        return int(self.beta.size)

    def index(self, names: Sequence[str]) -> list[int]:
        return [self.names.index(n) for n in names]


def fit(names: Sequence[str], x: Array, y: Array, controls: Array,
        country: Sequence[str], year: Sequence[str], weights: Array | None = None) -> Fit:
    """Weighted (optional) least squares of ``y`` on ``x`` with ``controls`` absorbed."""
    if weights is not None:
        root = np.sqrt(weights)
        x, y, controls = x * root[:, None], y * root, controls * root[:, None]
    both = partial_out(controls, np.column_stack([y, x]))
    yt, xt = both[:, 0], both[:, 1:]
    beta, *_ = np.linalg.lstsq(xt, yt, rcond=None)
    return Fit(tuple(names), np.asarray(beta), xt, yt, yt - xt @ beta,
               np.asarray(country), np.asarray(year))


def _groups(codes: NDArray[np.str_]) -> list[NDArray[np.int64]]:
    order: dict[str, list[int]] = {}
    for i, c in enumerate(codes.tolist()):
        order.setdefault(c, []).append(i)
    return [np.asarray(v, dtype=np.int64) for v in order.values()]


def cluster_vcov(x: Array, resid: Array, codes: NDArray[np.str_], k: int) -> tuple[Array, int]:
    """CR1 sandwich for one clustering: G/(G-1) (N-1)/(N-K) A meat A."""
    bread = np.linalg.inv(x.T @ x)
    groups = _groups(codes)
    meat = np.zeros((x.shape[1], x.shape[1]))
    for idx in groups:
        score = x[idx].T @ resid[idx]
        meat += np.outer(score, score)
    g, n = len(groups), x.shape[0]
    factor = g / (g - 1) * (n - 1) / (n - k)
    return factor * bread @ meat @ bread, g


def twoway_vcov(f: Fit) -> tuple[Array, int]:
    """V_country + V_year - V_country-year, negative eigenvalues set to zero.

    Returns the matrix and the degrees of freedom, min(G_country, G_year) - 1.
    """
    k = f.k
    vc, gc = cluster_vcov(f.x, f.resid, f.country, k)
    vt, gt = cluster_vcov(f.x, f.resid, f.year, k)
    cell = np.char.add(np.char.add(f.country, "|"), f.year)
    vct, _ = cluster_vcov(f.x, f.resid, cell, k)
    v = vc + vt - vct
    values, vectors = np.linalg.eigh((v + v.T) / 2)
    v = vectors @ np.diag(np.maximum(values, 0.0)) @ vectors.T
    return v, min(gc, gt) - 1


def t_test(beta: Array, v: Array, j: int, df: int) -> tuple[float, float, float]:
    """Estimate's t statistic, two-sided p-value and standard error."""
    se = float(np.sqrt(v[j, j]))
    t = float(beta[j] / se) if se > 0 else float("inf")
    return t, float(2 * stats.t.sf(abs(t), df)), se


def wald_test(beta: Array, v: Array, idx: Sequence[int], df: int) -> tuple[float, float]:
    """Wald statistic of ``beta[idx] = 0`` and its p-value from F(q, df)."""
    b = beta[list(idx)]
    w = float(b @ np.linalg.pinv(v[np.ix_(idx, idx)]) @ b)
    q = len(idx)
    return w, float(stats.f.sf(w / q, q, df))


def wild_bootstrap(f: Fit, test: Sequence[int], draws: int = BOOTSTRAP_DRAWS,
                   seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """Null-imposed wild cluster bootstrap over countries (Rademacher weights).

    The statistic is the t statistic (one coefficient) or the Wald statistic (several)
    with CR1 errors clustered by country. Returns the original statistic and the share
    of draws at least as extreme.
    """
    x, y = f.x, f.y
    keep = [j for j in range(f.k) if j not in set(test)]
    xr = x[:, keep]
    br, *_ = np.linalg.lstsq(xr, y, rcond=None)
    fitted_r, resid_r = xr @ br, y - xr @ br
    groups = _groups(f.country)
    g, n, k = len(groups), f.n, f.k
    cluster_of = np.empty(n, dtype=np.int64)
    for c, idx in enumerate(groups):
        cluster_of[idx] = c
    bread = np.linalg.inv(x.T @ x)
    factor = g / (g - 1) * (n - 1) / (n - k)
    rows = bread[list(test)]                       # q x k

    def statistics(yy: Array) -> Array:
        """Statistic for each row of ``yy`` (draws x n)."""
        beta = yy @ (x @ bread)                    # draws x k
        resid = yy - beta @ x.T
        scores = np.stack([resid[:, idx] @ x[idx] for idx in groups], axis=1)  # d x g x k
        u = scores @ rows.T                        # d x g x q
        v = factor * np.einsum("dgq,dgr->dqr", u, u)
        b = beta[:, list(test)]
        if len(test) == 1:
            return np.asarray(b[:, 0] / np.sqrt(v[:, 0, 0]))
        return np.asarray(np.einsum("dq,dqr,dr->d", b, np.linalg.pinv(v), b))

    original = float(statistics(y[None, :])[0])
    rng = np.random.default_rng(seed)
    extreme = 0
    done = 0
    while done < draws:
        m = min(BATCH, draws - done)
        w = rng.choice(np.array([-1.0, 1.0]), size=(m, g))
        ystar = fitted_r[None, :] + w[:, cluster_of] * resid_r[None, :]
        s = statistics(ystar)
        extreme += int(np.sum(np.abs(s) >= abs(original)) if len(test) == 1
                       else np.sum(s >= original))
        done += m
    return original, extreme / draws


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return [float(p) for p in adjusted]


def minimum_detectable(se: float, df: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest effect a two-sided test at ``alpha`` detects with ``power``."""
    return float((stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)) * se)
