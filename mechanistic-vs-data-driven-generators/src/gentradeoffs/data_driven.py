"""Data-driven synthesizers of the interval table.

Four estimators of the joint distribution of the nine columns, each written
out in full so that its behaviour at the edge of the data is visible:

* ``GaussianCopula`` -- empirical marginals glued by a Gaussian copula;
  marginals are inverted by interpolating order statistics, so no value
  outside the observed range of a column can come out.
* ``CartSynthesizer`` -- sequential CART in the style of synthpop: each
  column is drawn from the observed values in the leaf of a tree grown on
  the columns before it.
* ``Smote`` -- unconditional SMOTE-style interpolation between a row and
  one of its nearest neighbours.
* ``GaussianMixture`` -- a full-covariance mixture on standardised columns;
  smooth, and the only one of the four that will produce values outside the
  range it was fitted on.

All but SMOTE can also sample the remaining columns given fixed values of
some columns (``sample_given``), which is how a joint synthesizer is asked a
what-if question.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture as _SkGMM
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeRegressor

from .world import COLUMNS

INTEGER = ["dow", "interval", "campaign", "offered", "agents"]


def matrix(table: pd.DataFrame) -> np.ndarray:
    """The nine columns as floats, without rows that have a missing value
    (an interval in which nobody was answered has no handle time)."""
    return table[COLUMNS].dropna().to_numpy(float)


def standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = X.mean(axis=0), X.std(axis=0)
    return mu, np.where(sd > 0, sd, 1.0)


def tidy(frame: pd.DataFrame) -> pd.DataFrame:
    """Round integer columns and clip every column to its feasible range."""
    out = frame.copy()
    for c in INTEGER:
        out[c] = np.round(out[c])
    out["dow"] = out["dow"].clip(0, 4)
    out["interval"] = out["interval"].clip(0, 23)
    out["campaign"] = out["campaign"].clip(0, 1)
    out["offered"] = out["offered"].clip(lower=1)
    out["agents"] = out["agents"].clip(lower=1)
    out["abandon_rate"] = out["abandon_rate"].clip(0, 1)
    out["sl"] = out["sl"].clip(0, 1)
    out["awt"] = out["awt"].clip(lower=0)
    out["aht"] = out["aht"].clip(lower=1)
    return out[COLUMNS]


class GaussianCopula:
    name = "Gaussian copula"

    def fit(self, table: pd.DataFrame) -> "GaussianCopula":
        X = matrix(table)
        self.n = X.shape[0]
        self.sorted = np.sort(X, axis=0)
        u = stats.rankdata(X, axis=0) / (self.n + 1)
        Z = stats.norm.ppf(u)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.nan_to_num(np.corrcoef(Z, rowvar=False))  # constants
        np.fill_diagonal(corr, 1.0)
        self.corr = corr + 1e-6 * np.eye(X.shape[1])
        self.probs = np.arange(1, self.n + 1) / (self.n + 1)
        return self

    def _to_z(self, j: int, values: np.ndarray) -> np.ndarray:
        """Normal score of a value through the column's empirical CDF.

        Values beyond the observed range get the score of the extreme
        observation: the copula has no way to represent them.
        """
        xs = self.sorted[:, j]
        lo = np.searchsorted(xs, values, side="left")
        hi = np.searchsorted(xs, values, side="right")
        u = (0.5 * (lo + hi) + 0.5) / (self.n + 1)
        u = np.clip(u, 1 / (self.n + 1), self.n / (self.n + 1))
        return stats.norm.ppf(u)

    def _from_z(self, j: int, z: np.ndarray) -> np.ndarray:
        return np.interp(stats.norm.cdf(z), self.probs, self.sorted[:, j])

    def _decode(self, Z: np.ndarray) -> pd.DataFrame:
        cols = {c: self._from_z(j, Z[:, j]) for j, c in enumerate(COLUMNS)}
        return tidy(pd.DataFrame(cols))

    def sample(self, rng: np.random.Generator, n: int) -> pd.DataFrame:
        Z = rng.multivariate_normal(np.zeros(len(COLUMNS)), self.corr, size=n)
        return self._decode(Z)

    def sample_given(self, rng: np.random.Generator,
                     given: pd.DataFrame) -> pd.DataFrame:
        g = [COLUMNS.index(c) for c in given.columns]
        f = [j for j in range(len(COLUMNS)) if j not in g]
        Zg = np.column_stack([self._to_z(j, given[COLUMNS[j]].to_numpy(float))
                              for j in g])
        S = self.corr
        A = S[np.ix_(f, g)] @ np.linalg.inv(S[np.ix_(g, g)])
        cov = S[np.ix_(f, f)] - A @ S[np.ix_(g, f)]
        mean = Zg @ A.T
        L = np.linalg.cholesky(cov + 1e-9 * np.eye(len(f)))
        Zf = mean + rng.standard_normal(mean.shape) @ L.T
        Z = np.empty((given.shape[0], len(COLUMNS)))
        Z[:, g] = Zg
        Z[:, f] = Zf
        out = self._decode(Z)
        for c in given.columns:          # the conditioning values themselves
            out[c] = given[c].to_numpy()
        return out


class CartSynthesizer:
    name = "CART"

    def __init__(self, min_leaf: int = 5, seed: int = 0):
        self.min_leaf = min_leaf
        self.seed = seed

    def fit(self, table: pd.DataFrame) -> "CartSynthesizer":
        X = matrix(table)
        self.first = X[:, 0].copy()
        self.trees, self.donors = [], []
        for j in range(1, len(COLUMNS)):
            tree = DecisionTreeRegressor(min_samples_leaf=self.min_leaf,
                                         random_state=self.seed)
            tree.fit(X[:, :j], X[:, j])
            leaves = tree.apply(X[:, :j])
            donors = {leaf: X[leaves == leaf, j] for leaf in np.unique(leaves)}
            self.trees.append(tree)
            self.donors.append(donors)
        return self

    def _fill(self, rng: np.random.Generator, X: np.ndarray,
              start: int, fixed: set[int]) -> np.ndarray:
        for j in range(max(start, 1), len(COLUMNS)):
            if j in fixed:
                continue
            leaves = self.trees[j - 1].apply(X[:, :j])
            for leaf in np.unique(leaves):
                rows = np.flatnonzero(leaves == leaf)
                pool = self.donors[j - 1][leaf]
                X[rows, j] = pool[rng.integers(0, pool.size, rows.size)]
        return X

    def sample(self, rng: np.random.Generator, n: int) -> pd.DataFrame:
        X = np.zeros((n, len(COLUMNS)))
        X[:, 0] = self.first[rng.integers(0, self.first.size, n)]
        X = self._fill(rng, X, 1, set())
        return tidy(pd.DataFrame(X, columns=COLUMNS))

    def sample_given(self, rng: np.random.Generator,
                     given: pd.DataFrame) -> pd.DataFrame:
        n = given.shape[0]
        X = np.zeros((n, len(COLUMNS)))
        fixed = {COLUMNS.index(c) for c in given.columns}
        for c in given.columns:
            X[:, COLUMNS.index(c)] = given[c].to_numpy(float)
        if 0 not in fixed:
            X[:, 0] = self.first[rng.integers(0, self.first.size, n)]
        X = self._fill(rng, X, 1, fixed)
        return tidy(pd.DataFrame(X, columns=COLUMNS))


class Smote:
    name = "SMOTE"

    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, table: pd.DataFrame) -> "Smote":
        X = matrix(table)
        self.mu, self.sd = standardise(X)
        self.Xs = (X - self.mu) / self.sd
        nn = NearestNeighbors(n_neighbors=self.k + 1).fit(self.Xs)
        self.neigh = nn.kneighbors(self.Xs, return_distance=False)[:, 1:]
        return self

    def sample(self, rng: np.random.Generator, n: int) -> pd.DataFrame:
        i = rng.integers(0, self.Xs.shape[0], n)
        j = self.neigh[i, rng.integers(0, self.k, n)]
        u = rng.random((n, 1))
        Xs = self.Xs[i] + u * (self.Xs[j] - self.Xs[i])
        return tidy(pd.DataFrame(Xs * self.sd + self.mu, columns=COLUMNS))


class GaussianMixture:
    name = "Gaussian mixture"

    def __init__(self, n_components: int | None = None,
                 candidates: tuple[int, ...] = (2, 4, 6, 8, 10, 12),
                 seed: int = 0):
        self.n_components = n_components
        self.candidates = candidates
        self.seed = seed

    def fit(self, table: pd.DataFrame) -> "GaussianMixture":
        X = matrix(table)
        self.mu, self.sd = standardise(X)
        Xs = (X - self.mu) / self.sd
        if self.n_components is None:
            fits = [_SkGMM(k, covariance_type="full", reg_covar=1e-4,
                           random_state=self.seed, n_init=2).fit(Xs)
                    for k in self.candidates]
            self.model = min(fits, key=lambda m: m.bic(Xs))
            self.n_components = self.model.n_components
        else:
            self.model = _SkGMM(self.n_components, covariance_type="full",
                                reg_covar=1e-4, random_state=self.seed,
                                n_init=2).fit(Xs)
        return self

    def sample(self, rng: np.random.Generator, n: int) -> pd.DataFrame:
        m = self.model
        comp = rng.choice(m.n_components, size=n, p=m.weights_)
        Xs = np.empty((n, len(COLUMNS)))
        for k in range(m.n_components):
            rows = np.flatnonzero(comp == k)
            if rows.size:
                Xs[rows] = rng.multivariate_normal(m.means_[k],
                                                   m.covariances_[k],
                                                   size=rows.size)
        return tidy(pd.DataFrame(Xs * self.sd + self.mu, columns=COLUMNS))

    def sample_given(self, rng: np.random.Generator,
                     given: pd.DataFrame) -> pd.DataFrame:
        m = self.model
        g = [COLUMNS.index(c) for c in given.columns]
        f = [j for j in range(len(COLUMNS)) if j not in g]
        xg = (given[[COLUMNS[j] for j in g]].to_numpy(float)
              - self.mu[g]) / self.sd[g]
        n = xg.shape[0]
        logw = np.empty((n, m.n_components))
        cond = []
        for k in range(m.n_components):
            mu, S = m.means_[k], m.covariances_[k]
            Sgg = S[np.ix_(g, g)]
            logw[:, k] = (np.log(m.weights_[k])
                          + stats.multivariate_normal.logpdf(xg, mu[g], Sgg))
            A = S[np.ix_(f, g)] @ np.linalg.inv(Sgg)
            cond.append((mu[f], A, S[np.ix_(f, f)] - A @ S[np.ix_(g, f)]))
        w = np.exp(logw - logw.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        comp = (rng.random((n, 1)) > np.cumsum(w, axis=1)).sum(axis=1)
        comp = np.minimum(comp, m.n_components - 1)
        Xs = np.empty((n, len(COLUMNS)))
        Xs[:, g] = xg
        for k in range(m.n_components):
            rows = np.flatnonzero(comp == k)
            if not rows.size:
                continue
            mu_f, A, cov = cond[k]
            mean = mu_f + (xg[rows] - m.means_[k][g]) @ A.T
            L = np.linalg.cholesky(cov + 1e-9 * np.eye(len(f)))
            Xs[np.ix_(rows, f)] = mean + rng.standard_normal(
                (rows.size, len(f))) @ L.T
        out = tidy(pd.DataFrame(Xs * self.sd + self.mu, columns=COLUMNS))
        for c in given.columns:
            out[c] = given[c].to_numpy()
        return out


def all_synthesizers(seed: int = 0, gmm_components: int | None = None):
    return [GaussianCopula(), CartSynthesizer(seed=seed), Smote(),
            GaussianMixture(gmm_components, seed=seed)]
