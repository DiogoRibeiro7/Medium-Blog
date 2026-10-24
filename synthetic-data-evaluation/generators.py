"""Synthetic-data generators that share one interface.

Every generator works on a numeric code matrix produced by `Encoder`:
categorical columns become integer codes, numeric columns stay as they are.
A generator has `fit(M, rng)` and `sample(n, rng)`.

`synthesize` produces m synthetic data sets in one of two ways:

* simple synthesis: fit once on the real data, then draw m data sets from
  the fitted generator;
* proper synthesis: before each draw, refit the generator on a bootstrap
  resample of the real data, so that the uncertainty in the generator's own
  parameters is carried into the synthetic data (the bootstrap stands in for
  a draw from the posterior of those parameters).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


@dataclass
class Encoder:
    """Map a mixed-type data frame to a float matrix and back."""

    columns: list[str]
    categorical: dict[str, bool]
    levels: dict[str, np.ndarray] = field(default_factory=dict)
    integer: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "Encoder":
        columns = list(df.columns)
        categorical = {c: not pd.api.types.is_numeric_dtype(df[c]) for c in columns}
        enc = cls(columns=columns, categorical=categorical)
        for c in columns:
            if categorical[c]:
                enc.levels[c] = np.array(sorted(df[c].unique()), dtype=object)
            else:
                v = df[c].to_numpy(dtype=float)
                enc.integer[c] = bool(np.all(np.isclose(v, np.round(v))))
        return enc

    @property
    def cat_mask(self) -> np.ndarray:
        return np.array([self.categorical[c] for c in self.columns])

    def to_codes(self, df: pd.DataFrame) -> np.ndarray:
        out = np.empty((len(df), len(self.columns)))
        for j, c in enumerate(self.columns):
            if self.categorical[c]:
                lev = self.levels[c]
                pos = np.searchsorted(lev, df[c].to_numpy().astype(object))
                pos = np.clip(pos, 0, len(lev) - 1)
                if not np.all(lev[pos] == df[c].to_numpy()):
                    raise ValueError(f"unseen level in column {c}")
                out[:, j] = pos
            else:
                out[:, j] = df[c].to_numpy(dtype=float)
        return out

    def from_codes(self, M: np.ndarray) -> pd.DataFrame:
        data = {}
        for j, c in enumerate(self.columns):
            if self.categorical[c]:
                data[c] = self.levels[c][M[:, j].astype(int)]
            elif self.integer[c]:
                data[c] = np.round(M[:, j]).astype(np.int64)
            else:
                data[c] = M[:, j]
        return pd.DataFrame(data)


def _round_integers(S: np.ndarray, enc: Encoder) -> np.ndarray:
    for j, c in enumerate(enc.columns):
        if not enc.categorical[c] and enc.integer[c]:
            S[:, j] = np.round(S[:, j])
    return S


class IndependentMarginals:
    """Sample every column independently from its empirical marginal."""

    name = "independent"

    def __init__(self, enc: Encoder):
        self.enc = enc

    def fit(self, M: np.ndarray, rng: np.random.Generator) -> "IndependentMarginals":
        self.M = M.copy()
        return self

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        N, p = self.M.shape
        return np.column_stack([self.M[rng.integers(N, size=n), j] for j in range(p)])


class GaussianCopula:
    """Gaussian copula with empirical marginals.

    Numeric columns enter through the randomised probability integral
    transform, so ties and point masses (capital gain is zero for most
    people) are spread uniformly over their probability interval. Categorical
    columns are ordered by frequency and each level owns an interval of
    [0, 1] of length equal to its share, the device used by the Synthetic
    Data Vault. Sampling inverts both maps, so every synthetic numeric value
    is a value that occurs in the real data.
    """

    name = "copula"

    def __init__(self, enc: Encoder):
        self.enc = enc

    def fit(self, M: np.ndarray, rng: np.random.Generator) -> "GaussianCopula":
        N, p = M.shape
        Z = np.empty((N, p))
        self.parts: list[tuple] = []
        for j in range(p):
            x = M[:, j]
            v = rng.random(N)
            if self.enc.cat_mask[j]:
                k = len(self.enc.levels[self.enc.columns[j]])
                counts = np.bincount(x.astype(int), minlength=k).astype(float)
                order = np.argsort(-counts, kind="stable")
                prob = counts[order] / N
                lower = np.concatenate([[0.0], np.cumsum(prob)[:-1]])
                rank_of = np.empty(k, dtype=int)
                rank_of[order] = np.arange(k)
                r = rank_of[x.astype(int)]
                u = lower[r] + v * prob[r]
                self.parts.append(("cat", order, np.cumsum(prob)))
            else:
                xs = np.sort(x)
                lo = np.searchsorted(xs, x, side="left") / N
                hi = np.searchsorted(xs, x, side="right") / N
                u = lo + v * (hi - lo)
                self.parts.append(("num", xs))
            Z[:, j] = ndtri(np.clip(u, 1e-9, 1 - 1e-9))
        R = np.corrcoef(Z, rowvar=False)
        R = 0.999 * R + 0.001 * np.eye(p)
        self.L = np.linalg.cholesky(R)
        return self

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        p = self.L.shape[0]
        U = ndtr(rng.standard_normal((n, p)) @ self.L.T)
        S = np.empty((n, p))
        for j, part in enumerate(self.parts):
            if part[0] == "cat":
                _, order, upper = part
                pos = np.searchsorted(upper, U[:, j], side="right")
                S[:, j] = order[np.minimum(pos, len(order) - 1)]
            else:
                xs = part[1]
                idx = np.minimum((U[:, j] * len(xs)).astype(int), len(xs) - 1)
                S[:, j] = xs[idx]
        return S


class SequentialCART:
    """Sequential CART synthesis in the style of synthpop (Reiter 2005).

    The first column is drawn from its empirical marginal. Each later column
    is modelled by a regression or classification tree on the columns before
    it. To synthesise a value, a synthetic record is dropped down the tree and
    the value is copied from a real record drawn at random from the leaf it
    lands in, so the leaf's donors carry the conditional distribution.
    """

    name = "cart"

    def __init__(self, enc: Encoder, min_samples_leaf: int = 5):
        self.enc = enc
        self.min_samples_leaf = min_samples_leaf

    def fit(self, M: np.ndarray, rng: np.random.Generator) -> "SequentialCART":
        N, p = M.shape
        self.first = M[:, 0].copy()
        self.trees: list[tuple] = []
        for j in range(1, p):
            seed = int(rng.integers(2**31 - 1))
            if self.enc.cat_mask[j]:
                tree = DecisionTreeClassifier(
                    min_samples_leaf=self.min_samples_leaf, random_state=seed
                )
            else:
                tree = DecisionTreeRegressor(
                    min_samples_leaf=self.min_samples_leaf, random_state=seed
                )
            tree.fit(M[:, :j], M[:, j])
            leaf = tree.apply(M[:, :j])
            order = np.argsort(leaf, kind="stable")
            donors = M[order, j]
            n_nodes = tree.tree_.node_count
            count = np.bincount(leaf, minlength=n_nodes)
            start = np.concatenate([[0], np.cumsum(count)[:-1]])
            self.trees.append((tree, donors, start, count))
        return self

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        p = len(self.trees) + 1
        S = np.empty((n, p))
        S[:, 0] = self.first[rng.integers(len(self.first), size=n)]
        for j, (tree, donors, start, count) in enumerate(self.trees, start=1):
            leaf = tree.apply(S[:, :j])
            pick = start[leaf] + (rng.random(n) * count[leaf]).astype(int)
            S[:, j] = donors[pick]
        return S


def distance_features(M: np.ndarray, enc: Encoder, ref: np.ndarray) -> np.ndarray:
    """Embed records for nearest-neighbour search.

    Numeric columns are replaced by their mid-rank position in the reference
    data (a value in [0, 1]); categorical columns are one-hot encoded and
    scaled by 1/sqrt(2), so a mismatch on one categorical column adds exactly
    1 to the squared Euclidean distance. Identical records are at distance 0.
    """
    blocks = []
    for j, c in enumerate(enc.columns):
        if enc.categorical[c]:
            k = len(enc.levels[c])
            blocks.append(np.eye(k)[M[:, j].astype(int)] / np.sqrt(2.0))
        else:
            xs = np.sort(ref[:, j])
            lo = np.searchsorted(xs, M[:, j], side="left")
            hi = np.searchsorted(xs, M[:, j], side="right")
            blocks.append(((lo + hi) / (2.0 * len(xs)))[:, None])
    return np.hstack(blocks)


class SMOTEGenerator:
    """SMOTE used as a whole-table generator, as in the TabDDPM benchmarks.

    Within each class of `class_col` (or the whole table when None), a
    synthetic record lies on the segment between a real record and one of its
    k nearest real neighbours: numeric columns are interpolated with weight
    lambda ~ U(0, 1), and each categorical column is taken from the neighbour
    with probability lambda and from the base record otherwise.
    """

    name = "smote"

    def __init__(self, enc: Encoder, class_col: int | None = None, k: int = 5):
        self.enc = enc
        self.class_col = class_col
        self.k = k

    def fit(self, M: np.ndarray, rng: np.random.Generator) -> "SMOTEGenerator":
        F = distance_features(M, self.enc, M)
        if self.class_col is None:
            groups = [np.arange(len(M))]
        else:
            labels = M[:, self.class_col]
            groups = [np.flatnonzero(labels == v) for v in np.unique(labels)]
        self.groups = []
        for g in groups:
            k = min(self.k, len(g) - 1)
            nn = NearestNeighbors(n_neighbors=k + 1).fit(F[g])
            _, nbr = nn.kneighbors(F[g])
            self.groups.append((M[g], nbr[:, 1:]))
        sizes = np.array([len(g) for g in groups], dtype=float)
        self.share = sizes / sizes.sum()
        return self

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        counts = rng.multinomial(n, self.share)
        out = []
        cat = self.enc.cat_mask
        for (X, nbr), c in zip(self.groups, counts):
            if c == 0:
                continue
            base = rng.integers(len(X), size=c)
            other = nbr[base, rng.integers(nbr.shape[1], size=c)]
            lam = rng.random(c)[:, None]
            S = X[base] + lam * (X[other] - X[base])
            take_other = rng.random((c, X.shape[1])) < lam
            S[:, cat] = np.where(take_other[:, cat], X[other][:, cat], X[base][:, cat])
            out.append(S)
        S = np.vstack(out)
        S = S[rng.permutation(len(S))]
        return _round_integers(S, self.enc)


class TrainingCopy:
    """Release the training data itself: the degenerate 'perfect' generator."""

    name = "training copy"

    def __init__(self, enc: Encoder):
        self.enc = enc

    def fit(self, M: np.ndarray, rng: np.random.Generator) -> "TrainingCopy":
        self.M = M.copy()
        return self

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if n == len(self.M):
            return self.M.copy()
        return self.M[rng.integers(len(self.M), size=n)]


def synthesize(
    make: Callable[[], object],
    M: np.ndarray,
    n: int,
    m: int,
    rng: np.random.Generator,
    proper: bool,
) -> list[np.ndarray]:
    """Return m synthetic data sets of n rows each (see module docstring)."""
    if not proper:
        gen = make().fit(M, rng)
        return [gen.sample(n, rng) for _ in range(m)]
    out = []
    for _ in range(m):
        boot = M[rng.integers(len(M), size=len(M))]
        out.append(make().fit(boot, rng).sample(n, rng))
    return out


def factories(enc: Encoder, class_col: int | None) -> dict[str, Callable[[], object]]:
    """The non-deep generators used throughout, by name."""
    return {
        "independent": lambda: IndependentMarginals(enc),
        "copula": lambda: GaussianCopula(enc),
        "cart": lambda: SequentialCART(enc),
        "smote": lambda: SMOTEGenerator(enc, class_col=class_col),
    }


def column_index(enc: Encoder, names: Sequence[str]) -> list[int]:
    return [enc.columns.index(c) for c in names]
