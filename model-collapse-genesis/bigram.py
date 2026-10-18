"""A sparse bigram language model that can be fitted, decoded and refitted fast.

Counts are stored as sorted pair codes ``prev * V + next`` with their counts,
which is a CSR matrix in disguise. Sampling runs many documents in parallel:
row r's cumulative distribution is shifted to the interval (r, r + 1], so one
``searchsorted`` call draws the next token for every document at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Counts:
    codes: np.ndarray   # sorted unique pair codes prev * V + next
    n: np.ndarray       # counts (float, so policies can weight data)
    V: int

    @property
    def rows(self) -> np.ndarray:
        return self.codes // self.V

    @property
    def cols(self) -> np.ndarray:
        return self.codes % self.V


def count_docs(docs, V: int) -> Counts:
    """Bigram counts inside each document (no bigram crosses a document edge)."""
    prev = np.concatenate([d[:-1] for d in docs])
    nxt = np.concatenate([d[1:] for d in docs])
    codes, n = np.unique(prev * V + nxt, return_counts=True)
    return Counts(codes, n.astype(float), V)


def count_matrix(docs: np.ndarray, V: int) -> Counts:
    """Same as count_docs for a 2-D array of equal-length documents."""
    codes, n = np.unique((docs[:, :-1] * V + docs[:, 1:]).ravel(), return_counts=True)
    return Counts(codes, n.astype(float), V)


def add(a: Counts, b: Counts, wa: float = 1.0, wb: float = 1.0) -> Counts:
    codes = np.concatenate([a.codes, b.codes])
    n = np.concatenate([wa * a.n, wb * b.n])
    u, inv = np.unique(codes, return_inverse=True)
    return Counts(u, np.bincount(inv, weights=n), a.V)


def _row_starts(rows: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])


@dataclass
class Sampler:
    cdf: np.ndarray      # row r occupies (r, r + 1]
    cols: np.ndarray
    has_row: np.ndarray  # token has at least one continuation
    restart: int         # context used when a token has no continuation


def make_sampler(c: Counts, tau: float = 1.0, top_p: float = 1.0, restart: int = 0) -> Sampler:
    """Decode the fitted bigram with temperature tau and nucleus threshold top_p."""
    rows, cols, w = c.rows, c.cols, c.n.astype(float)
    if tau != 1.0:
        # p ** (1 / tau) within a row, computed in log space for stability
        lw = np.log(w) / tau
        starts = _row_starts(rows)
        mx = np.maximum.reduceat(lw, starts)
        lw -= np.repeat(mx, np.diff(np.r_[starts, len(lw)]))
        w = np.exp(lw)
    if top_p < 1.0:
        order = np.lexsort((-w, rows))
        rows, cols, w = rows[order], cols[order], w[order]
        starts = _row_starts(rows)
        lens = np.diff(np.r_[starts, len(w)])
        tot = np.repeat(np.add.reduceat(w, starts), lens)
        cs = np.cumsum(w)
        before = cs - w - np.repeat(cs[starts] - w[starts], lens)
        keep = before < top_p * tot * (1 - 1e-12)
        rows, cols, w = rows[keep], cols[keep], w[keep]
    starts = _row_starts(rows)
    lens = np.diff(np.r_[starts, len(w)])
    cs = np.cumsum(w)
    base = np.repeat(cs[starts] - w[starts], lens)
    tot = np.repeat(np.add.reduceat(w, starts), lens)
    frac = (cs - base) / tot
    last = starts + lens - 1
    frac[last] = 1.0
    cdf = rows + frac
    has_row = np.zeros(c.V, dtype=bool)
    has_row[rows[starts]] = True
    return Sampler(cdf=cdf, cols=cols, has_row=has_row, restart=restart)


def generate(s: Sampler, n_docs: int, length: int, rng: np.random.Generator) -> np.ndarray:
    """n_docs documents of `length` tokens, each started from the restart context."""
    out = np.empty((n_docs, length), dtype=np.int64)
    state = np.full(n_docs, s.restart, dtype=np.int64)
    for j in range(length):
        state = np.where(s.has_row[state], state, s.restart)
        # u in [1e-9, 1): keeps x strictly inside (r, r + 1] after rounding
        x = state + (1e-9 + rng.random(n_docs) * (1 - 2e-9))
        idx = np.searchsorted(s.cdf, x, side="left")
        state = s.cols[idx]
        out[:, j] = state
    return out
