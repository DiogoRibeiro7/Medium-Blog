"""Difference-in-differences estimators for a staggered adoption with a common end date.

Every estimator takes a long panel with columns ``year``, a unit id, an outcome, and ``g`` (the
first treated year of the unit). All states adopt by 1920, so after 1919 there is no untreated
comparison group; the clean estimators stop at ``last=1919``.

twfe          two-way fixed effects (optionally with group-specific linear trends), cluster SE
twfe_weights  the weight two-way fixed effects puts on each treated cell's own effect
att_gt        Callaway and Sant'Anna group-time effects, not-yet-treated controls
impute        Borusyak, Jaravel and Spiess imputation, optionally with unit-specific trends
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE_YEAR = 1918  # centring for linear trends; only affects conditioning


def project_out(cols, fes, slopes=(), tol=1e-11, maxit=10000):
    """Residualise columns on several sets of fixed effects and group-specific slopes.

    fes: list of integer code arrays; slopes: list of (codes, t) pairs. Alternating projections
    converge to the residual on the sum of the subspaces (Gaure 2013)."""
    X = np.column_stack(cols).astype(float)
    for _ in range(maxit):
        X0 = X.copy()
        for g in fes:
            cnt = np.bincount(g)
            for j in range(X.shape[1]):
                X[:, j] -= (np.bincount(g, weights=X[:, j]) / cnt)[g]
        for g, t in slopes:
            den = np.bincount(g, weights=t * t)
            for j in range(X.shape[1]):
                X[:, j] -= (np.bincount(g, weights=X[:, j] * t) / den)[g] * t
        if np.abs(X - X0).max() < tol:
            break
    return X


def _prepare(d, y, unit, trend, window, first):
    d = d[np.isfinite(d[y])]
    if window is not None:
        d = d[d.year <= window]
    if first is not None:
        d = d[d.year >= first]
    d = d.copy()
    codes = [pd.factorize(d[unit])[0], pd.factorize(d.year)[0]]
    slopes = []
    if trend:
        slopes = [(pd.factorize(d[trend])[0], (d.year - BASE_YEAR).to_numpy(float))]
    return d, codes, slopes


def twfe(d, y, unit="state", trend=None, cluster="state", window=None, first=None):
    """OLS of y on a post-suffrage dummy with unit and year fixed effects.

    trend: column naming the groups that get their own linear trend (Miller uses state).
    Returns (estimate, cluster-robust SE, observations, clusters)."""
    d, codes, slopes = _prepare(d, y, unit, trend, window, first)
    post = (d.year >= d.g).to_numpy(float)
    X = project_out([d[y].to_numpy(float), post], codes, slopes)
    yd, xd = X[:, 0], X[:, 1]
    b = xd @ yd / (xd @ xd)
    e = yd - b * xd
    cl = pd.factorize(d[cluster])[0]
    score = np.bincount(cl, weights=xd * e)
    G = cl.max() + 1
    se = np.sqrt(score @ score) / (xd @ xd) * np.sqrt(G / (G - 1))
    return float(b), float(se), len(d), int(G)


def twfe_weights(d, y, unit="state", trend=None, window=None, first=None):
    """Weights w_it with estimate = sum over treated cells of w_it * tau_it.

    They come from the residualised treatment dummy; they sum to one and can be negative."""
    d, codes, slopes = _prepare(d, y, unit, trend, window, first)
    post = (d.year >= d.g).to_numpy(float)
    xd = project_out([post], codes, slopes)[:, 0]
    w = pd.Series(np.where(post == 1, xd, np.nan), index=d.index)
    w = w[post == 1]
    w = w / w.sum()
    out = d.loc[w.index, [unit, "year", "g"]].copy()
    out["w"] = w
    out["e"] = out.year - out.g
    return out


def att_gt(df, unit, y, last=1919, anticipation=0):
    """Callaway and Sant'Anna (2021) ATT(g, t), not-yet-treated controls.

    The base year is b = g - 1 - anticipation. For cell (g, t) the controls are units first
    treated after max(t, b) + anticipation, so no control is treated, or anticipating, in t.
    Pre-period cells use the same fixed base, so they read as an event study around b."""
    base_offset = 1 + anticipation
    d = df[np.isfinite(df[y])]
    W = d.pivot_table(index=unit, columns="year", values=y)
    G = d.groupby(unit)["g"].first().reindex(W.index).to_numpy()
    cols = list(W.columns)
    A = W.to_numpy(float)
    rows = []
    for g in sorted(set(G)):
        if g > last or (g - base_offset) not in cols:
            continue
        b = g - base_offset
        base = A[:, cols.index(b)]
        tr = G == g
        for t in cols:
            if t > last or t == b:
                continue
            diff = A[:, cols.index(t)] - base
            ok = ~np.isnan(diff)
            ct = (G > max(t, b) + anticipation) & ok
            trk = tr & ok
            if trk.sum() == 0 or ct.sum() < 2:
                continue
            rows.append((g, t, t - g, diff[trk].mean() - diff[ct].mean(), int(trk.sum()), int(ct.sum())))
    return pd.DataFrame(rows, columns=["g", "t", "e", "att", "n_treat", "n_ctrl"])


def event(attgt, emin=-6, emax=6):
    """Aggregate ATT(g, t) to event time, weighting cells by their number of treated units."""
    a = attgt[(attgt.e >= emin) & (attgt.e <= emax)]
    rows = []
    for e, c in a.groupby("e"):
        rows.append(dict(e=e, att=np.average(c.att, weights=c.n_treat),
                         n_treat=int(c.n_treat.sum()), cohorts=int(c.g.nunique())))
    return pd.DataFrame(rows).set_index("e")


def overall(attgt, emax=None):
    """Average post-suffrage effect over all treated (g, t) cells, weighted by treated units."""
    a = attgt[attgt.e >= 0]
    if emax is not None:
        a = a[a.e <= emax]
    return float(np.average(a.att, weights=a.n_treat))


def impute(df, unit, y, last=1919, trend=None):
    """Borusyak, Jaravel and Spiess (2024) imputation estimator.

    Fits unit and year fixed effects (and, with trend=True, a linear trend per unit) on
    untreated observations only, predicts the untreated outcome for treated observations up to
    ``last``, and averages the gaps. Returns (overall effect, gaps by event time, units used)."""
    d = df[np.isfinite(df[y]) & (df.year <= last)].copy()
    d["treated"] = d.year >= d.g
    u0 = d[~d.treated]
    need = 2 if trend else 1
    counts = u0.groupby(unit).year.nunique()
    ok = counts.index[counts >= need]
    d = d[d[unit].isin(ok)]
    u0 = d[~d.treated]
    years0 = set(u0.year)
    d = d[d.year.isin(years0)]
    units = sorted(d[unit].unique())
    years = sorted(years0)
    ui = {k: i for i, k in enumerate(units)}
    yi = {k: i for i, k in enumerate(years)}

    def design(part):
        n = len(part)
        U = np.zeros((n, len(units)))
        U[np.arange(n), part[unit].map(ui).to_numpy()] = 1
        Y = np.zeros((n, len(years) - 1))
        idx = part.year.map(yi).to_numpy()
        has = idx > 0
        Y[np.arange(n)[has], idx[has] - 1] = 1
        mats = [U, Y]
        if trend:
            mats.append(U * (part.year - BASE_YEAR).to_numpy(float)[:, None])
        return np.hstack(mats)

    u0 = d[~d.treated]
    beta, *_ = np.linalg.lstsq(design(u0), u0[y].to_numpy(float), rcond=None)
    t1 = d[d.treated]
    gap = t1[y].to_numpy(float) - design(t1) @ beta
    by_e = pd.Series(gap, index=(t1.year - t1.g).to_numpy()).groupby(level=0).agg(["mean", "size"])
    return float(gap.mean()), by_e, int(t1[unit].nunique())


def resample(df, cluster, rng, relabel=("state",)):
    """One cluster bootstrap draw: sample clusters with replacement and relabel the copies so
    that fixed effects and trends treat each copy as a separate unit."""
    ids = df[cluster].unique()
    pick = rng.choice(ids, len(ids), replace=True)
    groups = dict(tuple(df.groupby(cluster)))
    parts = []
    for i, c in enumerate(pick):
        part = groups[c].copy()
        for col in set(relabel) | {cluster}:
            part[col] = part[col].astype(str) + f"#{i}"
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def bootstrap(df, fn, reps=499, seed=0, cluster="state", relabel=("state",)):
    """Percentile cluster bootstrap of a scalar or vector statistic fn(df)."""
    rng = np.random.default_rng(seed)
    return np.array([fn(resample(df, cluster, rng, relabel)) for _ in range(reps)])
