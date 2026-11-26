"""Competing-risks tools for the founder-exits cohort.

- outcomes: time from the chosen origin (first allotment of shares) to the first event after it;
- aalen_johansen: cumulative incidence of each cause, with optional case weights;
- bootstrap_cif: percentile intervals for cumulative incidence and group differences;
- cause_specific_cox: hazard ratios for one cause, the others treated as censoring (lifelines);
- fine_gray: subdistribution hazard ratios (Fine and Gray 1999) by weighted Newton-Raphson;
- states_at: where each company stands a fixed number of years after the origin, for companies
  followed that long (only administrative censoring, so these are plain proportions);
- balancing_weights: inverse-probability weights that balance covariates between two groups.
"""

import numpy as np
import pandas as pd

CAUSES = ["foreign_acquisition", "flip", "uk_acquisition", "offshore", "closure"]
ACQUISITIONS = ["foreign_acquisition", "flip", "uk_acquisition", "offshore"]
DAYS_PER_YEAR = 365.25


def _dates(s):
    return pd.to_datetime(s, errors="coerce")


def outcomes(df, entry="entry", causes=CAUSES, merge=None):
    """Add `time` (years from `entry`) and `cause` (first event after entry, or "censored").

    Companies with no `entry`, or with a change of control before it, are dropped: the origin is
    the first allotment while still independent. `merge` maps event names to analysis causes,
    e.g. {"flip": "foreign_acquisition"}; unmapped events keep their names.
    """
    merge = merge or {}
    out = df[_dates(df[entry]).notna()].copy()
    start = _dates(out[entry])
    censor = _dates(out["censor"])
    dates = {c: _dates(out[c]) for c in causes}
    before = np.zeros(len(out), dtype=bool)
    for c in ACQUISITIONS:
        if c in dates:
            before |= (dates[c] < start).to_numpy()
    out = out[~before]
    start, censor = start[~before], censor[~before]
    first = pd.Series(censor.to_numpy(), index=out.index)
    cause = pd.Series("censored", index=out.index, dtype=object)
    for c in causes:
        d = dates[c][~before]
        hit = d.notna() & (d >= start) & (d <= censor) & (d < first)
        first[hit] = d[hit]
        cause[hit] = merge.get(c, c)
    out["time"] = ((first - start).dt.days / DAYS_PER_YEAR).clip(lower=1 / DAYS_PER_YEAR / 2)
    out["cause"] = cause
    return out


def aalen_johansen(time, cause, causes, grid, weights=None):
    """Cumulative incidence of each cause at the times in `grid`.

    `cause` holds a cause name or "censored". Ties: events at a time are counted before
    censorings at the same time. Returns a DataFrame indexed by grid, one column per cause.
    """
    time = np.asarray(time, dtype=float)
    cause = np.asarray(cause, dtype=object)
    w = np.ones(len(time)) if weights is None else np.asarray(weights, dtype=float)
    order = np.argsort(time, kind="stable")
    time, cause, w = time[order], cause[order], w[order]
    uniq, first_idx = np.unique(time, return_index=True)
    at_risk = np.cumsum(w[::-1])[::-1][first_idx]
    surv_before = np.ones(len(uniq))
    d_total = np.zeros(len(uniq))
    inc = {}
    for c in causes:
        d = np.bincount(np.searchsorted(uniq, time[cause == c]), weights=w[cause == c], minlength=len(uniq))
        inc[c] = d
        d_total += d
    hazard_all = np.divide(d_total, at_risk, out=np.zeros_like(d_total), where=at_risk > 0)
    surv = np.cumprod(1 - hazard_all)
    surv_before[1:] = surv[:-1]
    pos = np.searchsorted(uniq, np.asarray(grid, dtype=float), side="right") - 1
    res = {}
    for c in causes:
        step = surv_before * np.divide(inc[c], at_risk, out=np.zeros_like(at_risk), where=at_risk > 0)
        cif = np.cumsum(step)
        res[c] = np.where(pos >= 0, cif[np.clip(pos, 0, None)], 0.0)
    return pd.DataFrame(res, index=pd.Index(grid, name="years"))


def bootstrap_cif(df, group, causes, horizons, b=500, seed=0, weights=None, levels=None):
    """Point estimates and 95% percentile intervals of CIF by group, and of differences from the first group.

    Resamples companies within each group. Returns long DataFrame with columns
    group, cause, years, estimate, low, high (group "A minus B" for differences).
    """
    rng = np.random.default_rng(seed)
    levels = levels or list(pd.unique(df[group]))
    parts = {g: df[df[group] == g] for g in levels}

    def est(frames):
        out = {}
        for g, f in frames.items():
            wt = None if weights is None else f[weights]
            out[g] = aalen_johansen(f["time"], f["cause"], causes, horizons, wt)
        return out

    point = est(parts)
    draws = {g: [] for g in levels}
    for _ in range(b):
        frames = {g: f.iloc[rng.integers(0, len(f), len(f))] for g, f in parts.items()}
        for g, v in est(frames).items():
            draws[g].append(v.to_numpy())
    rows = []
    for g in levels:
        arr = np.stack(draws[g])
        for i, c in enumerate(causes):
            for j, h in enumerate(horizons):
                lo, hi = np.percentile(arr[:, j, i], [2.5, 97.5])
                rows.append((g, c, h, point[g].iloc[j, i], lo, hi))
    ref = levels[0]
    for g in levels[1:]:
        diff = np.stack(draws[ref]) - np.stack(draws[g])
        for i, c in enumerate(causes):
            for j, h in enumerate(horizons):
                lo, hi = np.percentile(diff[:, j, i], [2.5, 97.5])
                rows.append((f"{ref} minus {g}", c, h, point[ref].iloc[j, i] - point[g].iloc[j, i], lo, hi))
    return pd.DataFrame(rows, columns=["group", "cause", "years", "estimate", "low", "high"])


def design(df, covariates, reference):
    """Dummy-coded design matrix; `reference` maps each categorical column to its reference level."""
    parts = []
    for c in covariates:
        if c in reference:
            present = sorted(df[c].astype(str).unique())
            ref = reference[c] if reference[c] in present else present[0]  # a subset may lack the reference
            levels = [lv for lv in present if lv != ref]
            for lv in levels:
                parts.append(pd.Series((df[c].astype(str) == lv).astype(float), name=f"{c}[{lv}]"))
        else:
            parts.append(df[c].astype(float).rename(c))
    x = pd.concat(parts, axis=1)
    return x.loc[:, x.std() > 0]


def cause_specific_cox(df, cause, covariates, reference):
    """Hazard ratios for one cause, other events censored. Returns lifelines' summary table."""
    from lifelines import CoxPHFitter

    x = design(df, covariates, reference)
    data = x.assign(time=df["time"].to_numpy(), event=(df["cause"] == cause).astype(int).to_numpy())
    cph = CoxPHFitter()
    cph.fit(data, duration_col="time", event_col="event")
    return cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].rename(
        columns={"exp(coef)": "hr", "exp(coef) lower 95%": "low", "exp(coef) upper 95%": "high"}
    )


def _censoring_left_limit(time, censored):
    """Kaplan-Meier of the censoring time evaluated just before each time: G(t-)."""
    order = np.argsort(time, kind="stable")
    t, c = time[order], censored[order]
    uniq, first_idx = np.unique(t, return_index=True)
    at_risk = len(t) - first_idx
    d = np.bincount(np.searchsorted(uniq, t[c]), minlength=len(uniq))
    g = np.cumprod(1 - d / at_risk)
    g_before = np.concatenate([[1.0], g[:-1]])

    def at(x):
        pos = np.searchsorted(uniq, x, side="left") - 1  # last censoring time strictly before x
        return np.where(pos >= 0, g[np.clip(pos, 0, None)], 1.0)

    return at, g_before


def fine_gray(time, cause, x, k, max_iter=60, tol=1e-10):
    """Subdistribution hazard regression for cause `k` (Fine and Gray 1999), Breslow ties.

    `cause` holds cause names or "censored". Subjects that failed from another cause stay in the
    risk set with weight G(t-)/G(T_i-), G the Kaplan-Meier of censoring. Returns coefficients and
    their model-based covariance (the inverse information; use a bootstrap for intervals).
    """
    time = np.asarray(time, dtype=float)
    cause = np.asarray(cause, dtype=object)
    x = np.asarray(x, dtype=float)
    n, p = x.shape
    order = np.argsort(time, kind="stable")
    time, cause, x = time[order], cause[order], x[order]
    event = cause == k
    competing = (cause != k) & (cause != "censored")
    g_at, _ = _censoring_left_limit(time, cause == "censored")
    g_own = g_at(time)  # G(T_i-)
    tk = np.unique(time[event])
    d = np.bincount(np.searchsorted(tk, time[event]), minlength=len(tk)).astype(float)
    xd = np.zeros((len(tk), p))
    np.add.at(xd, np.searchsorted(tk, time[event]), x[event])
    first_at = np.searchsorted(time, tk, side="left")  # index of first subject with T >= t_j
    g_tk = g_at(tk)
    beta = np.zeros(p)
    for _ in range(max_iter):
        e = np.exp(np.clip(x @ beta, -50, 50))
        ex = e[:, None] * x
        exx = ex[:, :, None] * x[:, None, :]
        # Risk set part 1: T_i >= t_j (reverse cumulative sums).
        r0 = np.cumsum(e[::-1])[::-1]
        r1 = np.cumsum(ex[::-1], axis=0)[::-1]
        r2 = np.cumsum(exx[::-1], axis=0)[::-1]
        # Part 2: competing failures before t_j, weighted G(t_j-)/G(T_i-) (forward cumulative sums).
        wc = np.where(competing, 1 / np.maximum(g_own, 1e-12), 0.0)
        c0 = np.concatenate([[0.0], np.cumsum(wc * e)])
        c1 = np.concatenate([np.zeros((1, p)), np.cumsum(wc[:, None] * ex, axis=0)])
        c2 = np.concatenate([np.zeros((1, p, p)), np.cumsum(wc[:, None, None] * exx, axis=0)])
        idx = np.minimum(first_at, n - 1)
        inside = first_at < n
        s0 = np.where(inside, r0[idx], 0.0) + g_tk * c0[first_at]
        s1 = np.where(inside[:, None], r1[idx], 0.0) + g_tk[:, None] * c1[first_at]
        s2 = np.where(inside[:, None, None], r2[idx], 0.0) + g_tk[:, None, None] * c2[first_at]
        mean = s1 / s0[:, None]
        score = (xd - d[:, None] * mean).sum(axis=0)
        info = (d[:, None, None] * (s2 / s0[:, None, None] - mean[:, :, None] * mean[:, None, :])).sum(axis=0)
        step = np.linalg.solve(info, score)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta, np.linalg.inv(info)


def states_at(df, years, causes, follow_on="next_allotment", entry="entry"):
    """State of each company `years` after entry, for companies whose entry is at least that long before censoring.

    States: each cause that happened by then, otherwise "independent, raised again" when a further
    allotment was filed by then, else "independent, no further allotment".
    """
    horizon = pd.to_timedelta(years * DAYS_PER_YEAR, unit="D")
    full = _dates(df[entry]) + horizon <= _dates(df["censor"])
    f = df[full].copy()
    st = f["cause"].where(f["time"] <= years, "independent")
    nxt = _dates(f[follow_on])
    raised = nxt.notna() & (nxt <= _dates(f[entry]) + horizon)
    quiet = np.where(raised, "independent, raised again", "independent, no further allotment")
    st = st.where(st != "independent", quiet)
    f["state"] = st
    return f


def balancing_weights(df, treat, covariates, reference, trim=0.01):
    """Inverse-probability (ATE) weights for a binary `treat` column, from a logistic model, trimmed."""
    import statsmodels.api as sm

    x = sm.add_constant(design(df, covariates, reference), has_constant="add")
    y = df[treat].astype(float)
    ps = sm.GLM(y, x, family=sm.families.Binomial()).fit().predict(x).to_numpy()
    lo, hi = np.quantile(ps, [trim, 1 - trim])
    ps = np.clip(ps, lo, hi)
    w = np.where(y == 1, 1 / ps, 1 / (1 - ps))
    return pd.Series(w, index=df.index), ps


def standardised_difference(df, col, treat, weights=None):
    """Weighted standardised mean difference of one numeric column between treat == 1 and 0."""
    w = np.ones(len(df)) if weights is None else np.asarray(weights)
    a, b = (df[treat] == 1).to_numpy(), (df[treat] == 0).to_numpy()
    v = df[col].astype(float).to_numpy()
    ma, mb = np.average(v[a], weights=w[a]), np.average(v[b], weights=w[b])
    va, vb = v[a].var(), v[b].var()
    return (ma - mb) / np.sqrt((va + vb) / 2) if va + vb > 0 else 0.0
