"""Survival Analysis for Data Scientists - Part 3.

The Cox proportional hazards model, fitted from scratch, applied to pricing
customer acquisition by lead segment for a simulated software subscription.

Everything the article reports is printed by ``python cox_model.py``.
The figures are produced by ``python figures.py``.

Requires numpy and scipy. If scikit-survival and statsmodels are installed,
the fitted coefficients and Schoenfeld residuals are also checked against them.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2, norm

SEED = 20261014

# ---------------------------------------------------------------------------
# 1. The business and its customer table
# ---------------------------------------------------------------------------
# A software subscription for small businesses, 36 months after launch.
# Customers arrive through three channels and come from three company sizes.
# At signup each chooses monthly billing (EUR 15 margin a month, can cancel at
# any time) or annual billing (EUR 150 margin a year, cancellation takes effect
# at the next renewal, so annual customers can only leave at months 12, 24, ...).
#
# Every customer has a latent time at which they decide to leave, with hazard
#     h(t | x) = h0(t) exp(x beta),   h0 Weibull with shape < 1 (falling)
# so proportional hazards holds exactly for the decision to leave. A monthly
# customer leaves at that time; an annual customer at the next renewal after it.

N_CUSTOMERS = 8000
SNAPSHOT = 36.0                  # months after launch
HORIZON = 24.0                   # business horizon for the value of a customer
MARGIN_MONTHLY = 15.0            # EUR per month, monthly billing
MARGIN_ANNUAL = 150.0            # EUR per year, annual billing
RENEWAL = 12.0                   # months between annual renewals
SHAPE, SCALE = 0.8, 30.0         # Weibull baseline: H0(t) = (t / SCALE) ** SHAPE

CHANNELS = ("search", "social", "referral")
CHANNEL_SHARE = np.array([0.5, 0.3, 0.2])
SIZES = ("1-9", "10-49", "50+")  # employees
SIZE_MIX = np.array([[0.50, 0.35, 0.15],     # search
                     [0.70, 0.25, 0.05],     # social
                     [0.35, 0.40, 0.25]])    # referral
ANNUAL_SHARE = np.array([0.15, 0.35, 0.55])  # by company size

COLUMNS = ("social", "referral", "size 10-49", "size 50+", "annual")
BETA_TRUE = np.array([0.40, -0.35, -0.25, -0.50, -0.40])
# The last coefficient acts on the latent decision of annual customers; the
# observed churn of annual customers is not proportional to that of monthly ones.


def design(channel, size, annual):
    """Columns: social, referral, size 10-49, size 50+, annual."""
    return np.column_stack([channel == 1, channel == 2, size == 1, size == 2,
                            annual == 1]).astype(float)


def simulate_table(rng, n=N_CUSTOMERS, snapshot=SNAPSHOT):
    signup = rng.uniform(0.0, snapshot, n)
    channel = rng.choice(3, size=n, p=CHANNEL_SHARE)
    u = rng.uniform(size=n)
    size = (u[:, None] > np.cumsum(SIZE_MIX[channel], axis=1)[:, :2]).sum(axis=1)
    annual = (rng.uniform(size=n) < ANNUAL_SHARE[size]).astype(int)
    X = design(channel, size, annual)
    lp = X @ BETA_TRUE
    decide = SCALE * (rng.exponential(1.0, n) * np.exp(-lp)) ** (1.0 / SHAPE)
    leave = np.where(annual == 1, np.ceil(decide / RENEWAL) * RENEWAL, decide)
    follow_up = snapshot - signup
    time = np.minimum(leave, follow_up)
    event = (leave <= follow_up).astype(int)
    return {"signup": signup, "channel": channel, "size": size, "annual": annual,
            "X": X, "time": time, "event": event}


def true_survival(t, channel, size, annual):
    """Probability of still being a customer at t months."""
    x = design(np.atleast_1d(channel), np.atleast_1d(size), np.atleast_1d(annual))[0]
    lp = x @ BETA_TRUE
    t = np.asarray(t, float)
    if annual:
        t = np.floor(t / RENEWAL + 1e-12) * RENEWAL
    return np.exp(-((t / SCALE) ** SHAPE) * np.exp(lp))


def true_value(channel, size, annual, horizon=HORIZON):
    """Expected margin per customer over the horizon."""
    if annual:
        renewals = np.arange(0.0, horizon, RENEWAL)
        return MARGIN_ANNUAL * true_survival(renewals, channel, size, 1).sum()
    grid = np.linspace(0.0, horizon, 20001)
    return MARGIN_MONTHLY * float(np.trapezoid(true_survival(grid, channel, size, 0), grid))


def true_segment_value(channel, size):
    a = ANNUAL_SHARE[size]
    return (1 - a) * true_value(channel, size, 0) + a * true_value(channel, size, 1)


# ---------------------------------------------------------------------------
# 2. The Cox model, from scratch
# ---------------------------------------------------------------------------

def _risk_sets(X, time, event, beta, ties):
    """Quantities needed at each event time of one stratum. Returns the event
    rows, the expanded (Efron) denominators and their first two moments."""
    order = np.argsort(time, kind="stable")
    t, e, x = time[order], event[order], X[order]
    w = np.exp(x @ beta)
    R = np.cumsum(w[::-1])[::-1]
    RX = np.cumsum((w[:, None] * x)[::-1], axis=0)[::-1]
    RXX = np.cumsum((w[:, None, None] * x[:, :, None] * x[:, None, :])[::-1], axis=0)[::-1]
    ev = np.flatnonzero(e == 1)
    ut, inv, d = np.unique(t[ev], return_inverse=True, return_counts=True)
    first = np.searchsorted(t, ut, side="left")
    m = len(ut)
    starts = np.r_[0, np.cumsum(d)[:-1]]                 # event rows are sorted by time
    we, xe = w[ev], x[ev]
    Dw = np.add.reduceat(we, starts)
    DX = np.add.reduceat(we[:, None] * xe, starts, axis=0)
    DXX = np.add.reduceat(we[:, None, None] * xe[:, :, None] * xe[:, None, :], starts, axis=0)
    j = np.repeat(np.arange(m), d)
    if ties == "efron":
        start = np.repeat(np.cumsum(d) - d, d)
        frac = (np.arange(len(j)) - start) / np.repeat(d, d)
    else:
        frac = np.zeros(len(j))
    phi = R[first][j] - frac * Dw[j]
    A = RX[first][j] - frac[:, None] * DX[j]
    B = RXX[first][j] - frac[:, None, None] * DXX[j]
    return {"x_ev": x[ev], "t_ev": t[ev], "rows": order[ev], "inv": inv, "d": d, "ut": ut,
            "j": j, "phi": phi, "A": A, "B": B}


def _loglik_grad_hess(X, time, event, strata, beta, ties):
    ll, grad, hess = 0.0, np.zeros(len(beta)), np.zeros((len(beta), len(beta)))
    for s in np.unique(strata):
        k = strata == s
        r = _risk_sets(X[k], time[k], event[k], beta, ties)
        mean = r["A"] / r["phi"][:, None]
        ll += float((r["x_ev"] @ beta).sum() - np.log(r["phi"]).sum())
        grad += r["x_ev"].sum(axis=0) - mean.sum(axis=0)
        hess -= (r["B"] / r["phi"][:, None, None]).sum(axis=0) - mean.T @ mean
    return ll, grad, hess


def cox_fit(X, time, event, strata=None, ties="efron", tol=1e-10, max_iter=50):
    """Maximise the Cox partial likelihood by Newton-Raphson."""
    strata = np.zeros(len(time), int) if strata is None else np.asarray(strata)
    beta = np.zeros(X.shape[1])
    ll0, grad, hess = _loglik_grad_hess(X, time, event, strata, beta, ties)
    ll = ll0
    for it in range(max_iter):
        step = np.linalg.solve(-hess, grad)
        for _ in range(30):                              # step halving if needed
            new = beta + step
            ll_new, grad_new, hess_new = _loglik_grad_hess(X, time, event, strata, new, ties)
            if ll_new >= ll - 1e-12:
                break
            step /= 2
        converged = abs(ll_new - ll) < tol
        beta, ll, grad, hess = new, ll_new, grad_new, hess_new
        if converged:
            break
    cov = np.linalg.inv(-hess)
    return {"beta": beta, "cov": cov, "se": np.sqrt(np.diag(cov)), "loglik": ll,
            "loglik0": ll0, "iterations": it + 1, "ties": ties, "strata": strata}


def baseline_cumhaz(fit, X, time, event):
    """Breslow (or Efron) estimate of the baseline cumulative hazard, per stratum."""
    out = {}
    for s in np.unique(fit["strata"]):
        k = fit["strata"] == s
        r = _risk_sets(X[k], time[k], event[k], fit["beta"], fit["ties"])
        steps = np.bincount(r["j"], weights=1.0 / r["phi"], minlength=len(r["ut"]))
        out[s] = (r["ut"], np.cumsum(steps))
    return out


def predicted_survival(t, x, base, beta):
    ut, H = base
    idx = np.searchsorted(ut, np.asarray(t, float), side="right") - 1
    Ht = np.where(idx >= 0, H[np.maximum(idx, 0)], 0.0)
    return np.exp(-Ht * np.exp(x @ beta))


def step_rmst(times, surv, horizon):
    """Area up to `horizon` under a right-continuous step function that starts at 1."""
    inside = times <= horizon
    knots = np.r_[0.0, times[inside], horizon]
    levels = np.r_[1.0, surv[inside]]
    return float((np.diff(knots) * levels).sum())


def predicted_value(x, base, beta, annual, horizon=HORIZON):
    """Expected margin over the horizon under a fitted curve: EUR 15 for every month
    under the curve for monthly billing; EUR 150 at signup and at each renewal the
    customer is predicted to reach for annual billing."""
    if annual:
        renewals = np.arange(0.0, horizon, RENEWAL)
        return MARGIN_ANNUAL * float(predicted_survival(renewals, x, base, beta).sum())
    ut, H = base
    surv = np.exp(-H * np.exp(x @ beta))
    return MARGIN_MONTHLY * step_rmst(ut, surv, horizon)


def schoenfeld(fit, X, time, event):
    """Schoenfeld residuals: covariates of each customer who churns minus the
    risk-set average at that time (Efron-adjusted for ties)."""
    times, resid, rows = [], [], []
    for s in np.unique(fit["strata"]):
        k = fit["strata"] == s
        r = _risk_sets(X[k], time[k], event[k], fit["beta"], fit["ties"])
        mean_rows = r["A"] / r["phi"][:, None]
        m = len(r["ut"])
        mean_t = np.add.reduceat(mean_rows, np.r_[0, np.cumsum(r["d"])[:-1]], axis=0) / r["d"][:, None]
        times.append(r["t_ev"])
        resid.append(r["x_ev"] - mean_t[r["inv"]])
        rows.append(np.flatnonzero(k)[r["rows"]])
    return np.concatenate(times), np.vstack(resid), np.concatenate(rows)


def ph_test(fit, X, time, event):
    """Grambsch-Therneau test of proportional hazards: correlation between the
    scaled Schoenfeld residuals and the rank of the event time."""
    t, r, _ = schoenfeld(fit, X, time, event)
    d = len(t)
    g = _average_rank(t)
    g = g - g.mean()
    cov = fit["cov"]
    U = g @ r
    per = d * (cov @ U) ** 2 / (np.diag(cov) * (g @ g))
    glob = d * U @ cov @ U / (g @ g)
    return {"chi2": per, "p": chi2.sf(per, 1), "global_chi2": float(glob),
            "global_p": float(chi2.sf(glob, len(U))),
            "scaled": d * r @ cov + fit["beta"], "times": t}


def _average_rank(t):
    order = np.argsort(t, kind="stable")
    ranks = np.empty(len(t))
    ranks[order] = np.arange(1, len(t) + 1)
    uniq, inv = np.unique(t, return_inverse=True)
    sums = np.bincount(inv, weights=ranks)
    counts = np.bincount(inv)
    return (sums / counts)[inv]


def kaplan_meier(time, event):
    order = np.argsort(time, kind="stable")
    t, e = time[order], event[order]
    uniq, first = np.unique(t, return_index=True)
    deaths = np.add.reduceat(e, first)
    at_risk = len(t) - first
    keep = deaths > 0
    surv = np.cumprod(1.0 - deaths[keep] / at_risk[keep])
    return uniq[keep], surv, float(t.max())


# ---------------------------------------------------------------------------
# 3. Segment values: three ways
# ---------------------------------------------------------------------------

def segment_values(tab, fits):
    """Value of a customer from each lead segment (channel x size), averaging
    over that segment's observed mix of billing plans."""
    X, time, event = tab["X"], tab["time"], tab["event"]
    out = {name: np.zeros((3, 3)) for name in ("km", "cox", "strat")}
    plan = {name: np.full((3, 3, 2), np.nan) for name in ("km", "cox", "strat")}
    short = 0
    bases = {"cox": baseline_cumhaz(fits["cox"], X, time, event),
             "strat": baseline_cumhaz(fits["strat"], X[:, :4], time, event)}
    for c in range(3):
        for s in range(3):
            seg = (tab["channel"] == c) & (tab["size"] == s)
            share = tab["annual"][seg].mean() if seg.any() else ANNUAL_SHARE[s]
            for name in ("km", "cox", "strat"):
                total = 0.0
                for a in (0, 1):
                    weight = share if a else 1 - share
                    if name == "km":
                        cell = seg & (tab["annual"] == a)
                        if cell.sum() == 0:
                            continue
                        kt, ks, last = kaplan_meier(time[cell], event[cell])
                        short += last < HORIZON
                        if a:
                            renewals = np.arange(0.0, HORIZON, RENEWAL)
                            idx = np.searchsorted(kt, renewals, side="right") - 1
                            value = MARGIN_ANNUAL * np.where(idx >= 0, ks[np.maximum(idx, 0)], 1.0).sum()
                        else:
                            value = MARGIN_MONTHLY * step_rmst(kt, ks, HORIZON)
                    elif name == "cox":
                        x = design(np.array([c]), np.array([s]), np.array([a]))[0]
                        value = predicted_value(x, bases["cox"][0], fits["cox"]["beta"], a)
                    else:
                        x = design(np.array([c]), np.array([s]), np.array([a]))[0][:4]
                        value = predicted_value(x, bases["strat"][a], fits["strat"]["beta"], a)
                    plan[name][c, s, a] = value
                    total += weight * value
                out[name][c, s] = total
    return out, short, plan


def fit_all(tab, ties="efron"):
    X, time, event = tab["X"], tab["time"], tab["event"]
    return {"cox": cox_fit(X, time, event, ties=ties),
            "strat": cox_fit(X[:, :4], time, event, strata=tab["annual"], ties=ties)}


def replicate(rng, n_rep):
    truth = np.array([[true_segment_value(c, s) for s in range(3)] for c in range(3)])
    truth_plan = np.array([[[true_value(c, s, a) for a in (0, 1)] for s in range(3)] for c in range(3)])
    errors = {name: [] for name in ("km", "cox", "strat")}
    plan_errors = {name: [] for name in ("km", "cox", "strat")}
    coefs = {"cox": [], "strat": [], "strat_breslow": []}
    ph_p = {"cox": [], "strat": []}
    covers = {"strat": []}
    short_total = 0
    for _ in range(n_rep):
        tab = simulate_table(rng)
        fits = fit_all(tab)
        fits_b = cox_fit(tab["X"][:, :4], tab["time"], tab["event"], strata=tab["annual"], ties="breslow")
        vals, short, plan = segment_values(tab, fits)
        short_total += short
        for name in errors:
            errors[name].append(vals[name] - truth)
            plan_errors[name].append(plan[name] - truth_plan)
        coefs["cox"].append(fits["cox"]["beta"])
        coefs["strat"].append(fits["strat"]["beta"])
        coefs["strat_breslow"].append(fits_b["beta"])
        for name in ("cox", "strat"):
            test = ph_test(fits[name], tab["X"] if name == "cox" else tab["X"][:, :4],
                           tab["time"], tab["event"])
            ph_p[name].append(np.r_[test["p"], test["global_p"]])
        b, se = fits["strat"]["beta"], fits["strat"]["se"]
        covers["strat"].append(np.abs(b - BETA_TRUE[:4]) <= 1.959964 * se)
    return truth, {k: np.array(v) for k, v in errors.items()}, {k: np.array(v) for k, v in plan_errors.items()}, \
        {k: np.array(v) for k, v in coefs.items()}, {k: np.array(v) for k, v in ph_p.items()}, \
        {k: np.array(v) for k, v in covers.items()}, short_total


def run_all(n_rep=500):
    """The example table, its two fits and the repeated tables, as in main()."""
    rng = np.random.default_rng(SEED)
    tab = simulate_table(rng)
    fits = fit_all(tab)
    return tab, fits, replicate(rng, n_rep)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def below(x, tol=1e-8):
    """Agreement with a library, reported against a tolerance: the exact size of
    differences this small depends on the machine's floating-point libraries."""
    return f"below {tol:.0e}" if x < tol else f"{x:.1e}, above {tol:.0e}"


def main(n_rep=500):
    rng = np.random.default_rng(SEED)
    tab = simulate_table(rng)
    X, time, event = tab["X"], tab["time"], tab["event"]
    annual = tab["annual"]

    print("=" * 72)
    print("1. THE TRUE PROCESS AND ONE CUSTOMER TABLE")
    print("=" * 72)
    print(f"Latent hazard of deciding to leave: Weibull baseline, shape {SHAPE}, scale {SCALE:g} months; "
          "hazard ratios " + ", ".join(f"{c} {np.exp(b):.3f}" for c, b in zip(COLUMNS, BETA_TRUE)))
    for a, plan in ((0, "monthly"), (1, "annual")):
        print(f"Search, 1-9 employees, {plan}: still a customer at 6 months {true_survival(6.0, 0, 0, a):.3f}, "
              f"12 {true_survival(12.0, 0, 0, a):.3f}, 24 {true_survival(24.0, 0, 0, a):.3f}; "
              f"value over {HORIZON:g} months EUR {true_value(0, 0, a):.2f}")
    print("True value of a customer by lead segment (EUR, at that segment's mix of plans):")
    for c, ch in enumerate(CHANNELS):
        print(f"  {ch:9s} " + "  ".join(f"{SIZES[s]:>5s}: {true_segment_value(c, s):6.2f}" for s in range(3)))
    print(f"Customers {len(time):,}; annual {annual.sum():,}; churned {event.sum():,} ({pct(event.mean())}): "
          f"monthly {event[annual == 0].sum():,}, annual {event[annual == 1].sum():,}")
    ut, counts = np.unique(time[event == 1], return_counts=True)
    print(f"Distinct churn times {len(ut):,}; annual cancellations at month 12: {counts[np.isclose(ut, 12)].sum():,}, "
          f"at month 24: {counts[np.isclose(ut, 24)].sum():,}")
    for c, ch in enumerate(CHANNELS):
        n_seg = [int(((tab["channel"] == c) & (tab["size"] == s)).sum()) for s in range(3)]
        print(f"  {ch:9s} customers by size: " + ", ".join(f"{SIZES[s]} {n_seg[s]:,}" for s in range(3)))
    cells = [(int(((tab["channel"] == c) & (tab["size"] == s) & (annual == a)).sum()), c, s, a)
             for c in range(3) for s in range(3) for a in (0, 1)]
    m, c, s, a = min(cells)
    print(f"Smallest cell of segment and plan: {CHANNELS[c]}, {SIZES[s]}, {'annual' if a else 'monthly'}, "
          f"{m} customers; median cell {int(np.median([x[0] for x in cells]))}")

    print()
    print("=" * 72)
    print("2. THE COX MODEL, FITTED FROM SCRATCH (BILLING PLAN AS A COVARIATE)")
    print("=" * 72)
    fits = fit_all(tab)
    f = fits["cox"]
    print(f"Efron ties: {f['iterations']} Newton steps; log partial likelihood {f['loglik0']:.1f} at beta = 0, "
          f"{f['loglik']:.1f} at the maximum")
    for i, col in enumerate(COLUMNS):
        lo, hi = np.exp(f["beta"][i] + np.array([-1, 1]) * 1.959964 * f["se"][i])
        print(f"  {col:11s} beta {f['beta'][i]:+.3f} (se {f['se'][i]:.3f})  hazard ratio {np.exp(f['beta'][i]):.3f} "
              f"[{lo:.3f}, {hi:.3f}]  true {np.exp(BETA_TRUE[i]):.3f}")
    fb_all = cox_fit(X, time, event, ties="breslow")
    print("Breslow ties: hazard ratios " + ", ".join(f"{c} {np.exp(b):.3f}" for c, b in zip(COLUMNS, fb_all["beta"])))
    try:
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        y = np.array(list(zip(event.astype(bool), time)), dtype=[("e", bool), ("t", float)])
        diffs = [np.max(np.abs(CoxPHSurvivalAnalysis(ties=ties, alpha=0.0).fit(X, y).coef_ - ours["beta"]))
                 for ties, ours in (("breslow", fb_all), ("efron", f))]
        print(f"Check against scikit-survival: largest coefficient difference, Breslow {below(diffs[0])}, "
              f"Efron {below(diffs[1])}")
    except ImportError:
        print("Check against scikit-survival: skipped (not installed)")

    print()
    print("=" * 72)
    print("3. CHECKING PROPORTIONAL HAZARDS, AND STRATIFYING BY PLAN")
    print("=" * 72)
    test = ph_test(f, X, time, event)
    print(f"Plan as a covariate: global chi-square {test['global_chi2']:.1f} on {len(COLUMNS)} df, "
          f"p {test['global_p']:.1e}")
    print("  " + "; ".join(f"{c} {x:.1f} (p {p:.2g})" for c, x, p in zip(COLUMNS, test["chi2"], test["p"])))
    t_s, r_s, rows_s = schoenfeld(f, X, time, event)
    at12, at24 = np.isclose(t_s, 12.0), np.isclose(t_s, 24.0)
    other = ~(at12 | at24)
    print(f"Schoenfeld residuals, annual column: mean at month 12 {r_s[at12, 4].mean():+.3f}, at month 24 "
          f"{r_s[at24, 4].mean():+.3f}, at all other churn times {r_s[other, 4].mean():+.3f}; "
          f"largest column sum {below(np.max(np.abs(r_s.sum(axis=0))))}")
    sc = test["scaled"]
    print(f"Scaled residuals (log hazard ratio scale), annual: month 12 {sc[at12, 4].mean():+.2f}, "
          f"month 24 {sc[at24, 4].mean():+.2f}, other times {sc[other, 4].mean():+.2f}")
    fs = fits["strat"]
    print(f"Stratified by plan (Efron ties): {fs['iterations']} Newton steps")
    for i, col in enumerate(COLUMNS[:4]):
        lo, hi = np.exp(fs["beta"][i] + np.array([-1, 1]) * 1.959964 * fs["se"][i])
        print(f"  {col:11s} beta {fs['beta'][i]:+.3f} (se {fs['se'][i]:.3f})  hazard ratio {np.exp(fs['beta'][i]):.3f} "
              f"[{lo:.3f}, {hi:.3f}]  true {np.exp(BETA_TRUE[i]):.3f}")
    test_s = ph_test(fs, X[:, :4], time, event)
    print(f"Stratified: global chi-square {test_s['global_chi2']:.1f} on 4 df, p {test_s['global_p']:.2f}; "
          + "; ".join(f"{c} p {p:.2f}" for c, p in zip(COLUMNS, test_s["p"])))
    fb = cox_fit(X[:, :4], time, event, strata=annual, ties="breslow")
    print("Stratified, Breslow ties: hazard ratios " + ", ".join(f"{c} {np.exp(b):.3f}" for c, b in zip(COLUMNS, fb["beta"])))
    try:
        import inspect
        import statsmodels.api as sm
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        res = sm.PHReg(time, X[:, :4], status=event, strata=annual, ties="efron").fit()
        res_b = sm.PHReg(time, X[:, :4], status=event, strata=annual, ties="breslow").fit()
        print(f"Check against statsmodels (stratified): largest difference, coefficients Efron "
              f"{below(np.max(np.abs(res.params - fs['beta'])))}, Breslow {below(np.max(np.abs(res_b.params - fb['beta'])))}; "
              f"standard errors {below(np.max(np.abs(res.bse - fs['se'])))}")
        sr = sm.PHReg(time, X, status=event, ties="breslow").fit().schoenfeld_residuals
        _, r_b, rows_b = schoenfeld(fb_all, X, time, event)
        print(f"Check against statsmodels (Breslow, plan as a covariate): largest difference in Schoenfeld "
              f"residuals {below(np.max(np.abs(sr[rows_b] - r_b)))}")
        defaults = (inspect.signature(CoxPHSurvivalAnalysis).parameters["ties"].default,
                    inspect.signature(sm.PHReg).parameters["ties"].default)
        print(f"Default handling of ties: scikit-survival '{defaults[0]}', statsmodels '{defaults[1]}'")
    except ImportError:
        print("Check against statsmodels: skipped (not installed)")

    print()
    print("=" * 72)
    print("4. FROM HAZARD RATIOS TO MONTHS AND EUROS (STRATIFIED MODEL)")
    print("=" * 72)
    base = baseline_cumhaz(fs, X[:, :4], time, event)
    base_all = baseline_cumhaz(f, X, time, event)[0]
    for s in (0, 2):
        for a in (0, 1):
            x_s = design(np.array([0]), np.array([s]), np.array([a]))[0][:4]
            x_r = design(np.array([2]), np.array([s]), np.array([a]))[0][:4]
            rate = MARGIN_ANNUAL / RENEWAL if a else MARGIN_MONTHLY
            est = (predicted_value(x_r, base[a], fs["beta"], a) - predicted_value(x_s, base[a], fs["beta"], a)) / rate
            tru = (true_value(2, s, a) - true_value(0, s, a)) / rate
            print(f"Referral vs search, {SIZES[s]} employees, {'annual' if a else 'monthly'}: "
                  f"{est:+.2f} paid months in the first {HORIZON:g} (true {tru:+.2f}); "
                  f"EUR {rate * est:+.2f} (true {rate * tru:+.2f})")
    print("Search, 10-49 employees, still a customer at 6 / 12 / 18 / 24 months: true, plan as covariate, stratified")
    for a in (0, 1):
        x_full = design(np.array([0]), np.array([1]), np.array([a]))[0]
        tr = [true_survival(m, 0, 1, a) for m in (6, 12, 18, 24)]
        pc = [predicted_survival(m, x_full, base_all, f["beta"]) for m in (6, 12, 18, 24)]
        ps = [predicted_survival(m, x_full[:4], base[a], fs["beta"]) for m in (6, 12, 18, 24)]
        print(f"  {'annual ' if a else 'monthly'}: " + " / ".join(f"{v:.3f}" for v in tr) + "; "
              + " / ".join(f"{float(v):.3f}" for v in pc) + "; " + " / ".join(f"{float(v):.3f}" for v in ps))
    vals, short, plan_vals = segment_values(tab, fits)
    truth = np.array([[true_segment_value(c, s) for s in range(3)] for c in range(3)])
    print("Value per customer by lead segment, this table (EUR): true / Kaplan-Meier per cell / plan as covariate / stratified")
    for c, ch in enumerate(CHANNELS):
        for s in range(3):
            print(f"  {ch:9s} {SIZES[s]:>5s}: {truth[c, s]:7.2f} / {vals['km'][c, s]:7.2f} / "
                  f"{vals['cox'][c, s]:7.2f} / {vals['strat'][c, s]:7.2f}")

    print()
    print("=" * 72)
    print(f"5. {n_rep:,} SIMULATED CUSTOMER TABLES")
    print("=" * 72)
    truth, errors, plan_errors, coefs, ph_p, covers, short_total = replicate(rng, n_rep)
    for name, label in (("km", "Kaplan-Meier per segment and plan"), ("cox", "Cox, plan as a covariate"),
                        ("strat", "Cox stratified by plan")):
        e = errors[name]
        rmse = np.sqrt((e ** 2).mean(axis=0))
        lo, hi = np.percentile(e, [2.5, 97.5], axis=0)
        print(f"{label}: mean error by segment from {e.mean(axis=0).min():+.2f} to {e.mean(axis=0).max():+.2f} EUR; "
              f"root mean squared error averaged over segments {rmse.mean():.2f}, worst segment {rmse.max():.2f}")
        for c, ch in enumerate(CHANNELS):
            print(f"    {ch:9s} " + "  ".join(
                f"{SIZES[s]:>5s}: mean {e[:, c, s].mean():+5.2f} rmse {rmse[c, s]:5.2f} "
                f"[{lo[c, s]:+6.2f}, {hi[c, s]:+6.2f}]" for s in range(3)))
        b = plan_errors[name].mean(axis=0)
        print(f"    mean error by plan: monthly from {b[..., 0].min():+.2f} to {b[..., 0].max():+.2f}; "
              f"annual from {b[..., 1].min():+.2f} to {b[..., 1].max():+.2f}")
    print(f"Kaplan-Meier cells whose curve ends before month {HORIZON:g}: {short_total} of {n_rep * 18}")
    print("Mean hazard ratios: " + "; ".join(
        f"{label} " + ", ".join(f"{c} {np.exp(b):.3f}" for c, b in zip(COLUMNS, coefs[name].mean(axis=0)))
        for name, label in (("cox", "plan as a covariate"), ("strat", "stratified (Efron)"),
                            ("strat_breslow", "stratified (Breslow)"))))
    print("Coverage of 95% intervals, stratified (Efron): " + ", ".join(
        f"{c} {pct(v)}" for c, v in zip(COLUMNS, covers["strat"].mean(axis=0))))
    for name, cols, label in (("cox", COLUMNS, "plan as a covariate"), ("strat", COLUMNS[:4], "stratified")):
        rej = (ph_p[name] < 0.05).mean(axis=0)
        print(f"Proportional-hazards test significant at 5%, {label}: " + ", ".join(
            f"{c} {pct(r)}" for c, r in zip(list(cols) + ["global"], rej)))
        print(f"  at least one of social, referral, 10-49, 50+ significant: "
              f"{pct((ph_p[name][:, :4] < 0.05).any(axis=1).mean())}")


if __name__ == "__main__":
    main()
