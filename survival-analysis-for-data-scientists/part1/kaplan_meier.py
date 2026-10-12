"""Survival Analysis for Data Scientists - Part 1.

Time to event, censoring and the Kaplan-Meier estimator, applied to a
question a subscription business asks of its customer table: how much can we
pay to acquire a customer?

Everything the article reports is printed by ``python kaplan_meier.py``.
The figures are produced by ``python figures.py``.

Requires numpy. If scikit-survival is installed, the Kaplan-Meier estimate is
also checked against its implementation.
"""

from __future__ import annotations

import numpy as np

SEED = 20261012

# ---------------------------------------------------------------------------
# 1. The customers
# ---------------------------------------------------------------------------
# Every customer has a constant monthly churn rate of their own, drawn from a
# gamma distribution with shape ALPHA and rate BETA. The population survival
# function is then S(t) = (1 + t / BETA) ** -ALPHA and its hazard,
# ALPHA / (BETA + t), falls with time even though no individual's does.

ALPHA, BETA = 1.5, 25.0
LAUNCH_TO_SNAPSHOT = 36.0         # months of acquisition before the snapshot
MARGIN = 15.0                     # monthly gross margin per customer, in euros
HORIZON = 24.0                    # payback horizon for the acquisition budget


def true_survival(t):
    return (1.0 + np.asarray(t, float) / BETA) ** -ALPHA


def true_hazard(t):
    return ALPHA / (BETA + np.asarray(t, float))


def true_rmst(tau):
    """Restricted mean survival time: the area under S(t) from 0 to tau."""
    return BETA / (1.0 - ALPHA) * ((1.0 + tau / BETA) ** (1.0 - ALPHA) - 1.0)


def simulate_customers(rng, n, pause_rate=0.0):
    """A snapshot of a customer table.

    Customers sign up uniformly over the LAUNCH_TO_SNAPSHOT months before the
    snapshot. Each has a churn rate, a lifetime drawn from it, and a time
    until the snapshot, after which nothing is known (administrative
    censoring). With pause_rate > 0, customers also pause their subscription
    at rate pause_rate times their own churn rate, and the analyst treats a
    pause as the end of the record (censoring), although paused customers
    go on to churn at their usual rate.

    Returns a dict with the signup month, the true lifetime, the observed
    time on book and the event indicator (1 = churn observed).
    """
    signup = rng.uniform(0.0, LAUNCH_TO_SNAPSHOT, n)
    rate = rng.gamma(ALPHA, 1.0 / BETA, n)
    lifetime = rng.exponential(1.0 / rate)
    follow_up = LAUNCH_TO_SNAPSHOT - signup
    end = follow_up
    if pause_rate > 0:
        pause = rng.exponential(1.0 / (pause_rate * rate))
        end = np.minimum(end, pause)
    time = np.minimum(lifetime, end)
    event = (lifetime <= end).astype(int)
    return {"signup": signup, "lifetime": lifetime, "time": time, "event": event,
            "follow_up": follow_up}


# ---------------------------------------------------------------------------
# 2. The Kaplan-Meier estimator
# ---------------------------------------------------------------------------

def kaplan_meier(time, event, z=1.959964):
    """Kaplan-Meier estimate of S(t) with Greenwood standard errors and
    pointwise log-log confidence intervals.

    Returns a dict of arrays indexed by the distinct event times: the time,
    the number at risk just before it, the number of events, S, its standard
    error and the lower and upper confidence limits.
    """
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    order = np.argsort(time, kind="stable")
    t, d = time[order], event[order]
    uniq, first, _ = np.unique(t, return_index=True, return_counts=True)
    deaths = np.add.reduceat(d, first)
    at_risk = len(t) - first
    keep = deaths > 0
    uniq, deaths, at_risk = uniq[keep], deaths[keep], at_risk[keep]

    surv = np.cumprod(1.0 - deaths / at_risk)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = deaths / (at_risk * (at_risk - deaths))
        green = np.cumsum(terms)                       # sum d / (n (n - d))
        se = surv * np.sqrt(green)
        theta_se = np.sqrt(green) / np.abs(np.log(surv))
        lower = surv ** np.exp(z * theta_se)
        upper = surv ** np.exp(-z * theta_se)
    return {"time": uniq, "at_risk": at_risk, "events": deaths, "surv": surv,
            "se": se, "lower": lower, "upper": upper}


def step_at(km, t, key="surv"):
    """Value of a Kaplan-Meier curve (a right-continuous step function) at t."""
    idx = np.searchsorted(km["time"], np.asarray(t, float), side="right") - 1
    values = np.r_[1.0, km[key]]
    return values[idx + 1]


def km_rmst(km, tau):
    """Area under the Kaplan-Meier curve from 0 to tau."""
    knots = np.r_[0.0, km["time"][km["time"] < tau], tau]
    levels = np.r_[1.0, km["surv"][km["time"] < tau]]
    return float(np.sum(np.diff(knots) * levels))


# ---------------------------------------------------------------------------
# 3. Three tempting shortcuts
# ---------------------------------------------------------------------------

def drop_censored(time, event, t):
    """Use only customers whose churn has been observed."""
    churned = time[event == 1]
    return np.mean(churned[:, None] > np.atleast_1d(t)[None, :], axis=0)


def censored_as_active(time, event, t):
    """Count every customer not yet seen to churn by t as still active at t."""
    churned_by = (event[:, None] == 1) & (time[:, None] <= np.atleast_1d(t)[None, :])
    return 1.0 - churned_by.mean(axis=0)


def fixed_horizon(customers, t):
    """Use only customers who signed up at least t months before the
    snapshot, so that everyone's status at t is known."""
    eligible = customers["follow_up"] >= t
    alive = customers["lifetime"][eligible] > t
    return float(alive.mean()), int(eligible.sum())


def rmst_shortcut(estimate, tau, grid=None):
    grid = np.linspace(0.0, tau, 2401) if grid is None else grid
    return float(np.trapezoid(estimate(grid), grid))


def fixed_horizon_rmst(customers, tau):
    """Average months active in the first tau months, among customers who
    signed up at least tau months before the snapshot."""
    eligible = customers["follow_up"] >= tau
    return float(np.minimum(customers["lifetime"][eligible], tau).mean()), int(eligible.sum())


# ---------------------------------------------------------------------------
# 4. Repeated snapshots
# ---------------------------------------------------------------------------

def replicate(rng, n_customers, n_rep=1000, tau=HORIZON, t_check=12.0):
    """Sampling distribution of each method's estimates of S(12) and of the
    restricted mean to tau, and the coverage of the Kaplan-Meier interval."""
    rows = []
    for _ in range(n_rep):
        c = simulate_customers(rng, n_customers)
        km = kaplan_meier(c["time"], c["event"])
        s12 = float(step_at(km, t_check))
        lo, hi = float(step_at(km, t_check, "lower")), float(step_at(km, t_check, "upper"))
        fh, _ = fixed_horizon(c, t_check)
        fh_rmst, _ = fixed_horizon_rmst(c, tau)
        rows.append((
            s12, fh,
            float(drop_censored(c["time"], c["event"], t_check)[0]),
            float(censored_as_active(c["time"], c["event"], t_check)[0]),
            km_rmst(km, tau), fh_rmst,
            rmst_shortcut(lambda g: drop_censored(c["time"], c["event"], g), tau),
            rmst_shortcut(lambda g: censored_as_active(c["time"], c["event"], g), tau),
            float(np.mean(c["time"])),
            lo <= true_survival(t_check) <= hi,
        ))
    return np.array(rows, dtype=float)


def snapshot_and_replications():
    """The snapshot of section 2 and the 1,000 snapshots of section 3, drawn
    in the same order as main() so that figures.py reproduces them."""
    rng = np.random.default_rng(SEED)
    customers = simulate_customers(rng, 5000)
    rep = replicate(rng, 5000)
    return customers, rep, rng


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def euros(months):
    return f"EUR {MARGIN * months:,.0f}"


def main():
    np.set_printoptions(precision=4, suppress=True)
    c, rep, rng = snapshot_and_replications()

    print("=" * 72)
    print("1. THE TRUE CHURN PROCESS")
    print("=" * 72)
    print(f"Individual churn rates ~ Gamma(shape {ALPHA}, rate {BETA}) per month; "
          f"mean rate {ALPHA / BETA:.3f}")
    print(f"S(t) = (1 + t/{BETA:g})^-{ALPHA}: S(6) {true_survival(6):.3f}, S(12) "
          f"{true_survival(12):.3f}, S(24) {true_survival(24):.3f}, S(36) {true_survival(36):.3f}")
    print(f"Median lifetime {BETA * (2 ** (1 / ALPHA) - 1):.1f} months; mean lifetime "
          f"{BETA / (ALPHA - 1):.1f} months")
    print(f"Population hazard: month 0 {true_hazard(0):.4f}, month 12 {true_hazard(12):.4f}, "
          f"month 24 {true_hazard(24):.4f} (falls by {100 * (1 - true_hazard(24) / true_hazard(0)):.0f}%)")
    print(f"Expected paid months in the first {HORIZON:g} (restricted mean): {true_rmst(HORIZON):.2f}")
    print(f"Break-even acquisition cost at EUR {MARGIN:g} margin per month: {euros(true_rmst(HORIZON))}")

    print()
    print("=" * 72)
    print("2. ONE SNAPSHOT OF 5,000 CUSTOMERS")
    print("=" * 72)
    n = len(c["time"])
    print(f"Customers {n:,}; churn observed {c['event'].sum():,} ({100 * c['event'].mean():.1f}%); "
          f"still active at the snapshot {n - c['event'].sum():,}")
    print(f"Customers who signed up at least 12 months before the snapshot: "
          f"{(c['follow_up'] >= 12).sum():,}; at least 24 months: {(c['follow_up'] >= 24).sum():,}")
    km = kaplan_meier(c["time"], c["event"])
    s12, lo12, hi12 = (float(step_at(km, 12, k)) for k in ("surv", "lower", "upper"))
    fh12, fh_n = fixed_horizon(c, 12)
    fh_r, fh_rn = fixed_horizon_rmst(c, HORIZON)
    rows = [
        ("truth", true_survival(12), true_rmst(HORIZON)),
        ("drop the censored", drop_censored(c["time"], c["event"], 12)[0],
         rmst_shortcut(lambda g: drop_censored(c["time"], c["event"], g), HORIZON)),
        ("censored as active", censored_as_active(c["time"], c["event"], 12)[0],
         rmst_shortcut(lambda g: censored_as_active(c["time"], c["event"], g), HORIZON)),
        ("fixed horizon", fh12, fh_r),
        ("Kaplan-Meier", s12, km_rmst(km, HORIZON)),
    ]
    print(f"{'method':22s} {'S(12)':>7s} {'paid months in 24':>18s} {'break-even CAC':>15s}")
    for name, s, r in rows:
        print(f"{name:22s} {s:7.3f} {r:18.2f} {euros(r):>15s}")
    print(f"Fixed horizon uses {fh_n:,} customers for S(12) and {fh_rn:,} for the 24-month figure")
    print(f"Kaplan-Meier 95% interval for S(12) (log-log): [{lo12:.3f}, {hi12:.3f}]; "
          f"Greenwood standard error {float(step_at(km, 12, 'se')):.4f}")
    print(f"Average time on book (ignores censoring): {c['time'].mean():.2f} months; "
          f"true mean lifetime {BETA / (ALPHA - 1):.1f}")
    print(f"Kaplan-Meier median lifetime: {km['time'][np.argmax(km['surv'] <= 0.5)]:.1f} months "
          f"(true {BETA * (2 ** (1 / ALPHA) - 1):.1f})")
    print("Customers still at risk (time on book at least m months) at m = 6, 12, 24, 35: "
          + ", ".join(f"{int((c['time'] >= m).sum()):,}" for m in (6, 12, 24, 35))
          + f"; last observed cancellation at month {c['time'][c['event'] == 1].max():.1f}")

    try:
        from sksurv.nonparametric import kaplan_meier_estimator
        t_sk, s_sk, ci_sk = kaplan_meier_estimator(c["event"].astype(bool), c["time"],
                                                   conf_type="log-log")
        mask = np.isin(t_sk, km["time"])
        diff_s = np.max(np.abs(s_sk[mask] - km["surv"]))
        ok = np.isfinite(km["lower"]) & np.isfinite(ci_sk[0][mask])
        diff_ci = np.max(np.abs(ci_sk[0][mask][ok] - km["lower"][ok]))
        print(f"Check against scikit-survival: largest difference in S {diff_s:.1e}, "
              f"in the lower confidence limit {diff_ci:.1e}")
    except ImportError:
        print("Check against scikit-survival: skipped (not installed)")

    print()
    print("=" * 72)
    print("3. 1,000 SNAPSHOTS OF 5,000 CUSTOMERS")
    print("=" * 72)
    names = ["Kaplan-Meier", "fixed horizon", "drop the censored", "censored as active"]
    truth_s, truth_r = true_survival(12), true_rmst(HORIZON)
    print(f"{'method':22s} {'S(12): mean':>12s} {'sd':>7s} {'paid months: mean':>18s} "
          f"{'sd':>6s} {'break-even CAC 95% range':>26s}")
    for i, name in enumerate(names):
        s, r = rep[:, i], rep[:, 4 + i]
        lo, hi = np.percentile(r, [2.5, 97.5])
        print(f"{name:22s} {s.mean():12.4f} {s.std(ddof=1):7.4f} {r.mean():18.2f} "
              f"{r.std(ddof=1):6.3f} {euros(lo):>12s} to {euros(hi)}")
    print(f"{'truth':22s} {truth_s:12.4f} {'':7s} {truth_r:18.2f}")
    sd_ratio_s = rep[:, 1].std(ddof=1) / rep[:, 0].std(ddof=1)
    sd_ratio_r = rep[:, 5].std(ddof=1) / rep[:, 4].std(ddof=1)
    print(f"Standard deviation, fixed horizon / Kaplan-Meier: S(12) {sd_ratio_s:.2f}, "
          f"paid months {sd_ratio_r:.2f}")
    print(f"Equivalent sample size: fixed horizon would need {sd_ratio_r ** 2:.1f} times "
          f"as many customers to match Kaplan-Meier on the 24-month figure")
    print(f"Coverage of the Kaplan-Meier 95% interval for S(12): {100 * rep[:, 9].mean():.1f}%")
    print(f"Average time on book across snapshots: {rep[:, 8].mean():.2f} months")

    print()
    print("=" * 72)
    print("4. WHEN CENSORING IS NOT INDEPENDENT OF CHURN")
    print("=" * 72)
    print("200 snapshots of 5,000 customers for each pause rate")
    for pause in (0.25, 0.5, 1.0):
        res = []
        for _ in range(200):
            cp = simulate_customers(rng, 5000, pause_rate=pause)
            kp = kaplan_meier(cp["time"], cp["event"])
            res.append((float(step_at(kp, 12)), km_rmst(kp, HORIZON),
                        1 - cp["event"].mean()))
        res = np.array(res)
        print(f"Pause rate {pause:.2f} x churn rate: Kaplan-Meier S(12) {res[:, 0].mean():.3f} "
              f"(truth {truth_s:.3f}), paid months {res[:, 1].mean():.2f} "
              f"(truth {truth_r:.2f}), break-even CAC {euros(res[:, 1].mean())} "
              f"(truth {euros(truth_r)}); censored share {100 * res[:, 2].mean():.1f}%")
    big = simulate_customers(rng, 200_000, pause_rate=0.5)
    kb = kaplan_meier(big["time"], big["event"])
    print(f"With 200,000 customers and pause rate 0.50: S(12) {float(step_at(kb, 12)):.3f}, "
          f"95% interval [{float(step_at(kb, 12, 'lower')):.3f}, {float(step_at(kb, 12, 'upper')):.3f}]")


if __name__ == "__main__":
    main()
