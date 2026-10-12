"""Survival Analysis for Data Scientists - Part 2.

Hazards, the Nelson-Aalen estimator and the log-rank test, applied to a
retention A/B test on the subscription product of Part 1.

Everything the article reports is printed by ``python hazards_logrank.py``.
The figures are produced by ``python figures.py``.

Requires numpy and scipy. If scikit-survival is installed, the Nelson-Aalen
estimate and the log-rank statistic are also checked against it.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

SEED = 20261013

# ---------------------------------------------------------------------------
# 1. The product and the experiment
# ---------------------------------------------------------------------------
# As in Part 1, every customer has a constant monthly churn rate of their own,
# drawn from a gamma distribution with shape ALPHA and rate BETA. Under the
# control onboarding, a customer with rate lam has cumulative hazard lam * t.
# The two treatments change each customer's cumulative hazard to lam * g(t):
#
#   "steady": g(t) = 0.8 t      every customer churns 20% more slowly, always
#   "delay":  g(t) = 0.5 t for t < 2, then 1 + 1.5 (t - 2)
#             churn is halved for two months and 50% faster afterwards
#
# New customers are randomised 50/50 as they sign up during the first
# ENROL_MONTHS months of the experiment, and the experiment is analysed at
# several dates, each customer being followed until that date.

ALPHA, BETA = 1.5, 25.0
ENROL_MONTHS = 3.0
SIGNUPS = 4500                   # customers enrolled, both arms together
LOOKS = (3.0, 6.0, 9.0, 12.0)    # months after the experiment starts
MARGIN = 15.0                    # monthly gross margin per customer, EUR
HORIZON = 12.0                   # business horizon for the decision, months


def g_control(t):
    return np.asarray(t, float)


def g_steady(t):
    return 0.8 * np.asarray(t, float)


def g_delay(t):
    t = np.asarray(t, float)
    return np.where(t < 2.0, 0.5 * t, 1.0 + 1.5 * (t - 2.0))


def g_delay_inverse(x):
    x = np.asarray(x, float)
    return np.where(x < 1.0, 2.0 * x, 2.0 + (x - 1.0) / 1.5)


SCENARIOS = {
    "control": (g_control, lambda x: x),
    "steady": (g_steady, lambda x: x / 0.8),
    "delay": (g_delay, g_delay_inverse),
}


def true_survival(t, arm="control"):
    g = SCENARIOS[arm][0]
    return (1.0 + g(t) / BETA) ** -ALPHA


def true_hazard(t, arm="control", dt=1e-5):
    t = np.asarray(t, float)
    g = SCENARIOS[arm][0]
    slope = (g(t + dt) - g(t)) / dt
    return ALPHA * slope / (BETA + g(t))


def true_rmst(tau, arm="control"):
    grid = np.linspace(0.0, tau, 20001)
    return float(np.trapezoid(true_survival(grid, arm), grid))


def simulate_experiment(rng, treatment, n=SIGNUPS):
    """Signup month, arm (1 = treatment) and lifetime of each customer."""
    signup = rng.uniform(0.0, ENROL_MONTHS, n)
    arm = rng.permutation(np.r_[np.zeros(n // 2, int), np.ones(n - n // 2, int)])
    rate = rng.gamma(ALPHA, 1.0 / BETA, n)
    unit = rng.exponential(1.0, n) / rate          # lam * g(T) = E  ->  g(T) = E / lam
    lifetime = np.where(arm == 1, SCENARIOS[treatment][1](unit), unit)
    return {"signup": signup, "arm": arm, "lifetime": lifetime}


def follow_up_at(experiment, look):
    """Months each customer has been followed by `look` months after the start."""
    return np.maximum(look - experiment["signup"], 0.0)


def observe(experiment, look):
    """What the customer table shows at `look` months after the start."""
    follow_up = follow_up_at(experiment, look)
    time = np.minimum(experiment["lifetime"], follow_up)
    event = (experiment["lifetime"] <= follow_up).astype(int)
    return time, event, experiment["arm"]


# ---------------------------------------------------------------------------
# 2. Estimators
# ---------------------------------------------------------------------------

def risk_table(time, event):
    """Distinct event times, number at risk just before each, number of events."""
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    order = np.argsort(time, kind="stable")
    t, d = time[order], event[order]
    uniq, first = np.unique(t, return_index=True)
    deaths = np.add.reduceat(d, first)
    at_risk = len(t) - first
    keep = deaths > 0
    return uniq[keep], at_risk[keep], deaths[keep]


def nelson_aalen(time, event):
    """Cumulative hazard H(t) = sum of d_j / n_j, with variance sum d_j / n_j^2."""
    t, n, d = risk_table(time, event)
    return {"time": t, "cumhaz": np.cumsum(d / n), "var": np.cumsum(d / n ** 2)}


def kaplan_meier(time, event):
    t, n, d = risk_table(time, event)
    return {"time": t, "at_risk": n, "events": d, "surv": np.cumprod(1.0 - d / n)}


def occurrence_exposure(time, event, edges):
    """Churn rate per customer-month in each interval [a, b): events in the
    interval divided by the customer-months at risk in it."""
    time = np.asarray(time, float)
    rates, lower, upper = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        exposure = np.clip(time, a, b) - a
        events = np.sum((event == 1) & (time >= a) & (time < b))
        months = exposure.sum()
        rate = events / months
        half = 1.959964 * np.sqrt(events) / months
        rates.append(rate)
        lower.append(rate - half)
        upper.append(rate + half)
    return np.array(rates), np.array(lower), np.array(upper)


def logrank(time, event, group):
    """Two-group log-rank test. Returns observed and expected events in group 1,
    the variance, the standardised statistic, the two-sided p-value and the
    one-step (Peto) estimate of the hazard ratio, group 1 against group 0."""
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    group = np.asarray(group, int)
    order = np.argsort(time, kind="stable")
    t, d, g = time[order], event[order], group[order]
    uniq, first = np.unique(t, return_index=True)
    deaths = np.add.reduceat(d, first)
    deaths1 = np.add.reduceat(d * g, first)
    at_risk = len(t) - first
    at_risk1 = np.cumsum(g[::-1])[::-1][first]
    keep = deaths > 0
    d_all, d1, n, n1 = deaths[keep], deaths1[keep], at_risk[keep], at_risk1[keep]
    expected = d_all * n1 / n
    with np.errstate(divide="ignore", invalid="ignore"):
        var = np.where(n > 1, d_all * (n1 / n) * (1 - n1 / n) * (n - d_all) / (n - 1), 0.0)
    o, e, v = d1.sum(), expected.sum(), var.sum()
    z = (o - e) / np.sqrt(v)
    return {"observed": float(o), "expected": float(e), "var": float(v), "z": float(z),
            "p": float(2 * norm.sf(abs(z))), "hr_peto": float(np.exp((o - e) / v))}


def rmst(time, event, tau):
    """Area under the Kaplan-Meier curve up to tau and its standard error,
    Var = sum over event times t_j <= tau of A_j^2 d_j / (n_j (n_j - d_j)),
    where A_j is the area under the curve from t_j to tau."""
    km = kaplan_meier(time, event)
    inside = km["time"] <= tau
    t, s, n, d = km["time"][inside], km["surv"][inside], km["at_risk"][inside], km["events"][inside]
    knots = np.r_[0.0, t, tau]
    levels = np.r_[1.0, s]
    pieces = np.diff(knots) * levels
    area = pieces.sum()
    tail = np.cumsum(pieces[::-1])[::-1][1:]             # area from t_j to tau
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(n > d, tail ** 2 * d / (n * (n - d)), 0.0)
    return float(area), float(np.sqrt(terms.sum()))


def naive_monthly_churn(time, event, edges):
    """Cancellations in each interval divided by customers active at its start,
    ignoring that some of them stopped being followed during the interval."""
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        events = np.sum((event == 1) & (time >= a) & (time < b))
        out.append(events / np.sum(time >= a))
    return np.array(out)


def month_one_test(time, event, group, follow_up, horizon=1.0):
    """Share churned within the first month among customers who signed up at
    least a month before the analysis, compared between arms with a
    two-proportion z-test."""
    eligible = follow_up >= horizon
    churned = (event == 1) & (time < horizon)
    p0 = churned[eligible & (group == 0)].mean()
    p1 = churned[eligible & (group == 1)].mean()
    n0, n1 = np.sum(eligible & (group == 0)), np.sum(eligible & (group == 1))
    pooled = churned[eligible].mean()
    se = np.sqrt(pooled * (1 - pooled) * (1 / n0 + 1 / n1))
    z = (p1 - p0) / se
    return {"p0": float(p0), "p1": float(p1), "n0": int(n0), "n1": int(n1),
            "z": float(z), "p": float(2 * norm.sf(abs(z)))}


# ---------------------------------------------------------------------------
# 3. Repeated experiments
# ---------------------------------------------------------------------------

def analyse_look(time, event, arm, look, follow_up):
    lr = logrank(time, event, arm)
    m1 = month_one_test(time, event, arm, follow_up)
    tau = min(look, HORIZON)
    r0, se0 = rmst(time[arm == 0], event[arm == 0], tau)
    r1, se1 = rmst(time[arm == 1], event[arm == 1], tau)
    diff, se = r1 - r0, np.sqrt(se0 ** 2 + se1 ** 2)
    return {"lr_p": lr["p"], "hr": lr["hr_peto"], "lr_z": lr["z"], "m1_p": m1["p"],
            "m1_diff": m1["p1"] - m1["p0"], "rmst_diff": diff, "rmst_se": se, "tau": tau}


def replicate(rng, treatment, n_rep=1000):
    """Each experiment is simulated once and analysed at every look."""
    out = {look: [] for look in LOOKS}
    for _ in range(n_rep):
        exp = simulate_experiment(rng, treatment)
        for look in LOOKS:
            time, event, arm = observe(exp, look)
            out[look].append(analyse_look(time, event, arm, look, follow_up_at(exp, look)))
    return out


def run_all(n_rep=1000):
    rng = np.random.default_rng(SEED)
    example = simulate_experiment(rng, "steady")
    reps = {name: replicate(rng, name, n_rep) for name in ("control", "steady", "delay")}
    return example, reps


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def main():
    np.set_printoptions(precision=4, suppress=True)
    example, reps = run_all()

    print("=" * 72)
    print("1. THE TRUE PROCESS")
    print("=" * 72)
    print(f"Control: individual churn rates ~ Gamma(shape {ALPHA}, rate {BETA}); "
          f"S(1) {true_survival(1):.3f}, S(6) {true_survival(6):.3f}, S(12) {true_survival(12):.3f}")
    print(f"Population hazard, control: month 0 {true_hazard(0):.4f}, month 6 {true_hazard(6):.4f}, "
          f"month 12 {true_hazard(12):.4f}")
    cum = lambda t: ALPHA * np.log1p(t / BETA)
    print(f"Average hazard within month 1 (H(1) - H(0)) {cum(1) - cum(0):.4f}; "
          f"within month 12 (H(12) - H(11)) {cum(12) - cum(11):.4f}")
    for arm, label in (("steady", "steady treatment (each customer's hazard x 0.8)"),
                       ("delay", "delay treatment (x 0.5 for two months, x 1.5 after)")):
        print(f"{label}: S(1) {true_survival(1, arm):.3f}, S(6) {true_survival(6, arm):.3f}, "
              f"S(12) {true_survival(12, arm):.3f}")
    ratios = [true_hazard(t, "steady") / true_hazard(t) for t in (0, 3, 6, 12, 24, 60)]
    print("Population hazard ratio, steady vs control, at months 0, 3, 6, 12, 24, 60: "
          + ", ".join(f"{r:.3f}" for r in ratios))
    rd = {arm: true_rmst(HORIZON, arm) - true_rmst(HORIZON) for arm in ("steady", "delay")}
    print(f"Expected paid months in the first {HORIZON:g}: control {true_rmst(HORIZON):.3f}, "
          f"steady {true_rmst(HORIZON, 'steady'):.3f} ({rd['steady']:+.3f}), "
          f"delay {true_rmst(HORIZON, 'delay'):.3f} ({rd['delay']:+.3f})")
    print(f"Value per customer over {HORIZON:g} months at EUR {MARGIN:g} margin: steady "
          f"EUR {MARGIN * rd['steady']:+.2f}, delay EUR {MARGIN * rd['delay']:+.2f}")
    print(f"Month-1 churn: control {1 - true_survival(1):.4f}, steady {1 - true_survival(1, 'steady'):.4f}, "
          f"delay {1 - true_survival(1, 'delay'):.4f}")
    t_cross = np.linspace(0.5, 12, 23001)
    cross = t_cross[np.argmax(true_survival(t_cross, "delay") < true_survival(t_cross))]
    print(f"Delay treatment: retention curves cross at month {cross:.2f}; hazards cross at month 2")

    print()
    print("=" * 72)
    print(f"2. ONE EXPERIMENT ({SIGNUPS:,} SIGNUPS, STEADY TREATMENT), ANALYSED AT MONTH 12")
    print("=" * 72)
    time, event, arm = observe(example, 12.0)
    print(f"Customers: control {np.sum(arm == 0):,}, treatment {np.sum(arm == 1):,}; churn observed "
          f"control {event[arm == 0].sum():,}, treatment {event[arm == 1].sum():,}")
    edges = np.arange(0.0, 13.0, 1.0)
    for a, name in ((0, "control"), (1, "treatment")):
        rates, lo, hi = occurrence_exposure(time[arm == a], event[arm == a], edges)
        print(f"Monthly churn rate per customer-month, {name}, months 1, 3, 6, 9, 12: "
              + ", ".join(f"{rates[i]:.4f}" for i in (0, 2, 5, 8, 11)))
        if a == 0:
            print(f"  95% interval in month 1: [{lo[0]:.4f}, {hi[0]:.4f}]; in month 12: "
                  f"[{lo[11]:.4f}, {hi[11]:.4f}]")
            naive = naive_monthly_churn(time[arm == 0], event[arm == 0], edges)
            print("  cancellations / customers active at the start of the month, months 1, 3, 6, 9, 12: "
                  + ", ".join(f"{naive[i]:.4f}" for i in (0, 2, 5, 8, 11)))
    na0 = nelson_aalen(time[arm == 0], event[arm == 0])
    at = np.searchsorted(na0["time"], 12.0, side="right") - 1
    km0 = kaplan_meier(time[arm == 0], event[arm == 0])
    print(f"Nelson-Aalen H(12), control: {na0['cumhaz'][at]:.4f} (true {ALPHA * np.log1p(12 / BETA):.4f}); "
          f"exp(-H) {np.exp(-na0['cumhaz'][at]):.4f} against Kaplan-Meier {km0['surv'][at]:.4f}")
    lr = logrank(time, event, arm)
    print(f"Log-rank: observed {lr['observed']:.0f} churns in treatment against {lr['expected']:.1f} "
          f"expected; z {lr['z']:.2f}, p {lr['p']:.2g}; one-step hazard ratio {lr['hr_peto']:.3f}")
    r1, se1 = rmst(time[arm == 1], event[arm == 1], HORIZON)
    r0, se0 = rmst(time[arm == 0], event[arm == 0], HORIZON)
    print(f"Expected paid months in the first {HORIZON:g}, treatment minus control: {r1 - r0:+.3f} "
          f"(standard error {np.hypot(se0, se1):.3f}; true {true_rmst(HORIZON, 'steady') - true_rmst(HORIZON):+.3f})")
    try:
        from sksurv.compare import compare_survival
        from sksurv.nonparametric import nelson_aalen_estimator
        y = np.array(list(zip(event.astype(bool), time)), dtype=[("e", bool), ("t", float)])
        chisq, p_sk = compare_survival(y, arm)
        t_sk, h_sk = nelson_aalen_estimator(event[arm == 0].astype(bool), time[arm == 0])
        mask = np.isin(t_sk, na0["time"])
        print(f"Check against scikit-survival: log-rank chi-square {chisq:.4f} vs {lr['z'] ** 2:.4f}; "
              f"largest difference in Nelson-Aalen {np.max(np.abs(h_sk[mask] - na0['cumhaz'])):.1e}")
    except ImportError:
        print("Check against scikit-survival: skipped (not installed)")

    print()
    print("=" * 72)
    print("3. 1,000 EXPERIMENTS PER SCENARIO, EACH ANALYSED AT MONTHS 3, 6, 9, 12")
    print("=" * 72)
    header = (f"{'scenario':9s} {'look':>4s} {'log-rank rejects':>17s} {'month-1 test rejects':>21s} "
              f"{'mean HR':>8s} {'RMST diff (tau)':>16s} {'true diff':>10s} {'RMST CI covers':>15s}")
    print(header)
    for name in ("control", "steady", "delay"):
        for look in LOOKS:
            rows = reps[name][look]
            lr_rej = np.mean([r["lr_p"] < 0.05 for r in rows])
            m1_rej = np.mean([r["m1_p"] < 0.05 for r in rows])
            hr = np.mean([r["hr"] for r in rows])
            diffs = np.array([r["rmst_diff"] for r in rows])
            ses = np.array([r["rmst_se"] for r in rows])
            tau = rows[0]["tau"]
            truth = 0.0 if name == "control" else true_rmst(tau, name) - true_rmst(tau)
            cover = np.mean(np.abs(diffs - truth) <= 1.959964 * ses)
            print(f"{name:9s} {look:4.0f} {pct(lr_rej):>17s} {pct(m1_rej):>21s} {hr:8.3f} "
                  f"{diffs.mean():+9.3f} ({tau:2.0f}) {truth:+10.3f} {pct(cover):>15s}")
    parts = []
    for look in LOOKS:
        rows = reps["delay"][look]
        fav = np.mean([(r["lr_p"] < 0.05) and (r["lr_z"] < 0) for r in rows])
        ag = np.mean([(r["lr_p"] < 0.05) and (r["lr_z"] > 0) for r in rows])
        parts.append(f"month {look:.0f} {pct(fav)} / {pct(ag)}")
    print("Delay scenario, log-rank significant in favour of / against the treatment: " + "; ".join(parts))
    parts = [f"month {look:.0f} {pct(np.mean([(r['m1_p'] < 0.05) and (r['m1_diff'] < 0) for r in reps['delay'][look]]))}"
             for look in LOOKS]
    print("Delay scenario, month-1 test significant in favour of the treatment: " + "; ".join(parts))
    rows = reps["delay"][12.0]
    favour = np.mean([(r["m1_p"] < 0.05) and (r["m1_diff"] < 0) for r in rows])
    harm = np.mean([r["rmst_diff"] + 1.959964 * r["rmst_se"] < 0 for r in rows])
    lr_dir = np.mean([(r["lr_p"] < 0.05) and (r["lr_z"] < 0) for r in rows])
    print(f"Delay scenario at month 12: month-1 test significant in favour of the treatment "
          f"{pct(favour)}; log-rank significant in favour {pct(lr_dir)}; RMST(12) difference "
          f"significantly negative {pct(harm)}")
    for key, name in (("lr_p", "log-rank"), ("m1_p", "month-1 test")):
        any_look = np.mean([any(reps["control"][look][i][key] < 0.05 for look in LOOKS)
                            for i in range(len(reps["control"][LOOKS[0]]))])
        print(f"No effect, {name} significant at one or more of the four analysis dates: {pct(any_look)}")
    rows3 = reps["delay"][3.0]
    print(f"Delay scenario at month 3: month-1 test significant in favour {pct(np.mean([(r['m1_p'] < 0.05) and (r['m1_diff'] < 0) for r in rows3]))}; "
          f"mean RMST(3) difference {np.mean([r['rmst_diff'] for r in rows3]):+.3f} months")


if __name__ == "__main__":
    main()
