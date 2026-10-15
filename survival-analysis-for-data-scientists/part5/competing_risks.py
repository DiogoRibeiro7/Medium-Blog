"""Survival Analysis for Data Scientists - Part 5.

Competing risks and recurrent events, applied to pricing a five-year service
contract for a fleet of industrial pumps.

Everything the article reports is printed by ``python competing_risks.py``.
The figures are produced by ``python figures.py``.

Requires numpy. If scikit-survival is installed, the Kaplan-Meier and
cause-specific Cox estimates are also checked against it.
"""

from __future__ import annotations

import numpy as np

SEED = 20261016

# ---------------------------------------------------------------------------
# 1. The fleet
# ---------------------------------------------------------------------------
# A manufacturer has installed N_PUMPS pumps at customer sites over the last
# YEARS years, at a steady rate. A pump leaves service in one of two ways:
#   breakdown (cause 1): a Weibull hazard that rises with age,
#       H(t) = (t / BREAKDOWN_SCALE) ** BREAKDOWN_SHAPE, times 1.8 for high-duty pumps;
#   planned replacement (cause 2): a constant hazard, 0.06 a year at most sites
#       and 0.24 a year at sites in a modernisation programme.
# The two are independent. While in service, a pump also needs unplanned repairs,
# which do not end its service: a Poisson process with rate z (0.5 + 0.2 t) a year
# at age t, times 1.5 for high-duty pumps, where z is the pump's own frailty
# (gamma, mean 1, variance 0.5). Repairs are independent of how the pump leaves.
#
# The decision: the break-even price of a service contract that covers a pump for
# CONTRACT years or until it leaves service, paying REPAIR_COST per repair and
# BREAKDOWN_COST if it breaks down.

N_PUMPS = 3000
YEARS = 10.0
HIGH_DUTY_SHARE, PROGRAMME_SHARE = 0.4, 0.3
BREAKDOWN_SCALE, BREAKDOWN_SHAPE, DUTY_HR = 9.0, 2.0, 1.8
REPLACE_RATE, PROGRAMME_HR = 0.06, 4.0
REPAIR_BASE, REPAIR_SLOPE, DUTY_REPAIR, FRAILTY_SHAPE = 0.5, 0.2, 1.5, 2.0
CONTRACT = 5.0
REPAIR_COST, BREAKDOWN_COST = 600.0, 9000.0
SEGMENTS = [(d, p) for d in (0, 1) for p in (0, 1)]       # (high duty, programme site)


def segment_name(duty, prog):
    return f"{'high-duty' if duty else 'standard'} pump, {'programme' if prog else 'other'} site"


def simulate_fleet(rng, n=N_PUMPS, years=YEARS):
    install = rng.uniform(0.0, years, n)
    duty = (rng.uniform(size=n) < HIGH_DUTY_SHARE).astype(int)
    prog = (rng.uniform(size=n) < PROGRAMME_SHARE).astype(int)
    breakdown = BREAKDOWN_SCALE * (rng.exponential(1.0, n) / np.where(duty == 1, DUTY_HR, 1.0)) ** (1 / BREAKDOWN_SHAPE)
    replace = rng.exponential(1.0, n) / (REPLACE_RATE * np.where(prog == 1, PROGRAMME_HR, 1.0))
    removal = np.minimum(breakdown, replace)
    follow = years - install
    time = np.minimum(removal, follow)
    status = np.where(removal > follow, 0, np.where(breakdown < replace, 1, 2))
    # recurrent repairs on [0, time]: cumulative intensity L(t) = z m (0.5 t + 0.1 t^2)
    z = rng.gamma(FRAILTY_SHAPE, 1 / FRAILTY_SHAPE, n)
    m = z * np.where(duty == 1, DUTY_REPAIR, 1.0)
    total = m * (REPAIR_BASE * time + REPAIR_SLOPE / 2 * time ** 2)
    count = rng.poisson(total)
    pump = np.repeat(np.arange(n), count)
    s = rng.uniform(size=count.sum()) * total[pump]           # uniform on [0, L(time)]
    age = (-REPAIR_BASE + np.sqrt(REPAIR_BASE ** 2 + 2 * REPAIR_SLOPE * s / m[pump])) / REPAIR_SLOPE
    return {"time": time, "status": status, "duty": duty, "prog": prog,
            "repair_pump": pump, "repair_age": age}


# True quantities ---------------------------------------------------------

def true_curves(duty, prog, grid):
    Hf = (grid / BREAKDOWN_SCALE) ** BREAKDOWN_SHAPE * (DUTY_HR if duty else 1.0)
    hf = BREAKDOWN_SHAPE / BREAKDOWN_SCALE * (grid / BREAKDOWN_SCALE) ** (BREAKDOWN_SHAPE - 1) * (DUTY_HR if duty else 1.0)
    mu = REPLACE_RATE * (PROGRAMME_HR if prog else 1.0)
    S = np.exp(-Hf - mu * grid)
    rate = (REPAIR_BASE + REPAIR_SLOPE * grid) * (DUTY_REPAIR if duty else 1.0)
    return {"S": S, "hf": hf, "mu": mu, "Hf": Hf, "rate": rate}


def true_values(duty, prog, horizon=CONTRACT):
    g = np.linspace(0.0, horizon, 50001)
    c = true_curves(duty, prog, g)
    cif_b = float(np.trapezoid(c["hf"] * c["S"], g))
    cif_r = float(np.trapezoid(c["mu"] * c["S"], g))
    net_b = float(1 - np.exp(-c["Hf"][-1]))
    repairs = float(np.trapezoid(c["rate"] * c["S"], g))
    mcf = float(np.trapezoid(c["rate"], g))
    return {"cif_b": cif_b, "cif_r": cif_r, "net_b": net_b, "repairs_in_service": repairs, "mcf": mcf,
            "price": BREAKDOWN_COST * cif_b + REPAIR_COST * repairs}


def true_mix_values(fleet_weights=None, horizon=CONTRACT):
    """True values for the whole fleet, averaging the four segments by their shares."""
    w = {(d, p): (HIGH_DUTY_SHARE if d else 1 - HIGH_DUTY_SHARE) * (PROGRAMME_SHARE if p else 1 - PROGRAMME_SHARE)
         for d, p in SEGMENTS}
    vals = {seg: true_values(*seg, horizon) for seg in SEGMENTS}
    return {k: sum(w[s] * vals[s][k] for s in SEGMENTS) for k in vals[SEGMENTS[0]]}


# ---------------------------------------------------------------------------
# 2. Estimators
# ---------------------------------------------------------------------------

def _event_table(time, status):
    """Distinct event times, numbers at risk, and events of each cause."""
    order = np.argsort(time, kind="stable")
    t, s = time[order], status[order]
    uniq, first = np.unique(t, return_index=True)
    at_risk = len(t) - first
    d1 = np.add.reduceat((s == 1).astype(int), first)
    d2 = np.add.reduceat((s == 2).astype(int), first)
    keep = (d1 + d2) > 0
    return uniq[keep], at_risk[keep], d1[keep], d2[keep]


def kaplan_meier(time, event):
    t, n, d, _ = _event_table(time, np.where(event == 1, 1, 0))
    keep = d > 0
    return t[keep], np.cumprod(1 - d[keep] / n[keep])


def aalen_johansen(time, status):
    """Cumulative incidence of each cause: the sum over event times of the
    probability of still being in service just before, times the cause's share of
    those at risk who leave at that time."""
    t, n, d1, d2 = _event_table(time, status)
    surv = np.cumprod(1 - (d1 + d2) / n)
    before = np.r_[1.0, surv[:-1]]
    return t, np.cumsum(before * d1 / n), np.cumsum(before * d2 / n), surv


def step_at(times, values, x, start=0.0):
    idx = np.searchsorted(times, np.asarray(x, float), side="right") - 1
    return np.where(idx >= 0, values[np.maximum(idx, 0)], start)


def repairs_in_service(fleet, mask, horizon=CONTRACT):
    """Mean cumulative function of repairs among pumps in service (Nelson), and the
    expected number of repairs per pump within the horizon once pumps that leave
    service stop accumulating them (each increment weighted by the probability of
    still being in service)."""
    time, status = fleet["time"][mask], fleet["status"][mask]
    sel = mask[fleet["repair_pump"]]
    ages = np.sort(fleet["repair_age"][sel])
    ages = ages[ages <= horizon]
    sorted_time = np.sort(time)
    at_risk = len(sorted_time) - np.searchsorted(sorted_time, ages, side="left")
    increments = 1.0 / at_risk
    t_all, _, _, surv = aalen_johansen(time, status)
    in_service = step_at(t_all, surv, ages - 1e-12, 1.0)          # just before each repair
    return {"ages": ages, "mcf": np.cumsum(increments), "expected": np.cumsum(in_service * increments)}


def cox_breslow(X, time, event, tol=1e-10):
    """Cox model (Breslow ties) by Newton-Raphson: used for the cause-specific hazard."""
    order = np.argsort(time, kind="stable")
    X, time, event = X[order], time[order], event[order]
    beta = np.zeros(X.shape[1])
    ll_old = -np.inf
    for _ in range(50):
        w = np.exp(X @ beta)
        R = np.cumsum(w[::-1])[::-1]
        RX = np.cumsum((w[:, None] * X)[::-1], axis=0)[::-1]
        RXX = np.cumsum((w[:, None, None] * X[:, :, None] * X[:, None, :])[::-1], axis=0)[::-1]
        first = np.searchsorted(time, time, side="left")
        ev = event == 1
        mean = RX[first[ev]] / R[first[ev], None]
        ll = float(np.sum(X[ev] @ beta - np.log(R[first[ev]])))
        grad = (X[ev] - mean).sum(axis=0)
        hess = -((RXX[first[ev]] / R[first[ev], None, None]).sum(axis=0) - mean.T @ mean)
        beta = beta - np.linalg.solve(hess, grad)
        if abs(ll - ll_old) < tol:
            break
        ll_old = ll
    return beta, np.sqrt(np.diag(np.linalg.inv(-hess)))


def fine_gray(X, time, status, tol=1e-10):
    """Fine-Gray model for the subdistribution hazard of cause 1 (Breslow ties).
    Pumps that left for the competing cause stay in the risk set after they leave,
    with weight G(t-) / G(their exit time-), where G is the Kaplan-Meier curve of the
    censoring times (here: the data cut). Sums over the risk set are cumulative sums
    over pumps sorted by time."""
    order = np.argsort(time, kind="stable")
    X, time, status = X[order], time[order], status[order]
    gt, gs = kaplan_meier(time, (status == 0).astype(int))
    G = lambda x: step_at(gt, gs, x, 1.0)
    comp = (status == 2) / G(time - 1e-12)                   # weight factor for competing exits
    ev = np.flatnonzero(status == 1)
    first = np.searchsorted(time, time[ev], side="left")       # first index at the event time
    G_ev = G(time[ev] - 1e-12)
    beta = np.zeros(X.shape[1])
    ll_old = -np.inf
    for _ in range(50):
        w = np.exp(X @ beta)
        wX, wXX = w[:, None] * X, w[:, None, None] * X[:, :, None] * X[:, None, :]
        R0, R1, R2 = (np.cumsum(a[::-1], axis=0)[::-1] for a in (w, wX, wXX))
        C0, C1, C2 = (np.cumsum(a, axis=0) for a in (comp * w, comp[:, None] * wX, comp[:, None, None] * wXX))
        prev = first - 1                                       # competing exits strictly before t
        has = prev >= 0
        S0 = R0[first] + np.where(has, G_ev * C0[np.maximum(prev, 0)], 0.0)
        S1 = R1[first] + np.where(has[:, None], G_ev[:, None] * C1[np.maximum(prev, 0)], 0.0)
        S2 = R2[first] + np.where(has[:, None, None], G_ev[:, None, None] * C2[np.maximum(prev, 0)], 0.0)
        mean = S1 / S0[:, None]
        ll = float(np.sum(X[ev] @ beta - np.log(S0)))
        grad = (X[ev] - mean).sum(axis=0)
        hess = -((S2 / S0[:, None, None]).sum(axis=0) - mean.T @ mean)
        beta = beta - np.linalg.solve(hess, grad)
        if abs(ll - ll_old) < tol:
            break
        ll_old = ll
    return beta


# ---------------------------------------------------------------------------
# 3. Contract prices, three ways
# ---------------------------------------------------------------------------

def prices(fleet, mask, horizon=CONTRACT):
    time, status = fleet["time"][mask], fleet["status"][mask]
    n = mask.sum()
    sel = mask[fleet["repair_pump"]]
    ages = fleet["repair_age"][sel]
    # (a) what a dashboard shows: shares and counts so far, with no regard to how long
    #     each pump has been followed
    dash_b = np.mean((status == 1) & (time <= horizon))
    dash_r = np.sum(ages <= horizon) / n
    # (b) one cause at a time: breakdown with replacements treated as censored, and
    #     repairs per pump in service, as if every pump stayed in service
    km_t, km_s = kaplan_meier(time, (status == 1).astype(int))
    net_b = 1 - float(step_at(km_t, km_s, horizon, 1.0))
    rep = repairs_in_service(fleet, mask, horizon)
    mcf = float(rep["mcf"][-1]) if len(rep["mcf"]) else 0.0
    # (c) competing risks: Aalen-Johansen cumulative incidence, and repairs weighted by
    #     the probability of still being in service
    t, c1, c2, _ = aalen_johansen(time, status)
    cif_b = float(step_at(t, c1, horizon))
    cif_r = float(step_at(t, c2, horizon))
    expected = float(rep["expected"][-1]) if len(rep["expected"]) else 0.0
    return {"dashboard": BREAKDOWN_COST * dash_b + REPAIR_COST * dash_r,
            "one cause at a time": BREAKDOWN_COST * net_b + REPAIR_COST * mcf,
            "competing risks": BREAKDOWN_COST * cif_b + REPAIR_COST * expected,
            "parts": {"dash_b": dash_b, "dash_r": dash_r, "net_b": net_b, "mcf": mcf,
                      "cif_b": cif_b, "cif_r": cif_r, "expected": expected}}


METHODS = ("dashboard", "one cause at a time", "competing risks")


def replicate(rng, n_rep):
    out = {seg: {m: [] for m in METHODS} for seg in SEGMENTS}
    cs, cs_se, fg = [], [], []
    for _ in range(n_rep):
        fleet = simulate_fleet(rng)
        for seg in SEGMENTS:
            mask = (fleet["duty"] == seg[0]) & (fleet["prog"] == seg[1])
            p = prices(fleet, mask)
            for m in METHODS:
                out[seg][m].append(p[m])
        X = np.column_stack([fleet["duty"], fleet["prog"]]).astype(float)
        b, se = cox_breslow(X, fleet["time"], (fleet["status"] == 1).astype(int))
        cs.append(b)
        cs_se.append(se)
        fg.append(fine_gray(X, fleet["time"], fleet["status"]))
    return ({seg: {m: np.array(v) for m, v in d.items()} for seg, d in out.items()},
            np.array(cs), np.array(cs_se), np.array(fg))


def run_all(n_rep=500):
    rng = np.random.default_rng(SEED)
    fleet = simulate_fleet(rng)
    return fleet, replicate(rng, n_rep)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def eur(x):
    return f"EUR {x:,.0f}"


def main(n_rep=500):
    rng = np.random.default_rng(SEED)
    fleet = simulate_fleet(rng)
    time, status = fleet["time"], fleet["status"]

    print("=" * 72)
    print("1. THE TRUE PROCESS")
    print("=" * 72)
    for seg in SEGMENTS:
        v = true_values(*seg)
        print(f"{segment_name(*seg):37s}: within {CONTRACT:g} years, breakdown {pct(v['cif_b'])} "
              f"(without replacements {pct(v['net_b'])}), planned replacement {pct(v['cif_r'])}; repairs per pump "
              f"{v['repairs_in_service']:.2f} in service (mean cumulative function {v['mcf']:.2f}); "
              f"break-even price {eur(v['price'])}")
    mix = true_mix_values()
    print(f"Whole fleet: breakdown within {CONTRACT:g} years {pct(mix['cif_b'])} (without replacements "
          f"{pct(mix['net_b'])}); repairs per contract {mix['repairs_in_service']:.2f}; break-even price {eur(mix['price'])}")
    ages = np.array([0.5, 3.0, 5.0, 10.0])
    sub = []
    for prog in (0, 1):
        g = np.linspace(0.0, 10.0, 100001)
        c = true_curves(0, prog, g)
        F1 = np.concatenate([[0.0], np.cumsum((c["hf"] * c["S"])[1:] * np.diff(g))])
        h_sub = c["hf"] * c["S"] / (1 - F1)
        sub.append(np.interp(ages, g, h_sub))
    print("True subdistribution hazard ratio for breakdown, programme against other sites (standard duty), at ages "
          + ", ".join(f"{a:g}: {r:.2f}" for a, r in zip(ages, sub[1] / sub[0])))
    print(f"Hazard ratios: breakdown {DUTY_HR} for high duty, 1.0 for programme sites; planned replacement "
          f"{PROGRAMME_HR} for programme sites, 1.0 for high duty")

    print()
    print("=" * 72)
    print(f"2. ONE FLEET: {N_PUMPS:,} PUMPS INSTALLED OVER {YEARS:g} YEARS")
    print("=" * 72)
    print(f"Pumps: high duty {fleet['duty'].sum():,}, at programme sites {fleet['prog'].sum():,}; broken down "
          f"{(status == 1).sum():,}, replaced as planned {(status == 2).sum():,}, still in service "
          f"{(status == 0).sum():,}; repairs recorded {len(fleet['repair_age']):,}")
    all_mask = np.ones(len(time), bool)
    p = prices(fleet, all_mask)["parts"]
    print(f"Whole fleet, within {CONTRACT:g} years: breakdown, Aalen-Johansen {pct(p['cif_b'])} vs one minus "
          f"Kaplan-Meier {pct(p['net_b'])}; planned replacement {pct(p['cif_r'])}; breakdowns so far "
          f"{pct(p['dash_b'])} of pumps")
    t, c1, c2, surv = aalen_johansen(time, status)
    km_t, km_s = kaplan_meier(time, (status == 1).astype(int))
    km2_t, km2_s = kaplan_meier(time, (status == 2).astype(int))
    for yr in (2.0, 5.0, 8.0):
        a, b = float(step_at(t, c1, yr)), float(step_at(t, c2, yr))
        s = float(step_at(t, surv, yr, 1.0))
        n1, n2 = 1 - float(step_at(km_t, km_s, yr, 1.0)), 1 - float(step_at(km2_t, km2_s, yr, 1.0))
        print(f"  at {yr:g} years: in service {pct(s)}, broken down {pct(a)}, replaced {pct(b)} (sum {pct(s + a + b)}); "
              f"one minus Kaplan-Meier: breakdown {pct(n1)}, replacement {pct(n2)} (sum with in service {pct(s + n1 + n2)})")
    try:
        from sksurv.nonparametric import kaplan_meier_estimator
        tt, ss = kaplan_meier_estimator((status == 1), time)
        print(f"Check against scikit-survival: largest difference in the Kaplan-Meier curve for breakdown "
              f"{np.max(np.abs(ss - step_at(km_t, km_s, tt, 1.0))):.1e}")
    except ImportError:
        print("Check against scikit-survival: skipped (not installed)")

    X = np.column_stack([fleet["duty"], fleet["prog"]]).astype(float)
    b_cs, se_cs = cox_breslow(X, time, (status == 1).astype(int))
    b_cr, se_cr = cox_breslow(X, time, (status == 2).astype(int))
    b_fg = fine_gray(X, time, status)
    print("Cause-specific Cox, breakdown: " + ", ".join(
        f"{n} hazard ratio {np.exp(b):.3f} [{np.exp(b - 1.96 * s):.3f}, {np.exp(b + 1.96 * s):.3f}]"
        for n, b, s in zip(("high duty", "programme site"), b_cs, se_cs)))
    print("Cause-specific Cox, planned replacement: " + ", ".join(
        f"{n} hazard ratio {np.exp(b):.3f}" for n, b in zip(("high duty", "programme site"), b_cr)))
    print("Fine-Gray, breakdown: subdistribution hazard ratios " + ", ".join(
        f"{n} {np.exp(b):.3f}" for n, b in zip(("high duty", "programme site"), b_fg)))
    try:
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        y = np.array(list(zip(status == 1, time)), dtype=[("e", bool), ("t", float)])
        sk = CoxPHSurvivalAnalysis(ties="breslow").fit(X, y)
        print(f"Check against scikit-survival: largest difference in cause-specific coefficients "
              f"{'below 1e-06' if np.max(np.abs(sk.coef_ - b_cs)) < 1e-6 else np.max(np.abs(sk.coef_ - b_cs))}")
    except ImportError:
        pass
    no_comp = np.where(status == 2, 0, status)
    b_check = fine_gray(X, time, no_comp)
    b_cox = cox_breslow(X, time, (no_comp == 1).astype(int))[0]
    print(f"Check: with no competing events, Fine-Gray equals Cox (largest difference "
          f"{'below 1e-08' if np.max(np.abs(b_check - b_cox)) < 1e-8 else np.max(np.abs(b_check - b_cox))})")
    for prog in (0, 1):
        m = fleet["prog"] == prog
        tt, cc1, _, _ = aalen_johansen(time[m], status[m])
        kt, ks = kaplan_meier(time[m], (status[m] == 1).astype(int))
        tn, nn, d1, _ = _event_table(time[m], status[m])
        na = float(np.sum((d1 / nn)[tn <= CONTRACT]))
        print(f"  {'programme' if prog else 'other'} sites: breakdown within {CONTRACT:g} years, Aalen-Johansen "
              f"{pct(float(step_at(tt, cc1, CONTRACT)))}, one minus Kaplan-Meier "
              f"{pct(1 - float(step_at(kt, ks, CONTRACT, 1.0)))}; Nelson-Aalen cumulative hazard of breakdown "
              f"at {CONTRACT:g} years {na:.3f}")

    print("Contract prices in this fleet (EUR): true / dashboard / one cause at a time / competing risks")
    for seg in SEGMENTS:
        mask = (fleet["duty"] == seg[0]) & (fleet["prog"] == seg[1])
        pr = prices(fleet, mask)
        q = pr["parts"]
        print(f"  {segment_name(*seg):37s} ({mask.sum():,} pumps): {eur(true_values(*seg)['price'])} / "
              + " / ".join(eur(pr[m]) for m in METHODS)
              + f"; repairs: so far {q['dash_r']:.2f}, mean cumulative function {q['mcf']:.2f}, in service "
              f"{q['expected']:.2f}")
    pr = prices(fleet, all_mask)
    print(f"  {'whole fleet':37s} ({N_PUMPS:,} pumps): {eur(mix['price'])} / " + " / ".join(eur(pr[m]) for m in METHODS))

    print()
    print("=" * 72)
    print(f"3. {n_rep} SIMULATED FLEETS")
    print("=" * 72)
    reps, cs, cs_se, fg = replicate(rng, n_rep)
    for seg in SEGMENTS:
        tv = true_values(*seg)["price"]
        print(f"{segment_name(*seg)} (true {eur(tv)}):")
        for m in METHODS:
            v = reps[seg][m]
            lo, hi = np.percentile(v, [2.5, 97.5])
            print(f"    {m:20s} mean {eur(v.mean())} ({pct(v.mean() / tv - 1)} off), middle 95% "
                  f"{eur(lo)} to {eur(hi)}")
    print("Mean hazard ratios: cause-specific " + ", ".join(
        f"{n} {np.exp(b):.3f}" for n, b in zip(("high duty", "programme site"), cs.mean(axis=0)))
          + "; Fine-Gray " + ", ".join(f"{n} {np.exp(b):.3f}" for n, b in zip(("high duty", "programme site"),
                                                                              fg.mean(axis=0))))
    truth = np.log([DUTY_HR, 1.0])
    cover = np.mean(np.abs(cs - truth) <= 1.959964 * cs_se, axis=0)
    print(f"Cause-specific 95% intervals cover the true hazard ratio: high duty {pct(cover[0])}, programme site "
          f"{pct(cover[1])}; Fine-Gray hazard ratio for programme sites below 1 in {pct(np.mean(fg[:, 1] < 0))} of fleets")


if __name__ == "__main__":
    main()
