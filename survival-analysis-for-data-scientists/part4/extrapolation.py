"""Survival Analysis for Data Scientists - Part 4.

Parametric survival models, accelerated failure time, and the value of a
customer beyond the period anyone has observed.

Everything the article reports is printed by ``python extrapolation.py``.
The figures are produced by ``python figures.py``.

Requires numpy and scipy. scipy's own censored-data fitting is used to check
the maximum likelihood fits; if scikit-survival is installed, it is used for a
Cox model on the same data.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gamma as gamma_fn
from scipy.stats import norm

SEED = 20261015

# ---------------------------------------------------------------------------
# 1. Three businesses that look the same for a year
# ---------------------------------------------------------------------------
# The subscription product of Parts 1 and 2, launched WINDOW months ago, with
# SIGNUPS_PER_MONTH new customers a month. 30% arrive by referral, and referral
# stretches every customer's time to churn by the factor TIME_RATIO (each
# customer's churn rate is multiplied by 0.7), which is an accelerated failure
# time effect.
#
# The retention of a customer from paid channels follows one of three curves:
#   "gamma mixture": each customer has a constant churn rate drawn from a gamma
#                    distribution (Parts 1 and 2): S(t) = (1 + t/25) ** -1.5
#   "Weibull":       S(t) = exp(-(t / 21.3) ** 0.9)
#   "loyal core":    30% never leave within the horizon; the rest leave at a
#                    constant rate: S(t) = 0.3 + 0.7 exp(-t / 12)
# The parameters of the second and third were chosen to match the first over
# its first 12 months.

SIGNUPS_PER_MONTH = 500
REFERRAL_SHARE = 0.3
TIME_RATIO = 1 / 0.7
MARGIN = 15.0                    # EUR per month
VALUE_HORIZON = 60.0             # months: the value of a customer over five years
WINDOWS = (12.0, 24.0, 36.0)     # months since launch at which the data are analysed

TRUTHS = {
    "gamma mixture": lambda t: (1.0 + t / 25.0) ** -1.5,
    "Weibull": lambda t: np.exp(-(np.asarray(t, float) / 21.3) ** 0.9),
    "loyal core": lambda t: 0.3 + 0.7 * np.exp(-np.asarray(t, float) / 12.0),
}


def simulate_table(rng, truth, window, per_month=SIGNUPS_PER_MONTH):
    n = int(per_month * window)
    signup = rng.uniform(0.0, window, n)
    referral = (rng.uniform(size=n) < REFERRAL_SHARE).astype(int)
    if truth == "gamma mixture":
        rate = rng.gamma(1.5, 1 / 25.0, n)
        base = rng.exponential(1.0, n) / rate
    elif truth == "Weibull":
        base = 21.3 * rng.exponential(1.0, n) ** (1 / 0.9)
    else:
        loyal = rng.uniform(size=n) < 0.3
        base = np.where(loyal, np.inf, rng.exponential(12.0, n))
    lifetime = base * np.where(referral == 1, TIME_RATIO, 1.0)
    follow_up = window - signup
    return {"time": np.minimum(lifetime, follow_up), "event": (lifetime <= follow_up).astype(int),
            "referral": referral}


def true_value(truth, horizon=VALUE_HORIZON, start=0.0, ratio=1.0):
    """Expected margin between `start` and `horizon` months; `ratio` stretches time
    (TIME_RATIO for a referral customer)."""
    grid = np.linspace(start, horizon, 60001)
    return MARGIN * float(np.trapezoid(TRUTHS[truth](grid / ratio), grid))


# ---------------------------------------------------------------------------
# 2. Six parametric families, as accelerated failure time models
# ---------------------------------------------------------------------------
# Each family is written through its baseline cumulative hazard H0(u) and log
# hazard log h0(u). The accelerated failure time version for a customer with
# covariate x is S(t | x) = S0(t exp(-g x)): the covariate stretches or shrinks
# time. The censored log-likelihood is sum of event * log h(t | x) - H(t | x).

def _gompertz_H(u, b, c):
    cu = c * u
    return b * np.where(np.abs(c) > 1e-9, np.expm1(cu) / np.where(c == 0, 1, c), u)


FAMILIES = {
    # name: (parameter names, log h0(u, p), H0(u, p))
    "exponential": (("log scale",),
                    lambda u, p: -p[0] + 0 * u,
                    lambda u, p: u * np.exp(-p[0])),
    "Weibull": (("log scale", "log shape"),
                lambda u, p: p[1] - p[0] + (np.exp(p[1]) - 1) * (np.log(u) - p[0]),
                lambda u, p: np.exp(np.exp(p[1]) * (np.log(u) - p[0]))),
    "Gompertz": (("log rate", "change"),
                 lambda u, p: p[0] + p[1] * u,
                 lambda u, p: _gompertz_H(u, np.exp(p[0]), p[1])),
    "log-normal": (("mean log", "log sd"),
                   lambda u, p: (norm.logpdf((np.log(u) - p[0]) / np.exp(p[1])) - p[1] - np.log(u)
                                 - norm.logsf((np.log(u) - p[0]) / np.exp(p[1]))),
                   lambda u, p: -norm.logsf((np.log(u) - p[0]) / np.exp(p[1]))),
    "log-logistic": (("log scale", "log shape"),
                     lambda u, p: (p[1] - p[0] + (np.exp(p[1]) - 1) * (np.log(u) - p[0])
                                   - np.log1p(np.exp(np.exp(p[1]) * (np.log(u) - p[0])))),
                     lambda u, p: np.log1p(np.exp(np.exp(p[1]) * (np.log(u) - p[0])))),
    "gamma mixture": (("log shape", "log scale"),
                      lambda u, p: p[0] - np.log(np.exp(p[1]) + u),
                      lambda u, p: np.exp(p[0]) * np.log1p(u / np.exp(p[1]))),
}
MODELS = tuple(FAMILIES)


def neg_loglik(theta, family, time, event, x):
    _, log_h0, H0 = FAMILIES[family]
    p, g = theta[:-1], theta[-1]
    u = time * np.exp(-g * x)
    ll = np.sum(event * (log_h0(u, p) - g * x)) - np.sum(H0(u, p))
    return -ll if np.isfinite(ll) else 1e300


def start_values(family, time, event):
    scale = time.sum() / max(event.sum(), 1)            # exponential fit
    return {"exponential": [np.log(scale)],
            "Weibull": [np.log(scale), 0.0],
            "Gompertz": [-np.log(scale), 0.0],
            "log-normal": [np.log(scale * 0.7), 0.3],
            "log-logistic": [np.log(scale * 0.7), 0.2],
            "gamma mixture": [np.log(2.0), np.log(scale)]}[family] + [0.0]


def fit_model(family, time, event, x):
    theta0 = np.array(start_values(family, time, event))
    best = minimize(neg_loglik, theta0, args=(family, time, event, x), method="BFGS", options={"gtol": 1e-6})
    if not best.success:                                  # polish with a derivative-free search
        alt = minimize(neg_loglik, best.x, args=(family, time, event, x), method="Nelder-Mead",
                       options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-10})
        if alt.fun < best.fun:
            best = alt
    k = len(theta0)
    return {"family": family, "theta": best.x, "loglik": -best.fun, "aic": 2 * k + 2 * best.fun, "k": k}


def survival(fit, t, x=0):
    _, _, H0 = FAMILIES[fit["family"]]
    p, g = fit["theta"][:-1], fit["theta"][-1]
    t = np.asarray(t, float)
    u = np.maximum(t, 1e-12) * np.exp(-g * x)
    return np.where(t > 0, np.exp(-H0(u, p)), 1.0)


def hazard(fit, t, x=0):
    log_h0 = FAMILIES[fit["family"]][1]
    p, g = fit["theta"][:-1], fit["theta"][-1]
    u = np.asarray(t, float) * np.exp(-g * x)
    return np.exp(log_h0(u, p) - g * x)


def model_value(fit, start=0.0, horizon=VALUE_HORIZON, x=0):
    grid = np.linspace(start, horizon, 6001)
    return MARGIN * float(np.trapezoid(survival(fit, grid, x), grid))


def full_mean(fit):
    """Mean lifetime of a paid-channel customer, in months (inf if the tail is too heavy)."""
    f, p = fit["family"], fit["theta"][:-1]
    if f == "exponential":
        return float(np.exp(p[0]))
    if f == "Weibull":
        return float(np.exp(p[0]) * gamma_fn(1 + np.exp(-p[1])))
    if f == "log-normal":
        return float(np.exp(p[0] + np.exp(2 * p[1]) / 2))
    if f == "log-logistic":
        k = np.exp(p[1])
        return float(np.exp(p[0]) * (np.pi / k) / np.sin(np.pi / k)) if k > 1 else np.inf
    if f == "gamma mixture":
        a, b = np.exp(p)
        return float(b / (a - 1)) if a > 1 else np.inf
    b, c = np.exp(p[0]), p[1]                             # Gompertz
    if c <= 0:
        return np.inf
    grid = np.linspace(0, 2000, 200001)
    return float(np.trapezoid(np.exp(-_gompertz_H(grid, b, c)), grid))


def kaplan_meier(time, event):
    order = np.argsort(time, kind="stable")
    t, e = time[order], event[order]
    uniq, first = np.unique(t, return_index=True)
    deaths = np.add.reduceat(e, first)
    at_risk = len(t) - first
    keep = deaths > 0
    return uniq[keep], np.cumprod(1.0 - deaths[keep] / at_risk[keep])


def km_at(km, t):
    times, surv = km
    idx = np.searchsorted(times, np.asarray(t, float), side="right") - 1
    return np.where(idx >= 0, surv[np.maximum(idx, 0)], 1.0)


def km_value(km, horizon):
    times, surv = km
    inside = times <= horizon
    knots = np.r_[0.0, times[inside], horizon]
    return MARGIN * float((np.diff(knots) * np.r_[1.0, surv[inside]]).sum())


# ---------------------------------------------------------------------------
# 3. Decision rules
# ---------------------------------------------------------------------------

def analyse_table(tab, window):
    time, event, x = tab["time"], tab["event"], tab["referral"]
    fits = {m: fit_model(m, time, event, x) for m in MODELS}
    aic = np.array([fits[m]["aic"] for m in MODELS])
    delta = aic - aic.min()
    weights = np.exp(-delta / 2)
    weights /= weights.sum()
    values = np.array([model_value(fits[m]) for m in MODELS])
    km = kaplan_meier(time[x == 0], event[x == 0])
    observed = km_value(km, window)
    tails = np.array([model_value(fits[m], start=window) for m in MODELS])
    plausible = delta <= 10
    return {"fits": fits, "delta": delta, "weights": weights, "values": values, "plausible": plausible,
            "best": MODELS[int(np.argmin(aic))],
            "rules": {"best AIC": values[np.argmin(aic)],
                      "AIC average": float(weights @ values),
                      "observed + lowest tail": observed + tails[plausible].min(),
                      "observed only": observed},
            "observed": observed, "tails": tails}


RULES = ("best AIC", "AIC average", "observed + lowest tail", "observed only")


def replicate(rng, n_rep):
    out = {}
    for truth in TRUTHS:
        for window in WINDOWS:
            rows = [analyse_table(simulate_table(rng, truth, window), window) for _ in range(n_rep)]
            out[(truth, window)] = {
                "rules": {r: np.array([row["rules"][r] for row in rows]) for r in RULES},
                "values": np.array([row["values"] for row in rows]),
                "plausible": np.array([row["plausible"] for row in rows]),
                "ratios": np.array([[np.exp(row["fits"][m]["theta"][-1]) for m in MODELS] for row in rows]),
                "best": [row["best"] for row in rows],
            }
    return out


def run_all(n_rep=100):
    rng = np.random.default_rng(SEED)
    example = simulate_table(rng, "gamma mixture", 12.0)
    return example, replicate(rng, n_rep)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def below(x, tol=1e-5):
    """Agreement with a library, reported against a tolerance."""
    return f"below {tol:.0e}" if x < tol else f"{x:.1e}, above {tol:.0e}"


def numerical_hessian(f, theta, h=1e-4):
    k = len(theta)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            e_i, e_j = np.eye(k)[i] * h, np.eye(k)[j] * h
            H[i, j] = (f(theta + e_i + e_j) - f(theta + e_i - e_j) - f(theta - e_i + e_j)
                       + f(theta - e_i - e_j)) / (4 * h * h)
    return H


def main(n_rep=100):
    rng = np.random.default_rng(SEED)
    tab = simulate_table(rng, "gamma mixture", 12.0)
    time, event, x = tab["time"], tab["event"], tab["referral"]

    print("=" * 72)
    print("1. THREE TRUE RETENTION CURVES (PAID CHANNELS)")
    print("=" * 72)
    g12 = np.linspace(0, 12, 1201)
    for name, S in TRUTHS.items():
        diff = np.max(np.abs(S(g12) - TRUTHS["gamma mixture"](g12)))
        print(f"{name:13s}: S(6) {float(S(6.0)):.3f}, S(12) {float(S(12.0)):.3f}, S(24) {float(S(24.0)):.3f}, "
              f"S(60) {float(S(60.0)):.3f}; largest gap to the gamma mixture in months 0-12: {diff:.4f}; "
              f"value over 60 months EUR {true_value(name):.2f} (months 0-12 EUR {true_value(name, 12.0):.2f}, "
              f"months 12-60 EUR {true_value(name, 60.0, 12.0):.2f}, {pct(true_value(name, 60.0, 12.0) / true_value(name))} "
              "of the total)")
    print("Mean lifetime of a paid-channel customer: gamma mixture 25 / (1.5 - 1) = 50.0 months; Weibull "
          f"{21.3 * gamma_fn(1 + 1 / 0.9):.1f} months; loyal core infinite (30% never leave)")
    print(f"Referral customers: every lifetime stretched by a time ratio of {TIME_RATIO:.3f}")

    print()
    print("=" * 72)
    print("2. ONE TABLE AT 12 MONTHS (GAMMA MIXTURE TRUTH): SIX MODELS")
    print("=" * 72)
    print(f"Customers {len(time):,} (referral {x.sum():,}); churned {event.sum():,} ({pct(event.mean())})")
    res = analyse_table(tab, 12.0)
    fits = res["fits"]
    km0 = kaplan_meier(time[x == 0], event[x == 0])
    grid = np.linspace(0.05, 12, 240)
    print(f"Kaplan-Meier, paid channels: S(12) {float(km_at(km0, 12.0)):.3f}; observed value over months 0-12 "
          f"EUR {res['observed']:.2f}")
    print(f"{'model':13s} {'params':>6s} {'log-lik':>10s} {'dAIC':>6s} {'weight':>7s} {'gap to KM':>9s} "
          f"{'S(60)':>6s} {'time ratio':>10s} {'value 60m':>9s} {'tail 12-60':>10s} {'mean life':>9s}")
    for i, m in enumerate(MODELS):
        f = fits[m]
        gap = np.max(np.abs(survival(f, grid) - km_at(km0, grid)))
        mean = full_mean(f)
        print(f"{m:13s} {f['k']:6d} {f['loglik']:10.1f} {res['delta'][i]:6.1f} {res['weights'][i]:7.3f} "
              f"{gap:9.4f} {float(survival(f, 60.0)):6.3f} {np.exp(f['theta'][-1]):10.3f} "
              f"{res['values'][i]:9.2f} {res['tails'][i]:10.2f} {'infinite' if not np.isfinite(mean) else f'{mean:.1f}':>9s}")
    print(f"Range of 60-month values across the six models: EUR {res['values'].min():.2f} to {res['values'].max():.2f}; "
          f"across models within 10 AIC points of the best: EUR {res['values'][res['plausible']].min():.2f} to "
          f"{res['values'][res['plausible']].max():.2f}; true EUR {true_value('gamma mixture'):.2f}")
    gz = fits["Gompertz"]["theta"]
    print(f"Gompertz fit: hazard {float(hazard(fits['Gompertz'], 0.0)):.3f} at signup, "
          f"{float(hazard(fits['Gompertz'], 12.0)):.3f} at 12 months, {float(hazard(fits['Gompertz'], 60.0)):.3f} at 60; "
          f"share predicted never to leave exp(b / c) = {np.exp(np.exp(gz[0]) / gz[1]):.3f}")
    gm = fits["gamma mixture"]
    H = numerical_hessian(lambda th: neg_loglik(th, "gamma mixture", time, event, x), gm["theta"])
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    lo, hi = np.exp(gm["theta"][-1] + np.array([-1, 1]) * 1.959964 * se[-1])
    print(f"Gamma mixture model: shape {np.exp(gm['theta'][0]):.3f}, scale {np.exp(gm['theta'][1]):.2f} months "
          f"(true 1.5 and 25); time ratio for referral {np.exp(gm['theta'][-1]):.3f} [{lo:.3f}, {hi:.3f}] "
          f"(true {TIME_RATIO:.3f})")
    print(f"Value over 60 months of a referral customer: gamma mixture model EUR {model_value(gm, x=1):.2f}, "
          f"true EUR {true_value('gamma mixture', ratio=TIME_RATIO):.2f}; paid channels: model EUR "
          f"{model_value(gm):.2f}, true EUR {true_value('gamma mixture'):.2f}")
    t_grid = np.array([0.0, 6.0, 12.0, 24.0, 60.0])
    S = TRUTHS["gamma mixture"]
    ratio_t = [float((S(t / TIME_RATIO + 1e-6) - S(t / TIME_RATIO)) / TIME_RATIO / S(t / TIME_RATIO)
                     / ((S(t + 1e-6) - S(t)) / S(t))) for t in t_grid]
    print("True hazard ratio, referral against paid, at months 0, 6, 12, 24, 60: "
          + ", ".join(f"{r:.3f}" for r in ratio_t))
    for m in MODELS:
        print(f"  {m:13s} parameters " + ", ".join(f"{n} {v:+.4f}" for n, v in zip(FAMILIES[m][0] + ("log time ratio",), fits[m]["theta"])))
    # check against scipy's censored-data fitting (paid channels only, no covariate)
    from scipy import stats
    data = stats.CensoredData.right_censored(time[x == 0], event[x == 0] == 0)
    checks = []
    for m, dist in (("exponential", stats.expon), ("Weibull", stats.weibull_min), ("log-normal", stats.lognorm),
                    ("log-logistic", stats.fisk), ("gamma mixture", stats.lomax)):
        ours = fit_model(m, time[x == 0], event[x == 0], np.zeros((x == 0).sum()))
        params = dist.fit(data, floc=0)
        sp = dist(*params)
        ours_s = survival(ours, np.array([6.0, 12.0, 60.0]))
        checks.append((m, np.max(np.abs(ours_s - sp.sf(np.array([6.0, 12.0, 60.0])))),
                       ours["loglik"], float(np.sum(sp.logpdf(time[x == 0][event[x == 0] == 1]))
                                             + np.sum(sp.logsf(time[x == 0][event[x == 0] == 0])))))
    print("Check against scipy.stats fit to censored data (paid channels only, no covariate): largest "
          "difference in S(6), S(12), S(60) " + ", ".join(f"{m} {below(d)}" for m, d, _, _ in checks)
          + "; log-likelihoods " + ", ".join(f"{a:.2f} vs {b:.2f}" for _, _, a, b in checks))
    try:
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        y = np.array(list(zip(event.astype(bool), time)), dtype=[("e", bool), ("t", float)])
        hr = float(np.exp(CoxPHSurvivalAnalysis(ties="efron").fit(x[:, None].astype(float), y).coef_[0]))
        print(f"Cox model on the same table (scikit-survival): referral hazard ratio {hr:.3f}")
    except ImportError:
        print("Cox model on the same table: skipped (scikit-survival not installed)")
    print(f"Best AIC: {res['best']}; AIC-weighted value EUR {res['rules']['AIC average']:.2f}; "
          f"observed + lowest plausible tail EUR {res['rules']['observed + lowest tail']:.2f}")

    print()
    print("=" * 72)
    print(f"3. {n_rep} SIMULATED TABLES PER TRUTH AND WINDOW")
    print("=" * 72)
    reps = replicate(rng, n_rep)
    for truth in TRUTHS:
        tv = true_value(truth)
        print(f"Truth: {truth} (true 60-month value EUR {tv:.2f})")
        for window in WINDOWS:
            r = reps[(truth, window)]
            share = {m: np.mean([b == m for b in r["best"]]) for m in MODELS}
            vals = np.where(r["plausible"], r["values"], np.nan)
            spread = (np.nanmax(vals, axis=1) - np.nanmin(vals, axis=1)).mean()
            print(f"  window {window:4.0f} months: AIC picks " + ", ".join(f"{m} {pct(share[m], 0)}" for m in MODELS)
                  + f"; models within 10 AIC points: {r['plausible'].sum(axis=1).mean():.1f} on average, "
                  f"mean spread of their values EUR {spread:.2f}")
            if window == 12.0:
                print("      mean time ratio by model: " + ", ".join(
                    f"{m} {v:.3f}" for m, v in zip(MODELS, r["ratios"].mean(axis=0))))
            for rule in RULES:
                v = r["rules"][rule]
                err = v - tv
                print(f"      {rule:24s} mean error EUR {err.mean():+7.2f}, rmse {np.sqrt((err ** 2).mean()):6.2f}, "
                      f"middle 95% [{np.percentile(err, 2.5):+7.2f}, {np.percentile(err, 97.5):+7.2f}], "
                      f"overpays by >10% in {pct(np.mean(v > 1.1 * tv))}")


if __name__ == "__main__":
    main()
