"""Part 2 of the statistical-process-control series: the mechanism, from one chart to a production line.

Every number quoted in the article is printed by this script. Run:

    python mechanism_numbers.py > output.txt

No data are downloaded. Every chart is built from a simulated in-control Phase I sample, and its
conditional false-alarm probability is then computed exactly given its estimates (by the normal
distribution function, by numerical integration, or by Imhof's method for quadratic forms).

Sections
1. One chart, exactly: the conditional ARL by conditioning, its sensitivity to sigma_hat, the choice of
   estimator, and the condition under which the average conditional ARL is infinite.
2. Machines in parallel: the line-level conditional ARL of K charts, with limits widened (Sidak) so that the
   line would have an in-control ARL of 370.4 if the parameters were known.
3. Short runs and high mix: fixed limits from a handful of units after a changeover, the self-starting
   (Q) chart, what it does to a shift it does not catch, and pooling sigma across products.
4. Stations in series: the cause-selecting chart, with the regression of stage 2 on stage 1 estimated in
   Phase I, and what an upstream fault then does to the downstream chart.
5. Many sensors: Hotelling's T^2 with an estimated mean vector and covariance matrix.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from functools import lru_cache

import numpy as np
from scipy import integrate, optimize, special, stats

SEED = 20261107
CHILDREN = np.random.SeedSequence(SEED).spawn(16)
RNG = {name: np.random.default_rng(CHILDREN[i]) for i, name in enumerate(
    ["est_mc", "parallel", "parallel_pool", "parallel_het", "short_fixed", "short_q", "short_shift",
     "short_pool", "cause", "t2", "spare1", "spare2", "spare3", "spare4", "spare5", "spare6"])}

FIG: dict = {}  # data for figures.py, written to figure_data.json
QGRID = np.linspace(0.5, 99.5, 199)

PHI = stats.norm.cdf
PHIBAR = stats.norm.sf
P0 = 2 * PHIBAR(3.0)          # 0.0026998, the nominal false-alarm probability
ARL0 = 1 / P0                 # 370.398


def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def pct(x: np.ndarray, qs=(10, 50, 90)) -> list[float]:
    return [float(v) for v in np.percentile(x, qs)]


@lru_cache(maxsize=None)
def d2(n: int) -> float:
    val, _ = integrate.quad(lambda x: 1.0 - PHI(x) ** n - PHIBAR(x) ** n, -12, 12, limit=200)
    return val


def c4(k: int) -> float:
    """E[S]/sigma for a sample standard deviation with k - 1 degrees of freedom."""
    nu = k - 1
    return math.sqrt(2.0 / nu) * math.exp(special.gammaln(k / 2) - special.gammaln(nu / 2))


def p_cond(z: np.ndarray | float, s: np.ndarray | float, k: float = 3.0) -> np.ndarray:
    """Conditional false-alarm probability of a chart whose standardized centre-line error is z
    (in standard errors of the plotted statistic) and whose sigma estimate is s (in units of sigma)."""
    return PHIBAR(np.asarray(z) + k * np.asarray(s)) + PHI(np.asarray(z) - k * np.asarray(s))


def describe(label: str, carl: np.ndarray, with_mean: bool = True) -> dict[str, float]:
    q10, q50, q90 = pct(carl)
    out = {"p10": q10, "median": q50, "p90": q90, "below370": float(np.mean(carl < ARL0)),
           "below200": float(np.mean(carl < 200)), "below100": float(np.mean(carl < 100))}
    mean_part = ""
    if with_mean:
        out["mean"] = float(carl.mean())
        mean_part = f"mean {carl.mean():8.1f}; "
    print(f"    {label}: {mean_part}median {q50:7.1f}; 10th-90th {q10:7.1f}-{q90:8.1f};"
          f" below 370.4 {out['below370']:.3f}; below 200 {out['below200']:.3f}; below 100 {out['below100']:.3f}")
    return out


# ---------------------------------------------------------------------------------------------------
# 1. One chart, exactly
# ---------------------------------------------------------------------------------------------------


def s_dist(nu: int, unbiased: bool = True):
    """Distribution of s = sigma_hat/sigma when sigma_hat^2 = sigma^2 chi2_nu / nu (optionally / c4)."""
    scale = 1.0 / c4(nu + 1) if unbiased else 1.0
    # s = scale * sqrt(chi2_nu / nu)  ->  P(s <= x) = F_chi2(nu * (x/scale)^2)
    return scale


def carl_cdf_exact(t: float, m: int, n: int, nu: int, k: float = 3.0, unbiased: bool = True) -> float:
    """P(conditional ARL <= t) for a chart with centre line from m subgroups of n and an independent
    sigma estimate with nu degrees of freedom (pooled or sample standard deviation)."""
    scale = s_dist(nu, unbiased)
    sd_z = math.sqrt(1.0 / m)  # z = sqrt(n)(mu_hat - mu)/sigma ~ N(0, 1/m)
    target = 1.0 / t

    def inner(z: float) -> float:
        f = lambda s: float(p_cond(z, s, k)) - target  # noqa: E731  decreasing in s
        if f(1e-9) <= 0:
            return 0.0
        hi = 1.0
        while f(hi) > 0:
            hi *= 2
        s_star = optimize.brentq(f, 1e-9, hi, xtol=1e-12)
        return stats.chi2.cdf(nu * (s_star / scale) ** 2, nu)

    nodes, weights = np.polynomial.hermite_e.hermegauss(48)
    return float(sum(w * inner(sd_z * x) for x, w in zip(nodes, weights)) / math.sqrt(2 * math.pi))


def carl_quantile_exact(q: float, m: int, n: int, nu: int, k: float = 3.0) -> float:
    return math.exp(optimize.brentq(lambda lt: carl_cdf_exact(math.exp(lt), m, n, nu, k) - q,
                                    math.log(1.0), math.log(1e9), xtol=1e-9))


def aarl_truncated(m: int, nu: int, s_max: float, k: float = 3.0) -> float:
    """E[ 1/p ; s <= s_max ] for the same chart, by quadrature (the AARL as s_max -> infinity)."""
    scale = s_dist(nu, True)
    sd_z = math.sqrt(1.0 / m)
    nodes, weights = np.polynomial.hermite_e.hermegauss(48)
    dens = lambda s: stats.chi2.pdf(nu * (s / scale) ** 2, nu) * 2 * nu * s / scale ** 2  # noqa: E731

    def g(s: float) -> float:
        inv_p = sum(w / float(p_cond(sd_z * x, s, k)) for x, w in zip(nodes, weights)) / math.sqrt(2 * math.pi)
        return inv_p * dens(s)

    val, _ = integrate.quad(g, 0, s_max, limit=500, epsrel=1e-10)
    return val


def section_one_chart() -> None:
    header("1. ONE CHART, EXACTLY: THE CONDITIONAL ARL AND THE ESTIMATOR OF SIGMA")
    for kk in (3.0, sidak_k(10), sidak_k(50), sidak_k(500)):
        el = kk * stats.norm.pdf(kk) / PHIBAR(kk)
        print(f"  limits at {kk:.2f} sigma: d log ARL / d log sigma_hat = {el:.2f} (exact centre line); a 1% error in sigma_hat"
              f" moves the ARL by about {100 * (math.exp(el * math.log(1.01)) - 1):.1f}%")
    print("  X-bar chart, n = 5, m = 25 subgroups: three estimators of sigma, 200,000 simulated Phase I samples each")
    rng = RNG["est_mc"]
    reps, m, n = 200_000, 25, 5
    mu = np.empty(reps)
    s_pool = np.empty(reps)
    s_bar = np.empty(reps)
    r_bar = np.empty(reps)
    chunk = 20_000
    for st in range(0, reps, chunk):
        x = rng.standard_normal((chunk, m, n))
        mu[st:st + chunk] = x.mean(axis=(1, 2))
        sd = x.std(axis=2, ddof=1)
        s_pool[st:st + chunk] = np.sqrt((sd ** 2).mean(axis=1))
        s_bar[st:st + chunk] = sd.mean(axis=1)
        r_bar[st:st + chunk] = (x.max(axis=2) - x.min(axis=2)).mean(axis=1)
    z = math.sqrt(n) * mu
    ests = {
        "pooled S / c4(nu+1), nu = 100": s_pool / c4(m * (n - 1) + 1),
        "S-bar / c4(5)": s_bar / c4(n),
        "R-bar / d2(5)": r_bar / d2(n),
    }
    for name, s in ests.items():
        carl = 1 / p_cond(z, s)
        print(f"    {name}: sigma_hat/sigma mean {s.mean():.4f}, SD {s.std(ddof=1):.4f}")
        describe(f"  {name}", carl)
    print("  exact (by conditioning and quadrature), pooled S / c4, same chart:")
    nu = m * (n - 1)
    qx = [carl_quantile_exact(q, m, n, nu) for q in (0.10, 0.50, 0.90)]
    print(f"    10th / 50th / 90th percentile: {qx[0]:.1f} / {qx[1]:.1f} / {qx[2]:.1f}; P(conditional ARL < 370.4) ="
          f" {carl_cdf_exact(ARL0, m, n, nu):.4f}; P(< 200) = {carl_cdf_exact(200, m, n, nu):.4f}")
    print(f"    exact mean (AARL): {aarl_truncated(m, nu, 6.0):.1f}")
    print("  individuals chart, sigma from the sample standard deviation S/c4 of m values (nu = m - 1):")
    for m_i in (10, 11, 25, 50):
        nu_i = m_i - 1
        q = [carl_quantile_exact(qq, m_i, 1, nu_i) for qq in (0.10, 0.50, 0.90)]
        print(f"    m = {m_i:3d}: exact 10th / 50th / 90th percentile {q[0]:8.1f} / {q[1]:7.1f} / {q[2]:10.1f};"
              f" P(< 370.4) = {carl_cdf_exact(ARL0, m_i, 1, nu_i):.4f}")
    print("  the mean conditional ARL, integrated over sigma_hat up to s_max (individuals chart, S/c4):")
    for m_i in (10, 11, 25):
        vals = [aarl_truncated(m_i, m_i - 1, smax) for smax in (2.0, 3.0, 4.0, 5.0, 6.0)]
        print(f"    m = {m_i:3d} (nu = {m_i - 1:2d}): s_max 2 / 3 / 4 / 5 / 6 -> " + " / ".join(f"{v:.4g}" for v in vals))
    print("  divergence condition: 1/p grows like exp(4.5 s^2) while the density of s falls like exp(-nu c4^2 s^2 / 2);"
          " the mean is finite only if nu > 9 / c4(nu+1)^2:")
    for nu_i in (8, 9, 10, 11):
        print(f"    nu = {nu_i:2d}: 9 / c4^2 = {9 / c4(nu_i + 1) ** 2:.3f} -> {'finite' if nu_i > 9 / c4(nu_i + 1) ** 2 else 'infinite'}")


# ---------------------------------------------------------------------------------------------------
# 2. Machines in parallel
# ---------------------------------------------------------------------------------------------------


def sidak_k(K: int) -> float:
    per_chart = 1 - (1 - P0) ** (1 / K)
    return float(stats.norm.isf(per_chart / 2))


def phase1_xbar(rng, reps: int, m: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    mu = np.empty(reps)
    sig = np.empty(reps)
    per = max(1, 2_000_000 // (m * n))
    for st in range(0, reps, per):
        kk = min(per, reps - st)
        x = rng.standard_normal((kk, m, n))
        mu[st:st + kk] = x.mean(axis=(1, 2))
        sig[st:st + kk] = (x.max(axis=2) - x.min(axis=2)).mean(axis=1) / d2(n)
    return mu, sig


def section_parallel() -> None:
    header("2. MACHINES IN PARALLEL: THE LINE-LEVEL CONDITIONAL ARL")
    print("  K machines, one X-bar chart each (n = 5); the line signals when any chart signals.")
    print("  known parameters:")
    for K in (1, 10, 50, 500):
        line3 = 1 / (1 - (1 - P0) ** K)
        print(f"    K = {K:3d}: every chart at 3 sigma -> line ARL {line3:7.2f}; Sidak limits for a line ARL of 370.4:"
              f" k = {sidak_k(K):.3f} sigma per chart (per-chart ARL {1 / (1 - (1 - P0) ** (1 / K)):,.0f})")
    print("  estimated limits: each machine's own Phase I of m = 25 subgroups, sigma from Rbar/d2, limits at the Sidak k")
    rng = RNG["parallel"]
    n_lines = {1: 20_000, 10: 4_000, 50: 2_000, 500: 400}
    summary = {}
    for K, L in n_lines.items():
        k = sidak_k(K)
        mu, sig = phase1_xbar(rng, L * K, 25, 5)
        pk = p_cond(math.sqrt(5) * mu, sig, k).reshape(L, K)
        line_p = 1 - np.prod(1 - pk, axis=1)
        carl = 1 / line_p
        ep_nominal = 1 - (1 - P0) ** (1 / K)
        top1 = pk.max(axis=1) / pk.sum(axis=1)
        ntop = max(1, K // 10)
        top10 = np.sort(pk, axis=1)[:, -ntop:].sum(axis=1) / pk.sum(axis=1)
        summary[K] = describe(f"K = {K:3d} ({L:,} simulated lines)", carl, with_mean=False)
        FIG.setdefault("parallel", {})[str(K)] = np.percentile(carl, QGRID).tolist()
        print(f"      mean per-chart false-alarm probability {pk.mean():.2e} against nominal {ep_nominal:.2e} (ratio {pk.mean() / ep_nominal:.2f});"
              f" the narrowest chart carries a median {np.median(top1):.1%} of the line's false-alarm probability"
              + (f", the narrowest 10% of charts {np.median(top10):.1%}" if K >= 10 else ""))
    print("  pooled Phase I when the K = 50 machines are identical: sigma from all 50 x 25 subgroups, machine means own:")
    rng = RNG["parallel_pool"]
    K, L = 50, 2_000
    k = sidak_k(K)
    mu, sig = phase1_xbar(rng, L * K, 25, 5)
    mu = mu.reshape(L, K)
    sig_pool = sig.reshape(L, K).mean(axis=1, keepdims=True)
    pk = p_cond(math.sqrt(5) * mu, sig_pool, k)
    describe("K =  50, pooled sigma, identical machines", 1 / (1 - np.prod(1 - pk, axis=1)), with_mean=False)
    print("  pooled sigma when the machines differ: true sigma_k lognormal with log-SD 0.10 around the pooled value:")
    rng = RNG["parallel_het"]
    sig_true = np.exp(0.10 * rng.standard_normal((L, K)))
    sig_true /= np.exp(0.5 * 0.10 ** 2)  # mean 1 across machines
    mu2, sig2 = phase1_xbar(rng, L * K, 25, 5)
    mu2 = mu2.reshape(L, K)                          # in units of each machine's own sigma
    sig_pool2 = (sig2.reshape(L, K) * sig_true).mean(axis=1, keepdims=True)  # pooled in common units
    pk = p_cond(math.sqrt(5) * mu2, sig_pool2 / sig_true, k)
    describe("K =  50, pooled sigma, machines differ", 1 / (1 - np.prod(1 - pk, axis=1)), with_mean=False)
    print(f"    a machine whose sigma is 10% above the pooled value has limits at {k / 1.1:.3f} of its own sigma: ARL"
          f" {1 / (2 * PHIBAR(k / 1.1)):,.0f} against the per-chart target {1 / (1 - (1 - P0) ** (1 / K)):,.0f}")


# ---------------------------------------------------------------------------------------------------
# 3. Short runs and high mix
# ---------------------------------------------------------------------------------------------------


def q_threshold(r: int) -> float:
    """Threshold on the standardized prediction error of point r of a self-starting (Q) chart so that
    |Q_r| > 3 exactly: T has r - 2 degrees of freedom."""
    return float(stats.t.isf(PHIBAR(3.0), r - 2))


def self_starting_run(rng, x0: np.ndarray, inner: int, delta: float = 0.0, shift_at: int | None = None,
                      max_steps: int = 30_000) -> np.ndarray:
    """Run lengths of a self-starting individuals chart (Quesenberry's Q chart for unknown mean and
    variance) after the first m values x0 (shape (outer, m)), for `inner` continuations of each.
    Monitoring starts at point m + 1; running mean and variance are updated after every point."""
    outer, m = x0.shape
    n_ch = outer * inner
    cnt = np.full(n_ch, m, dtype=np.int64)
    mean = np.repeat(x0.mean(axis=1), inner)
    m2 = np.repeat(((x0 - x0.mean(axis=1, keepdims=True)) ** 2).sum(axis=1), inner)
    rl = np.zeros(n_ch, dtype=np.int64)
    active = np.arange(n_ch)
    step = 0
    while active.size and step < max_steps:
        step += 1
        r = m + step
        x = rng.standard_normal(active.size)
        if delta and shift_at is not None and r >= shift_at:
            x = x + delta
        c = cnt[active]
        mu = mean[active]
        s = np.sqrt(m2[active] / (c - 1))
        stat = np.sqrt((r - 1) / r) * (x - mu) / s
        sig = np.abs(stat) > q_threshold(r)
        # update the running estimates with the new point (Welford)
        c1 = c + 1
        d = x - mu
        mu_new = mu + d / c1
        m2[active] += d * (x - mu_new)
        mean[active] = mu_new
        cnt[active] = c1
        rl[active[sig]] = step
        active = active[~sig]
    rl[active] = max_steps
    return rl.reshape(outer, inner)


def section_short_runs() -> None:
    header("3. SHORT RUNS AND HIGH MIX: A HANDFUL OF UNITS AFTER A CHANGEOVER")
    print("  fixed limits from the first m individual values after a changeover, sigma from MRbar/d2(2); 100,000 runs each:")
    rng = RNG["short_fixed"]
    for m in (5, 10, 20, 30):
        x = rng.standard_normal((100_000, m))
        mu = x.mean(axis=1)
        s = np.abs(np.diff(x, axis=1)).mean(axis=1) / d2(2)
        p = p_cond(mu, s)
        carl = 1 / p
        describe(f"m = {m:2d}, MRbar", carl, with_mean=False)
        print(f"      share of runs whose conditional false-alarm probability exceeds 1%: {np.mean(p > 0.01):.3f};"
              f" probability of at least one false alarm in the next 50 units {np.mean(1 - (1 - p) ** 50):.3f} (known limits"
              f" {1 - (1 - P0) ** 50:.3f})")
    print("  self-starting (Q) chart: unconditionally the Q statistics are independent standard normals, so the in-control ARL")
    print(f"  is exactly {ARL0:.1f}. Conditionally on the first m values, nested simulation: 300 starting samples x 2,000"
          " continuations each:")
    rng = RNG["short_q"]
    rng_f = RNG["short_pool"]
    for m in (5, 10, 20):
        x0 = rng.standard_normal((300, m))
        rl = self_starting_run(rng, x0, 2_000)
        carl = rl.mean(axis=1)
        se = rl.std(axis=1, ddof=1) / math.sqrt(rl.shape[1])
        noise_var = float(np.mean(se ** 2))
        sd_true = math.sqrt(max(carl.var(ddof=1) - noise_var, 0.0))
        q10, q50, q90 = pct(carl)
        # fixed MRbar limits from the same starting samples
        mu = x0.mean(axis=1)
        s = np.abs(np.diff(x0, axis=1)).mean(axis=1) / d2(2)
        fixed = 1 / p_cond(mu, s)
        f10, f50, f90 = pct(fixed)
        print(f"    m = {m:2d}: self-starting conditional ARL mean {carl.mean():.1f}, median {q50:.1f}, 10th-90th {q10:.1f}-{q90:.1f};"
              f" SD across starts {carl.std(ddof=1):.1f} of which simulation noise {math.sqrt(noise_var):.1f}"
              f" (noise-corrected SD {sd_true:.1f}); share below 370.4 {np.mean(carl < ARL0):.3f}")
        print(f"            fixed MRbar limits from the same {m} values: median {f50:.1f}, 10th-90th {f10:.1f}-{f90:.1f}")
        FIG.setdefault("selfstart", {})[str(m)] = {"q": carl.tolist(), "fixed": fixed.tolist()}
    print("  a shift the self-starting chart does not catch at once is absorbed into its estimates. Shift of delta (in sigma)")
    print("  from unit 21 onwards; 20,000 runs; counted only for runs with no signal before the shift:")
    rng = RNG["short_shift"]
    for delta in (1.0, 2.0):
        x0 = rng.standard_normal((20_000, 2))
        rl = self_starting_run(rng, x0, 1, delta=delta, shift_at=21, max_steps=400).ravel()
        signal_unit = rl + 2
        ok = signal_unit >= 21
        after = signal_unit[ok] - 21
        # fixed limits from units 1..20 (known to be in control), same shift from unit 21
        x = rng.standard_normal((20_000, 20))
        mu = x.mean(axis=1)
        s = np.abs(np.diff(x, axis=1)).mean(axis=1) / d2(2)
        p_shift = p_cond(mu - delta, s)
        us = np.arange(1, 381)
        FIG.setdefault("shift", {})[str(int(delta))] = {
            "u": us.tolist(),
            "q": [float(np.mean(after < u)) for u in us],
            "fixed": [float(np.mean(1 - (1 - p_shift) ** u)) for u in us]}
        p50 = 1 - (1 - p_shift) ** 50
        p200 = 1 - (1 - p_shift) ** 200
        print(f"    delta = {delta:.0f}: self-starting, P(signal within 50 units of the shift) {np.mean(after < 50):.3f};"
              f" P(no signal within 200 units) {np.mean(after >= 200):.3f}; P(no signal within 380 units) {np.mean(after >= 380):.3f}"
              f" ({ok.sum():,} runs without a signal before the shift)")
        print(f"               fixed limits from the 20 in-control units: P(signal within 50) {p50.mean():.3f};"
              f" P(no signal within 200) {(1 - p200).mean():.3f}")
    print("  why: after k shifted points the running estimates (from 20 in-control points plus k shifted ones) have expected")
    print("  mean delta k/(20+k) and variance about 1 + delta^2 * 20 k/(20+k)^2, so the shift is absorbed:")
    for delta in (1.0, 2.0):
        cells = []
        for kk in (10, 50, 100):
            mean_k = delta * kk / (20 + kk)
            var_k = 1 + delta ** 2 * 20 * kk / (20 + kk) ** 2
            z_next = (delta - mean_k) / math.sqrt(var_k)
            cells.append(f"k = {kk:3d}: mean {mean_k:.2f}, SD {math.sqrt(var_k):.2f}, next shifted point sits {z_next:.2f} SD from the mean")
        print(f"    delta = {delta:.0f}: " + "; ".join(cells))
    print("  high mix: P products, m = 5 units each, sigma pooled across products (nu = P(m - 1)), product means own:")
    rng = RNG["short_pool"]
    for P in (1, 4, 10, 40):
        reps = 100_000
        x = rng.standard_normal((reps, P, 5))
        mu = x[:, 0, :].mean(axis=1)                     # the product being charted is product 0
        s = np.sqrt(x.var(axis=2, ddof=1).mean(axis=1)) / c4(P * 4 + 1)
        describe(f"P = {P:2d} products (nu = {4 * P:3d})", 1 / p_cond(mu, s), with_mean=False)
    for r in (0.8, 0.9, 1.1, 1.2):
        print(f"    a product whose sigma is {r:.1f} times the pooled value (estimates exact): limits at {3 / r:.2f} of its own sigma,"
              f" ARL {1 / (2 * PHIBAR(3 / r)):,.1f}")


# ---------------------------------------------------------------------------------------------------
# 4. Stations in series: cause-selecting charts
# ---------------------------------------------------------------------------------------------------


def section_cause_selecting() -> None:
    header("4. STATIONS IN SERIES: THE CAUSE-SELECTING CHART WITH AN ESTIMATED REGRESSION")
    beta, sig_e = 0.8, 0.6
    print(f"  stage 1: Y1 ~ N(0, 1); stage 2: Y2 = {beta} Y1 + e, e ~ N(0, {sig_e}^2), so Y2 has standard deviation"
          f" {math.sqrt(beta ** 2 + sig_e ** 2):.2f}; correlation {beta / math.sqrt(beta ** 2 + sig_e ** 2):.2f}")
    print("  a fault at stage 1 shifts the mean of Y1 by delta1; stage 2 is unchanged.")
    print("  known parameters:")
    for d1 in (0.0, 1.0, 2.0, 3.0):
        sy = math.sqrt(beta ** 2 + sig_e ** 2)
        p_naive = p_cond(-beta * d1 / sy, 1.0)
        print(f"    delta1 = {d1:.0f}: chart of Y2 itself ARL {1 / p_naive:8.2f}; cause-selecting chart of the residual ARL {ARL0:.2f}")
    rng = RNG["cause"]
    reps = 100_000
    for m in (25, 50, 100):
        y1 = rng.standard_normal((reps, m))
        y2 = beta * y1 + sig_e * rng.standard_normal((reps, m))
        xbar = y1.mean(axis=1, keepdims=True)
        ybar = y2.mean(axis=1, keepdims=True)
        sxx = ((y1 - xbar) ** 2).sum(axis=1)
        b_hat = ((y1 - xbar) * (y2 - ybar)).sum(axis=1) / sxx
        a_hat = ybar[:, 0] - b_hat * xbar[:, 0]
        resid = y2 - a_hat[:, None] - b_hat[:, None] * y1
        se_hat = np.sqrt((resid ** 2).sum(axis=1) / (m - 2))
        print(f"  estimated regression from m = {m} Phase I pairs; residual chart limits at +-3 sigma_e_hat; 100,000 charts:")
        print(f"    slope error beta_hat - beta: SD {np.std(b_hat - beta, ddof=1):.4f}; sigma_e_hat/sigma_e SD {np.std(se_hat / sig_e, ddof=1):.4f}")
        for d1 in (0.0, 1.0, 2.0, 3.0):
            mean_r = -a_hat + (beta - b_hat) * d1     # residual mean when Y1 ~ N(d1, 1)
            sd_r = np.sqrt((beta - b_hat) ** 2 + sig_e ** 2)
            p = PHIBAR((3 * se_hat - mean_r) / sd_r) + PHI((-3 * se_hat - mean_r) / sd_r)
            out = describe(f"delta1 = {d1:.0f}", 1 / p, with_mean=False)
            FIG.setdefault("cause", {}).setdefault(str(m), {})[str(int(d1))] = [out["p10"], out["median"], out["p90"]]
            if d1 > 0:
                print(f"      share of charts that blame stage 2 at least twice as often as when stage 1 is in control"
                      f" (per-point rate ratio >= 2): {np.mean(p >= 2 * (PHIBAR((3 * se_hat + a_hat) / sd_r) + PHI((-3 * se_hat + a_hat) / sd_r))):.3f}")
        if m == 25:
            # comparison: an individuals chart of Y1 alone with the same m, S/c4
            s_y1 = y1.std(axis=1, ddof=1) / c4(m)
            describe("for comparison, individuals chart of Y1, m = 25, S/c4", 1 / p_cond(xbar[:, 0], s_y1), with_mean=False)


# ---------------------------------------------------------------------------------------------------
# 5. Many sensors: Hotelling's T^2 with estimated parameters
# ---------------------------------------------------------------------------------------------------


IMHOF_FALLBACKS = [0, 0.0]  # number of evaluations where QUADPACK warned, largest disagreement between splits


def _imhof_once(x: float, lam: np.ndarray, c2: np.ndarray, a: float) -> float:
    def th0(u: float) -> float:
        lu = lam * u
        return 0.5 * float(np.sum(np.arctan(lu) + c2 * lu / (1 + lu ** 2)))

    def rho(u: float) -> float:
        lu = lam * u
        return math.exp(0.25 * float(np.sum(np.log1p(lu ** 2))) + 0.5 * float(np.sum(c2 * lu ** 2 / (1 + lu ** 2))))

    w = 0.5 * x
    v1, _ = integrate.quad(lambda u: math.sin(th0(u) - w * u) / (u * rho(u)), 0, a, limit=500,
                           epsabs=1e-13, epsrel=1e-11)
    v2, _ = integrate.quad(lambda u: math.sin(th0(u)) / (u * rho(u)), a, np.inf, weight="cos", wvar=w,
                           limlst=200, limit=500, epsabs=1e-13)
    v3, _ = integrate.quad(lambda u: math.cos(th0(u)) / (u * rho(u)), a, np.inf, weight="sin", wvar=w,
                           limlst=200, limit=500, epsabs=1e-13)
    return 0.5 + (v1 + v2 - v3) / math.pi


def imhof_sf(x: float, lam: np.ndarray, c2: np.ndarray) -> float:
    """P(sum_i lam_i W_i^2 > x) with W_i ~ N(c_i, 1) independent (Imhof 1961); c2 = c_i^2.
    The integral is split at u = a: [0, a] by ordinary quadrature, [a, inf) as Fourier integrals
    (QUADPACK QAWF) of the slowly varying parts against cos(x u / 2) and sin(x u / 2). If QUADPACK warns,
    the value is recomputed with two other split points and the median of the three is used."""
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        v = _imhof_once(x, lam, c2, 1.0)
    if caught:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alts = [_imhof_once(x, lam, c2, 0.5), _imhof_once(x, lam, c2, 2.0)]
        IMHOF_FALLBACKS[0] += 1
        IMHOF_FALLBACKS[1] = max(IMHOF_FALLBACKS[1], max(abs(v - t) for t in alts))
        v = float(np.median([v] + alts))
    return v


def t2_limits(p: int, m: int) -> tuple[float, float]:
    chi = float(stats.chi2.isf(P0, p))
    f = p * (m + 1) * (m - 1) / (m * (m - p)) * float(stats.f.isf(P0, p, m - p))
    return chi, f


def section_t2() -> None:
    header("5. MANY SENSORS: HOTELLING'S T^2 WITH AN ESTIMATED MEAN VECTOR AND COVARIANCE MATRIX")
    print("  individual observations in p dimensions; Phase I of m observations gives x_bar and S; Phase II statistic")
    print("  (x - x_bar)' S^-1 (x - x_bar). Limits: the known-parameter chi-square limit, or the F-based limit")
    print("  p(m+1)(m-1)/(m(m-p)) F(p, m-p) that is right on average over Phase I samples. Conditional false-alarm")
    print("  probability of each chart by Imhof's method; 400 Phase I samples per cell (invariance: Sigma = I wlog).")
    rng = RNG["t2"]
    sanity = imhof_sf(float(stats.chi2.isf(P0, 10)), np.ones(10), np.zeros(10))
    print(f"  check: Imhof's method with known parameters, p = 10, gives {sanity:.7f} (exact {P0:.7f})")
    results = {}
    for p in (2, 5, 10, 20, 50):
        for m in (50, 100, 250, 1000):
            if m <= p + 5:
                continue
            chi, flim = t2_limits(p, m)
            pc, pf = [], []
            for _ in range(400):
                x = rng.standard_normal((m, p))
                xb = x.mean(axis=0)
                S = np.cov(x, rowvar=False)
                lam, U = np.linalg.eigh(np.linalg.inv(S))
                c2 = (U.T @ xb) ** 2
                pc.append(imhof_sf(chi, lam, c2))
                pf.append(imhof_sf(flim, lam, c2))
            carl_c = 1 / np.array(pc)
            carl_f = 1 / np.array(pf)
            qc = pct(carl_c)
            qf = pct(carl_f)
            results[(p, m)] = (qc, qf, float(np.mean(carl_c < ARL0)), float(np.mean(carl_f < ARL0)))
            FIG.setdefault("t2", []).append({"p": p, "m": m, "chi": qc, "f": qf})
            print(f"    p = {p:2d}, m = {m:4d}: chi-square limit {chi:6.2f} -> median {qc[1]:7.1f} (10th-90th {qc[0]:7.1f}-{qc[2]:7.1f},"
                  f" below 370.4 {np.mean(carl_c < ARL0):.2f}); F limit {flim:6.2f} -> median {qf[1]:7.1f}"
                  f" (10th-90th {qf[0]:7.1f}-{qf[2]:8.1f}, below 370.4 {np.mean(carl_f < ARL0):.2f})")
    print(f"  Imhof evaluations on which QUADPACK warned and the value was recomputed: {IMHOF_FALLBACKS[0]};"
          f" largest disagreement between split points {IMHOF_FALLBACKS[1]:.1e}")
    return results


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    print("Statistical process control, Part 2: the mechanism, from one chart to a production line")
    print(f"numpy {np.__version__}; master seed {SEED}")
    print(f"nominal false-alarm probability {P0:.7f}; nominal in-control ARL {ARL0:.3f}")
    section_one_chart()
    section_parallel()
    section_short_runs()
    section_cause_selecting()
    section_t2()
    (Path(__file__).parent / "figure_data.json").write_text(json.dumps(FIG, indent=0))


if __name__ == "__main__":
    main()
