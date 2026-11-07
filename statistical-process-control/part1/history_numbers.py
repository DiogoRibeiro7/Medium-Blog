"""Part 1 of the statistical-process-control series: where the 370 comes from, with the numbers recomputed.

Every number quoted in the article is printed by this script. Run:

    python history_numbers.py > output.txt

No data are downloaded. The "data" of this part are the numbers that the history of the control chart
handed down -- Shewhart's three sigma, the 0.0027 and the 370, the Western Electric and Nelson tests,
the British action and warning limits, and the first results on limits estimated from a short record --
and the script recomputes each of them from its definition.

Sections
1. Three sigma: the normal tail against the distribution-free bounds Shewhart could appeal to.
2. The run length of a chart with known parameters: geometric, with mean 370.4 and median 257.
3. The Western Electric zone tests: exact average run lengths by Markov chain, checked by simulation;
   Nelson's eight tests by simulation.
4. The British convention: action limits at 3.09 sigma, warning limits at 1.96 sigma.
5. Known parameters, other distributions: the in-control ARL of three-sigma limits for individual values
   and for means of five.
6. Estimated parameters, a first look: the conditional in-control ARL of X-bar and individuals charts
   whose limits come from 20 to 1000 Phase I subgroups, the adjustment that restores the average
   false-alarm probability, and the correlation that estimated limits put between alarms.
"""

from __future__ import annotations

import math
import sys
from functools import lru_cache

import numpy as np
from scipy import integrate, optimize, sparse, stats
from scipy.sparse.linalg import spsolve

SEED = 19240516  # the date of Shewhart's memo, 16 May 1924
SS = np.random.SeedSequence(SEED)
RNG_WE, RNG_NELSON, RNG_XBAR, RNG_IND, RNG_GRID, RNG_EXTRA, RNG_PS, RNG_1200 = (np.random.default_rng(s) for s in SS.spawn(8))

PHI = stats.norm.cdf
PHIBAR = stats.norm.sf


def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fmt_arl(x: float) -> str:
    if not np.isfinite(x):
        return "infinite"
    if x >= 1e6:
        return f"{x:.3g}"
    return f"{x:,.1f}"


# ---------------------------------------------------------------------------------------------------
# Constants of the range
# ---------------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def d2(n: int) -> float:
    """Expected range of n standard normal observations: integral of 1 - Phi^n - (1 - Phi)^n."""
    val, _ = integrate.quad(lambda x: 1.0 - PHI(x) ** n - PHIBAR(x) ** n, -12, 12, limit=200)
    return val


# ---------------------------------------------------------------------------------------------------
# 1. Three sigma and the bounds
# ---------------------------------------------------------------------------------------------------


def section_bounds() -> None:
    header("1. THREE SIGMA: THE NORMAL TAIL AGAINST DISTRIBUTION-FREE BOUNDS")
    print("  P(|X - mu| >= k sigma) for limits at k sigma, and the in-control ARL = 1/P for independent points")
    print("  bound 'Chebyshev' holds for every distribution with finite variance: 1/k^2")
    print("  bound 'unimodal' holds when the distribution is unimodal (Vysochanskii-Petunin, k >= sqrt(8/3));")
    print("    with mode equal to mean it is the Gauss / Camp-Meidell bound, the same 4/(9 k^2) for k >= 2/sqrt(3)")
    for k in (1.0, 1.5, 1.96, 2.0, 2.5, 3.0, 3.09, 3.5, 4.0):
        p_norm = 2 * PHIBAR(k)
        cheb = min(1.0, 1 / k ** 2)
        uni = 4 / (9 * k ** 2) if k >= math.sqrt(8 / 3) else float("nan")
        uni_s = f"{uni:.5f} (ARL >= {1 / uni:7.2f})" if np.isfinite(uni) else "n/a (k < sqrt(8/3))"
        print(f"  k = {k:4.2f}: normal {p_norm:.6f} (ARL {1 / p_norm:9.2f});  Chebyshev {cheb:.5f} (ARL >= {1 / cheb:6.2f});"
              f"  unimodal {uni_s}")
    p3 = 2 * PHIBAR(3.0)
    print(f"  normal, k = 3, two-sided: p = {p3:.7f}; 1/p = {1 / p3:.3f}")
    print(f"  normal, k = 3, one side only: p = {PHIBAR(3.0):.7f}; 1/p = {1 / PHIBAR(3.0):.3f}")
    print(f"  ratio of the Chebyshev bound to the normal tail at k = 3: {(1 / 9) / p3:.1f}; of the unimodal bound: {(4 / 81) / p3:.1f}")
    # what k the bounds would need to deliver the normal 0.0027
    k_cheb = math.sqrt(1 / p3)
    k_uni = math.sqrt(4 / (9 * p3))
    print(f"  limits needed for a guaranteed false-alarm probability of {p3:.4f}: Chebyshev k = {k_cheb:.2f}; unimodal k = {k_uni:.2f}")


# ---------------------------------------------------------------------------------------------------
# 2. Run length with known parameters
# ---------------------------------------------------------------------------------------------------


def geometric_summary(p: float) -> tuple[float, float, int]:
    arl = 1 / p
    sdrl = math.sqrt(1 - p) / p
    median = math.ceil(math.log(0.5) / math.log(1 - p))
    return arl, sdrl, median


def section_geometric() -> None:
    header("2. THE RUN LENGTH OF A THREE-SIGMA CHART WITH KNOWN PARAMETERS")
    p = 2 * PHIBAR(3.0)
    arl, sdrl, med = geometric_summary(p)
    print(f"  in control: p = {p:.6f}; run length geometric with mean {arl:.2f}, SD {sdrl:.2f}, median {med}")
    for q in (0.05, 0.10, 0.25, 0.75, 0.90, 0.95):
        print(f"    {int(q * 100):2d}th percentile of the run length: {math.ceil(math.log(1 - q) / math.log(1 - p))}")
    print(f"  probability that the first false alarm comes before point 370: {1 - (1 - p) ** 369:.4f}")
    for horizon in (25, 50, 100, 250, 350, 500, 1000, 2000):
        print(f"  probability of at least one false alarm in {horizon:4d} in-control points: {1 - (1 - p) ** horizon:.4f};"
              f" expected number {horizon * p:.3f}")
    print("  out of control, X-bar chart of subgroups of n, process mean shifted by delta (sigma of individual values):")
    for n in (1, 4, 5):
        row = []
        for delta in (0.5, 1.0, 1.5, 2.0, 3.0):
            s = delta * math.sqrt(n)
            pp = PHIBAR(3 - s) + PHI(-3 - s)
            row.append(f"delta {delta:.1f}: ARL {1 / pp:7.2f}")
        print(f"    n = {n}:  " + "; ".join(row))


# ---------------------------------------------------------------------------------------------------
# 3. Western Electric zone tests (Markov chain) and Nelson's tests (simulation)
# ---------------------------------------------------------------------------------------------------

ZONES = (3, 2, 1, -1, -2, -3)  # A+, B+, C+, C-, B-, A-  (|code| = 3 means between 2 and 3 sigma)


def zone_probs(delta: float) -> dict[int, float]:
    """Probabilities of the six zones inside the 3-sigma limits for a statistic N(delta, 1)."""
    edges = {3: (2, 3), 2: (1, 2), 1: (0, 1), -1: (-1, 0), -2: (-2, -1), -3: (-3, -2)}
    return {z: PHI(b - delta) - PHI(a - delta) for z, (a, b) in edges.items()}


def we_signal(window: tuple[int, ...], run: int, rules: frozenset[int]) -> bool:
    """Do the supplementary tests fire on the window that ends with the newest point?
    `run` is the signed number of consecutive points on the newest point's side of the centre line."""
    if 2 in rules:  # 2 of the last 3 beyond 2 sigma on the same side
        last3 = window[-3:]
        if sum(z == 3 for z in last3) >= 2 or sum(z == -3 for z in last3) >= 2:
            return True
    if 3 in rules:  # 4 of the last 5 beyond 1 sigma on the same side
        last5 = window[-5:]
        if sum(z >= 2 for z in last5) >= 4 or sum(z <= -2 for z in last5) >= 4:
            return True
    if 4 in rules and abs(run) >= 8:  # 8 in a row on the same side of the centre line
        return True
    return False


def we_arl(rules: frozenset[int], delta: float = 0.0) -> tuple[float, int]:
    """Exact ARL of a Shewhart chart with test 1 plus the given supplementary tests.

    The state is the zones of the last few points (as many as tests 2 and 3 can still use) and the
    signed length of the current run on one side of the centre line (for test 4); signalling
    transitions are absorbing.  Returns the ARL and the number of transient states.
    """
    keep = 0
    if 2 in rules:
        keep = max(keep, 2)
    if 3 in rules:
        keep = max(keep, 4)
    track_run = 4 in rules
    probs = zone_probs(delta)
    start: tuple[tuple[int, ...], int] = ((), 0)
    index: dict[tuple[tuple[int, ...], int], int] = {start: 0}
    frontier = [start]
    rows, cols, vals = [], [], []
    while frontier:
        nxt = []
        for state in frontier:
            i = index[state]
            last, run = state
            for z in ZONES:
                window = last + (z,)
                side = 1 if z > 0 else -1
                new_run = run + side if (run > 0) == (side > 0) and run != 0 else side
                if we_signal(window, new_run, rules):
                    continue
                new = (window[-keep:] if keep else (), new_run if track_run else 0)
                if new not in index:
                    index[new] = len(index)
                    nxt.append(new)
                rows.append(i)
                cols.append(index[new])
                vals.append(probs[z])
        frontier = nxt
    size = len(index)
    q = sparse.csr_matrix((vals, (rows, cols)), shape=(size, size))
    arl = spsolve((sparse.identity(size, format="csr") - q).tocsc(), np.ones(size))
    return float(arl[0]), size


def simulate_run_lengths(rng: np.random.Generator, chains: int, tests: str, delta: float = 0.0,
                         max_steps: int = 20000) -> np.ndarray:
    """Simulate run lengths of a standardized Shewhart chart under the Western Electric ('we')
    tests 1-4 or Nelson's ('nelson') tests 1-8, for N(delta, 1) points."""
    rl = np.zeros(chains, dtype=np.int64)
    active = np.arange(chains)
    hist = np.zeros((chains, 15))  # last 15 standardized values of each active chain (newest last)
    count = np.zeros(chains, dtype=np.int64)
    run_side = np.zeros(chains, dtype=np.int64)       # signed run length on one side of the centre
    run_up = np.zeros(chains, dtype=np.int64)          # consecutive increases
    run_down = np.zeros(chains, dtype=np.int64)        # consecutive decreases
    run_alt = np.zeros(chains, dtype=np.int64)         # consecutive alternations of the differences
    last_diff = np.zeros(chains)
    run_inner = np.zeros(chains, dtype=np.int64)       # consecutive points within 1 sigma
    run_outer = np.zeros(chains, dtype=np.int64)       # consecutive points beyond 1 sigma (either side)
    t = 0
    while active.size and t < max_steps:
        t += 1
        z = rng.standard_normal(active.size) + delta
        h = hist[active]
        h[:, :-1] = h[:, 1:]
        h[:, -1] = z
        hist[active] = h
        c = count[active] + 1
        count[active] = c
        sig = np.abs(z) > 3
        # same-side counts in windows of 3 and 5 (partial windows at the start)
        w3 = h[:, -3:]
        w5 = h[:, -5:]
        valid3 = np.arange(3)[None, :] >= 3 - np.minimum(c, 3)[:, None]
        valid5 = np.arange(5)[None, :] >= 5 - np.minimum(c, 5)[:, None]
        sig |= (((w3 > 2) & valid3).sum(1) >= 2) | (((w3 < -2) & valid3).sum(1) >= 2)
        sig |= (((w5 > 1) & valid5).sum(1) >= 4) | (((w5 < -1) & valid5).sum(1) >= 4)
        side = np.sign(z).astype(np.int64)
        rs = run_side[active]
        rs = np.where(np.sign(rs) == side, rs + side, side)
        run_side[active] = rs
        if tests == "we":
            sig |= np.abs(rs) >= 8
        elif tests == "nelson":
            sig |= np.abs(rs) >= 9
            d = z - h[:, -2]
            first = c == 1
            up = np.where(first, 0, np.where(d > 0, run_up[active] + 1, 0))
            down = np.where(first, 0, np.where(d < 0, run_down[active] + 1, 0))
            run_up[active] = up
            run_down[active] = down
            sig |= (up >= 5) | (down >= 5)               # six points steadily increasing or decreasing
            ld = last_diff[active]
            alt = np.where(c <= 2, 0, np.where(d * ld < 0, run_alt[active] + 1, 0))
            run_alt[active] = alt
            last_diff[active] = np.where(first, 0.0, d)
            sig |= alt >= 12                              # fourteen points alternating up and down
            inner = np.where(np.abs(z) < 1, run_inner[active] + 1, 0)
            run_inner[active] = inner
            sig |= inner >= 15                            # fifteen in a row within one sigma
            outer = np.where(np.abs(z) > 1, run_outer[active] + 1, 0)
            run_outer[active] = outer
            sig |= outer >= 8                             # eight in a row beyond one sigma, either side
        elif tests != "shewhart":
            raise ValueError(tests)
        done = active[sig]
        rl[done] = t
        active = active[~sig]
    if active.size:
        raise RuntimeError("some chains did not signal")
    return rl


def section_runs_rules() -> None:
    header("3. THE WESTERN ELECTRIC ZONE TESTS AND NELSON'S EIGHT TESTS")
    print("  tests: 1 = one point beyond 3 sigma; 2 = two of three beyond 2 sigma, same side;")
    print("         3 = four of five beyond 1 sigma, same side; 4 = eight in a row on one side of the centre line")
    combos = [frozenset({1}), frozenset({1, 2}), frozenset({1, 3}), frozenset({1, 4}), frozenset({1, 2, 3}),
              frozenset({1, 2, 4}), frozenset({1, 3, 4}), frozenset({1, 2, 3, 4})]
    deltas = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
    print("  exact ARL (Markov chain); shift delta in units of the standard deviation of the plotted statistic")
    for rules in combos:
        cells = []
        for delta in deltas:
            arl, size = we_arl(rules, delta)
            cells.append(f"{arl:8.2f}")
        name = "+".join(str(r) for r in sorted(rules))
        print(f"    tests {name:8s} ({size:6d} states): " + " ".join(f"d={d:.1f}:{c}" for d, c in zip(deltas, cells)))
    arl1, _ = we_arl(frozenset({1}))
    arl_all, _ = we_arl(frozenset({1, 2, 3, 4}))
    print(f"  in control, all four tests: ARL {arl_all:.2f}, i.e. {arl1 / arl_all:.2f} times as many false alarms as test 1 alone;"
          f" per-point false-alarm rate about {1 / arl_all:.5f}")
    for name, nchains in (("we", 200_000),):
        rl = simulate_run_lengths(RNG_WE, nchains, name)
        se = rl.std(ddof=1) / math.sqrt(nchains)
        print(f"  simulation check, tests 1-4, {nchains:,} chains: mean run length {rl.mean():.2f} (MC SE {se:.2f}); median {np.median(rl):.0f}")
    rl = simulate_run_lengths(RNG_NELSON, 200_000, "nelson")
    se = rl.std(ddof=1) / math.sqrt(rl.size)
    print("  Nelson's eight tests (1: beyond 3 sigma; 2: nine in a row on one side; 3: six steadily increasing or")
    print("  decreasing; 4: fourteen alternating; 5: two of three beyond 2 sigma; 6: four of five beyond 1 sigma;")
    print("  7: fifteen in a row within 1 sigma; 8: eight in a row beyond 1 sigma, either side), by simulation:")
    print(f"    in control, 200,000 chains: mean run length {rl.mean():.2f} (MC SE {se:.2f}); median {np.median(rl):.0f};"
          f" {arl1 / rl.mean():.2f} times the false alarms of test 1 alone")
    for delta in (1.0, 2.0):
        rl_d = simulate_run_lengths(RNG_NELSON, 100_000, "nelson", delta)
        print(f"    shift {delta:.1f}: mean run length {rl_d.mean():.2f} (MC SE {rl_d.std(ddof=1) / math.sqrt(rl_d.size):.2f})")


# ---------------------------------------------------------------------------------------------------
# 4. British action and warning limits
# ---------------------------------------------------------------------------------------------------


def warning_scheme_arl(action: float, warning: float, delta: float = 0.0) -> float:
    """ARL when the chart signals on one point beyond an action limit or on two successive points
    between the same warning and action limits (states: last point inside, in upper band, in lower band)."""
    p_up = PHI(action - delta) - PHI(warning - delta)
    p_lo = PHI(-warning - delta) - PHI(-action - delta)
    p_in = PHI(warning - delta) - PHI(-warning - delta)
    q = np.array([[p_in, p_up, p_lo],
                  [p_in, 0.0, p_lo],
                  [p_in, p_up, 0.0]])
    arl = np.linalg.solve(np.eye(3) - q, np.ones(3))
    return float(arl[0])


def warning_scheme_arl_either(action: float, warning: float, delta: float = 0.0) -> float:
    """As warning_scheme_arl, but two successive points in the warning zones signal whichever side they fall on."""
    p_warn = PHI(action - delta) - PHI(warning - delta) + PHI(-warning - delta) - PHI(-action - delta)
    p_in = PHI(warning - delta) - PHI(-warning - delta)
    q = np.array([[p_in, p_warn], [p_in, 0.0]])
    arl = np.linalg.solve(np.eye(2) - q, np.ones(2))
    return float(arl[0])


def section_british() -> None:
    header("4. THE BRITISH CONVENTION: ACTION LIMITS AT 3.09 SIGMA, WARNING LIMITS AT 1.96 SIGMA")
    for k in (3.09, 1.96):
        print(f"  limits at {k:.2f} sigma: one tail {PHIBAR(k):.5f} (1 in {1 / PHIBAR(k):.0f}); both tails {2 * PHIBAR(k):.5f};"
              f" ARL {1 / (2 * PHIBAR(k)):.1f}")
    print(f"  3-sigma limits on one tail: 1 in {1 / PHIBAR(3):.0f}; the 3.09 action limit is the one-tail 1-in-1000 point:"
          f" z = {stats.norm.isf(0.001):.4f}; the 1.96 warning line is the 1-in-40 point: z = {stats.norm.isf(0.025):.4f}")
    print("  action at 3.09 sigma, plus a signal on two successive points beyond the same 1.96 warning line:")
    for delta in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        print(f"    shift {delta:.1f}: ARL {warning_scheme_arl(3.09, 1.96, delta):8.2f}   (action limits alone:"
              f" {1 / (PHIBAR(3.09 - delta) + PHI(-3.09 - delta)):8.2f}; American 3-sigma alone: {1 / (PHIBAR(3 - delta) + PHI(-3 - delta)):8.2f})")
    print("  the same scheme when two successive points in the warning zones signal whichever side they fall on:")
    print("    " + "; ".join(f"shift {d:.1f}: ARL {warning_scheme_arl_either(3.09, 1.96, d):.2f}" for d in (0.0, 1.0, 2.0)))


# ---------------------------------------------------------------------------------------------------
# 5. Known parameters, other distributions
# ---------------------------------------------------------------------------------------------------


def standardized(dist) -> tuple[float, float]:
    m, v = dist.stats(moments="mv")
    return float(m), math.sqrt(float(v))


def tail_individual(dist) -> tuple[float, float]:
    mu, sd = standardized(dist)
    return float(dist.sf(mu + 3 * sd)), float(dist.cdf(mu - 3 * sd))


def tail_mean_of_n(dist, n: int, step: float = 0.002, half_width: float = 60.0) -> tuple[float, float, float]:
    """Tails beyond +-3 sigma/sqrt(n) of the mean of n i.i.d. draws, by numerical convolution of the
    standardized density on a grid (trapezoid-consistent discretization of the CDF)."""
    from scipy.signal import fftconvolve
    mu, sd = standardized(dist)
    edges = np.arange(-half_width, half_width + step / 2, step)
    cdf = dist.cdf(mu + sd * edges)
    mass = np.diff(cdf)
    mass = np.clip(mass, 0, None)
    lost = 1 - mass.sum()
    total = mass.copy()
    for _ in range(n - 1):
        total = fftconvolve(total, mass)
        total = np.clip(total, 0, None)
    # cell j of the n-fold sum has midpoint n*(-half_width) + (j + n/2) * step
    mids = n * (-half_width) + (np.arange(total.size) + n / 2) * step
    cut = 3 * math.sqrt(n)  # the sum of n standardized values, with limits at 3 sigma of the mean
    upper = total[mids > cut].sum()
    lower = total[mids < -cut].sum()
    return float(upper), float(lower), float(lost)


def section_nonnormal() -> None:
    header("5. KNOWN PARAMETERS, OTHER DISTRIBUTIONS: IN-CONTROL ARL OF THREE-SIGMA LIMITS")
    dists = [
        ("normal", stats.norm()),
        ("uniform", stats.uniform()),
        ("logistic", stats.logistic()),
        ("Laplace (double exponential)", stats.laplace()),
        ("Student t, 5 df", stats.t(5)),
        ("gamma, shape 4 (skewness 1)", stats.gamma(4)),
        ("lognormal, log-sd 0.5", stats.lognorm(0.5)),
        ("exponential (skewness 2)", stats.expon()),
    ]
    print("  individual values (n = 1): exact tail probabilities")
    rows = {}
    for name, d in dists:
        up, lo = tail_individual(d)
        p = up + lo
        rows[name] = p
        sk, ku = (float(x) for x in d.stats(moments="sk"))
        print(f"    {name:30s} skew {sk:5.2f}, excess kurtosis {ku:6.2f}: upper {up:.5f}, lower {lo:.5f}, total {p:.5f},"
              f" ARL {fmt_arl(1 / p) if p > 0 else 'infinite'}")
    print("  means of n = 5: numerical convolution (grid step 0.002); exact gamma result shown for the check")
    for name, d in dists:
        up, lo, lost = tail_mean_of_n(d, 5)
        p = up + lo
        extra = ""
        if name.startswith("exponential"):
            exact = stats.gamma(5, scale=1 / 5)
            ex_up = exact.sf(1 + 3 / math.sqrt(5))
            ex_lo = exact.cdf(1 - 3 / math.sqrt(5))
            extra = f"  [exact gamma(5): upper {ex_up:.5f}, lower {ex_lo:.5f}, ARL {1 / (ex_up + ex_lo):.1f}]"
        elif name.startswith("gamma"):
            exact = stats.gamma(20, scale=1 / 5)
            ex_up = exact.sf(4 + 3 * 2 / math.sqrt(5))
            ex_lo = exact.cdf(4 - 3 * 2 / math.sqrt(5))
            extra = f"  [exact gamma(20): upper {ex_up:.5f}, lower {ex_lo:.5f}, ARL {1 / (ex_up + ex_lo):.1f}]"
        elif name == "normal":
            extra = f"  [exact: {2 * PHIBAR(3):.5f}]"
        print(f"    {name:30s} upper {up:.5f}, lower {lo:.5f}, total {p:.5f}, ARL {fmt_arl(1 / p) if p > 0 else 'infinite'}{extra}")


# ---------------------------------------------------------------------------------------------------
# 6. Estimated parameters: a first look
# ---------------------------------------------------------------------------------------------------


def phase1_xbar(rng: np.random.Generator, reps: int, m: int, n: int, chunk: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    """Grand mean and Rbar/d2 from `reps` simulated in-control Phase I samples of m subgroups of n."""
    mu_hat = np.empty(reps)
    sigma_hat = np.empty(reps)
    c2 = d2(n)
    per = max(1, min(chunk, 4_000_000 // (m * n)))
    for start in range(0, reps, per):
        k = min(per, reps - start)
        x = rng.standard_normal((k, m, n))
        mu_hat[start:start + k] = x.mean(axis=(1, 2))
        r = x.max(axis=2) - x.min(axis=2)
        sigma_hat[start:start + k] = r.mean(axis=1) / c2
    return mu_hat, sigma_hat


def phase1_individuals(rng: np.random.Generator, reps: int, m: int, chunk: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    """Mean and MRbar/d2(2) from `reps` simulated in-control Phase I samples of m individual values."""
    mu_hat = np.empty(reps)
    sigma_hat = np.empty(reps)
    c2 = d2(2)
    per = max(1, min(chunk, 4_000_000 // m))
    for start in range(0, reps, per):
        k = min(per, reps - start)
        x = rng.standard_normal((k, m))
        mu_hat[start:start + k] = x.mean(axis=1)
        sigma_hat[start:start + k] = np.abs(np.diff(x, axis=1)).mean(axis=1) / c2
    return mu_hat, sigma_hat


def conditional_p(mu_hat: np.ndarray, sigma_hat: np.ndarray, n: int, k: float = 3.0, delta: float = 0.0) -> np.ndarray:
    """Probability that a future plotted mean of n falls outside mu_hat +- k sigma_hat / sqrt(n),
    for an in-control process N(delta, 1) -- the false-alarm probability of one particular chart."""
    rn = math.sqrt(n)
    return PHIBAR(rn * (mu_hat - delta) + k * sigma_hat) + PHI(rn * (mu_hat - delta) - k * sigma_hat)


def describe_carl(label: str, p: np.ndarray, show_percentiles: bool = True, heavy: bool = False) -> dict[str, float]:
    """Summaries of the conditional ARL across simulated Phase I samples.  With heavy=True the mean and SD
    of the conditional ARL are not printed: a handful of charts with an overestimated sigma carry most of
    the sum, so the Monte Carlo mean does not settle; the concentration of the sum is printed instead."""
    carl = 1 / p
    reps = p.size
    ep = p.mean()
    out = {"E[p]": ep, "1/E[p]": 1 / ep, "median": float(np.median(carl))}
    qs = np.percentile(carl, [5, 10, 25, 50, 75, 90, 95])
    print(f"  {label}")
    print(f"    {reps:,} simulated Phase I samples; average false-alarm probability E[p] = {ep:.5f} (known limits 0.00270);"
          f" 1/E[p] = {1 / ep:.1f}")
    if heavy:
        top = np.sort(carl)[::-1]
        share = top[: max(1, reps // 1000)].sum() / carl.sum()
        print(f"    conditional in-control ARL: median {out['median']:.1f}; the mean is not a usable summary here -- the top 0.1% of"
              f" charts carry {share:.1%} of the summed ARL (largest simulated value {top[0]:.3g})")
    else:
        out["AARL"] = carl.mean()
        out["AARL_se"] = carl.std(ddof=1) / math.sqrt(reps)
        out["SDARL"] = carl.std(ddof=1)
        print(f"    conditional in-control ARL: mean (AARL) {out['AARL']:.1f} (MC SE {out['AARL_se']:.1f}), SD (SDARL) {out['SDARL']:.1f},"
              f" median {out['median']:.1f}")
    if show_percentiles:
        print("    percentiles 5/10/25/50/75/90/95: " + " / ".join(f"{q:.0f}" for q in qs))
    for thr in (100, 200, 300, 370.4):
        print(f"    share of charts with conditional ARL below {thr:5.1f}: {np.mean(carl < thr):.3f}")
    print(f"    share above 1,000: {np.mean(carl > 1000):.3f}")
    out["p10"], out["p90"] = float(qs[1]), float(qs[5])
    out["below370"] = float(np.mean(carl < 370.4))
    out["below200"] = float(np.mean(carl < 200))
    return out


def alarm_correlation(p: np.ndarray) -> float:
    """Correlation between the alarm indicators of two different future points of the same chart:
    they are conditionally independent given the estimates, so Cov = Var(p)."""
    ep = p.mean()
    return float(p.var() / (ep * (1 - ep)))


def alarm_clustering(label: str, p: np.ndarray) -> None:
    ep = p.mean()
    ep2 = (p ** 2).mean()
    print(f"    {label}: correlation between the alarm indicators of two future points of the same chart {alarm_correlation(p):.5f};"
          f" P(alarm at a later point | alarm now) = {ep2 / ep:.5f} against P(alarm) = {ep:.5f} (ratio {ep2 / ep ** 2:.2f})")


def section_estimated() -> None:
    header("6. ESTIMATED PARAMETERS, A FIRST LOOK: THE CONDITIONAL IN-CONTROL ARL")
    for n in (2, 4, 5):
        print(f"  d2({n}) = {d2(n):.4f}; A2({n}) = 3/(d2 sqrt n) = {3 / (d2(n) * math.sqrt(n)):.4f}")
    print("  average false-alarm probability per future subgroup, E[p], subgroups of 4, limits from m subgroups")
    print("  (the quantity of Proschan and Savage 1952/1960 and Hillier 1964/1969); Rbar/d2 by simulation (100,000")
    print("  Phase I samples per m), pooled standard deviation exactly: P(|t_nu| > 3/sqrt(1 + 1/m)), nu = 3m")
    first_r = first_s = None
    for m in (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 19, 20, 25, 30):
        mh, sh = phase1_xbar(RNG_PS, 100_000, m, 4)
        ep_r = conditional_p(mh, sh, 4).mean()
        nu = 3 * m
        ep_s = 2 * stats.t(nu).sf(3 / math.sqrt(1 + 1 / m))
        if first_r is None and ep_r <= 0.01:
            first_r = m
        if first_s is None and ep_s <= 0.005:
            first_s = m
        print(f"    m = {m:2d}: Rbar/d2 E[p] {ep_r:.5f}; pooled s E[p] {ep_s:.5f}")
    print(f"  smallest m in this list with E[p] <= 0.01 (Rbar/d2): {first_r}; with E[p] <= 0.005 (pooled s): {first_s}")
    print("  every chart below: in-control normal process, limits grand mean +- 3 sigma_hat/sqrt(n) estimated from a")
    print("  Phase I sample assumed clean; the conditional false-alarm probability p of each chart computed exactly")
    print("  given its estimates; conditional ARL = 1/p.")

    reps = 200_000
    mu_hat, s_hat = phase1_xbar(RNG_XBAR, reps, 25, 5)
    p = conditional_p(mu_hat, s_hat, 5)
    r = describe_carl("X-bar chart, 25 subgroups of 5, sigma from Rbar/d2 (the textbook A2*Rbar limits)", p)
    alarm_clustering("alarms of one chart", p)
    for horizon in (100, 370, 1000):
        print(f"    probability of at least one false alarm in {horizon} future in-control subgroups: {np.mean(1 - (1 - p) ** horizon):.4f}"
              f"  (known limits {1 - (1 - 2 * PHIBAR(3)) ** horizon:.4f})")
    sd_rel = s_hat / 1.0
    print(f"    sigma_hat/sigma: mean {sd_rel.mean():.4f}, SD {sd_rel.std(ddof=1):.4f}; 5th and 95th percentiles"
          f" {np.percentile(sd_rel, 5):.3f} and {np.percentile(sd_rel, 95):.3f}")
    lo_s, hi_s = np.percentile(sd_rel, [10, 90])
    print(f"    a chart whose sigma_hat is 10% low (0.90 sigma) and mean exact: ARL {1 / conditional_p(np.array([0.0]), np.array([0.9]), 5)[0]:.1f};"
          f" 10% high (1.10 sigma): {1 / conditional_p(np.array([0.0]), np.array([1.1]), 5)[0]:.1f}")
    a_lo = 1 / conditional_p(np.array([0.0]), np.array([0.9]), 5)[0]
    a_hi = 1 / conditional_p(np.array([0.0]), np.array([1.1]), 5)[0]
    print(f"    two charts only, sigma_hat 10% low and 10% high: reciprocal of their average false-alarm probability"
          f" {1 / ((1 / a_lo + 1 / a_hi) / 2):.1f}; their average ARL {(a_lo + a_hi) / 2:.1f}")
    print(f"    10th / 90th percentile of sigma_hat/sigma ({lo_s:.3f} / {hi_s:.3f}) as limits with exact mean:"
          f" ARL {1 / conditional_p(np.array([0.0]), np.array([lo_s]), 5)[0]:.1f} / {1 / conditional_p(np.array([0.0]), np.array([hi_s]), 5)[0]:.1f}")

    # a Hillier-type correction: widen the limits so that the AVERAGE false-alarm probability is 0.0027
    target = 2 * PHIBAR(3)
    k_adj = optimize.brentq(lambda k: conditional_p(mu_hat, s_hat, 5, k).mean() - target, 2.8, 3.4)
    p_adj = conditional_p(mu_hat, s_hat, 5, k_adj)
    print(f"  widening the limits to k = {k_adj:.4f} sigma_hat restores the average false-alarm probability to {target:.5f}"
          f" (A2 becomes {k_adj / (d2(5) * math.sqrt(5)):.4f} instead of {3 / (d2(5) * math.sqrt(5)):.4f})")
    describe_carl(f"X-bar chart, 25 subgroups of 5, Rbar/d2, limits at k = {k_adj:.3f}", p_adj)
    k_med = optimize.brentq(lambda k: np.median(1 / conditional_p(mu_hat, s_hat, 5, k)) - 1 / target, 2.8, 3.4)
    print(f"  limits that make the MEDIAN conditional ARL equal to 370.4: k = {k_med:.4f}")
    k_q = optimize.brentq(lambda k: np.percentile(1 / conditional_p(mu_hat, s_hat, 5, k), 10) - 1 / target, 2.8, 4.0)
    print(f"  limits that make 90% of charts have a conditional ARL of at least 370.4: k = {k_q:.4f}")
    # out-of-control ARL for a 1-sigma shift of the process mean (sigma of individual values)
    for k_lim, lab in ((3.0, "3 sigma_hat"), (k_q, f"{k_q:.3f} sigma_hat")):
        rn = math.sqrt(5)
        pp = PHIBAR(rn * mu_hat - rn * 1.0 + k_lim * s_hat) + PHI(rn * mu_hat - rn * 1.0 - k_lim * s_hat)
        oarl = 1 / pp
        print(f"    1-sigma shift, limits at {lab}: out-of-control ARL median {np.median(oarl):.2f}, 10th-90th percentile"
              f" {np.percentile(oarl, 10):.2f}-{np.percentile(oarl, 90):.2f}  (known limits at 3 sigma: {1 / (PHIBAR(3 - rn) + PHI(-3 - rn)):.2f})")

    mu4, s4 = phase1_xbar(RNG_EXTRA, reps, 25, 4)
    describe_carl("X-bar chart, 25 subgroups of 4, sigma from Rbar/d2", conditional_p(mu4, s4, 4), show_percentiles=True)

    mu_i, s_i = phase1_individuals(RNG_IND, reps, 25)
    p_i = conditional_p(mu_i, s_i, 1)
    describe_carl("individuals chart, 25 observations, sigma from MRbar/d2(2)", p_i, heavy=True)
    alarm_clustering("alarms of one chart", p_i)
    for horizon in (100, 370):
        print(f"    probability of at least one false alarm in {horizon} future in-control points: {np.mean(1 - (1 - p_i) ** horizon):.4f}")

    print("  how the conditional ARL tightens with the size of Phase I (100,000 Phase I samples per row):")
    mh, sh = phase1_xbar(RNG_1200, 100_000, 1200, 5)
    c1200 = 1 / conditional_p(mh, sh, 5)
    q10, q50, q90 = np.percentile(c1200, [10, 50, 90])
    print(f"    X-bar, n = 5, m = 1200 (the Phase I size reported for Saleh et al. 2015): AARL {c1200.mean():.1f}; SDARL"
          f" {c1200.std(ddof=1):.1f} ({c1200.std(ddof=1) / 370.4:.1%} of 370.4); median {q50:.1f}; 10th-90th percentile"
          f" {q10:.1f}-{q90:.1f}; share below 300: {np.mean(c1200 < 300):.3f}; share outside 333-407 (370.4 +- 10%):"
          f" {np.mean((c1200 < 333.4) | (c1200 > 407.4)):.3f}")
    grid_reps = 100_000
    for kind, n, ms in (("X-bar, n = 5, Rbar/d2", 5, (20, 25, 30, 50, 75, 100, 150, 300, 1000)),
                        ("individuals, MRbar/d2", 1, (20, 25, 30, 50, 75, 100, 150, 300, 1000))):
        print(f"    {kind}:")
        for m in ms:
            if n == 1:
                mh, sh = phase1_individuals(RNG_GRID, grid_reps, m)
            else:
                mh, sh = phase1_xbar(RNG_GRID, grid_reps, m, n)
            pp = conditional_p(mh, sh, n)
            carl = 1 / pp
            q10, q50, q90 = np.percentile(carl, [10, 50, 90])
            mean_part = f"AARL {carl.mean():6.1f}; SDARL {carl.std(ddof=1):6.1f}; " if n > 1 else ""
            print(f"      m = {m:4d}: E[p] {pp.mean():.5f}; {mean_part}median {q50:6.1f}; 10th-90th percentile"
                  f" {q10:6.1f}-{q90:7.1f}; share below 370.4: {np.mean(carl < 370.4):.3f};"
                  f" share below 200: {np.mean(carl < 200):.3f}")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    print("Statistical process control, Part 1: where the 370 comes from (numbers recomputed)")
    print(f"numpy {np.__version__}; master seed {SEED}")
    section_bounds()
    section_geometric()
    section_runs_rules()
    section_british()
    section_nonnormal()
    section_estimated()


if __name__ == "__main__":
    main()
