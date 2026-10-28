"""Part 1 of the Simpson's paradox series: the history, with every table recomputed.

Every number quoted in the article is printed by this script. Run:

    python history_numbers.py > output.txt

Data downloaded on first run from a pinned commit of the Rdatasets repository (Arel-Bundock),
checked against SHA-256 digests and cached in data/:
  - UCBAdmissions: the six largest departments of Bickel, Hammel and O'Connell (1975);
  - Lahman Batting: season batting lines, for the Jeter-Justice example.
The other tables are printed in the sources cited next to them and are written into the script.
"""

from __future__ import annotations

import csv
import hashlib
import math
import urllib.request
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import brentq

SEED = 1951
HERE = Path(__file__).parent
DATA = HERE / "data"
RDATASETS = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/1dcc2bf5f955cc1224a3e1307256e1fe86b68dae/"
FILES = {
    "csv/datasets/UCBAdmissions.csv": "92ae5a362fd3a09847ee388823286381b847b48fd1c9a1eca387c987a1b5eba3",
    "csv/Lahman/Batting.csv": "a4083e2d4bacf411c352686fd7c0696eaa4bc8719b718db5e9c45ac90fd95b35",
}


def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fetch(name: str) -> Path:
    path = DATA / Path(name).name
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(RDATASETS + name, timeout=300) as r:
            path.write_bytes(r.read())
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != FILES[name]:
        raise RuntimeError(f"{name}: SHA-256 {got} does not match the pinned digest {FILES[name]}")
    return path


# ---------------------------------------------------------------------------
# Tables. A stratified two-group table is an array of shape (K, 2, 2):
# [stratum, group (0 = first, 1 = second), (successes, trials)].
# ---------------------------------------------------------------------------

def load_berkeley() -> tuple[list[str], np.ndarray]:
    rows = list(csv.DictReader(open(fetch("csv/datasets/UCBAdmissions.csv"))))
    depts = sorted({r["Dept"] for r in rows})
    t = np.zeros((len(depts), 2, 2), dtype=int)
    for r in rows:
        k = depts.index(r["Dept"])
        g = 0 if r["Gender"] == "Male" else 1
        n = int(r["Freq"])
        t[k, g, 1] += n
        if r["Admit"] == "Admitted":
            t[k, g, 0] += n
    return depts, t


def load_batting(players: dict[str, str], years: list[int]) -> np.ndarray:
    """Hits and at-bats, summed over stints, for each year and player."""
    path = fetch("csv/Lahman/Batting.csv")
    ids = list(players)
    t = np.zeros((len(years), len(ids), 2), dtype=int)
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["playerID"] in players and int(r["yearID"]) in years:
                k = years.index(int(r["yearID"]))
                g = ids.index(r["playerID"])
                t[k, g, 0] += int(r["H"])
                t[k, g, 1] += int(r["AB"])
    return t


# ---------------------------------------------------------------------------
# Statistics for a stratified 2 x 2 x K table
# ---------------------------------------------------------------------------

def rates(t: np.ndarray):
    """Stratum rates (K, 2) and aggregate rates (2,)."""
    r = t[:, :, 0] / t[:, :, 1]
    agg = t[:, :, 0].sum(axis=0) / t[:, :, 1].sum(axis=0)
    return r, agg


def two_by_two(t_k: np.ndarray) -> np.ndarray:
    """[[a, b], [c, d]]: rows = groups, columns = success, failure."""
    return np.array([[t_k[0, 0], t_k[0, 1] - t_k[0, 0]], [t_k[1, 0], t_k[1, 1] - t_k[1, 0]]])


def fisher(t_k: np.ndarray):
    tab = two_by_two(t_k)
    p = stats.fisher_exact(tab).pvalue
    res = stats.contingency.odds_ratio(tab)
    ci = res.confidence_interval(0.95)
    return res.statistic, (ci.low, ci.high), p


def mantel_haenszel(t: np.ndarray):
    """Mantel-Haenszel common odds ratio (group 0 vs group 1), Robins-Breslow-Greenland 95% CI,
    Cochran-Mantel-Haenszel chi-square (no continuity correction) and its p-value."""
    a = t[:, 0, 0].astype(float)
    b = (t[:, 0, 1] - t[:, 0, 0]).astype(float)
    c = t[:, 1, 0].astype(float)
    d = (t[:, 1, 1] - t[:, 1, 0]).astype(float)
    n = a + b + c + d
    R = a * d / n
    S = b * c / n
    or_mh = R.sum() / S.sum()
    P = (a + d) / n
    Q = (b + c) / n
    var = ((P * R).sum() / (2 * R.sum() ** 2) + ((P * S + Q * R).sum()) / (2 * R.sum() * S.sum())
           + (Q * S).sum() / (2 * S.sum() ** 2))
    se = math.sqrt(var)
    ci = (or_mh * math.exp(-1.96 * se), or_mh * math.exp(1.96 * se))
    m1 = a + c
    n1 = a + b
    E = n1 * m1 / n
    V = n1 * (c + d) * m1 * (b + d) / (n ** 2 * (n - 1))
    chi2 = (a.sum() - E.sum()) ** 2 / V.sum()
    return or_mh, ci, chi2, stats.chi2.sf(chi2, 1)


def breslow_day(t: np.ndarray, or_common: float):
    """Breslow-Day test of homogeneous odds ratios (with Tarone's correction)."""
    a = t[:, 0, 0].astype(float)
    n1 = t[:, 0, 1].astype(float)
    n2 = t[:, 1, 1].astype(float)
    m1 = a + t[:, 1, 0]
    stat, ea_sum, var_sum, a_sum = 0.0, 0.0, 0.0, 0.0
    for k in range(t.shape[0]):
        lo = max(0.0, m1[k] - n2[k])
        hi = min(n1[k], m1[k])

        def f(x):
            return x * (n2[k] - m1[k] + x) - or_common * (n1[k] - x) * (m1[k] - x)

        ea = brentq(f, lo + 1e-12, hi - 1e-12)
        var = 1 / (1 / ea + 1 / (n1[k] - ea) + 1 / (m1[k] - ea) + 1 / (n2[k] - m1[k] + ea))
        stat += (a[k] - ea) ** 2 / var
        ea_sum += ea
        var_sum += var
        a_sum += a[k]
    stat_t = stat - (a_sum - ea_sum) ** 2 / var_sum
    df = t.shape[0] - 1
    return stat_t, df, stats.chi2.sf(stat_t, df)


def logistic_irls(X: np.ndarray, y: np.ndarray, w: np.ndarray, iters: int = 50):
    """Binomial logistic regression on grouped data (y successes out of w trials) by IRLS.
    Returns coefficients and standard errors."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        W = w * p * (1 - p)
        z = eta + (y - w * p) / W
        XtW = X.T * W
        beta_new = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(beta_new - beta)) < 1e-12:
            beta = beta_new
            break
        beta = beta_new
    p = 1 / (1 + np.exp(-(X @ beta)))
    cov = np.linalg.inv((X.T * (w * p * (1 - p))) @ X)
    return beta, np.sqrt(np.diag(cov))


def decomposition(t: np.ndarray):
    """Kitagawa decomposition of the aggregate difference (group 0 minus group 1) into a mix
    component (different allocation across strata) and a rate component (different rates within
    strata), using average weights and average rates, so the two add up exactly."""
    r, agg = rates(t)
    w = t[:, :, 1] / t[:, :, 1].sum(axis=0)
    mix = np.sum((w[:, 0] - w[:, 1]) * (r[:, 0] + r[:, 1]) / 2)
    rate = np.sum((w[:, 0] + w[:, 1]) / 2 * (r[:, 0] - r[:, 1]))
    return agg[0] - agg[1], mix, rate


def standardised_difference(t: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Difference in rates (group 0 minus group 1) directly standardised to a common stratum mix,
    by default the combined mix of both groups."""
    r, _ = rates(t)
    if weights is None:
        weights = t[:, :, 1].sum(axis=1) / t[:, :, 1].sum()
    return float(np.sum(weights * (r[:, 0] - r[:, 1])))


def standardised_interval(t: np.ndarray, draws: int = 20_000, seed: int = 7):
    """Standardised difference with a 95% parametric-bootstrap interval: stratum counts resampled
    from binomials at the observed rates, common mix held at the observed combined mix."""
    rng = np.random.default_rng(seed)
    est = standardised_difference(t)
    w = t[:, :, 1].sum(axis=1) / t[:, :, 1].sum()
    r, _ = rates(t)
    sims = rng.binomial(t[:, :, 1][None], r[None], size=(draws,) + r.shape) / t[:, :, 1][None]
    vals = (w[None] * (sims[:, :, 0] - sims[:, :, 1])).sum(axis=1)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return est, lo, hi


def posterior_reversal(t: np.ndarray, draws: int = 200_000, seed: int = SEED):
    """Posterior probabilities, under uniform Beta priors on every stratum rate and uniform Dirichlet
    priors on each group's allocation across strata, that in the population:
      - the aggregate rate favours the group the sample aggregate favours (call it the winner);
      - the standardised (common-mix) difference favours the other group;
      - every stratum favours the other group;
      - the aggregate favours the winner while every stratum favours the other group (a full reversal).
    """
    _, agg_obs = rates(t)
    if agg_obs[1] > agg_obs[0]:
        t = t[:, ::-1, :]
    rng = np.random.default_rng(seed)
    K = t.shape[0]
    s = t[:, :, 0]
    f = t[:, :, 1] - t[:, :, 0]
    p = rng.beta(1 + s[None], 1 + f[None], size=(draws, K, 2))
    mix = np.stack([rng.dirichlet(1 + t[:, g, 1], size=draws) for g in range(2)], axis=2)
    agg = (mix * p).sum(axis=1)
    agg_0 = agg[:, 0] > agg[:, 1]
    comb = rng.dirichlet(1 + t[:, :, 1].sum(axis=1), size=draws)
    std_1 = (comb * (p[:, :, 0] - p[:, :, 1])).sum(axis=1) < 0
    all_1 = np.all(p[:, :, 1] > p[:, :, 0], axis=1)
    return agg_0.mean(), std_1.mean(), all_1.mean(), (agg_0 & all_1).mean()


def chi2_2x2(tab: np.ndarray, yates: bool) -> tuple[float, float]:
    res = stats.chi2_contingency(tab, correction=yates)
    return float(res.statistic), float(res.pvalue)


def n_per_arm(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Patients per arm for a two-sided test of two proportions (normal approximation, pooled
    variance under the null)."""
    za = stats.norm.ppf(1 - alpha / 2)
    zb = stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return num / (p1 - p2) ** 2


def assoc(cells: np.ndarray) -> tuple[float, float, float]:
    """For a 2 x 2 table of parent attribute (rows: with, without) by offspring attribute
    (columns: with, without), return P(offspring with | parent with), P(offspring with | parent
    without) and the cross-product ratio."""
    a, b, c, d = cells.ravel()
    return a / (a + b), c / (c + d), (a * d) / (b * c)


def critical_mix(t: np.ndarray) -> float:
    """Two strata: keeping the first group's mix and every stratum rate fixed, the second group's
    share in stratum 2 at which the two aggregate rates would be equal."""
    r, agg = rates(t)
    return float((r[0, 1] - agg[0]) / (r[0, 1] - r[1, 1]))


def report_table(name: str, t: np.ndarray, strata: list[str], groups: tuple[str, str],
                 unit: str = "success") -> None:
    r, agg = rates(t)
    print(f"  {name}")
    for k, lab in enumerate(strata):
        or_, ci, p = fisher(t[k])
        print(f"    {lab:<22} {groups[0]} {t[k, 0, 0]}/{t[k, 0, 1]} = {r[k, 0]:.3f};"
              f" {groups[1]} {t[k, 1, 0]}/{t[k, 1, 1]} = {r[k, 1]:.3f};"
              f" difference {r[k, 0] - r[k, 1]:+.3f}; odds ratio {or_:.3f} [{ci[0]:.3f}, {ci[1]:.3f}];"
              f" Fisher p = {p:.3g}")
    tot = t.sum(axis=0)
    or_, ci, p = fisher(tot)
    print(f"    {'all strata combined':<22} {groups[0]} {tot[0, 0]}/{tot[0, 1]} = {agg[0]:.3f};"
          f" {groups[1]} {tot[1, 0]}/{tot[1, 1]} = {agg[1]:.3f};"
          f" difference {agg[0] - agg[1]:+.3f}; odds ratio {or_:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]; Fisher p = {p:.3g}")
    if t.shape[0] > 1:
        or_mh, ci_mh, chi2, p_mh = mantel_haenszel(t)
        print(f"    Mantel-Haenszel odds ratio {or_mh:.3f} [{ci_mh[0]:.3f}, {ci_mh[1]:.3f}];"
              f" CMH chi-square {chi2:.3f}, p = {p_mh:.3g}")
        bd, df, p_bd = breslow_day(t, or_mh)
        print(f"    Breslow-Day test of a common odds ratio: chi-square {bd:.3f} on {df} df, p = {p_bd:.3g}")
        total, mix, rate = decomposition(t)
        print(f"    aggregate difference {total:+.4f} = allocation across strata {mix:+.4f}"
              f" + rates within strata {rate:+.4f}")
        est, lo, hi = standardised_interval(t)
        print(f"    difference standardised to the combined mix: {est:+.4f} (95% bootstrap interval {lo:+.4f} to {hi:+.4f})")
        pa, ps, pall, prev = posterior_reversal(t)
        win, other = (groups[0], groups[1]) if agg[0] >= agg[1] else (groups[1], groups[0])
        print(f"    posterior P(aggregate favours {win}) = {pa:.4f};"
              f" P(standardised difference favours {other}) = {ps:.4f};"
              f" P(every stratum favours {other}) = {pall:.4f}; P(full reversal) = {prev:.4f}")


def main() -> None:
    header("1. YULE'S MIXED RECORDS (Yule 1903; numbers from his textbook, 8th ed. 1927, pp. 50-51)")
    print("  An attribute present in 50% of fathers and of sons and in 10% of mothers and of daughters,")
    print("  independent between parent and offspring within each line (cells per 100 pairs:")
    print("  parent with & offspring with, with & without, without & with, without & without):")
    lines = {
        "fathers and sons": np.outer([0.5, 0.5], [0.5, 0.5]) * 100,
        "mothers and daughters": np.outer([0.1, 0.9], [0.1, 0.9]) * 100,
        "fathers and daughters": np.outer([0.5, 0.5], [0.1, 0.9]) * 100,
        "mothers and sons": np.outer([0.1, 0.9], [0.5, 0.5]) * 100,
    }
    for name, cells in lines.items():
        p1, p0, cr = assoc(cells)
        print(f"    {name:<22} cells {' '.join(f'{x:g}' for x in cells.ravel())}: P(offspring | parent with) {p1:.3f},"
              f" P(offspring | parent without) {p0:.3f}, cross-product ratio {cr:.3f}")
    for label, keys in (("same-sex lines mixed", ("fathers and sons", "mothers and daughters")),
                        ("cross-sex lines mixed", ("fathers and daughters", "mothers and sons")),
                        ("all four lines mixed", tuple(lines))):
        cells = sum(lines[k] for k in keys)
        p1, p0, cr = assoc(cells)
        print(f"    {label:<22} cells {' '.join(f'{x:g}' for x in cells.ravel())}: P(offspring | parent with) {p1:.3f},"
              f" P(offspring | parent without) {p0:.3f}, cross-product ratio {cr:.3f}")
    print()
    print("  Yule's hospital: 100 men and 100 women; death rate 30% for men and 60% for women whether treated or not;")
    print("  treatment given to 80% of the men and 40% of the women:")
    th = np.array([[[0.8 * 100 * 0.3, 0.8 * 100], [0.2 * 100 * 0.3, 0.2 * 100]],
                   [[0.4 * 100 * 0.6, 0.4 * 100], [0.6 * 100 * 0.6, 0.6 * 100]]])
    for k, lab in enumerate(("men", "women")):
        print(f"    {lab}: treated died {th[k, 0, 0]:g} of {th[k, 0, 1]:g}; untreated died {th[k, 1, 0]:g} of {th[k, 1, 1]:g}")
    tot = th.sum(axis=0)
    print(f"    combined: treated died {tot[0, 0]:g} of {tot[0, 1]:g} = {tot[0, 0] / tot[0, 1]:.3f};"
          f" untreated died {tot[1, 0]:g} of {tot[1, 1]:g} = {tot[1, 0] / tot[1, 1]:.3f}")

    header("2. SIMPSON (1951): THE PACK OF CARDS AND THE MEDICAL TABLE (cells printed as fractions of 52)")
    ts = np.array([[[8, 13], [4, 7]], [[12, 27], [2, 5]]])  # [sex, (treated, untreated), (alive, total)]
    for k, lab in enumerate(("males (dirty cards)", "females (clean cards)")):
        a, b = ts[k, 0, 0], ts[k, 0, 1] - ts[k, 0, 0]
        c, d = ts[k, 1, 0], ts[k, 1, 1] - ts[k, 1, 0]
        print(f"    {lab:<22} treated alive {a}/{a + b} = {a / (a + b):.3f}; untreated alive {c}/{c + d} = {c / (c + d):.3f};"
              f" cross-product ratio {(a * d) / (b * c):.3f}")
    tt = ts.sum(axis=0)
    a, b = tt[0, 0], tt[0, 1] - tt[0, 0]
    c, d = tt[1, 0], tt[1, 1] - tt[1, 0]
    print(f"    {'combined':<22} treated alive {a}/{a + b} = {a / (a + b):.3f}; untreated alive {c}/{c + d} = {c / (c + d):.3f};"
          f" cross-product ratio {(a * d) / (b * c):.3f}")
    print(f"    if the 52 cells were counts: Fisher p for males {stats.fisher_exact(two_by_two(ts[0])).pvalue:.3f},"
          f" females {stats.fisher_exact(two_by_two(ts[1])).pvalue:.3f}")

    header("3. BLYTH (1972), AS REPRODUCED IN LATER EXPOSITIONS")
    tbly = np.array([[[1000, 10000], [50, 1000]], [[95, 100], [5000, 10000]]])  # [group, (new, standard), (alive, total)]
    r, agg = rates(tbly)
    print(f"    group C: new {r[0, 0]:.3f}, standard {r[0, 1]:.3f}; group C': new {r[1, 0]:.3f}, standard {r[1, 1]:.3f};"
          f" combined: new {tbly[:, 0, 0].sum()}/{tbly[:, 0, 1].sum()} = {agg[0]:.3f},"
          f" standard {tbly[:, 1, 0].sum()}/{tbly[:, 1, 1].sum()} = {agg[1]:.3f}")
    total, mix, rate = decomposition(tbly)
    print(f"    aggregate difference {total:+.4f} = allocation {mix:+.4f} + rates within groups {rate:+.4f}")

    header("4. BERKELEY, FALL 1973 (Bickel, Hammel and O'Connell, 1975)")
    agg_tab = np.array([[3738, 8442 - 3738], [1494, 4321 - 1494]])
    c_y, p_y = chi2_2x2(agg_tab, True)
    c_n, p_n = chi2_2x2(agg_tab, False)
    print(f"  all applicants (their Table 1): men 3738/8442 = {3738 / 8442:.3f}; women 1494/4321 = {1494 / 4321:.3f};"
          f" difference {3738 / 8442 - 1494 / 4321:+.4f}")
    exp_w = 4321 * (3738 + 1494) / (8442 + 4321)
    print(f"    chi-square with Yates correction {c_y:.1f} (without {c_n:.1f}), p = {p_y:.2g};"
          f" women admitted 1494 against {exp_w:.1f} expected if both sexes had the same rate")
    hyp = np.array([[[200, 400], [100, 200]], [[50, 150], [150, 450]]])  # their hypothetical Table 2
    r, agg = rates(hyp)
    c_y, p_y = chi2_2x2(two_by_two(hyp.sum(axis=0)), True)
    print(f"  their hypothetical two departments: 'Machismatics' men {r[0, 0]:.3f}, women {r[0, 1]:.3f};"
          f" 'Social warfare' men {r[1, 0]:.3f}, women {r[1, 1]:.3f}; combined men {agg[0]:.3f}, women {agg[1]:.3f};"
          f" chi-square (Yates) {c_y:.2f}, two-sided p = {p_y:.3f}, one-sided {p_y / 2:.3f}")
    print()
    print("  The six largest departments (UCBAdmissions; as in Freedman, Pisani and Purves):")
    depts, tb = load_berkeley()
    report_table("admitted / applied, men against women", tb, [f"department {d}" for d in depts], ("men", "women"))
    w = tb[:, :, 1] / tb[:, :, 1].sum(axis=0)
    r, _ = rates(tb)
    print("    share of each sex's applications going to each department, and the department's overall admission rate:")
    for k, d in enumerate(depts):
        overall = tb[k, :, 0].sum() / tb[k, :, 1].sum()
        print(f"      {d}: men {w[k, 0]:.3f}, women {w[k, 1]:.3f}; admitted overall {overall:.3f}")
    corr = np.corrcoef(w[:, 1] - w[:, 0], tb[:, :, 0].sum(axis=1) / tb[:, :, 1].sum(axis=1))[0, 1]
    print(f"    correlation across departments between women's excess share and admission rate: {corr:.3f}")
    print(f"    departments where women's admission rate is higher: {int(np.sum(r[:, 1] > r[:, 0]))} of {len(depts)}")
    no_a = tb[1:]
    or_mh, ci_mh, chi2, p_mh = mantel_haenszel(no_a)
    bd, df, p_bd = breslow_day(no_a, or_mh)
    print(f"    without department A: Mantel-Haenszel odds ratio {or_mh:.3f} [{ci_mh[0]:.3f}, {ci_mh[1]:.3f}],"
          f" p = {p_mh:.3g}; Breslow-Day p = {p_bd:.3g}")
    # Logistic regression with department effects, on the grouped data.
    K = len(depts)
    X, y, n = [], [], []
    for k in range(K):
        for g in range(2):
            row = [1.0, float(g == 1)] + [float(k == j) for j in range(1, K)]
            X.append(row)
            y.append(tb[k, g, 0])
            n.append(tb[k, g, 1])
    X, y, n = np.array(X), np.array(y, float), np.array(n, float)
    beta, se = logistic_irls(X[:, :2], y, n)
    print(f"    logistic regression, sex only: woman coefficient {beta[1]:+.4f} (SE {se[1]:.4f});"
          f" odds ratio men/women {math.exp(-beta[1]):.3f}")
    beta, se = logistic_irls(X, y, n)
    print(f"    logistic regression with department effects: woman coefficient {beta[1]:+.4f} (SE {se[1]:.4f});"
          f" odds ratio men/women {math.exp(-beta[1]):.3f}"
          f" [{math.exp(-beta[1] - 1.96 * se[1]):.3f}, {math.exp(-beta[1] + 1.96 * se[1]):.3f}]")

    header("5. KIDNEY STONES (Charig, Webb, Payne and Wickham, 1986, Tables I and II)")
    tk = np.array([[[81, 87], [234, 270]], [[192, 263], [55, 80]]])  # [size, (open, PCNL), (success, total)]
    report_table("successes / treated, open surgery (1972-80) against percutaneous nephrolithotomy (1980-85)",
                 tk, ["stones < 2 cm", "stones >= 2 cm or multiple"], ("open", "PCNL"))
    sh = tk[1, :, 1] / tk[:, :, 1].sum(axis=0)
    print(f"    share of patients with stones >= 2 cm or multiple: open {sh[0]:.3f}, PCNL {sh[1]:.3f}")
    print(f"    PCNL share of large or multiple stones at which the two overall rates would be equal: {critical_mix(tk):.3f}")
    print(f"    ESWL (1985): < 2 cm 200/204 = {200 / 204:.3f}; >= 2 cm or multiple 101/124 = {101 / 124:.3f};"
          f" overall 301/328 = {301 / 328:.3f}")
    print("    patients per arm needed for 80% power at a two-sided 5% level, if the observed rates were true:")
    r, agg = rates(tk)
    for lab, p1, p2 in (("< 2 cm", r[0, 0], r[0, 1]), (">= 2 cm or multiple", r[1, 0], r[1, 1]),
                        ("combined", agg[0], agg[1])):
        print(f"      {lab:<22} {p1:.3f} vs {p2:.3f}: {n_per_arm(p1, p2):.0f}")

    header("6. DEREK JETER AND DAVID JUSTICE, 1995-1997 (Lahman database)")
    years = [1995, 1996, 1997]
    tj = load_batting({"jeterde01": "Jeter", "justida01": "Justice"}, years)
    report_table("hits / at-bats, Jeter against Justice, 1995 and 1996", tj[:2], ["1995", "1996"], ("Jeter", "Justice"))
    report_table("hits / at-bats, Jeter against Justice, 1995 to 1997", tj, ["1995", "1996", "1997"], ("Jeter", "Justice"))
    share95 = tj[0, :, 1] / tj[:2, :, 1].sum(axis=0)
    print(f"    share of 1995-96 at-bats that came in 1995: Jeter {share95[0]:.3f}, Justice {share95[1]:.3f}")
    games = {}
    with open(fetch("csv/Lahman/Batting.csv")) as f:
        for row in csv.DictReader(f):
            if row["playerID"] in ("jeterde01", "justida01") and row["yearID"] in ("1995", "1996"):
                games[(row["playerID"], row["yearID"])] = games.get((row["playerID"], row["yearID"]), 0) + int(row["G"])
    print("    games played: " + "; ".join(f"{k[0]} {k[1]} {v}" for k, v in sorted(games.items())))
    tj2 = tj[:2][::-1]  # strata ordered 1996, 1995 so that stratum 2 is the bad season
    print(f"    Justice's share of 1995 at-bats at which the two 1995-96 averages would be equal: {critical_mix(tj2):.3f}")


if __name__ == "__main__":
    main()
