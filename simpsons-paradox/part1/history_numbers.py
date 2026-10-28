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


def posterior_reversal(t: np.ndarray, draws: int = 200_000, seed: int = SEED):
    """Posterior probabilities, under uniform Beta priors on every stratum rate and uniform Dirichlet
    priors on each group's allocation across strata, that in the population:
      - the aggregate rate is higher for group 0;
      - the standardised (common-mix) difference favours group 1;
      - every stratum favours group 1;
      - the aggregate favours group 0 while every stratum favours group 1 (a full reversal).
    """
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
        print(f"    difference standardised to the combined mix: {standardised_difference(t):+.4f}")
        pa, ps, pall, prev = posterior_reversal(t)
        print(f"    posterior P(aggregate favours {groups[0]}) = {pa:.4f};"
              f" P(standardised difference favours {groups[1]}) = {ps:.4f};"
              f" P(every stratum favours {groups[1]}) = {pall:.4f}; P(full reversal) = {prev:.4f}")


def main() -> None:
    header("3. BERKELEY, FALL 1973: THE SIX LARGEST DEPARTMENTS (Bickel, Hammel and O'Connell, 1975)")
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

    header("5. DEREK JETER AND DAVID JUSTICE, 1995-1997 (Lahman database)")
    years = [1995, 1996, 1997]
    tj = load_batting({"jeterde01": "Jeter", "justida01": "Justice"}, years)
    report_table("hits / at-bats, Jeter against Justice, 1995 and 1996", tj[:2], ["1995", "1996"], ("Jeter", "Justice"))
    report_table("hits / at-bats, Jeter against Justice, 1995 to 1997", tj, ["1995", "1996", "1997"], ("Jeter", "Justice"))


if __name__ == "__main__":
    main()
