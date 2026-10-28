"""Part 3 of the Simpson's paradox series: real data where the reversal matters.

Every number quoted in the article is printed by this script. Run:

    python real_data_numbers.py > output.txt

Data downloaded on first run from pinned commits on GitHub, checked against SHA-256 digests and cached
in data/:
  - Palmer penguins (Horst, Hill and Gorman; data from Gorman, Williams and Fraser, 2014);
  - mosaicData SAT (state averages for 1994-95, assembled by Guber, 1999), via Rdatasets;
  - Lahman Batting, via Rdatasets.
The COVID-19 case and death counts by age are short and are written into the script with their sources.

Sections
1. COVID-19 case fatality by age, China and Italy.
2. SAT scores and teacher salaries across US states.
3. Bill length and bill depth in three penguin species.
4. How often reversals happen by chance: random tables, sampling noise, and every pair of baseball
   players over consecutive seasons.
"""

from __future__ import annotations

import csv
import hashlib
import math
import urllib.request
from pathlib import Path

import numpy as np
from scipy import stats

SEED = 2020
HERE = Path(__file__).parent
DATA = HERE / "data"
RD = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/1dcc2bf5f955cc1224a3e1307256e1fe86b68dae/"
# Age-stratified COVID-19 counts compiled by von Kuegelgen, Gresele and Schoelkopf (2021), pinned commit.
COVID = "https://raw.githubusercontent.com/Juliusvk/Covid19-age-related-causal-effects/496b734a7072234edc1f28962879f019d8bc9da4/"
FILES = {
    "penguins.csv": ("https://raw.githubusercontent.com/allisonhorst/palmerpenguins/"
                     "8957207b78d6ccd1b4654a9dd9c9041b657478ab/inst/extdata/penguins.csv",
                     "f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93"),
    "SAT.csv": (RD + "csv/mosaicData/SAT.csv", "2370aa1e09529d69e3b4d805f617811d6641c46772a52bfd6a292e3d1cae457b"),
    "Batting.csv": (RD + "csv/Lahman/Batting.csv", "a4083e2d4bacf411c352686fd7c0696eaa4bc8719b718db5e9c45ac90fd95b35"),
    "covid_countries.csv": (COVID + "datasets/Different_countries_latest/"
                            "Covid-19_age-stratified_dataset_for_different_countries.csv",
                            "7c328a71e57ccab9fa922bacad4afe5f7e41a0c34a7ba8dcc8f6d32314e92803"),
    "covid_italy_time.csv": (COVID + "datasets/Italy_longitudinal/"
                             "Covid-19_age-stratified_dataset_for_Italy_at_different_points_in_time.csv",
                             "ae96f932e2bcd5932e525f50eb14a780bc27a5a2ac6fd7baf3988a5692d67287"),
}
AGES = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]


def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fetch(name: str) -> Path:
    url, digest = FILES[name]
    path = DATA / name
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=300) as r:
            path.write_bytes(r.read())
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != digest:
        raise RuntimeError(f"{name}: SHA-256 {got} does not match the pinned digest {digest}")
    return path


def ols(y: np.ndarray, X: np.ndarray):
    """Ordinary least squares with an intercept column expected in X. Returns coefficients and
    conventional standard errors."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    return beta, np.sqrt(np.diag(cov)), dof


def ci(b, se, dof):
    t = stats.t.ppf(0.975, dof)
    return b - t * se, b + t * se


# ---------------------------------------------------------------------------
# 1. COVID-19
# ---------------------------------------------------------------------------

def load_covid(name: str):
    """{(country, date): (cases[9], deaths[9])} in the age order of AGES."""
    out: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    with open(fetch(name)) as f:
        for r in csv.DictReader(f):
            key = (r["Country"], r["Date of reporting"].replace("March2020", " March 2020")
                   .replace("February2020", " February 2020").replace("April2020", " April 2020")
                   .replace("May2020", " May 2020").replace("  ", " "))
            c, dd = out.setdefault(key, (np.zeros(9), np.zeros(9)))
            k = AGES.index(r["Age group"])
            c[k] = float(r["Confirmed cases"])
            dd[k] = float(r["Confirmed fatalities"])
    return out


def cfr_effects(cases_a, deaths_a, cases_b, deaths_b):
    """Effects of country b against reference country a on the case fatality rate (in proportions):
    total; direct with a's age distribution of cases (b's age-specific CFRs against a's); indirect
    with a's age-specific CFRs (b's age distribution against a's); direct with b's age distribution."""
    pa = cases_a / cases_a.sum()
    pb = cases_b / cases_b.sum()
    fa = deaths_a / cases_a
    fb = deaths_b / cases_b
    total = deaths_b.sum() / cases_b.sum() - deaths_a.sum() / cases_a.sum()
    direct_a = np.sum(pa * (fb - fa))
    indirect_a = np.sum(fa * (pb - pa))
    direct_b = np.sum(pb * (fb - fa))
    return total, direct_a, indirect_a, direct_b


# ---------------------------------------------------------------------------
# 2. SAT
# ---------------------------------------------------------------------------

def load_sat():
    rows = list(csv.DictReader(open(fetch("SAT.csv"))))
    keys = ("expend", "ratio", "salary", "frac", "verbal", "math", "sat")
    d = {k: np.array([float(r[k]) for r in rows]) for k in keys}
    d["state"] = [r["state"] for r in rows]
    return d


# ---------------------------------------------------------------------------
# 3. Penguins
# ---------------------------------------------------------------------------

def load_penguins():
    rows = [r for r in csv.DictReader(open(fetch("penguins.csv")))
            if r["bill_length_mm"] != "NA" and r["bill_depth_mm"] != "NA"]
    return {
        "species": np.array([r["species"] for r in rows]),
        "sex": np.array([r["sex"] for r in rows]),
        "length": np.array([float(r["bill_length_mm"]) for r in rows]),
        "depth": np.array([float(r["bill_depth_mm"]) for r in rows]),
    }


# ---------------------------------------------------------------------------
# 4. Reversals by chance
# ---------------------------------------------------------------------------

def reversal_in_table(cells: np.ndarray) -> np.ndarray:
    """cells: (..., 2, 2, 2) probabilities or counts indexed [stratum, group, outcome(1 = success)].
    Returns a boolean array: True where both strata favour one group and the aggregate the other."""
    s = cells[..., 1]
    n = cells.sum(axis=-1)
    r = s / n
    agg = s.sum(axis=-2) / n.sum(axis=-2)
    d0 = r[..., 0, 0] - r[..., 0, 1]
    d1 = r[..., 1, 0] - r[..., 1, 1]
    da = agg[..., 0] - agg[..., 1]
    return ((d0 > 0) & (d1 > 0) & (da < 0)) | ((d0 < 0) & (d1 < 0) & (da > 0))


def batting_seasons(first: int, last: int, min_ab: int):
    """Hits and at-bats by player and season (stints summed) for players with at least min_ab at-bats."""
    totals: dict[tuple[str, int], list[int]] = {}
    with open(fetch("Batting.csv")) as f:
        for r in csv.DictReader(f):
            y = int(r["yearID"])
            if first <= y <= last:
                key = (r["playerID"], y)
                h, ab = int(r["H"] or 0), int(r["AB"] or 0)
                t = totals.setdefault(key, [0, 0])
                t[0] += h
                t[1] += ab
    return {k: v for k, v in totals.items() if v[1] >= min_ab}


def reversal_pairs_detail(h1, ab1, h2, ab2):
    """For reversing pairs, the z statistics of the two within-season differences and of the combined
    difference (difference over its binomial standard error)."""
    a1, a2 = h1 / ab1, h2 / ab2
    ac = (h1 + h2) / (ab1 + ab2)
    d1 = a1[:, None] - a1[None, :]
    d2 = a2[:, None] - a2[None, :]
    dc = ac[:, None] - ac[None, :]
    rev = (d1 > 0) & (d2 > 0) & (dc < 0)
    i, j = np.nonzero(rev)
    se1 = np.sqrt(a1[i] * (1 - a1[i]) / ab1[i] + a1[j] * (1 - a1[j]) / ab1[j])
    se2 = np.sqrt(a2[i] * (1 - a2[i]) / ab2[i] + a2[j] * (1 - a2[j]) / ab2[j])
    abc = ab1 + ab2
    sec = np.sqrt(ac[i] * (1 - ac[i]) / abc[i] + ac[j] * (1 - ac[j]) / abc[j])
    return d1[i, j] / se1, d2[i, j] / se2, dc[i, j] / sec


def count_pair_reversals(h1, ab1, h2, ab2):
    """For every pair of players, both seasons: reversal if one player has the higher average in both
    seasons and the lower average over the two combined. Ties count as no reversal."""
    a1 = h1 / ab1
    a2 = h2 / ab2
    ac = (h1 + h2) / (ab1 + ab2)
    d1 = a1[:, None] - a1[None, :]
    d2 = a2[:, None] - a2[None, :]
    dc = ac[:, None] - ac[None, :]
    rev = ((d1 > 0) & (d2 > 0) & (dc < 0))
    n = len(a1)
    return int(rev.sum()), n * (n - 1) // 2


def main() -> None:
    header("1. COVID-19 CASE FATALITY BY AGE: CHINA (17 FEBRUARY 2020) AND ITALY (9 MARCH 2020)")
    cov = load_covid("covid_countries.csv")
    ita_t = load_covid("covid_italy_time.csv")
    ch_c, ch_d = cov[("China", "17 February 2020")]
    it_c, it_d = ita_t[("Italy", "9 March 2020")]
    print(f"  China: {ch_d.sum():.0f} deaths among {ch_c.sum():,.0f} cases, CFR {ch_d.sum() / ch_c.sum():.4f};"
          f" Italy: {it_d.sum():.0f} deaths among {it_c.sum():,.0f} cases of known age, CFR {it_d.sum() / it_c.sum():.4f}")
    print("  age band: share of cases China / Italy; CFR China / Italy; 95% exact intervals")
    lower = 0
    for k, a in enumerate(AGES):
        ci_c = stats.binomtest(int(ch_d[k]), int(ch_c[k])).proportion_ci() if ch_c[k] else None
        ci_i = stats.binomtest(int(it_d[k]), int(it_c[k])).proportion_ci() if it_c[k] else None
        fc, fi = ch_d[k] / ch_c[k], it_d[k] / it_c[k]
        lower += fi < fc
        print(f"    {a:<6} {ch_c[k] / ch_c.sum():.3f} / {it_c[k] / it_c.sum():.3f};"
              f" CFR {fc:.4f} [{ci_c.low:.4f}, {ci_c.high:.4f}] / {fi:.4f} [{ci_i.low:.4f}, {ci_i.high:.4f}]")
    print(f"  bands where Italy's CFR is lower: {lower} of 9 (equal at zero in 0-9)")
    print(f"  share of cases aged 70 or more: China {ch_c[7:].sum() / ch_c.sum():.3f}, Italy {it_c[7:].sum() / it_c.sum():.3f}")
    tot, dir_a, ind_a, dir_b = cfr_effects(ch_c, ch_d, it_c, it_d)
    print(f"  Italy against China, in percentage points: total {100 * tot:+.3f}; direct (China's age mix) {100 * dir_a:+.3f};"
          f" indirect (China's age-specific CFRs) {100 * ind_a:+.3f}; direct (Italy's age mix) {100 * dir_b:+.3f}")
    print(f"    direct (China's mix) + indirect (China's CFRs) = {100 * (dir_a + ind_a):+.3f};"
          f" direct (Italy's mix) + indirect (China's CFRs) = {100 * (dir_b + ind_a):+.3f}")
    # posterior probability that Italy is lower in every band with at least one death, and higher overall
    rng = np.random.default_rng(SEED + 1)
    draws = 200_000
    bands = [k for k in range(9) if ch_d[k] + it_d[k] > 0]
    fc = rng.beta(1 + ch_d[None, bands], 1 + (ch_c - ch_d)[None, bands], size=(draws, len(bands)))
    fi = rng.beta(1 + it_d[None, bands], 1 + (it_c - it_d)[None, bands], size=(draws, len(bands)))
    mix_c = rng.dirichlet(1 + ch_c, size=draws)
    mix_i = rng.dirichlet(1 + it_c, size=draws)
    fc_all = rng.beta(1 + ch_d[None], 1 + (ch_c - ch_d)[None], size=(draws, 9))
    fi_all = rng.beta(1 + it_d[None], 1 + (it_c - it_d)[None], size=(draws, 9))
    agg_i_higher = (mix_i * fi_all).sum(axis=1) > (mix_c * fc_all).sum(axis=1)
    every_lower = np.all(fi < fc, axis=1)
    old = [bands.index(k) for k in (5, 6, 7, 8)]
    lower_old = np.all(fi[:, old] < fc[:, old], axis=1)
    direct = (mix_c * (fi_all - fc_all)).sum(axis=1)
    lo_d, hi_d = np.percentile(direct, [2.5, 97.5])
    print(f"  posterior P(Italy lower in each of the four bands from 50 up) {lower_old.mean():.4f};"
          f" direct effect with China's age mix: 95% interval {100 * lo_d:+.3f} to {100 * hi_d:+.3f} points,"
          f" P(negative) {(direct < 0).mean():.4f}")
    print(f"  posterior (uniform priors): P(Italy's overall CFR higher) {agg_i_higher.mean():.4f};"
          f" P(Italy lower in all {len(bands)} bands with a death) {every_lower.mean():.4f};"
          f" P(both) {(agg_i_higher & every_lower).mean():.4f}")
    print()
    print("  Italy at later dates against China (17 February), percentage points:")
    order = sorted(ita_t, key=lambda k: (["March", "April", "May"].index(k[1].split()[1]), int(k[1].split()[0])))
    for key in order:
        c, dd = ita_t[key]
        tot, dir_a, ind_a, dir_b = cfr_effects(ch_c, ch_d, c, dd)
        print(f"    {key[1]:<14} cases {c.sum():>8,.0f}, CFR {dd.sum() / c.sum():.4f}: total {100 * tot:+.3f},"
              f" direct (China's mix) {100 * dir_a:+.3f}, indirect {100 * ind_a:+.3f}")
    print()
    print("  Stylised testing model: country B has exactly China's infections and deaths by age, but confirms only")
    print("  a share of the surviving infections under 60 (every death is confirmed); effects of B against China:")
    for share in (1.0, 0.75, 0.5, 0.25):
        tested = np.where(np.arange(9) < 6, share, 1.0)
        cb = ch_d + (ch_c - ch_d) * tested
        tot, dir_a, ind_a, dir_b = cfr_effects(ch_c, ch_d, cb, ch_d)
        print(f"    share of surviving under-60 infections confirmed {share:.2f}: crude CFR {ch_d.sum() / cb.sum():.4f};"
              f" share of confirmed cases aged 70+ {cb[7:].sum() / cb.sum():.3f}; total {100 * tot:+.3f},"
              f" direct (China's mix) {100 * dir_a:+.3f}, indirect {100 * ind_a:+.3f} points")
    print()
    print("  Every pair of countries in the compiled data set (latest date for each; Diamond Princess excluded):")
    latest = {}
    for (ctry, date), v in cov.items():
        if ctry != "Diam. Princ.":
            latest[ctry] = (date, v)
    names = sorted(latest)
    pooled_flip, full_rev, pairs = 0, 0, 0
    flips = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ca, da_ = latest[names[i]][1]
            cb, db_ = latest[names[j]][1]
            fa, fb = da_ / ca, db_ / cb
            crude = db_.sum() / cb.sum() - da_.sum() / ca.sum()
            w = (ca / ca.sum() + cb / cb.sum()) / 2
            std = np.sum(w * (fb - fa))
            pairs += 1
            if np.sign(crude) != np.sign(std):
                pooled_flip += 1
                flips.append((names[i], names[j], crude, std))
            m = (da_ + db_) > 0
            if (np.all(fb[m] < fa[m]) and crude > 0) or (np.all(fb[m] > fa[m]) and crude < 0):
                full_rev += 1
    print(f"    {len(names)} countries, {pairs} pairs: crude and age-standardised comparisons point in opposite"
          f" directions for {pooled_flip}; a full reversal (every band with a death one way, the crude rate the other)"
          f" for {full_rev}")
    for a, b, cr, sd in sorted(flips, key=lambda x: -abs(x[2] - x[3]))[:5]:
        print(f"      {b} against {a}: crude {100 * cr:+.2f} points, age-standardised {100 * sd:+.2f} points")
    for ctry in names:
        c, dd = latest[ctry][1]
        print(f"    {ctry:<13} {latest[ctry][0]:<17} cases {c.sum():>9,.0f}; crude CFR {dd.sum() / c.sum():.4f};"
              f" share of cases aged 70+ {c[7:].sum() / c.sum():.3f}")
    std_pop = sum(latest[c][1][0] / latest[c][1][0].sum() for c in names) / len(names)
    crude = np.array([latest[c][1][1].sum() / latest[c][1][0].sum() for c in names])
    stdz = np.array([np.sum(std_pop * latest[c][1][1] / latest[c][1][0]) for c in names])
    rk_c = stats.rankdata(-crude)
    rk_s = stats.rankdata(-stdz)
    print("  league table, crude against standardised to the average age distribution of the eleven countries:")
    for i in np.argsort(-stdz):
        print(f"    {names[i]:<13} crude {100 * crude[i]:5.2f}% (rank {rk_c[i]:.0f}); standardised {100 * stdz[i]:5.2f}% (rank {rk_s[i]:.0f})")
    print(f"    Spearman correlation of the two rankings {stats.spearmanr(crude, stdz).statistic:+.3f};"
          f" largest rank change {int(np.max(np.abs(rk_c - rk_s)))} places")

    header("2. SAT SCORES AND TEACHER SALARIES, US STATES, 1994-95 (Guber, 1999; mosaicData::SAT)")
    d = load_sat()
    n = len(d["sat"])
    one = np.ones(n)
    print(f"  {n} states; mean SAT {d['sat'].mean():.1f}; mean salary {d['salary'].mean():.2f} thousand dollars;"
          f" share taking the SAT from {d['frac'].min():.0f}% to {d['frac'].max():.0f}% (median {np.median(d['frac']):.0f}%)")
    print(f"  correlations: SAT-salary {np.corrcoef(d['sat'], d['salary'])[0, 1]:+.3f};"
          f" SAT-share taking {np.corrcoef(d['sat'], d['frac'])[0, 1]:+.3f};"
          f" salary-share taking {np.corrcoef(d['salary'], d['frac'])[0, 1]:+.3f}")
    for label, cols in (("SAT on salary", ["salary"]), ("SAT on salary and share taking", ["salary", "frac"]),
                        ("SAT on expenditure per pupil", ["expend"]),
                        ("SAT on expenditure and share taking", ["expend", "frac"])):
        X = np.column_stack([one] + [d[c] for c in cols])
        b, se, dof = ols(d["sat"], X)
        lo, hi = ci(b[1], se[1], dof)
        extra = ""
        if len(cols) == 2:
            lo2, hi2 = ci(b[2], se[2], dof)
            extra = f"; share taking {b[2]:+.3f} per point [{lo2:+.3f}, {hi2:+.3f}]"
        print(f"    {label:<38} {cols[0]} {b[1]:+.3f} per thousand dollars [{lo:+.3f}, {hi:+.3f}]{extra}")
    for label, extra_cols in (("log of share taking", [np.log(d["frac"])]),
                              ("share taking and its square", [d["frac"], d["frac"] ** 2])):
        X = np.column_stack([one, d["salary"]] + extra_cols)
        b, se, dof = ols(d["sat"], X)
        lo, hi = ci(b[1], se[1], dof)
        print(f"    SAT on salary and {label:<28} salary {b[1]:+.3f} per thousand dollars [{lo:+.3f}, {hi:+.3f}]")
    order = np.argsort(d["frac"])
    for lab, idx in (("lowest participation", order[:3]), ("highest participation", order[-3:][::-1])):
        print(f"  {lab}: " + "; ".join(f"{d['state'][i]} {d['frac'][i]:.0f}% taking, SAT {d['sat'][i]:.0f}, salary {d['salary'][i]:.1f}"
                                       for i in idx))
    # thirds by share taking
    q = np.quantile(d["frac"], [1 / 3, 2 / 3])
    groups = np.digitize(d["frac"], q)
    print(f"  states split into thirds by share taking (cut points {q[0]:.1f}% and {q[1]:.1f}%):")
    for g, name in enumerate(("low participation", "middle", "high participation")):
        m = groups == g
        X = np.column_stack([np.ones(m.sum()), d["salary"][m]])
        b, se, dof = ols(d["sat"][m], X)
        print(f"    {name:<18} {m.sum():>2} states: mean share {d['frac'][m].mean():5.1f}%, mean SAT {d['sat'][m].mean():6.1f},"
              f" mean salary {d['salary'][m].mean():5.2f}; slope of SAT on salary {b[1]:+.2f} (SE {se[1]:.2f})")
    X = np.column_stack([one, d["salary"], (groups == 1).astype(float), (groups == 2).astype(float)])
    b, se, dof = ols(d["sat"], X)
    lo, hi = ci(b[1], se[1], dof)
    print(f"    salary slope with third-of-participation effects: {b[1]:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    header("3. BILL LENGTH AND BILL DEPTH IN THREE PENGUIN SPECIES (palmerpenguins)")
    p = load_penguins()
    n = len(p["length"])
    print(f"  {n} penguins with both measurements: " + ", ".join(
        f"{s} {np.sum(p['species'] == s)}" for s in ("Adelie", "Chinstrap", "Gentoo")))
    X = np.column_stack([np.ones(n), p["length"]])
    b, se, dof = ols(p["depth"], X)
    lo, hi = ci(b[1], se[1], dof)
    print(f"  all penguins: slope of depth on length {b[1]:+.4f} mm per mm [{lo:+.4f}, {hi:+.4f}];"
          f" correlation {np.corrcoef(p['length'], p['depth'])[0, 1]:+.3f}")
    for s in ("Adelie", "Chinstrap", "Gentoo"):
        m = p["species"] == s
        X = np.column_stack([np.ones(m.sum()), p["length"][m]])
        b, se, dof = ols(p["depth"][m], X)
        lo, hi = ci(b[1], se[1], dof)
        print(f"  {s:<10} mean length {p['length'][m].mean():.1f}, mean depth {p['depth'][m].mean():.2f};"
              f" slope {b[1]:+.4f} [{lo:+.4f}, {hi:+.4f}]; correlation {np.corrcoef(p['length'][m], p['depth'][m])[0, 1]:+.3f}")
    tot_cov = np.cov(p["length"], p["depth"], bias=True)[0, 1]
    within, between = 0.0, 0.0
    for s in ("Adelie", "Chinstrap", "Gentoo"):
        m = p["species"] == s
        w_s = m.mean()
        within += w_s * np.cov(p["length"][m], p["depth"][m], bias=True)[0, 1]
        between += w_s * (p["length"][m].mean() - p["length"].mean()) * (p["depth"][m].mean() - p["depth"].mean())
    print(f"  covariance of length and depth {tot_cov:+.3f} mm^2 = within species {within:+.3f} + between species means {between:+.3f}")
    sp = [(p["species"] == s).astype(float) for s in ("Chinstrap", "Gentoo")]
    X = np.column_stack([np.ones(n), p["length"]] + sp)
    b, se, dof = ols(p["depth"], X)
    lo, hi = ci(b[1], se[1], dof)
    print(f"  common slope with species effects: {b[1]:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    known = p["sex"] != "NA"
    groups = [(s, x) for s in ("Adelie", "Chinstrap", "Gentoo") for x in ("female", "male")]
    dummies = [((p["species"] == s) & (p["sex"] == x)).astype(float) for s, x in groups[1:]]
    X = np.column_stack([np.ones(n), p["length"]] + dummies)[known]
    b, se, dof = ols(p["depth"][known], X)
    lo, hi = ci(b[1], se[1], dof)
    print(f"  common slope with species-by-sex effects ({known.sum()} penguins of known sex): {b[1]:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    header("4. HOW OFTEN REVERSALS HAPPEN BY CHANCE")
    rng = np.random.default_rng(SEED)
    draws = 2_000_000
    cells = rng.dirichlet(np.ones(8), size=draws).reshape(draws, 2, 2, 2)
    rev = reversal_in_table(cells)
    pr = rev.mean()
    print(f"  2 x 2 x 2 tables drawn uniformly from the probability simplex ({draws:,} draws):"
          f" reversal in {pr:.5f} (SE {math.sqrt(pr * (1 - pr) / draws):.5f}); 1/60 = {1 / 60:.5f}")
    print("  Sampling noise: populations drawn uniformly from the simplex WITHOUT a reversal, samples of n")
    print("  drawn from each, share of samples showing a reversal the population does not have:")
    pops = cells[~rev][:20_000]
    flat = pops.reshape(len(pops), 8)
    for n_s in (50, 100, 500, 1000, 5000):
        counts = np.array([rng.multinomial(n_s, f) for f in flat]).reshape(len(pops), 2, 2, 2).astype(float)
        valid = np.all(counts.sum(axis=-1) > 0, axis=(-1, -2))
        rev_s = reversal_in_table(np.where(counts.sum(axis=-1, keepdims=True) > 0, counts, 1.0))
        print(f"    n = {n_s:>5}: {np.mean(rev_s[valid]):.4f} of {valid.sum():,} samples with every cell group non-empty")
    print()
    print("  Every pair of regular players over consecutive seasons, 1990-2018 (Lahman; at least 100 at-bats in")
    print("  each season of the pair; 1994-95 strike seasons included as they are):")
    seasons = batting_seasons(1990, 2019, 100)
    obs_rev, obs_pairs, null_rev = 0, 0, []
    d_avg, d_ab = [], []
    z_all = []
    null_sims = 20
    null_counts = np.zeros(null_sims)
    per_year = []
    for y in range(1990, 2019):
        ids = sorted({pid for (pid, yy) in seasons if yy == y} & {pid for (pid, yy) in seasons if yy == y + 1})
        h1 = np.array([seasons[(i, y)][0] for i in ids], float)
        ab1 = np.array([seasons[(i, y)][1] for i in ids], float)
        h2 = np.array([seasons[(i, y + 1)][0] for i in ids], float)
        ab2 = np.array([seasons[(i, y + 1)][1] for i in ids], float)
        r, pairs = count_pair_reversals(h1, ab1, h2, ab2)
        z1, z2, zc = reversal_pairs_detail(h1, ab1, h2, ab2)
        z_all.append((z1, z2, zc))
        obs_rev += r
        obs_pairs += pairs
        per_year.append((y, len(ids), pairs, r))
        d_avg.append(h2 / ab2 - h1 / ab1)
        d_ab.append(ab2 - ab1)
        p_true = (h1 + h2) / (ab1 + ab2)
        for s in range(null_sims):
            s1 = rng.binomial(ab1.astype(int), p_true)
            s2 = rng.binomial(ab2.astype(int), p_true)
            null_counts[s] += count_pair_reversals(s1.astype(float), ab1, s2.astype(float), ab2)[0]
    print(f"    {len(per_year)} season pairs, {obs_pairs:,} player pairs; observed reversals {obs_rev:,}"
          f" ({obs_rev / obs_pairs:.5f} of pairs, about 1 in {obs_pairs / obs_rev:.0f})")
    print(f"    null model (each player's true average the same in both seasons, hits binomial):"
          f" mean {null_counts.mean():,.0f} reversals (range {null_counts.min():,.0f} to {null_counts.max():,.0f}"
          f" over {null_sims} simulations); observed / null {obs_rev / null_counts.mean():.3f}")
    z1 = np.concatenate([z[0] for z in z_all])
    z2 = np.concatenate([z[1] for z in z_all])
    zc = np.concatenate([z[2] for z in z_all])
    both = (z1 > 1.96) & (z2 > 1.96)
    print(f"    of the observed reversals: both within-season differences beyond 1.96 standard errors {both.sum()};"
          f" at least one {np.sum((z1 > 1.96) | (z2 > 1.96))}; combined difference beyond 1.96 {np.sum(np.abs(zc) > 1.96)};"
          f" all three {np.sum(both & (np.abs(zc) > 1.96))}")
    print(f"    median absolute z of the within-season differences {np.median(np.concatenate([z1, z2])):.3f};"
          f" of the combined difference {np.median(np.abs(zc)):.3f}")
    da, db = np.concatenate(d_avg), np.concatenate(d_ab)
    rho = stats.spearmanr(da, db)
    print(f"    player season-pairs: {len(da):,}; rank correlation between the change in a player's average and the change"
          f" in his at-bats from one season to the next {rho.statistic:+.3f}")
    y, k, pairs, r = max(per_year, key=lambda t: t[3] / t[2])
    y2, k2, pairs2, r2 = min(per_year, key=lambda t: t[3] / t[2])
    print(f"    highest share: {y}-{y + 1}, {r} of {pairs:,} pairs; lowest: {y2}-{y2 + 1}, {r2} of {pairs2:,}")


if __name__ == "__main__":
    main()
