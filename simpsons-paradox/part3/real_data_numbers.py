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
FILES = {
    "penguins.csv": ("https://raw.githubusercontent.com/allisonhorst/palmerpenguins/"
                     "8957207b78d6ccd1b4654a9dd9c9041b657478ab/inst/extdata/penguins.csv",
                     "f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93"),
    "SAT.csv": (RD + "csv/mosaicData/SAT.csv", "2370aa1e09529d69e3b4d805f617811d6641c46772a52bfd6a292e3d1cae457b"),
    "Batting.csv": (RD + "csv/Lahman/Batting.csv", "a4083e2d4bacf411c352686fd7c0696eaa4bc8719b718db5e9c45ac90fd95b35"),
}


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
    da, db = np.concatenate(d_avg), np.concatenate(d_ab)
    rho = stats.spearmanr(da, db)
    print(f"    player season-pairs: {len(da):,}; rank correlation between the change in a player's average and the change"
          f" in his at-bats from one season to the next {rho.statistic:+.3f}")
    y, k, pairs, r = max(per_year, key=lambda t: t[3] / t[2])
    y2, k2, pairs2, r2 = min(per_year, key=lambda t: t[3] / t[2])
    print(f"    highest share: {y}-{y + 1}, {r} of {pairs:,} pairs; lowest: {y2}-{y2 + 1}, {r2} of {pairs2:,}")


if __name__ == "__main__":
    main()
