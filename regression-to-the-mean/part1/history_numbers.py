"""Part 1 of the regression-to-the-mean series: the history, with the data recomputed.

Every number quoted in the article is printed by this script. Run:

    python history_numbers.py > output.txt

Data downloaded on first run from a pinned commit of the Rdatasets repository, checked against SHA-256
digests and cached in data/:
  - HistData::Galton: Galton's (1886) table of 928 adult children and the mid-heights of their parents;
  - HistData::GaltonFamilies: Hanley's (2004) transcription of Galton's notebook, child by child;
  - HistData::PearsonLee: Pearson and Lee's (1903) frequency tables of parents' and children's heights;
  - Lahman::Teams: season records of major league teams, for a replication of Secrist's (1933) design.

Galton's sweet-pea table is seven rows long and is written into the script with its source.

Sections
0. Galton's sweet peas: parent seed diameter against the mean diameter of the offspring.
1. Galton's mid-parents and children: both regressions, the correlation, the spread of each generation.
2. Galton's family records: fathers, mothers, sons and daughters, with standard errors that respect families.
3. Pearson and Lee: fathers and sons, the two regressions and the two standard deviations.
4. Secrist's design on baseball teams: groups formed in one season, followed forwards and backwards.
"""

from __future__ import annotations

import csv
import hashlib
import math
import urllib.request
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
DATA = HERE / "data"
RD = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/1dcc2bf5f955cc1224a3e1307256e1fe86b68dae/csv/"
FILES = {
    "Galton.csv": RD + "HistData/Galton.csv",
    "GaltonFamilies.csv": RD + "HistData/GaltonFamilies.csv",
    "PearsonLee.csv": RD + "HistData/PearsonLee.csv",
    "Teams.csv": RD + "Lahman/Teams.csv",
}
DIGESTS = {
    "Galton.csv": "9c31eab984878ee7bc5eefe592b2f92b9bc19ad11c53bf3132282ff28fb669f0",
    "GaltonFamilies.csv": "b1fc91e4d4b9604661967dcc62ff1055eaa3020c4d5c42d1f9d588f74b9b3317",
    "PearsonLee.csv": "ff4e6c4e02c00eb90834951de9111f87034fd9d2729fd2590c081c66f3b7c332",
    "Teams.csv": "9d1c390df5d2a422175f6bcfb56d099cfb9c5805586ba5e3f8c895a9952423ad",
}


def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fetch(name: str) -> Path:
    url = FILES[name]
    path = DATA / name
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=300) as r:
            path.write_bytes(r.read())
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != DIGESTS[name]:
        raise RuntimeError(f"{name}: SHA-256 {got} does not match the pinned digest {DIGESTS[name]}")
    return path


def read(name: str) -> list[dict[str, str]]:
    with open(fetch(name)) as f:
        return list(csv.DictReader(f))


def weighted_moments(x, y, w):
    """Weighted means, standard deviations (dividing by the total weight), covariance and correlation."""
    w = np.asarray(w, float)
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    sx = math.sqrt(np.average((x - mx) ** 2, weights=w))
    sy = math.sqrt(np.average((y - my) ** 2, weights=w))
    cxy = np.average((x - mx) * (y - my), weights=w)
    return mx, my, sx, sy, cxy, cxy / (sx * sy)


def ols_slope(x, y):
    """Least-squares slope and intercept of y on x with the conventional standard error of the slope."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = resid @ resid / (len(y) - 2)
    se = math.sqrt(s2 / np.sum((x - x.mean()) ** 2))
    return beta[1], beta[0], se


def cluster_se(x, y, groups):
    """Slope of y on x with a standard error that allows correlation within groups (families): the
    sandwich estimator with the usual small-sample factor G / (G - 1) * (n - 1) / (n - 2)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    bread = np.linalg.inv(X.T @ X)
    meat = np.zeros((2, 2))
    labels = np.asarray(groups)
    uniq = np.unique(labels)
    for g in uniq:
        m = labels == g
        s = X[m].T @ resid[m]
        meat += np.outer(s, s)
    n, G = len(y), len(uniq)
    cov = bread @ meat @ bread * G / (G - 1) * (n - 1) / (n - 2)
    return beta[1], math.sqrt(cov[1, 1])


# Galton (1886), Appendix I, Table VI, p. 260 (the same table is Table 2 of Natural Inheritance, 1889, p. 226):
# mean diameter of the produce, in hundredths of an inch, for parent seeds of each diameter; each row of the
# original is scaled to 100 seeds. "Observed" and "smoothed" are Galton's two columns. Pearson (1930, Life,
# Letters and Labours of Francis Galton, IIIA, p. 7) recomputed the observed means from Galton's full table.
PEAS_PARENT = np.array([21, 20, 19, 18, 17, 16, 15], float)
PEAS_OBSERVED = np.array([17.5, 17.3, 16.0, 16.3, 15.6, 16.0, 15.3])
PEAS_SMOOTHED = np.array([17.3, 17.0, 16.6, 16.3, 16.0, 15.7, 15.4])
PEAS_PEARSON = np.array([17.26, 17.07, 16.37, 16.40, 16.13, 16.17, 15.98])


def ols_cluster(X, y, groups=None):
    """Least squares with conventional standard errors and, if groups are given, cluster-robust ones."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(bread * (resid @ resid) / (n - k)))
    if groups is None:
        return beta, se, None, resid
    labels = np.asarray(groups)
    meat = np.zeros((k, k))
    uniq = np.unique(labels)
    for g in uniq:
        sg = X[labels == g].T @ resid[labels == g]
        meat += np.outer(sg, sg)
    G = len(uniq)
    cov = bread @ meat @ bread * G / (G - 1) * (n - 1) / (n - k)
    return beta, se, np.sqrt(np.diag(cov)), resid


def icc_oneway(values, groups):
    """Intraclass correlation from a one-way analysis of variance with unequal group sizes."""
    labels = np.asarray(groups)
    uniq = np.unique(labels)
    gs = [values[labels == g] for g in uniq]
    N, G = len(values), len(gs)
    ni = np.array([len(v) for v in gs])
    grand = values.mean()
    msb = sum(len(v) * (v.mean() - grand) ** 2 for v in gs) / (G - 1)
    msw = sum(((v - v.mean()) ** 2).sum() for v in gs) / (N - G)
    n0 = (N - (ni ** 2).sum() / N) / (G - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw), n0


def main() -> None:
    header("0. GALTON'S SWEET PEAS (Galton 1886, Table VI; Pearson 1930)")
    for label, ys in (("Galton's observed means", PEAS_OBSERVED), ("Galton's smoothed means", PEAS_SMOOTHED),
                      ("Pearson's recomputed means", PEAS_PEARSON)):
        b, a, se = ols_slope(PEAS_PARENT, ys)
        print(f"  {label:<27} slope on parent diameter {b:.4f} (SE {se:.4f}); intercept {a:.3f};"
              f" fixed point (filial = parent) {a / (1 - b):.2f} hundredths of an inch")
    print(f"  parents 21 and 15: produce {PEAS_OBSERVED[0]:.1f} and {PEAS_OBSERVED[-1]:.1f}; a 6-unit difference in"
          f" parents becomes {PEAS_OBSERVED[0] - PEAS_OBSERVED[-1]:.1f} in the observed means ({(PEAS_OBSERVED[0] - PEAS_OBSERVED[-1]) / 6:.3f} per unit)")

    header("1. GALTON (1886): 928 ADULT CHILDREN AND THEIR MID-PARENTS (HistData::Galton)")
    g = read("Galton.csv")
    parent = np.array([float(r["parent"]) for r in g])
    child = np.array([float(r["child"]) for r in g])
    n = len(child)
    r = np.corrcoef(parent, child)[0, 1]
    b_cp, a_cp, se_cp = ols_slope(parent, child)
    b_pc, a_pc, se_pc = ols_slope(child, parent)
    print(f"  {n} children; mean mid-parent {parent.mean():.3f} in, SD {parent.std(ddof=1):.3f};"
          f" mean child {child.mean():.3f} in, SD {child.std(ddof=1):.3f}")
    print(f"  correlation {r:.4f}; 95% interval {np.tanh(np.arctanh(r) - 1.96 / math.sqrt(n - 3)):.4f} to"
          f" {np.tanh(np.arctanh(r) + 1.96 / math.sqrt(n - 3)):.4f}")
    print(f"  slope of child on mid-parent {b_cp:.4f} (SE {se_cp:.4f}), intercept {a_cp:.2f};"
          f" slope of mid-parent on child {b_pc:.4f} (SE {se_pc:.4f})")
    print(f"  product of the two slopes {b_cp * b_pc:.4f} = r^2 {r ** 2:.4f}")
    print("  mean child height by mid-parent height (Galton's grouping):")
    for v in sorted(set(parent)):
        m = parent == v
        print(f"    mid-parent {v:5.1f}: {m.sum():>3} children, mean {child[m].mean():.2f}, deviation from"
              f" the overall mean: parents {v - parent.mean():+.2f}, children {child[m].mean() - child.mean():+.2f}")
    print("  mean mid-parent height by child height (the regression in the other direction):")
    for v in sorted(set(child)):
        m = child == v
        print(f"    child {v:5.1f}: {m.sum():>3} children, mean mid-parent {parent[m].mean():.2f}; deviations from the"
              f" overall means: children {v - child.mean():+.2f}, mid-parents {parent[m].mean() - parent.mean():+.2f}")
    top = parent >= 70.5
    bottom = parent <= 66.5
    print(f"  mid-parents of 70.5 in or more: {top.sum()} children, parents {parent[top].mean() - parent.mean():+.2f} in above"
          f" the mean, children {child[top].mean() - child.mean():+.2f} ({(child[top].mean() - child.mean()) / (parent[top].mean() - parent.mean()):.3f} of it)")
    xc = parent - parent.mean()
    beta, se, _, _ = ols_cluster(np.column_stack([np.ones(n), xc, xc ** 2]), child)
    print(f"  quadratic term in the regression of child on mid-parent: {beta[2]:+.4f} per square inch (SE {se[2]:.4f}),"
          f" t = {beta[2] / se[2]:.2f}, p = {2 * stats.t.sf(abs(beta[2] / se[2]), n - 3):.3f}")
    print(f"  mid-parents of 66.5 in or less: {bottom.sum()} children, parents {parent[bottom].mean() - parent.mean():+.2f} in,"
          f" children {child[bottom].mean() - child.mean():+.2f} ({(child[bottom].mean() - child.mean()) / (parent[bottom].mean() - parent.mean()):.3f} of it)")

    header("2. GALTON'S FAMILY RECORDS (HistData::GaltonFamilies, Hanley's transcription)")
    fam = read("GaltonFamilies.csv")
    father = np.array([float(r["father"]) for r in fam])
    mother = np.array([float(r["mother"]) for r in fam])
    mid = np.array([float(r["midparentHeight"]) for r in fam])
    ch = np.array([float(r["childHeight"]) for r in fam])
    sex = np.array([r["gender"] for r in fam])
    famid = np.array([r["family"] for r in fam])
    print(f"  {len(ch)} children in {len(set(famid))} families: {np.sum(sex == 'male')} sons, {np.sum(sex == 'female')} daughters")
    print(f"  largest |mid-parent - (father + 1.08 x mother) / 2|: {np.max(np.abs(mid - (father + 1.08 * mother) / 2)):.4f}")
    print(f"  fathers mean {father[np.unique(famid, return_index=True)[1]].mean():.2f} in, mothers mean"
          f" {mother[np.unique(famid, return_index=True)[1]].mean():.2f} in (one record per family)")
    print(f"  sons mean {ch[sex == 'male'].mean():.2f}, SD {ch[sex == 'male'].std(ddof=1):.2f};"
          f" daughters mean {ch[sex == 'female'].mean():.2f}, SD {ch[sex == 'female'].std(ddof=1):.2f};"
          f" ratio of means {ch[sex == 'male'].mean() / ch[sex == 'female'].mean():.4f}")
    for label, m in (("sons", sex == "male"), ("daughters", sex == "female")):
        for xname, x in (("mid-parent", mid), ("father", father), ("mother", mother)):
            b, se_cl = cluster_se(x[m], ch[m], famid[m])
            _, _, se_naive = ols_slope(x[m], ch[m])
            rr = np.corrcoef(x[m], ch[m])[0, 1]
            print(f"    {label:<9} on {xname:<10} slope {b:.4f} (SE {se_naive:.4f} ignoring families, {se_cl:.4f} by family);"
                  f" correlation {rr:.4f}")
    for label, m in (("sons", sex == "male"), ("daughters", sex == "female")):
        xc = mid[m] - mid[m].mean()
        beta, se, se_cl, _ = ols_cluster(np.column_stack([np.ones(m.sum()), xc, xc ** 2]), ch[m], famid[m])
        print(f"    {label:<9} quadratic term on mid-parent {beta[2]:+.4f} (SE {se[2]:.4f}, {se_cl[2]:.4f} by family);"
              f" t = {beta[2] / se[2]:.2f} ignoring families, {beta[2] / se_cl[2]:.2f} by family")
    pooled = np.where(sex == "female", 1.08 * ch, ch)
    xc = mid - mid.mean()
    beta, se, se_cl, _ = ols_cluster(np.column_stack([np.ones(len(ch)), xc, xc ** 2]), pooled, famid)
    print(f"    all children, daughters x 1.08 as Galton pooled them: quadratic term {beta[2]:+.4f}; t = {beta[2] / se[2]:.2f}"
          f" ignoring families, {beta[2] / se_cl[2]:.2f} by family")
    X = np.column_stack([np.ones(len(ch)), father, mother, (sex == "male").astype(float)])
    beta, se, se_cl, resid = ols_cluster(X, ch, famid)
    icc, n0 = icc_oneway(resid, famid)
    sizes = np.array([np.sum(famid == f) for f in np.unique(famid)])
    print(f"  child on father, mother and sex: father {beta[1]:.4f} (SE {se_cl[1]:.4f} by family), mother {beta[2]:.4f}"
          f" ({se_cl[2]:.4f}), son minus daughter {beta[3]:.3f} in; residual SD {resid.std(ddof=4):.3f} in")
    print(f"  intraclass correlation of those residuals within families {icc:.4f} (one-way ANOVA, average family size"
          f" {n0:.2f}); families with two or more children {np.sum(sizes >= 2)}")
    fu = np.unique(famid, return_index=True)[1]
    print(f"  correlation between fathers and mothers (one record per family): {np.corrcoef(father[fu], mother[fu])[0, 1]:+.4f}")
    print(f"  SD of fathers {father[fu].std(ddof=1):.3f}; of 1.08 x mothers {(1.08 * mother[fu]).std(ddof=1):.3f};"
          f" of mid-parents {mid[fu].std(ddof=1):.3f}; ratio mid-parent / father {mid[fu].std(ddof=1) / father[fu].std(ddof=1):.3f}")

    header("3. PEARSON AND LEE (1903): FATHERS AND SONS (HistData::PearsonLee)")
    pl = read("PearsonLee.csv")
    for code, lab in (("fs", "father-son"), ("fd", "father-daughter"), ("ms", "mother-son"), ("md", "mother-daughter")):
        rows = [r for r in pl if r["gp"] == code]
        x = np.array([float(r["parent"]) for r in rows])
        y = np.array([float(r["child"]) for r in rows])
        w = np.array([float(r["frequency"]) for r in rows])
        mx, my, sx, sy, cxy, rr = weighted_moments(x, y, w)
        print(f"  {lab:<16} {w.sum():,.0f} pairs ({len(rows)} cells): parents mean {mx:.2f}, SD {sx:.3f};"
              f" children mean {my:.2f}, SD {sy:.3f}; correlation {rr:.4f}; slope child on parent {cxy / sx ** 2:.4f},"
              f" parent on child {cxy / sy ** 2:.4f}")
        if code == "fs":
            fs = (x, y, w, mx, my, sx, sy, rr)
        if code == "md":
            nn_ = w.sum()
            t_md = (sy ** 2 - sx ** 2) * math.sqrt(nn_ - 2) / math.sqrt(4 * (1 - rr ** 2) * sx ** 2 * sy ** 2)
            print(f"    Pitman-Morgan test of equal variances of mothers and daughters: t = {t_md:.3f}, p = {2 * stats.t.sf(abs(t_md), nn_ - 2):.4f}")
    x, y, w, mx, my, sx, sy, rr = fs
    for cls in (71.5, 73.5):
        m = x >= cls
        dev_f = np.average(x[m], weights=w[m]) - mx
        dev_s = np.average(y[m], weights=w[m]) - my
        print(f"  fathers in the {cls} class or above ({w[m].sum():,.1f} pairs, {w[m].sum() / w.sum():.3f} of fathers):"
              f" {dev_f:+.2f} in above the fathers' mean; their sons {dev_s:+.2f} in above the sons' mean,"
              f" {dev_s / dev_f:.3f} of the fathers' deviation; in standardised units {dev_s / sy / (dev_f / sx):.3f}")
        m = y >= cls
        dev_s = np.average(y[m], weights=w[m]) - my
        dev_f = np.average(x[m], weights=w[m]) - mx
        print(f"  sons in the {cls} class or above ({w[m].sum():,.1f} pairs, {w[m].sum() / w.sum():.3f} of sons):"
              f" {dev_s:+.2f} in above the sons' mean; their fathers {dev_f:+.2f} in above the fathers' mean ({dev_f / dev_s:.3f} of it)")
    for cf, cs in ((70.5, 71.5), (71.5, 72.5)):
        tf, ts = x >= cf, y >= cs
        print(f"  'tall' = fathers in the {cf:.1f} class or above ({w[tf].sum() / w.sum():.3f} of fathers), sons in the {cs:.1f}"
              f" class or above ({w[ts].sum() / w.sum():.3f} of sons): P(tall son | tall father) {w[tf & ts].sum() / w[tf].sum():.3f};"
              f" P(tall father | tall son) {w[tf & ts].sum() / w[ts].sum():.3f}; tall sons whose fathers were not tall"
              f" {w[~tf & ts].sum():.2f} of {w[ts].sum():.1f} ({w[~tf & ts].sum() / w[ts].sum():.3f})")
    for q in (0.2, 0.1, 0.05):
        z = stats.norm.ppf(1 - q)
        both = stats.multivariate_normal(mean=[0, 0], cov=[[1, rr], [rr, 1]]).cdf([-z, -z])
        print(f"  bivariate normal with correlation {rr:.3f}: P(son in top {q:.0%} | father in top {q:.0%}) {both / q:.3f}")
    # the Pitman-Morgan test of equal variances in paired data, treating the weighted pairs as independent
    nn = w.sum()
    t = (sy ** 2 - sx ** 2) * math.sqrt(nn - 2) / math.sqrt(4 * (1 - rr ** 2) * sx ** 2 * sy ** 2)
    print(f"  Pitman-Morgan test of equal variances of fathers and sons: t = {t:.3f} on {nn - 2:,.0f} df,"
          f" p = {2 * stats.t.sf(abs(t), nn - 2):.3f}")

    header("4. SECRIST'S DESIGN ON MAJOR LEAGUE TEAMS (Lahman::Teams)")
    teams = read("Teams.csv")
    wp: dict[tuple[str, int], float] = {}
    for r in teams:
        y_ = int(r["yearID"])
        if r["lgID"] in ("AL", "NL") and y_ >= 1901:
            W, L = int(r["W"]), int(r["L"])
            wp[(r["franchID"], y_)] = W / (W + L)
    years = sorted({y_ for _, y_ in wp})
    print(f"  {len(wp):,} team seasons in the American and National Leagues, {years[0]}-{years[-1]}")
    sd_by_year = {y_: np.std([v for (f, yy), v in wp.items() if yy == y_]) for y_ in years}
    games = {}
    for r in teams:
        y_ = int(r["yearID"])
        if r["lgID"] in ("AL", "NL") and y_ >= 1901:
            games.setdefault(y_, []).append(int(r["W"]) + int(r["L"]))
    for lo, hi in ((1901, 1930), (1931, 1960), (1961, 1990), (1991, 2025)):
        vals = [sd_by_year[y_] for y_ in years if lo <= y_ <= hi]
        noise = [math.sqrt(0.25 / np.mean(games[y_])) for y_ in years if lo <= y_ <= hi]
        true_sd = [math.sqrt(max(sd_by_year[y_] ** 2 - 0.25 / np.mean(games[y_]), 0)) for y_ in years if lo <= y_ <= hi]
        print(f"    SD of winning percentage across teams, {lo}-{hi}: mean {np.mean(vals):.4f}, range {min(vals):.4f} to {max(vals):.4f};"
              f" coin-toss SD for the games played {np.mean(noise):.4f}; SD left after removing it {np.mean(true_sd):.4f}")
    share = [1 - 0.25 / np.mean(games[y_]) / sd_by_year[y_] ** 2 for y_ in years if 1961 <= y_ <= 2025 and y_ != 2020]
    print(f"    1961-2025 excluding 2020: share of the variance of winning percentage not explained by coin-toss luck,"
          f" mean {np.mean(share):.3f} (range {min(share):.3f} to {max(share):.3f})")
    print(f"    2020 (a {np.mean(games[2020]):.0f}-game season): SD {sd_by_year[2020]:.4f}, coin-toss SD {math.sqrt(0.25 / np.mean(games[2020])):.4f}")
    k = 5
    bases = [y_ for y_ in range(1961, 2026 - k) if y_ - k >= 1901]
    offsets = list(range(-k, k + 1))
    groups = {q: {o: [] for o in offsets} for q in range(4)}
    for y0 in bases:
        franchises = sorted(f for (f, yy) in wp if yy == y0)
        vals = np.array([wp[(f, y0)] for f in franchises])
        cuts = np.quantile(vals, [0.25, 0.5, 0.75])
        quart = np.digitize(vals, cuts)
        for f, q in zip(franchises, quart):
            if all((f, y0 + o) in wp for o in offsets):
                for o in offsets:
                    groups[q][o].append(wp[(f, y0 + o)] - np.mean([v for (ff, yy), v in wp.items() if yy == y0 + o]))
    print(f"  base seasons {bases[0]}-{bases[-1]}; teams grouped into quarters by winning percentage in the base season;"
          f" mean deviation from that season's league mean, in points of winning percentage (x 1000):")
    for q in range(4):
        line = "; ".join(f"{o:+d}: {1000 * np.mean(groups[q][o]):+.1f}" for o in offsets)
        print(f"    quarter {q + 1} ({len(groups[q][0]):,} team-seasons): {line}")
    top0, top1 = np.mean(groups[3][0]), np.mean(groups[3][1])
    bot0, bot1 = np.mean(groups[0][0]), np.mean(groups[0][1])
    print(f"    top quarter keeps {top1 / top0:.3f} of its deviation one season later and {np.mean(groups[3][k]) / top0:.3f}"
          f" after {k}; bottom quarter {bot1 / bot0:.3f} and {np.mean(groups[0][k]) / bot0:.3f}")
    print(f"    one season earlier the top quarter had {np.mean(groups[3][-1]) / top0:.3f} of its base-season deviation,"
          f" the bottom quarter {np.mean(groups[0][-1]) / bot0:.3f}")
    stay, from_other, n_top, n_top_next = 0, 0, 0, 0
    sd_top0, sd_top1 = [], []
    for y0 in range(1961, 2025):
        fr = sorted(f for (f, yy) in wp if yy == y0 and (f, y0 + 1) in wp)
        v0 = np.array([wp[(f, y0)] for f in fr])
        v1 = np.array([wp[(f, y0 + 1)] for f in fr])
        top0 = v0 >= np.quantile(v0, 0.75)
        top1 = v1 >= np.quantile(v1, 0.75)
        stay += np.sum(top0 & top1)
        n_top += top0.sum()
        from_other += np.sum(~top0 & top1)
        n_top_next += top1.sum()
        sd_top0.append(v0[top0].std())
        sd_top1.append(v1[top0].std())
    print(f"  top quarter in one season (1961-2024, {n_top:,} team-seasons): still in the top quarter the next season {stay / n_top:.3f};"
          f" of the next season's top quarter ({n_top_next:,}), from outside the top quarter {from_other / n_top_next:.3f}")
    print(f"  spread of winning percentage within the top quarter: {np.mean(sd_top0):.4f} in the season it was formed,"
          f" {np.mean(sd_top1):.4f} one season later (averages over base seasons)")
    pairs = [(wp[(f, y_)], wp[(f, y_ + 1)]) for (f, y_) in wp if (f, y_ + 1) in wp and y_ >= 1961]
    a = np.array(pairs)
    rho = np.corrcoef(a[:, 0], a[:, 1])[0, 1]
    m0 = np.array([np.mean(groups[q][0]) for q in range(4)])
    m1 = np.array([np.mean(groups[q][1]) for q in range(4)])
    nq = np.array([len(groups[q][0]) for q in range(4)], float)
    v0 = np.average((m0 - np.average(m0, weights=nq)) ** 2, weights=nq)
    v1 = np.average((m1 - np.average(m1, weights=nq)) ** 2, weights=nq)
    print(f"  Hotelling's check: variance of the four quarter means one season later / in the base season {v1 / v0:.3f};"
          f" square of the season-to-season correlation {rho ** 2:.3f}")
    print(f"  season-to-season correlation of a team's winning percentage, 1961 on: {np.corrcoef(a[:, 0], a[:, 1])[0, 1]:.4f}"
          f" ({len(a):,} pairs); SD in the first season {a[:, 0].std():.4f}, in the second {a[:, 1].std():.4f}")


if __name__ == "__main__":
    main()
