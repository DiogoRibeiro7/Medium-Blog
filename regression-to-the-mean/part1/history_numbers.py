"""Part 1 of the regression-to-the-mean series: the history, with the data recomputed.

Every number quoted in the article is printed by this script. Run:

    python history_numbers.py > output.txt

Data downloaded on first run from a pinned commit of the Rdatasets repository, checked against SHA-256
digests and cached in data/:
  - HistData::Galton: Galton's (1886) table of 928 adult children and the mid-heights of their parents;
  - HistData::GaltonFamilies: Hanley's (2004) transcription of Galton's notebook, child by child;
  - HistData::PearsonLee: Pearson and Lee's (1903) frequency tables of parents' and children's heights;
  - Lahman::Teams: season records of major league teams, for a replication of Secrist's (1933) design.

Sections
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


def main() -> None:
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
        print(f"    child {v:5.1f}: {m.sum():>3} children, mean mid-parent {parent[m].mean():.2f}")
    top = parent >= 70.5
    bottom = parent <= 66.5
    print(f"  mid-parents of 70.5 in or more: {top.sum()} children, parents {parent[top].mean() - parent.mean():+.2f} in above"
          f" the mean, children {child[top].mean() - child.mean():+.2f} ({(child[top].mean() - child.mean()) / (parent[top].mean() - parent.mean()):.3f} of it)")
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
    x, y, w, mx, my, sx, sy, rr = fs
    for cut in (72.0, 74.0):
        m = x >= cut - 0.5
        dev_f = np.average(x[m], weights=w[m]) - mx
        dev_s = np.average(y[m], weights=w[m]) - my
        print(f"  fathers of {cut:.0f} in or more ({w[m].sum():,.1f} pairs): {dev_f:+.2f} in above the fathers' mean; their sons"
              f" {dev_s:+.2f} in above the sons' mean, {dev_s / dev_f:.3f} of the fathers' deviation in standardised units"
              f" {dev_s / sy / (dev_f / sx):.3f}")
        m = y >= cut - 0.5
        dev_s = np.average(y[m], weights=w[m]) - my
        dev_f = np.average(x[m], weights=w[m]) - mx
        print(f"  sons of {cut:.0f} in or more ({w[m].sum():,.1f} pairs): {dev_s:+.2f} in above the sons' mean; their fathers"
              f" {dev_f:+.2f} in above the fathers' mean ({dev_f / dev_s:.3f} of it)")
    # an F test of equal variances, treating the weighted pairs as independent and using the correlation
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
    pairs = [(wp[(f, y_)], wp[(f, y_ + 1)]) for (f, y_) in wp if (f, y_ + 1) in wp and y_ >= 1961]
    a = np.array(pairs)
    print(f"  season-to-season correlation of a team's winning percentage, 1961 on: {np.corrcoef(a[:, 0], a[:, 1])[0, 1]:.4f}"
          f" ({len(a):,} pairs); SD in the first season {a[:, 0].std():.4f}, in the second {a[:, 1].std():.4f}")


if __name__ == "__main__":
    main()
