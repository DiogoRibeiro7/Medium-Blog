"""Was European industrialisation regional rather than national? Tests on the Rosés-Wolf
database of regional GDP (version 7, 2025).

    python check_data.py     # the two hand-downloaded files are present and match the registry
    python regions.py        # writes results/summary.txt and the CSVs figures.py reads
    python regions.py 50     # the same with 50 random partitions instead of 1000, for a quick look

Three tests, each against a null model built from random contiguous partitions of the map
(groups of neighbouring regions with the same sizes as the 16 countries):

1. How much of the variation in the share of industry in employment across regions lies between
   countries, and is that more than any random grouping of neighbouring regions would capture?
2. Are neighbouring regions on opposite sides of a national border more different than
   neighbouring regions inside one country (the border premium)?
3. Where was industry concentrated in 1900, do those clusters cross borders, and what became of
   them by 2015?

A last section lists income levels by country and, border by border, how far apart neighbouring
regions are in income and in industry, in 1900 and 2021.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
RES = HERE / "results"
WORKBOOK = RAW / "RosesWolf_RegionalGDP_v7.xlsx"
SHAPES = RAW / "RosesWolf_RegionalGDP_shapefile.zip"
N_PARTITIONS = 1000
SEED = 20261122
CORE_N = 35  # the top fifth of the 173 regions by industrial share in 1900
MERGED = {"AT12+AT13": "AT123", "DE71+DE72": "DE712", "DE91+DE92": "DE912"}
SHEETS = {"ind": "A5 Employment Share Industry", "agr": "A4 Employment Share Agriculture",
          "srv": "A6 Employment Share Services", "pop": "A3 Population",
          "gdp": "A1b Regional GDP (2011PPP)", "gdp90": "A1 Regional GDP"}
# whole regions that sat in another state in 1900 and 1910; regions split by a border of the time
# (Lorraine, Southern Denmark, Friuli-Venezia Giulia, Liège) keep their present country
BORDERS_1900 = {"FR42": "Germany", "IE0": "United Kingdom", "ITH1+ITH2": "Austria"}
SPLIT_1900 = {"FR41": "Moselle German", "DK03": "North Schleswig German",
              "ITH4": "Gorizia and Trieste Austrian", "BE33": "Eupen-Malmedy German"}
OUT: list[str] = []


def say(line: str = "") -> None:
    OUT.append(line)


# --------------------------------------------------------------------------- data

def read_sheet(xls: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    """One sheet of the workbook as a long table (code, country, region, year, value).

    Every sheet has six header rows; missing values are text such as 'n.a.'."""
    d = xls.parse(sheet, header=5)
    d = d.dropna(subset=[d.columns[1]])
    d = d.rename(columns={d.columns[0]: "country", d.columns[1]: "code", d.columns[2]: "region"})
    d["code"] = d.code.astype(str).str.strip()
    d["country"] = d.country.astype(str).str.strip().str.title()
    d["region"] = d.region.astype(str).str.strip()
    years = [c for c in d.columns if isinstance(c, (int, float)) and not isinstance(c, bool)]
    long = d.melt(id_vars=["code", "country", "region"], value_vars=years, var_name="year", value_name="v")
    long["year"] = long.year.astype(int)
    long["v"] = pd.to_numeric(long.v, errors="coerce")
    return long


def load_panel() -> pd.DataFrame:
    xls = pd.ExcelFile(WORKBOOK)
    parts = []
    for key, sheet in SHEETS.items():
        s = read_sheet(xls, sheet).rename(columns={"v": key})
        parts.append(s if not parts else s[["code", "year", key]])
    df = parts[0]
    for p in parts[1:]:
        df = df.merge(p, on=["code", "year"], how="outer")
    df["country"] = df.groupby("code").country.transform("first")
    df["region"] = df.groupby("code").region.transform("first")
    df["lgdppc"] = np.log(df.gdp / df["pop"])  # GDP in millions, population in thousands
    # the same regional GDP in 1990 international dollars (to 2015): within a country and year it
    # differs from the 2011 series by one factor, so only the national levels change
    df["lgdppc90"] = np.log(df.gdp90 / df["pop"])
    return df.sort_values(["code", "year"]).reset_index(drop=True)


def load_map(codes: pd.Series):
    """Regions as polygons in the ETRS89 Lambert equal-area projection (metres)."""
    import geopandas as gpd

    with zipfile.ZipFile(SHAPES) as z:
        shp = [n for n in z.namelist() if n.endswith(".shp")][0]
    g = gpd.read_file(f"zip://{SHAPES}!{shp}")
    want = pd.DataFrame({"code": codes, "shp": codes.replace(MERGED)})
    g = g.merge(want, left_on="NUTS_CODE", right_on="shp")
    if len(g) != len(want):
        missing = sorted(set(want.shp) - set(g.NUTS_CODE))
        raise ValueError(f"regions without a polygon: {missing}")
    return g.to_crs(3035).set_index("code").sort_index()


def land_pairs(g, tol: float = 500.0) -> list[tuple[str, str]]:
    """Pairs of regions whose boundaries touch, allowing a 500 m tolerance for slivers."""
    import geopandas as gpd

    b = gpd.GeoDataFrame({"code": list(g.index)}, geometry=list(g.geometry.buffer(tol)), crs=g.crs)
    j = gpd.sjoin(b, b, predicate="intersects")
    pairs = {tuple(sorted((a, c))) for a, c in zip(j.code_left, j.code_right) if a != c}
    return sorted(pairs)


def connect(codes: list[str], pairs: list[tuple[str, str]], xy: pd.DataFrame) -> list[tuple[str, str]]:
    """Add the shortest centroid-to-centroid links needed to make the graph connected
    (islands, Britain and Ireland, Scandinavia). Used only to draw random partitions."""
    links = list(pairs)
    while True:
        comp = components(codes, links)
        if len(set(comp.values())) == 1:
            return links
        labels = pd.Series(comp)
        main = labels.value_counts().index[0]
        inside = labels.index[labels == main]
        outside = labels.index[labels != main]
        a = xy.loc[inside].to_numpy()
        b = xy.loc[outside].to_numpy()
        d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        i, k = np.unravel_index(np.argmin(d), d.shape)
        links.append(tuple(sorted((inside[i], outside[k]))))


def components(codes, pairs) -> dict[str, int]:
    nb = {c: [] for c in codes}
    for a, b in pairs:
        nb[a].append(b)
        nb[b].append(a)
    comp, k = {}, 0
    for c in codes:
        if c in comp:
            continue
        stack = [c]
        comp[c] = k
        while stack:
            u = stack.pop()
            for v in nb[u]:
                if v not in comp:
                    comp[v] = k
                    stack.append(v)
        k += 1
    return comp


# --------------------------------------------------------------------------- statistics

def between_share(y: np.ndarray, groups: np.ndarray, w: np.ndarray) -> float:
    """Weighted share of the variance of y that lies between groups (a weighted R2)."""
    m = np.average(y, weights=w)
    total = np.average((y - m) ** 2, weights=w)
    gid, inv = np.unique(groups, return_inverse=True)
    gm = np.bincount(inv, weights=w * y) / np.bincount(inv, weights=w)
    within = np.average((y - gm[inv]) ** 2, weights=w)
    return float(1 - within / total)


def random_partition(nodes: list[str], nb: dict[str, list[str]], sizes: list[int],
                     rng: np.random.Generator) -> dict[str, int]:
    """Split a connected graph into contiguous groups with about the given sizes.

    Groups are grown one at a time, in random order, each from a random unassigned region that
    borders the territory already assigned (so what is left stays compact), by adding random
    neighbours until the group reaches its size or is boxed in. Regions left over at the end
    join a neighbouring group. Every group is contiguous; sizes can drift from the targets,
    and the drift is reported."""
    label: dict[str, int] = {}
    for k in rng.permutation(len(sizes)):
        free = [c for c in nodes if c not in label]
        if not free:
            break
        if label:
            edge = [c for c in free if any(v in label for v in nb[c])]
            free = edge or free
        seed = free[rng.integers(len(free))]
        label[seed] = int(k)
        size = 1
        frontier = {v for v in nb[seed] if v not in label}
        while size < sizes[k] and frontier:
            cand = sorted(frontier)
            v = cand[rng.integers(len(cand))]
            label[v] = int(k)
            size += 1
            frontier.discard(v)
            frontier.update(u for u in nb[v] if u not in label)
    while len(label) < len(nodes):
        for c in nodes:
            if c in label:
                continue
            nbrs = sorted(v for v in nb[c] if v in label)
            if nbrs:
                label[c] = label[nbrs[rng.integers(len(nbrs))]]
    return label


def draw_partitions(nodes, nb, sizes, n, rng, best_of: int = 50) -> list[dict[str, int]]:
    """n random contiguous partitions; each is the draw with the smallest size drift out of
    best_of independent draws."""
    target = np.sort(sizes)

    def drift(p):
        return np.abs(np.sort(np.bincount(list(p.values()), minlength=len(sizes))) - target).mean()

    return [min((random_partition(nodes, nb, sizes, rng) for _ in range(best_of)), key=drift)
            for _ in range(n)]


def size_drift(parts: list[dict[str, int]], sizes: list[int]) -> float:
    """Mean absolute difference between drawn and target group sizes, sorted by size."""
    target = np.sort(sizes)
    d = [np.abs(np.sort(np.bincount(list(p.values()), minlength=len(sizes))) - target).mean() for p in parts]
    return float(np.mean(d))


def dyadic_ols(y: np.ndarray, X: np.ndarray, a: np.ndarray, b: np.ndarray):
    """OLS with Fafchamps-Gubert dyadic-robust standard errors: residuals of two pairs that
    share a region are allowed to be correlated."""
    XtX = np.linalg.inv(X.T @ X)
    beta = XtX @ X.T @ y
    e = y - X @ beta
    s = X * e[:, None]
    meat = np.zeros((X.shape[1], X.shape[1]))
    nodes = np.unique(np.concatenate([a, b]))
    for node in nodes:  # pairs sharing this node; the pair itself is counted once below
        m = (a == node) | (b == node)
        t = s[m].sum(0)
        meat += np.outer(t, t) - (s[m].T @ s[m])
    meat += s.T @ s
    V = XtX @ meat @ XtX
    return beta, np.sqrt(np.diag(V))


def border_premium(diff: np.ndarray, cross: np.ndarray, logdist: np.ndarray, a, b, se=True):
    X = np.column_stack([np.ones_like(diff), cross.astype(float), logdist - logdist.mean()])
    if not se:
        return float(np.linalg.lstsq(X, diff, rcond=None)[0][1])
    beta, sd = dyadic_ols(diff, X, a, b)
    return float(beta[1]), float(sd[1]), float(beta[0])


# --------------------------------------------------------------------------- sections

def section_data(df, g, pairs, sea):
    say("== Data")
    reg = df.drop_duplicates("code")
    say(f"regions {reg.code.nunique()}, countries {reg.country.nunique()}: " +
        ", ".join(f"{c} {n}" for c, n in reg.country.value_counts().sort_index().items()))
    cov = df.groupby("year").ind.count()
    say("regions with an industry share, by year: " + " ".join(f"{y} {n}" for y, n in cov.items()))
    cross = sum(g.country[a] != g.country[b] for a, b in pairs)
    iso = sorted(set(g.index) - {c for p in pairs for c in p})
    say(f"pairs of regions sharing a land boundary: {len(pairs)}; across a national border: {cross}; "
        f"regions with no land neighbour: {' '.join(iso)}")
    say("links added to connect the map for random partitions: " +
        ", ".join(f"{a}-{b}" for a, b in sea))
    m = df[df.year.isin([1900, 1960, 2015])].dropna(subset=["ind", "pop"]).groupby("year").apply(
        lambda d: np.average(d.ind, weights=d["pop"]), include_groups=False)
    say("population-weighted share of industry in employment: " +
        " ".join(f"{y} {v:.3f}" for y, v in m.items()))
    say()


def section_variance(df, parts, codes, sizes):
    say(f"== 1. Share of regional variation that lies between countries "
        f"(population-weighted; null: {N_PARTITIONS} random contiguous partitions of the same sizes, "
        f"and {N_PARTITIONS} random relabellings ignoring geography)")
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for var in ["ind", "lgdppc", "lgdppc90"]:
        for year in sorted(df.year.unique()):
            d = df[(df.year == year) & df[var].notna() & df["pop"].notna()].set_index("code")
            if len(d) < 100:
                continue
            y, w = d[var].to_numpy(), d["pop"].to_numpy()
            obs = between_share(y, d.country.to_numpy(), w)
            contiguous = np.array([between_share(y, np.array([p[c] for c in d.index]), w) for p in parts])
            lab = d.country.to_numpy()
            shuffled = np.array([between_share(y, rng.permutation(lab), w) for _ in range(N_PARTITIONS)])
            rows.append(dict(var=var, year=year, n=len(d), country=obs,
                             contiguous_mean=contiguous.mean(), contiguous_p05=np.percentile(contiguous, 5),
                             contiguous_p95=np.percentile(contiguous, 95),
                             p_contiguous=(contiguous >= obs).mean(),
                             shuffled_mean=shuffled.mean(), shuffled_p95=np.percentile(shuffled, 95)))
    t = pd.DataFrame(rows)
    t.to_csv(RES / "variance_between_countries.csv", index=False, float_format="%.4f")
    for var, label in [("ind", "industry share"), ("lgdppc", "log GDP per head"),
                       ("lgdppc90", "log GDP per head in 1990 international dollars instead of 2011 PPP")]:
        say(f"{label}:")
        for r in t[t["var"] == var].itertuples():
            say(f"  {r.year}  n={r.n}  countries {r.country:.3f} | random contiguous mean {r.contiguous_mean:.3f} "
                f"[5-95%: {r.contiguous_p05:.3f}, {r.contiguous_p95:.3f}], share >= countries {r.p_contiguous:.3f} | "
                f"relabelled mean {r.shuffled_mean:.3f}")
    say()
    return t


def section_border(df, g, pairs, parts):
    say("== 2. Border premium: |difference| between neighbours across a border minus within a country, "
        "holding centroid distance fixed (dyadic-robust SE; null from random contiguous partitions)")
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    xy = np.column_stack([g.geometry.centroid.x, g.geometry.centroid.y])
    pos = {c: i for i, c in enumerate(g.index)}
    dist = np.array([np.hypot(*(xy[pos[i]] - xy[pos[j]])) / 1000 for i, j in pairs])
    logd = np.log(dist)
    cross = np.array([g.country[i] != g.country[j] for i, j in pairs])
    say(f"median centroid distance: within-country pairs {np.median(dist[~cross]):.0f} km, "
        f"cross-border pairs {np.median(dist[cross]):.0f} km")
    rows = []
    for var, scale in [("ind", 100.0), ("lgdppc", 100.0), ("lgdppc90", 100.0)]:
        for year in sorted(df.year.unique()):
            v = df[df.year == year].set_index("code")[var]
            ok = np.array([np.isfinite(v.get(i, np.nan)) and np.isfinite(v.get(j, np.nan)) for i, j in pairs])
            if ok.sum() < 250:
                continue
            diff = scale * np.abs(v.reindex(a[ok]).to_numpy() - v.reindex(b[ok]).to_numpy())
            prem, se, base = border_premium(diff, cross[ok], logd[ok], a[ok], b[ok])
            null = np.array([border_premium(diff, np.array([p[i] != p[j] for i, j in zip(a[ok], b[ok])]),
                                            logd[ok], a[ok], b[ok], se=False) for p in parts])
            rows.append(dict(var=var, year=year, pairs=int(ok.sum()), within_mean=diff[~cross[ok]].mean(),
                             cross_mean=diff[cross[ok]].mean(), premium=prem, se=se,
                             null_mean=null.mean(), null_p05=np.percentile(null, 5),
                             null_p95=np.percentile(null, 95), p_null=(null >= prem).mean()))
    t = pd.DataFrame(rows)
    t.to_csv(RES / "border_premium.csv", index=False, float_format="%.4f")
    for var, label in [("ind", "industry share, percentage points"), ("lgdppc", "log GDP per head x 100"),
                       ("lgdppc90", "log GDP per head x 100, 1990 international dollars instead of 2011 PPP")]:
        say(f"{label}:")
        for r in t[t["var"] == var].itertuples():
            say(f"  {r.year}  pairs={r.pairs}  within {r.within_mean:.2f}  across {r.cross_mean:.2f}  "
                f"premium {r.premium:+.2f} (SE {r.se:.2f}) | random lines mean {r.null_mean:+.2f} "
                f"[5-95%: {r.null_p05:+.2f}, {r.null_p95:+.2f}], share >= observed {r.p_null:.3f}")
    say()
    return t


def section_core(df, g, pairs):
    say(f"== 3. The industrial core of 1900: the {CORE_N} regions with the highest industry share")
    s1900 = df[df.year == 1900].set_index("code").ind.dropna()
    core = s1900.sort_values(ascending=False).index[:CORE_N]
    cut = s1900.sort_values(ascending=False).iloc[CORE_N - 1]
    say(f"threshold: industry share >= {cut:.3f} in 1900; median of all regions {s1900.median():.3f}")
    sub = [p for p in pairs if p[0] in core and p[1] in core]
    comp = components(list(core), sub)
    lab = pd.Series(comp)
    info = []
    for k, members in lab.groupby(lab):
        ms = sorted(members.index)
        cs = sorted(set(g.country[ms]))
        info.append((len(ms), cs, ms))
    info.sort(key=lambda r: -r[0])
    multi = [r for r in info if len(r[1]) > 1]
    pop1900 = df[df.year == 1900].set_index("code")["pop"]
    say(f"land-connected clusters among them: {len(info)}; spanning more than one country: {len(multi)}, "
        f"holding {sum(r[0] for r in multi)} of the {CORE_N} regions and "
        f"{pop1900.reindex([c for r in multi for c in r[2]]).sum() / pop1900.reindex(core).sum():.3f} "
        f"of their 1900 population")
    names = df.drop_duplicates("code").set_index("code").region
    for size, cs, ms in info:
        say(f"  {size:2d} regions, {', '.join(cs)}: " + "; ".join(f"{names[m]} ({m})" for m in ms))
    pd.DataFrame([dict(code=m, cluster=i, countries="+".join(cs)) for i, (_, cs, ms) in enumerate(info)
                  for m in ms]).to_csv(RES / "core_1900.csv", index=False)
    ind = df.pivot(index="code", columns="year", values="ind")
    say("industry share in 1900, 1960 and 2015 (%), regions of the clusters that cross a border:")
    for size, cs, ms in multi:
        say("  " + "; ".join(f"{names[m]} {100 * ind.loc[m, 1900]:.1f}, {100 * ind.loc[m, 1960]:.1f}, "
                             f"{100 * ind.loc[m, 2015]:.1f}" for m in ms))
    for y in [1900, 2015]:
        top = ind[y].dropna().sort_values(ascending=False).head(5)
        say(f"highest industry shares in {y} (%): " + "; ".join(f"{names[c]} {100 * v:.1f}" for c, v in top.items()))

    w = df.pivot(index="code", columns="year", values="ind")
    w = w.loc[:, w.notna().sum() >= 100]
    pw = df.pivot(index="code", columns="year", values="pop")
    is_core = w.index.isin(core)
    rows = []
    for y in w.columns:
        ok = w[y].notna() & pw[y].notna()
        c = ok & is_core
        r = ok & ~is_core
        rows.append(dict(year=y, core=np.average(w.loc[c, y], weights=pw.loc[c, y]),
                         rest=np.average(w.loc[r, y], weights=pw.loc[r, y]),
                         spearman=w[[1900, y]].dropna().corr(method="spearman").iloc[0, 1],
                         n=int(ok.sum())))
    t = pd.DataFrame(rows)
    t.to_csv(RES / "core_vs_rest.csv", index=False, float_format="%.4f")
    say("population-weighted industry share, core vs the other regions, and the rank correlation "
        "of each year with 1900:")
    for r in t.itertuples():
        say(f"  {r.year}  core {r.core:.3f}  rest {r.rest:.3f}  gap {r.core - r.rest:+.3f}  "
            f"Spearman with 1900 {r.spearman:+.2f}  (n={r.n})")
    peak = w.loc[:, [y for y in w.columns if y <= 2015]].idxmax(axis=1)
    say(f"year of the highest industry share, core regions: " +
        " ".join(f"{y} {n}" for y, n in peak[is_core].value_counts().sort_index().items()))
    say(f"year of the highest industry share, other regions: " +
        " ".join(f"{y} {n}" for y, n in peak[~is_core].value_counts().sort_index().items()))
    # standing inside their own country: log GDP per head minus the national population-weighted mean
    rel = []
    for y in [1900, 1960, 2015]:
        d = df[df.year == y].dropna(subset=["lgdppc", "pop"]).copy()
        nat = d.groupby("country").apply(lambda x: np.average(x.lgdppc, weights=x["pop"]), include_groups=False)
        d["rel"] = d.lgdppc - d.country.map(nat)
        d = d[d.code.isin(core) & (d.country.map(d.groupby("country").size()) > 1)]
        rel.append(f"{y} {100 * d.rel.mean():+.1f}")
    say("core regions' GDP per head relative to their own country (log points x 100, mean): " + "; ".join(rel))
    say()


def section_steps(df, g, pairs):
    say("== 4. Steps at national borders: income levels by country and the land borders one by one")
    names = df.drop_duplicates("code").set_index("code").region
    for y, col, base in [(1900, "gdp", "2011 PPP"), (1900, "gdp90", "1990 international dollars"),
                         (2015, "gdp", "2011 PPP"), (2015, "gdp90", "1990 international dollars"),
                         (2021, "gdp", "2011 PPP")]:
        d = df[df.year == y].dropna(subset=[col, "pop"])
        europe = d[col].sum() / d["pop"].sum()
        nat = d.groupby("country").apply(lambda x: x[col].sum() / x["pop"].sum(), include_groups=False) / europe
        say(f"{y} GDP per head by country, {base}, % of the 16-country level: " +
            ", ".join(f"{c} {100 * v:.0f}" for c, v in nat.sort_values(ascending=False).items()))
    for y, col, base in [(1900, "lgdppc", "2011 PPP"), (1900, "lgdppc90", "1990 international dollars"),
                         (2021, "lgdppc", "2011 PPP")]:
        v = df[df.year == y].set_index("code")[col]
        rows = []
        for a, b in pairs:
            if g.country[a] == g.country[b] or not (np.isfinite(v.get(a, np.nan)) and np.isfinite(v.get(b, np.nan))):
                continue
            hi, lo = (a, b) if v[a] >= v[b] else (b, a)
            rows.append((np.exp(v[hi] - v[lo]), hi, lo))
        rows.sort(reverse=True)
        say(f"{y} largest income steps across a border, {base} (richer side / poorer side, GDP per head): " +
            "; ".join(f"{names[h]} ({g.country[h]}) / {names[l]} ({g.country[l]}) {r:.2f}" for r, h, l in rows[:6]))
    say("land borders one by one: pairs of neighbouring regions; mean gap in log GDP per head x 100, 2011 PPP "
        "(first country minus second; in brackets the same in 1990 international dollars), and mean absolute gap "
        "in the industry share (percentage points)")
    cp = {}
    for a, b in pairs:
        ca, cb = g.country[a], g.country[b]
        if ca != cb:
            cp.setdefault(tuple(sorted((ca, cb))), []).append((a, b) if ca < cb else (b, a))
    for (ca, cb), ps in sorted(cp.items()):
        parts = []
        for y in [1900, 2015]:
            v = df[df.year == y].set_index("code")
            inc = [100 * (v.lgdppc.get(a, np.nan) - v.lgdppc.get(b, np.nan)) for a, b in ps]
            inc90 = [100 * (v.lgdppc90.get(a, np.nan) - v.lgdppc90.get(b, np.nan)) for a, b in ps]
            ind = [100 * abs(v.ind.get(a, np.nan) - v.ind.get(b, np.nan)) for a, b in ps]
            parts.append(f"{y} income {np.nanmean(inc):+.0f} (1990 $ {np.nanmean(inc90):+.0f}), industry " +
                         (f"{np.nanmean(ind):.1f}" if np.isfinite(ind).any() else "n/a"))
        say(f"  {ca}-{cb} ({len(ps)} pairs): " + " | ".join(parts))
    say()


def section_borders_1900(df, g, pairs, parts):
    say("== 5. The same border test with the borders of 1900 for 1900 and 1910 (Alsace in Germany, "
        "Ireland in the United Kingdom, Trentino-South Tyrol in Austria)")
    say("regions split by a border of the time keep their present country: " +
        ", ".join(f"{c} ({why})" for c, why in SPLIT_1900.items()) +
        "; Norway and Sweden (in union until 1905) and Finland (a grand duchy of the Russian Empire with "
        "its own customs and currency) stay separate")
    c1900 = g.country.copy()
    for code, country in BORDERS_1900.items():
        c1900[code] = country
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    xy = np.column_stack([g.geometry.centroid.x, g.geometry.centroid.y])
    pos = {c: i for i, c in enumerate(g.index)}
    logd = np.log(np.array([np.hypot(*(xy[pos[i]] - xy[pos[j]])) / 1000 for i, j in pairs]))
    cross = np.array([c1900[i] != c1900[j] for i, j in pairs])
    now = np.array([g.country[i] != g.country[j] for i, j in pairs])
    say(f"pairs across a border: {cross.sum()} with the borders of 1900, {now.sum()} with today's")
    for var, label in [("ind", "industry share, percentage points"), ("lgdppc", "log GDP per head x 100"),
                       ("lgdppc90", "log GDP per head x 100, 1990 international dollars")]:
        out = []
        for year in [1900, 1910]:
            v = df[df.year == year].set_index("code")[var]
            ok = np.array([np.isfinite(v.get(i, np.nan)) and np.isfinite(v.get(j, np.nan)) for i, j in pairs])
            diff = 100 * np.abs(v.reindex(a[ok]).to_numpy() - v.reindex(b[ok]).to_numpy())
            prem, se, _ = border_premium(diff, cross[ok], logd[ok], a[ok], b[ok])
            null = np.array([border_premium(diff, np.array([p[i] != p[j] for i, j in zip(a[ok], b[ok])]),
                                            logd[ok], a[ok], b[ok], se=False) for p in parts])
            out.append(f"{year} premium {prem:+.2f} (SE {se:.2f}), random lines share >= observed "
                       f"{(null >= prem).mean():.3f}")
        say(f"  {label}: " + " | ".join(out))
    say()


def benchmark_years(df: pd.DataFrame) -> list[int]:
    """Years in which at least 100 regions report an industry share: the benchmark years
    1900-2015 and 2021."""
    n = df.groupby("year").ind.count()
    return sorted(n.index[n >= 100])


def main() -> int:
    global N_PARTITIONS
    if len(sys.argv) > 1:  # python regions.py 50 for a quick look
        N_PARTITIONS = int(sys.argv[1])
    if not WORKBOOK.exists() or not SHAPES.exists():
        print("data/raw is missing the Rosés-Wolf files: see check_data.py", file=sys.stderr)
        return 1
    RES.mkdir(exist_ok=True)
    df = load_panel()
    df = df[df.year.isin(benchmark_years(df))].reset_index(drop=True)
    codes = pd.Series(sorted(df.code.unique()))
    g = load_map(codes)
    g["country"] = df.drop_duplicates("code").set_index("code").country.reindex(g.index)
    pairs = land_pairs(g)
    xy = pd.DataFrame({"x": g.geometry.centroid.x, "y": g.geometry.centroid.y}, index=g.index)
    links = connect(list(g.index), pairs, xy)
    sea = [p for p in links if p not in set(pairs)]
    section_data(df, g, pairs, sea)
    nb = {c: [] for c in g.index}
    for p, q in links:
        nb[p].append(q)
        nb[q].append(p)
    for c in nb:
        nb[c].sort()
    sizes = sorted(g.country.value_counts().tolist())
    rng = np.random.default_rng(SEED)
    parts = draw_partitions(list(g.index), nb, sizes, N_PARTITIONS, rng)
    say(f"random contiguous partitions: {N_PARTITIONS}, each the best of 50 draws; mean absolute drift "
        f"from the country sizes {size_drift(parts, sizes):.2f} regions per group (country sizes average "
        f"{np.mean(sizes):.1f}); number of groups always {len(sizes)}")
    say()
    section_variance(df, parts, list(g.index), sizes)
    section_border(df, g, pairs, parts)
    section_core(df, g, pairs)
    section_steps(df, g, pairs)
    section_borders_1900(df, g, pairs, parts)
    (RES / "summary.txt").write_text("\n".join(OUT) + "\n")
    print("\n".join(OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
