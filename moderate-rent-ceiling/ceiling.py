"""Where the EUR 2,300 "moderate rent" ceiling sat against the rents of new leases, 2017-2025.

    python ceiling.py            # reads data/, writes results/

Every number quoted in the article is printed to ``results/summary.txt``; the tables behind it are
in ``results/*.csv``. Nothing after December 2025 is used: the ceiling is the 2026 value (EUR 2,300,
2.5 times the 2026 minimum wage), compared with rents and incomes observed before it existed, and
the same rule (2.5 times each year's minimum wage) is also applied year by year.

Four measurements, all by municipality and year:

1. The floor area a new lease would need to reach the ceiling at the local 1st quartile, median
   and 3rd quartile of rent per m2 (INE, new leases), set against the floor area of rented homes
   (Census 2021).
2. A model of the share of new leases whose monthly rent exceeded the ceiling: rent per m2
   log-normal with the published quartiles, floor area distributed as the municipality's rented
   homes in the Census, and rent per m2 falling with floor area at an elasticity estimated from
   INE's rents by dwelling type. Checked against the Census and the UTAO's 0.9% for 2024.
3. The ceiling against local incomes: twelve months at EUR 2,300 as a share of the median gross
   declared income of a fiscal household, and the share of fiscal households whose income reaches
   the government's own reference (about EUR 5,750 a month).
4. The ceiling rule and rents over time: how many square metres the ceiling buys at the median
   rent per m2, 2017-2025, splicing INE's 2021 methodology (2017-2019) onto the 2026 one.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"

CEILING = 2300.0                     # EUR a month, 2026: 2.5 x the minimum wage of EUR 920
# Minimum monthly wage (RMMG, mainland Portugal), EUR
RMMG = {2017: 557, 2018: 580, 2019: 600, 2020: 635, 2021: 665, 2022: 705,
        2023: 760, 2024: 820, 2025: 870, 2026: 920}
RULE_MULTIPLE = 2.5
GOV_REFERENCE_MONTHLY_INCOME = 5750.0  # household income the government said the ceiling assumed
EFFORT = 0.35                          # benchmark housing-cost burden
YEARS_NEW = list(range(2020, 2026))    # Metodologia 2026, quartiles available
YEARS_OLD = [2017, 2018, 2019]         # Metodologia 2021, medians only
Z75 = norm.ppf(0.75)

# Census 2021 floor-area classes (m2) of rented main residences: code -> (low, high)
AREA_CLASSES = {"1": (16, 30), "2": (30, 40), "3": (40, 50), "4": (50, 60), "5": (60, 80),
                "6": (80, 100), "7": (100, 120), "8": (120, 150), "9": (150, 200), "10": (200, 300)}
# Census 2021 monthly-rent classes (EUR): code -> (low, high)
RENT_CLASSES = {"1": (5, 20), "2": (20, 50), "3": (50, 75), "4": (75, 100), "5": (100, 150),
                "6": (150, 200), "7": (200, 300), "8": (300, 400), "9": (400, 500),
                "10": (500, 650), "11": (650, 1000), "12": (1000, 2000)}
BIG = ["Lisboa", "Cascais", "Oeiras", "Porto", "Funchal", "Almada", "Sintra", "Braga", "Coimbra"]

LINES: list[str] = []


def say(text: str = "") -> None:
    LINES.append(text)
    print(text)


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{name}.csv", dtype={"geocod": str, "period": str, "dim_3": str,
                                                   "dim_4": str, "dim_5": str})
    return df


def municipalities(df: pd.DataFrame, width: int = 7) -> pd.DataFrame:
    """Rows for municipalities only, with the 4-digit municipality code as ``dtmn``."""
    out = df[df["geocod"].str.len() == width].copy()
    out["dtmn"] = out["geocod"].str[-4:]
    return out


def geo_mid(lo: float, hi: float) -> float:
    return math.sqrt(lo * hi)


# --- data ---------------------------------------------------------------------------------------

def quartiles_2026() -> pd.DataFrame:
    q = municipalities(load("rent_quartiles_2026"))
    q["q"] = q["dim_3"].map({"1": "p25", "2": "p50", "3": "p75"})
    wide = q.pivot_table(index=["dtmn", "geodsg", "period"], columns="q", values="value").reset_index()
    wide["year"] = wide["period"].astype(int)
    return wide.drop(columns="period")


def national_quartiles_2026() -> pd.DataFrame:
    q = load("rent_quartiles_2026")
    q = q[q["geocod"] == "PT"].copy()
    q["q"] = q["dim_3"].map({"1": "p25", "2": "p50", "3": "p75"})
    w = q.pivot_table(index="period", columns="q", values="value").reset_index()
    w["year"] = w["period"].astype(int)
    return w.drop(columns="period").set_index("year")


def leases_2026() -> pd.DataFrame:
    d = municipalities(load("new_leases_2026"))
    d["year"] = d["period"].astype(int)
    return d[["dtmn", "year", "value"]].rename(columns={"value": "leases"})


def medians_old() -> pd.DataFrame:
    m = municipalities(load("median_nuts2013"))
    n = municipalities(load("new_leases_nuts2013"))
    m["year"] = m["period"].astype(int)
    n["year"] = n["period"].astype(int)
    out = m[["dtmn", "geodsg", "year", "value"]].rename(columns={"value": "p50_old"})
    return out.merge(n[["dtmn", "year", "value"]].rename(columns={"value": "leases_old"}),
                     on=["dtmn", "year"], how="left")


def income() -> pd.DataFrame:
    d = municipalities(load("income_median_household"))
    d["year"] = d["period"].astype(int)
    return d[["dtmn", "year", "value"]].rename(columns={"value": "income_median"})


def wmedian(values: np.ndarray, weights: np.ndarray) -> float:
    o = np.argsort(values)
    v, w = values[o], weights[o]
    c = np.cumsum(w) / w.sum()
    return float(v[np.searchsorted(c, 0.5)])


def wquantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    o = np.argsort(values)
    v, w = values[o], weights[o]
    c = np.cumsum(w) / w.sum()
    return float(v[min(np.searchsorted(c, q), len(v) - 1)])


def census_area() -> pd.DataFrame:
    """Rented main residences by municipality and floor-area class (all rent classes), Census 2021."""
    c = load("census_area_rent")
    c = c[(c["dim_5"] == "T") & (c["dim_3"] != "T")]
    c = c[(c["geocod"].str.len() == 4) | (c["geocod"] == "PT")].copy()
    c["dtmn"] = np.where(c["geocod"] == "PT", "PT", c["geocod"])
    c["area"] = c["dim_3"].map(lambda k: geo_mid(*AREA_CLASSES[k]))
    return c[["dtmn", "geodsg", "dim_3", "area", "value"]].rename(columns={"value": "homes"})


def census_rent_area() -> pd.DataFrame:
    """Rented main residences by floor-area class and monthly-rent class, Census 2021."""
    c = load("census_area_rent")
    c = c[(c["dim_5"] != "T") & (c["dim_3"] != "T")]
    c = c[(c["geocod"].str.len() == 4) | (c["geocod"] == "PT")].copy()
    c["dtmn"] = np.where(c["geocod"] == "PT", "PT", c["geocod"])
    return c[["dtmn", "geodsg", "dim_3", "dim_5", "value"]].rename(columns={"value": "homes"})


TYPOLOGY_ROOMS = {"T0 ou T1": ["1", "2"], "T2": ["3"], "T3": ["4"],
                  "T4 ou mais": ["5", "6", "7", "8", "9", "10"]}


def typology_areas() -> dict[str, float]:
    """Median floor area of main residences by dwelling type, Census 2021, Portugal.

    The Census counts rooms without the kitchen, bathrooms and corridors, so a dwelling of type Tn
    (n bedrooms) has n + 1 rooms: T0-T1 = 1-2 rooms, T2 = 3, T3 = 4, T4 or more = 5 or more. The
    median is interpolated linearly inside the floor-area class that contains it.
    """
    c = load("census_rooms_area")
    c = c[(c["geocod"] == "PT") & c["dim_4"].isin(AREA_CLASSES)]
    out = {}
    for typ, rooms in TYPOLOGY_ROOMS.items():
        g = c[c["dim_3"].isin(rooms)].groupby("dim_4")["value"].sum()
        lows = np.array([AREA_CLASSES[k][0] for k in g.index], float)
        highs = np.array([AREA_CLASSES[k][1] for k in g.index], float)
        o = np.argsort(lows)
        lows, highs, w = lows[o], highs[o], g.to_numpy(float)[o]
        cum = np.cumsum(w) / w.sum()
        i = int(np.searchsorted(cum, 0.5))
        before = cum[i - 1] if i > 0 else 0.0
        out[typ] = float(lows[i] + (0.5 - before) / (cum[i] - before) * (highs[i] - lows[i]))
    return out


# --- 1. floor area needed to reach the ceiling ----------------------------------------------------

def area_needed(q: pd.DataFrame, leases: pd.DataFrame, homes: pd.DataFrame) -> pd.DataFrame:
    say("## 1. Floor area a new lease needed to reach the ceiling")
    d = q.merge(leases, on=["dtmn", "year"], how="left")
    d["rule"] = d["year"].map(lambda y: RULE_MULTIPLE * RMMG[y])
    for k in ("p25", "p50", "p75"):
        d[f"A_{k}"] = CEILING / d[k]
        d[f"Arule_{k}"] = d["rule"] / d[k]
    nat = national_quartiles_2026()
    for y in (2020, 2025):
        r = nat.loc[y]
        say(f"Portugal {y}: rent per m2 Q1 {r.p25:.2f}, median {r.p50:.2f}, Q3 {r.p75:.2f}; "
            f"floor area to reach EUR 2,300: {CEILING / r.p25:.0f} / {CEILING / r.p50:.0f} / {CEILING / r.p75:.0f} m2; "
            f"to reach the rule ceiling of that year (EUR {RULE_MULTIPLE * RMMG[y]:,.1f}): "
            f"{RULE_MULTIPLE * RMMG[y] / r.p50:.0f} m2 at the median, {RULE_MULTIPLE * RMMG[y] / r.p75:.0f} at Q3")
    for y in YEARS_NEW:
        g = d[(d["year"] == y) & d["A_p75"].notna()]
        gw = g[g["leases"].notna()]
        say(f"{y}: {len(g)} municipalities with quartiles; lease-weighted median of the Q3 floor area "
            f"{wmedian(gw.A_p75.to_numpy(), gw.leases.to_numpy()):.0f} m2 (10th percentile "
            f"{wquantile(gw.A_p75.to_numpy(), gw.leases.to_numpy(), 0.10):.0f}); Q3 area below 100 m2 in "
            f"{(g.A_p75 < 100).sum()}, below 120 m2 in {(g.A_p75 < 120).sum()}, below 150 m2 in {(g.A_p75 < 150).sum()}; "
            f"median area below 150 m2 in {(g.A_p50 < 150).sum()}")
    g = d[(d["year"] == 2025) & (d["A_p75"] < 150)].sort_values("A_p75")
    say("2025, municipalities where Q3 needs less than 150 m2: " +
        "; ".join(f"{r.geodsg} {r.A_p75:.0f} (median {r.A_p50:.0f})" for r in g.itertuples()))
    # rented homes of those sizes, Census 2021
    tot = homes.groupby("dtmn")["homes"].sum()
    big = homes[homes["dim_3"].isin(["7", "8", "9", "10"])].groupby("dtmn")["homes"].sum() / tot
    big120 = homes[homes["dim_3"].isin(["8", "9", "10"])].groupby("dtmn")["homes"].sum() / tot
    big150 = homes[homes["dim_3"].isin(["9", "10"])].groupby("dtmn")["homes"].sum() / tot
    say(f"Census 2021, rented homes of 100 m2 or more: Portugal {big['PT']:.1%}; 120 m2 or more {big120['PT']:.1%}; "
        f"150 m2 or more {big150['PT']:.1%}")
    names = homes.drop_duplicates("dtmn").set_index("dtmn")["geodsg"]
    for n in BIG:
        code = names[names == n].index
        if len(code):
            c = code[0]
            r = d[(d["dtmn"] == c) & (d["year"] == 2025)]
            if len(r):
                r = r.iloc[0]
                say(f"  {n} 2025: Q1/median/Q3 {r.p25:.2f}/{r.p50:.2f}/{r.p75:.2f} EUR/m2; area to EUR 2,300 "
                    f"{r.A_p25:.0f}/{r.A_p50:.0f}/{r.A_p75:.0f} m2; rented homes >=100 m2 {big[c]:.1%}, >=120 {big120[c]:.1%}, >=150 {big150[c]:.1%}")
    d.to_csv(RESULTS / "area_needed.csv", index=False)
    say()
    return d


# --- 2. modelled share of new leases above the ceiling ------------------------------------------

def elasticity(areas: dict[str, float]) -> tuple[float, float, pd.DataFrame]:
    """Elasticity of rent per m2 with respect to floor area, from INE medians by dwelling type.

    Within each municipality and quarter, log median rent per m2 of T0-T1, T2, T3 and T4+ leases is
    regressed on the log of the typical floor area of a dwelling with that many rooms (Census 2021:
    T0-T1 ~ 1-2 rooms, T2 ~ 3, T3 ~ 4, T4+ ~ 5 or more). Standard error clustered by municipality.
    """
    t = load("rent_by_type_quarterly")
    t = t[(t["geocod"].str.len() == 7) & (t["dim_3"] != "T") & t["value"].notna()].copy()
    t["year"] = t["period"].str[-4:].astype(int)
    t = t[t["year"] <= 2025]
    t["area"] = t["dim_3_t"].map(areas)
    t = t[t["area"].notna()]
    t["y"] = np.log(t["value"])
    t["x"] = np.log(t["area"])
    t["cell"] = t["geocod"] + "|" + t["period"]
    t["yd"] = t["y"] - t.groupby("cell")["y"].transform("mean")
    t["xd"] = t["x"] - t.groupby("cell")["x"].transform("mean")
    beta = float((t["xd"] * t["yd"]).sum() / (t["xd"] ** 2).sum())
    e = t["yd"] - beta * t["xd"]
    sxx = (t["xd"] ** 2).sum()
    meat = sum(((g["xd"] * e.loc[g.index]).sum()) ** 2 for _, g in t.groupby("geocod"))
    se = float(math.sqrt(meat) / sxx)
    return beta, se, t


def share_above(p50: float, p25: float, p75: float, areas: np.ndarray, w: np.ndarray, cap: float,
                beta: float, sigma_scale: float = 1.0, area_scale: float = 1.0) -> float:
    """P(monthly rent > cap) when log rent per m2 is normal given floor area (see module docstring)."""
    la = np.log(areas * area_scale)
    w = w / w.sum()
    m = float((w * la).sum())
    var_a = float((w * (la - m) ** 2).sum())
    sp = sigma_scale * (math.log(p75) - math.log(p25)) / (2 * Z75)
    s2 = max(sp ** 2 - beta ** 2 * var_a, 0.05 ** 2)
    mu = math.log(p50) + beta * (la - m)
    z = (math.log(cap) - la - mu) / math.sqrt(s2)
    return float((w * norm.sf(z)).sum())


def rent_cdf(grid: np.ndarray, p50: float, p25: float, p75: float, areas: np.ndarray, w: np.ndarray,
             beta: float) -> np.ndarray:
    """P(monthly rent <= r) for each r in ``grid``, same model as ``share_above``."""
    la = np.log(areas)
    w = w / w.sum()
    m = float((w * la).sum())
    var_a = float((w * (la - m) ** 2).sum())
    sp = (math.log(p75) - math.log(p25)) / (2 * Z75)
    s = math.sqrt(max(sp ** 2 - beta ** 2 * var_a, 0.05 ** 2))
    mu = math.log(p50) + beta * (la - m)
    z = (np.log(grid)[:, None] - la[None, :] - mu[None, :]) / s
    return (norm.cdf(z) * w[None, :]).sum(axis=1)


def national_distribution(d: pd.DataFrame, homes: pd.DataFrame, beta: float) -> pd.DataFrame:
    """Lease-weighted mixture of the municipal rent distributions, by year: quantiles and the
    percentile at which EUR 2,300 and the rule ceiling sit."""
    areas = {k: g.sort_values("dim_3") for k, g in homes.groupby("dtmn")}
    grid = np.exp(np.linspace(math.log(50), math.log(20000), 4000))
    rows = []
    curves = {"rent": grid}
    for y in YEARS_NEW:
        g = d[(d["year"] == y) & d["leases"].notna()]
        cdf = np.zeros_like(grid)
        for r in g.itertuples():
            cdf += r.leases * rent_cdf(grid, r.p50, r.p25, r.p75, areas[r.dtmn]["area"].to_numpy(float),
                                       areas[r.dtmn]["homes"].to_numpy(float), beta)
        cdf /= g["leases"].sum()
        curves[str(y)] = cdf
        row = {"year": y}
        for qq in (0.25, 0.5, 0.75, 0.9, 0.95, 0.99):
            row[f"q{int(qq * 100)}"] = float(np.interp(qq, cdf, grid))
        row["pct_2300"] = float(np.interp(CEILING, grid, cdf))
        row["pct_rule"] = float(np.interp(RULE_MULTIPLE * RMMG[y], grid, cdf))
        rows.append(row)
        say(f"{y}: modelled monthly rent of new leases: P25 EUR {row['q25']:,.0f}, median {row['q50']:,.0f}, "
            f"P75 {row['q75']:,.0f}, P90 {row['q90']:,.0f}, P95 {row['q95']:,.0f}, P99 {row['q99']:,.0f}; "
            f"EUR 2,300 at percentile {100 * row['pct_2300']:.2f}, the rule ceiling at {100 * row['pct_rule']:.2f}; "
            f"EUR 2,300 = {CEILING / row['q50']:.1f} x the median")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "national_distribution.csv", index=False)
    pd.DataFrame(curves).iloc[::8].to_csv(RESULTS / "national_cdf.csv", index=False, float_format="%.6g")
    return out


def model_shares(q: pd.DataFrame, leases: pd.DataFrame, homes: pd.DataFrame, beta: float, beta_se: float) -> pd.DataFrame:
    say("## 2. Modelled share of new leases with a monthly rent above the ceiling")
    say(f"Elasticity of rent per m2 with respect to floor area, INE medians by dwelling type, "
        f"municipalities over 100,000 inhabitants, 2020-2025: {beta:.3f} (s.e. {beta_se:.3f})")
    areas = {k: g.sort_values("dim_3") for k, g in homes.groupby("dtmn")}
    d = q.merge(leases, on=["dtmn", "year"], how="left")
    d = d[d[["p25", "p50", "p75"]].notna().all(axis=1) & d["dtmn"].isin(areas)].copy()
    d["rule"] = d["year"].map(lambda y: RULE_MULTIPLE * RMMG[y])
    variants = {
        "main": dict(beta=beta),
        "beta0": dict(beta=0.0),
        "beta_lo": dict(beta=beta - 1.96 * beta_se),
        "beta_hi": dict(beta=min(beta + 1.96 * beta_se, 0.0)),
        "area90": dict(beta=beta, area_scale=0.9),
        "area110": dict(beta=beta, area_scale=1.1),
        "sigma120": dict(beta=beta, sigma_scale=1.2),
    }
    for name, kw in variants.items():
        d[f"share_{name}"] = [share_above(r.p50, r.p25, r.p75, areas[r.dtmn]["area"].to_numpy(float),
                                          areas[r.dtmn]["homes"].to_numpy(float), CEILING, **kw)
                              for r in d.itertuples()]
    d["share_rule"] = [share_above(r.p50, r.p25, r.p75, areas[r.dtmn]["area"].to_numpy(float),
                                   areas[r.dtmn]["homes"].to_numpy(float), r.rule, beta=beta)
                       for r in d.itertuples()]
    for c in (1000.0, 1500.0):
        d[f"share_{int(c)}"] = [share_above(r.p50, r.p25, r.p75, areas[r.dtmn]["area"].to_numpy(float),
                                            areas[r.dtmn]["homes"].to_numpy(float), c, beta=beta)
                                for r in d.itertuples()]
    nat_leases = leases_national()
    for y in YEARS_NEW:
        g = d[(d["year"] == y) & d["leases"].notna()]
        cover = g["leases"].sum() / nat_leases[y]
        parts = []
        for name in variants:
            parts.append(f"{name} {np.average(g[f'share_{name}'], weights=g['leases']):.2%}")
        say(f"{y}: municipalities {len(g)}, covering {cover:.1%} of new leases; share above EUR 2,300: " + ", ".join(parts))
        say(f"      above the rule ceiling of {y} (EUR {RULE_MULTIPLE * RMMG[y]:,.1f}): "
            f"{np.average(g['share_rule'], weights=g['leases']):.2%}; above EUR 1,000: "
            f"{np.average(g['share_1000'], weights=g['leases']):.1%}; above EUR 1,500: "
            f"{np.average(g['share_1500'], weights=g['leases']):.1%}; leases above EUR 2,300 (main): "
            f"{(g['share_main'] * g['leases']).sum():,.0f}")
    g = d[(d["year"] == 2025) & d["leases"].notna()].copy()
    g["above"] = g["share_main"] * g["leases"]
    tot = g["above"].sum()
    top = g.sort_values("above", ascending=False).head(8)
    say("2025, municipalities with most modelled leases above EUR 2,300: " +
        "; ".join(f"{r.geodsg} {r.share_main:.1%} ({r.above:,.0f})" for r in top.itertuples()))
    say(f"2025, share of those leases in Lisboa, Cascais and Oeiras: "
        f"{g[g.geodsg.isin(['Lisboa', 'Cascais', 'Oeiras'])]['above'].sum() / tot:.1%}; "
        f"municipalities with a modelled share below 1%: {(g.share_main < 0.01).sum()} of {len(g)}, "
        f"below 0.1%: {(g.share_main < 0.001).sum()}")
    d.to_csv(RESULTS / "share_above.csv", index=False)
    say("National distribution of monthly rents of new leases (model, main elasticity):")
    national_distribution(d, homes, beta)
    say()
    return d


def leases_national() -> dict[int, float]:
    d = load("new_leases_2026")
    d = d[d["geocod"] == "PT"]
    return {int(p): float(v) for p, v in zip(d["period"], d["value"])}


def tail_check(rent_area: pd.DataFrame) -> None:
    """Does a log-normal fitted to the middle of the Census rent distribution get its tail right?"""
    say("## 2b. Log-normal tail check on the Census 2021 rent distribution (rented homes, Portugal)")
    g = rent_area[rent_area["dtmn"] == "PT"].groupby("dim_5")["homes"].sum()
    lows = np.array([RENT_CLASSES[k][0] for k in g.index], float)
    order = np.argsort(lows)
    codes = g.index.to_numpy()[order]
    w = g.to_numpy(float)[order]
    cum = np.cumsum(w) / w.sum()
    class_lows = lows[order]

    def quant(p):
        i = int(np.searchsorted(cum, p))
        lo = RENT_CLASSES[codes[i]][0]
        hi = RENT_CLASSES[codes[i]][1]
        before = cum[i - 1] if i > 0 else 0.0
        return lo * (hi / lo) ** ((p - before) / (cum[i] - before))   # log-linear within the class

    for lo_q, hi_q in ((0.25, 0.75), (0.50, 0.90)):
        a, b = quant(lo_q), quant(hi_q)
        sig = (math.log(b) - math.log(a)) / (norm.ppf(hi_q) - norm.ppf(lo_q))
        mu = math.log(a) - norm.ppf(lo_q) * sig
        for cut in (650.0, 1000.0):
            actual = w[class_lows >= cut].sum() / w.sum()
            fitted = norm.sf((math.log(cut) - mu) / sig)
            say(f"  log-normal through the {lo_q:.0%} and {hi_q:.0%} quantiles (EUR {a:,.0f}, {b:,.0f}): "
                f"share at or above EUR {cut:,.0f} fitted {fitted:.1%}, Census {actual:.1%}")
    say()


# --- 3. the ceiling against local incomes -----------------------------------------------------------

def incomes(q: pd.DataFrame) -> pd.DataFrame:
    say("## 3. The ceiling against the declared incomes of fiscal households")
    inc = income()
    qu = municipalities(load("income_quintiles_household"))
    qu["year"] = qu["period"].astype(int)
    qu["k"] = qu["dim_3"].astype(str).str[-1]
    p = qu.pivot_table(index=["dtmn", "year"], columns="k", values="value").reset_index()
    p = p.rename(columns={"1": "P20", "2": "P40", "3": "P60", "4": "P80"})
    d = inc.merge(p, on=["dtmn", "year"], how="left")
    names = q.drop_duplicates("dtmn").set_index("dtmn")["geodsg"]
    d["geodsg"] = d["dtmn"].map(names)
    nat = load("income_median_household")
    nat = nat[nat["geocod"] == "PT"].set_index("period")["value"]
    natq = load("income_quintiles_household")
    natq = natq[natq["geocod"] == "PT"]
    year_rent = 12 * CEILING
    gov = 12 * GOV_REFERENCE_MONTHLY_INCOME
    for y in (2018, 2024):
        y_inc = float(nat[str(y)])
        say(f"Portugal {y}: median gross declared income per fiscal household EUR {y_inc:,.0f} "
            f"(EUR {y_inc / 12:,.0f} a month); twelve months at EUR 2,300 = {year_rent / y_inc:.0%} of it; "
            f"at the rule ceiling of {y} (EUR {RULE_MULTIPLE * RMMG[y]:,.0f}): {12 * RULE_MULTIPLE * RMMG[y] / y_inc:.0%}; "
            f"rent at a {EFFORT:.0%} burden: EUR {EFFORT * y_inc / 12:,.0f} a month")
    qn = natq[natq["period"] == "2024"]
    qn = qn.set_index(qn["dim_3"].astype(str).str[-1])["value"]
    say(f"Portugal 2024 quintile boundaries (gross, fiscal households): P20 {qn['1']:,.0f}, P40 {qn['2']:,.0f}, "
        f"P60 {qn['3']:,.0f}, P80 {qn['4']:,.0f}; EUR 2,300 a month = {year_rent / qn['4']:.0%} of P80 income; "
        f"the government's reference household (EUR {GOV_REFERENCE_MONTHLY_INCOME:,.0f} a month, EUR {gov:,.0f} a year) "
        f"= {gov / qn['4']:.2f} x P80")
    g = d[d["year"] == 2024].dropna(subset=["income_median"])
    g = g.assign(r_med=year_rent / g["income_median"], r_p80=year_rent / g["P80"])
    say(f"2024, {len(g)} municipalities: twelve months at EUR 2,300 exceed the median household's gross income in "
        f"{(g.r_med > 1).sum()}; the smallest ratio is {g.r_med.min():.2f} ({g.loc[g.r_med.idxmin(), 'geodsg']}), "
        f"the largest {g.r_med.max():.2f} ({g.loc[g.r_med.idxmax(), 'geodsg']})")
    h = g.dropna(subset=["P80"])
    say(f"2024, {len(h)} municipalities with quintiles: EUR 27,600 a year exceeds the gross income at P80 in "
        f"{(h.r_p80 > 1).sum()}; the highest P80 is EUR {h.P80.max():,.0f} ({h.loc[h.P80.idxmax(), 'geodsg']}); "
        f"the reference household's EUR {gov:,.0f} exceeds P80 in {(gov > h.P80).sum()} of {len(h)}")
    for n in BIG:
        r = g[g["geodsg"] == n]
        if len(r):
            r = r.iloc[0]
            say(f"  {n} 2024: median EUR {r.income_median:,.0f}, P80 EUR {r.P80:,.0f}; rent at 35% of the median "
                f"EUR {EFFORT * r.income_median / 12:,.0f}, at 35% of P80 EUR {EFFORT * r.P80 / 12:,.0f}; ceiling = "
                f"{year_rent / r.income_median:.2f} x median income")
    d.to_csv(RESULTS / "income_ratios.csv", index=False)
    say()
    return d


# --- 4. the ceiling rule and rents over time, 2017-2025 -------------------------------------------

def over_time(q: pd.DataFrame, old: pd.DataFrame, leases: pd.DataFrame) -> pd.DataFrame:
    say("## 4. The ceiling and rents over time, 2017-2025")
    nat_new = national_quartiles_2026()["p50"]
    o = load("median_nuts2013")
    o = o[o["geocod"] == "PT"]
    nat_old = o.set_index(o["period"].astype(int))["value"]
    overlap = [y for y in nat_old.index if y in nat_new.index]
    ratios = {y: nat_new[y] / nat_old[y] for y in overlap}
    say("Portugal, median rent per m2, 2026 methodology over 2021 methodology: " +
        ", ".join(f"{y} {r:.3f}" for y, r in ratios.items()))
    splice = ratios[2020]
    series = {y: nat_old[y] * splice for y in YEARS_OLD}
    series.update({y: nat_new[y] for y in YEARS_NEW})
    for y in sorted(series):
        rule = RULE_MULTIPLE * RMMG[y]
        say(f"{y}: median EUR {series[y]:.2f}/m2{' (spliced)' if y in YEARS_OLD else ''}; rule ceiling EUR {rule:,.1f} "
            f"buys {rule / series[y]:.0f} m2, EUR 2,300 buys {CEILING / series[y]:.0f} m2; an 80 m2 home at the median "
            f"costs EUR {80 * series[y]:,.0f} = {80 * series[y] / rule:.0%} of the rule ceiling")
    pd.DataFrame([{"year": y, "p50": series[y], "rule": RULE_MULTIPLE * RMMG[y],
                   "m2_rule": RULE_MULTIPLE * RMMG[y] / series[y], "m2_fixed": CEILING / series[y]}
                  for y in sorted(series)]).to_csv(RESULTS / "national_series.csv", index=False)
    say(f"2017 to 2025: median rent per m2 x{series[2025] / series[2017]:.2f}; rule ceiling x"
        f"{RMMG[2025] / RMMG[2017]:.2f}; (2017 to 2026 rule ceiling x{RMMG[2026] / RMMG[2017]:.2f})")
    # municipality splice and persistence
    om = old[old["year"] == 2020][["dtmn", "p50_old"]]
    nm = q[q["year"] == 2020][["dtmn", "p50"]]
    m = om.merge(nm, on="dtmn").dropna()
    m["ratio"] = m["p50"] / m["p50_old"]
    say(f"Municipal splice ratios in 2020 ({len(m)} municipalities): median {m.ratio.median():.3f}, "
        f"interquartile range {m.ratio.quantile(0.25):.3f} to {m.ratio.quantile(0.75):.3f}")
    panel = []
    for y in YEARS_OLD:
        g = old[old["year"] == y].merge(m[["dtmn", "ratio"]], on="dtmn")
        g = g.assign(p50=g["p50_old"] * g["ratio"], year=y, leases=g["leases_old"])
        panel.append(g[["dtmn", "geodsg", "year", "p50", "leases"]])
    newp = q.merge(leases, on=["dtmn", "year"], how="left")[["dtmn", "geodsg", "year", "p50", "leases"]]
    panel.append(newp)
    p = pd.concat(panel, ignore_index=True).dropna(subset=["p50"])
    p["rule"] = p["year"].map(lambda y: RULE_MULTIPLE * RMMG[y])
    p["m2_rule"] = p["rule"] / p["p50"]
    p["m2_fixed"] = CEILING / p["p50"]
    p["home80_over_rule"] = 80 * p["p50"] / p["rule"]
    for y in sorted(p["year"].unique()):
        g = p[(p["year"] == y) & p["leases"].notna()]
        lw = np.sqrt(np.average((np.log(g.p50) - np.average(np.log(g.p50), weights=g.leases)) ** 2, weights=g.leases))
        say(f"  {y}: {len(g)} municipalities; lease-weighted median of m2 bought by the rule ceiling "
            f"{wmedian(g.m2_rule.to_numpy(), g.leases.to_numpy()):.0f}, by EUR 2,300 "
            f"{wmedian(g.m2_fixed.to_numpy(), g.leases.to_numpy()):.0f}; smallest {g.m2_rule.min():.0f} "
            f"({g.loc[g.m2_rule.idxmin(), 'geodsg']}); municipalities where the rule ceiling buys under 120 m2: "
            f"{(g.m2_rule < 120).sum()}; lease-weighted s.d. of log median rent {lw:.3f}")
    a = p[p["year"] == 2017][["dtmn", "p50"]].merge(p[p["year"] == 2025][["dtmn", "p50", "geodsg"]], on="dtmn", suffixes=("_17", "_25"))
    rho = a[["p50_17", "p50_25"]].rank().corr().iloc[0, 1]
    a["growth"] = a["p50_25"] / a["p50_17"]
    say(f"Rank correlation of municipal median rent per m2, 2017 and 2025: {rho:.3f} ({len(a)} municipalities); "
        f"median growth x{a.growth.median():.2f}, interquartile x{a.growth.quantile(0.25):.2f} to x{a.growth.quantile(0.75):.2f}; "
        f"rule ceiling x{RMMG[2025] / RMMG[2017]:.2f}; municipalities where rents grew faster than the rule: "
        f"{(a.growth > RMMG[2025] / RMMG[2017]).sum()} of {len(a)}")
    p.to_csv(RESULTS / "panel_2017_2025.csv", index=False)
    say()
    return p


def by_dwelling_type(areas: dict[str, float]) -> None:
    say("## 5. Monthly rent at the median rent per m2, by dwelling type, 2025 (municipalities over 100,000)")
    say("Typical floor area used (Census 2021 median by rooms): " + ", ".join(f"{k} {v:.0f} m2" for k, v in areas.items()))
    t = load("rent_by_type_quarterly")
    t = t[t["period"].str.endswith("2025") & (t["dim_3"] != "T")]
    t["area"] = t["dim_3_t"].map(areas)
    t["rent"] = t["value"] * t["area"]
    yearly = t.groupby(["geodsg", "dim_3_t"])["rent"].mean().unstack()
    yearly.to_csv(RESULTS / "rent_by_type_2025.csv", float_format="%.1f")
    for n in ["Portugal"] + BIG:
        if n in yearly.index:
            r = yearly.loc[n]
            say(f"  {n}: " + ", ".join(f"{k} EUR {r[k]:,.0f}" for k in areas if k in r and pd.notna(r[k])))
    mx = yearly.max().max()
    where = yearly.stack().idxmax()
    say(f"Highest typical monthly rent of any type in any of these municipalities in 2025: EUR {mx:,.0f} ({where[1]}, {where[0]}) "
        f"= {mx / CEILING:.0%} of the ceiling")
    say()


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    say("# Where the EUR 2,300 moderate-rent ceiling sat against real rents, 2017-2025")
    say(f"Ceiling EUR {CEILING:,.0f} a month (2.5 x the 2026 minimum wage of EUR {RMMG[2026]}); rule ceiling by year "
        + ", ".join(f"{y} EUR {RULE_MULTIPLE * RMMG[y]:,.1f}" for y in sorted(RMMG)))
    say()
    q = quartiles_2026()
    leases = leases_2026()
    homes = census_area()
    areas = typology_areas()
    area_needed(q, leases, homes)
    beta, se, _ = elasticity(areas)
    model_shares(q, leases, homes, beta, se)
    tail_check(census_rent_area())
    incomes(q)
    over_time(q, medians_old(), leases)
    by_dwelling_type(areas)
    (RESULTS / "summary.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
