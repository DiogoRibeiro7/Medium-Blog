"""Integrated assessment: prices, income, construction, population and rents in one audited panel.

Reads only the snapshots already pinned by the earlier analyses (nothing is
downloaded), audits them, builds a municipality-year panel on the NUTS 2024
geography (municipalities matched by DICO code), and estimates descriptive
associations at three levels of ambition:

1. robust descriptive facts (levels, growth, cross-sectional rank correlations
   with cluster-bootstrap intervals);
2. model-dependent within-municipality associations (two-way fixed effects, first
   differences, long differences, region-by-year effects), each with cluster-robust
   and wild-cluster-bootstrap intervals, serial-correlation and influence checks;
3. questions that need a causal design, which this data cannot supply.

Only compatible observations enter a model: one price method, one population
basis and one geography per sample; no series is spliced across a method break.
Nothing here estimates a causal effect.

Run:  python integrated.py   # writes results_integrated.json, integrated_panel.csv, data_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import affordability as F
import panel_stats as S
import regional as R
import rental as RT
from analysis import (
    annual_mean_rates,
    parse_completed_dwellings,
    parse_demographic_totals,
    parse_hpi,
    parse_income,
    parse_mortgage_rates,
)

HERE = Path(__file__).parent
DATA = HERE / "data"
DATE = "2026-10-29"
YEARS = list(range(2011, 2026))
METRO = {"1A0", "1B0", "11A"}  # Grande Lisboa, Península de Setúbal, Área Metropolitana do Porto
BOOT_REPS = 1999
Key = tuple[str, int]
Panel = dict[Key, dict[str, float]]

# Variables of the municipal panel: name -> (description, unit, alignment).
VARIABLES: dict[str, tuple[str, str, str]] = {
    "population": ("Resident population, 31 December (census-based to 2020, administrative-based from 2021)", "persons", "end of year"),
    "completions": ("Dwellings completed in new construction for family housing", "dwellings", "calendar year"),
    "permits": ("Dwellings licensed in new construction for family housing", "dwellings", "calendar year"),
    "price_m2": ("Median value per m2 of family dwellings sold, 2022 method", "EUR/m2", "12 months to Q4 = calendar year"),
    "price_m2_existing": ("Same, existing dwellings only", "EUR/m2", "12 months to Q4 = calendar year"),
    "price_m2_2018m": ("Median value per m2 of family dwellings sold, 2018 method (NUTS 2013 table)", "EUR/m2", "12 months to Q4 = calendar year"),
    "appraisal_m2": ("Median bank appraisal value per m2 (all construction types)", "EUR/m2", "calendar year"),
    "income": ("Median declared gross income per tax household (IRS)", "EUR/year", "tax year"),
    "income_net": ("Median declared income net of IRS per tax household", "EUR/year", "tax year"),
    "rent_m2": ("Median monthly rent per m2 of new lease contracts, 2026 method", "EUR/m2/month", "contracts of the calendar year"),
    "rent_m2_2021m": ("Same, 2021 method (NUTS 2013 table)", "EUR/m2/month", "contracts of the calendar year"),
}


# ---------------------------------------------------------------- data ----

def load_inputs(folder: Path, date: str) -> dict[str, Any]:
    """Parse every pinned source the panel needs (each response checked against its SHA-256)."""
    reg = R.load_sources(folder, date)
    aff = F.load(folder, date)
    rent = F.parse_series(folder / f"ine_0014698_{date}.jsonl.gz")
    rent_old = F.parse_series(folder / f"ine_0009631_{date}.jsonl.gz")
    return {"reg": reg, "aff": aff, "rent": rent, "rent_old": rent_old}


def series_by_dico(series: F.Series, dims: dict[str, str] | None = None) -> dict[Key, float]:
    """Municipal values keyed by (DICO, year)."""
    return RT.by_dico(series, dims)


def build_panel(inp: dict[str, Any]) -> tuple[Panel, R.Geography, dict[str, Any]]:
    """Municipality-year panel on NUTS 2024 (DICO), plus the checks made while joining sources."""
    reg, aff = inp["reg"], inp["aff"]
    geo = R.nuts2024_geography(F.as_source(aff["price"], {"3": "H.1"}))
    checks: dict[str, Any] = {}
    pop13, pop24 = R.by_dico(reg["pop13"]), R.by_dico(reg["pop24"])
    overlap = [k for k in pop13 if k[1] >= 2021 and k in pop24]
    checks["population_nuts2013_vs_nuts2024_2021_2023_identical"] = all(pop13[k] == pop24[k] for k in overlap)
    inc13, inc24 = series_by_dico(aff["income_2013"]), series_by_dico(aff["income"])
    shared = [k for k in inc13 if k in inc24]
    diffs = [abs(inc13[k] - inc24[k]) for k in shared]
    checks["income_nuts2013_vs_nuts2024_overlap"] = {"cells": len(shared), "identical": sum(1 for d in diffs if d == 0),
                                                    "max_abs_diff_eur": max(diffs)}
    if max(diffs) > 1:
        raise ValueError("Municipal income differs between the NUTS 2013 and NUTS 2024 tables by more than EUR 1")
    sources: dict[str, dict[Key, float]] = {
        "population": {**{k: v for k, v in pop13.items() if k[1] <= 2020}, **pop24},
        "completions": R.by_dico(reg["comp24"]),
        "permits": R.by_dico(reg["perm24"]),
        "price_m2": series_by_dico(aff["price"], {"3": "H.1"}),
        "price_m2_existing": series_by_dico(aff["price"], {"3": "H.1.2"}),
        "price_m2_2018m": series_by_dico(aff["price_2018"]),
        "appraisal_m2": series_by_dico(aff["appraisal"], {"3": "T"}),
        # Same statistic in two geography tables; identical in the overlap (checked above), so 2015-2017 come from the old table.
        "income": {**{k: v for k, v in inc13.items() if k[1] <= 2017}, **inc24},
        "income_net": series_by_dico(aff["income_net"]),
        "rent_m2": series_by_dico(inp["rent"]),
        "rent_m2_2021m": series_by_dico(inp["rent_old"]),
    }
    panel: Panel = defaultdict(dict)
    for name, values in sources.items():
        for (dico, year), v in values.items():
            if dico in geo.municipality and year in YEARS:
                panel[(dico, year)][name] = v
    return dict(panel), geo, checks


def audit(inp: dict[str, Any], panel: Panel, geo: R.Geography) -> list[dict[str, Any]]:
    """One row per variable: coverage, missing cells, alignment, units and documented breaks."""
    meta = {
        "population": (inp["reg"]["pop13"], "INE 0008273 (to 2020) + 0012917 (from 2021)",
                       "2020/2021: census-based to administrative-based estimates (+2.0% for Portugal); never compared across"),
        "completions": (inp["reg"]["comp24"], "INE 0013248",
                        "Amadora under-reported 2011-2017; 2021+ preliminary; 2024-2025 estimated"),
        "permits": (inp["reg"]["perm24"], "INE 0013235", "2021+ preliminary"),
        "price_m2": (inp["aff"]["price"], "INE 0012234", "2022 method from 2019 Q4; not spliced with the 2018 method"),
        "price_m2_existing": (inp["aff"]["price"], "INE 0012234 (existing)", "as price_m2"),
        "price_m2_2018m": (inp["aff"]["price_2018"], "INE 0009490", "2018 method, closed at 2021 Q3; NUTS 2013 table"),
        "appraisal_m2": (inp["aff"]["appraisal"], "INE 0012256", "survey of bank valuations; suppressed where few appraisals"),
        "income": (inp["aff"]["income"], "INE 0012712 (2018+) + 0009763 (2015-2017)", "two geography tables of one statistic, identical in 2018-2021"),
        "income_net": (inp["aff"]["income_net"], "INE 0012741", "net of IRS"),
        "rent_m2": (inp["rent"], "INE 0014698", "2026 method from 2020; not spliced with the 2021 method"),
        "rent_m2_2021m": (inp["rent_old"], "INE 0009631", "2021 method, NUTS 2013 table, 2017-2023"),
    }
    rows = []
    for name, (desc, unit, align) in VARIABLES.items():
        years = sorted({y for (_d, y), vals in panel.items() if name in vals})
        per_year = {y: sum(1 for d in geo.municipality if name in panel.get((d, y), {})) for y in years}
        series, code, breaks = meta[name]
        values = [vals[name] for vals in panel.values() if name in vals]
        rows.append({
            "variable": name, "source": code, "description": desc, "unit": unit, "time_alignment": align,
            "geography": "NUTS 2024 municipalities (DICO); source table " + series.title.split(" por ")[-1].split(";")[0][:60],
            "first_year": years[0], "last_year": years[-1],
            "municipalities_min": min(per_year.values()), "municipalities_max": max(per_year.values()),
            "missing_cells": sum(308 - c for c in per_year.values()),
            "min_value": min(values), "max_value": max(values), "non_positive_values": sum(1 for v in values if v <= 0),
            "breaks_and_notes": breaks,
        })
    return rows


# ------------------------------------------------------------- samples ----

def balanced_units(panel: Panel, variables: Sequence[str], years: Sequence[int], units: Sequence[str],
                   positive: Sequence[str] = ()) -> list[str]:
    """Units with every variable present in every year (and positive where it will be logged)."""
    return [u for u in units if all(v in panel.get((u, t), {}) for t in years for v in variables)
            and all(panel[(u, t)][v] > 0 for t in years for v in positive)]


def transform(values: dict[Key, float], units: Sequence[str], years: Sequence[int], kind: str,
              region_of: dict[str, str]) -> dict[Key, float]:
    """Remove fixed effects from a balanced panel variable.

    twoway: unit and year effects; region_year: unit effects and region-by-year
    effects (x_it - mean_i - mean_rt + mean_r); fd: first differences with year
    effects (demeaned by year).
    """
    if kind == "twoway":
        return S.two_way_within(values, units, years)
    if kind == "region_year":
        unit_mean = {u: sum(values[(u, t)] for t in years) / len(years) for u in units}
        members: dict[str, list[str]] = defaultdict(list)
        for u in units:
            members[region_of[u]].append(u)
        rt = {(r, t): sum(values[(u, t)] for u in us) / len(us) for r, us in members.items() for t in years}
        rm = {r: sum(rt[(r, t)] for t in years) / len(years) for r in members}
        return {(u, t): values[(u, t)] - unit_mean[u] - rt[(region_of[u], t)] + rm[region_of[u]] for u in units for t in years}
    if kind == "fd":
        diff = {(u, t1): values[(u, t1)] - values[(u, t0)] for u in units for t0, t1 in zip(years, years[1:], strict=False)}
        year_mean = {t: sum(diff[(u, t)] for u in units) / len(units) for t in years[1:]}
        return {(u, t): diff[(u, t)] - year_mean[t] for u in units for t in years[1:]}
    raise ValueError(f"Unknown transform {kind!r}")


def variable(panel: Panel, name: str, units: Sequence[str], years: Sequence[int],
             base_pop: dict[str, float] | None = None) -> dict[Key, float]:
    """A model variable: log_<x> = ln(x); <x>_per_1000 = 1000 * x / base-year population (fixed denominator)."""
    if name.startswith("log_"):
        src = name[4:]
        return {(u, t): math.log(panel[(u, t)][src]) for u in units for t in years}
    if name.endswith("_per_1000"):
        src = name[:-9]
        if base_pop is None:
            raise ValueError("A base-year population is needed for per-1000 rates")
        return {(u, t): 1000 * panel[(u, t)][src] / base_pop[u] for u in units for t in years}
    raise ValueError(f"Unknown variable {name!r}")


def raw_needed(names: Sequence[str]) -> list[str]:
    """Panel columns a list of model variables needs."""
    return [n[4:] if n.startswith("log_") else n[:-9] for n in names]


def fit_model(panel: Panel, y: str, xs: Sequence[str], years: Sequence[int], units: Sequence[str],
              geo: R.Geography, kind: str = "twoway", bootstrap: bool = True, base_year: int | None = None) -> dict[str, Any]:
    """Estimate one specification on a balanced sample; CR1 by municipality, wild bootstrap by NUTS 3."""
    needed = raw_needed([y, *xs]) + (["population"] if any(x.endswith("_per_1000") for x in xs) else [])
    logged = [n[4:] for n in [y, *xs] if n.startswith("log_")]
    sample = balanced_units(panel, needed, years, units, positive=logged)
    by = base_year if base_year is not None else years[0]
    base_pop = {u: panel[(u, by)]["population"] for u in sample if "population" in panel.get((u, by), {})}
    nuts3 = {u: geo.nuts3[u] for u in sample}
    nuts2 = {u: geo.nuts2_of3[geo.nuts3[u]] for u in sample}
    data = {name: transform(variable(panel, name, sample, years, base_pop), sample, years, kind, nuts2) for name in [y, *xs]}
    keys = sorted(data[y])
    x_rows = [[data[x][k] for x in xs] for k in keys]
    y_vec = [data[y][k] for k in keys]
    fit = S.ols(x_rows, y_vec, [k[0] for k in keys], list(xs))
    out: dict[str, Any] = {"spec": {"y": y, "x": list(xs), "years": [years[0], years[-1]], "transform": kind},
                           "municipalities": len(sample), **fit.as_dict()}
    out["within_variance_share"] = {name: round(S.within_share(variable(panel, name, sample, years, base_pop), sample, years), 4)
                                    for name in [y, *xs]} if kind == "twoway" else None
    if bootstrap:
        out["wild_cluster_bootstrap_nuts3"] = {
            x: {k: round(v, 4) for k, v in S.wild_cluster_bootstrap(x_rows, y_vec, [nuts3[k[0]] for k in keys], j, reps=BOOT_REPS).items()}
            for j, x in enumerate(xs)}
    out["_internal"] = {"keys": keys, "x": x_rows, "y": y_vec, "fit": fit, "sample": sample, "nuts3": nuts3}
    return out


def strip(model: dict[str, Any]) -> dict[str, Any]:
    """Model results without the arrays kept for diagnostics."""
    return {k: v for k, v in model.items() if k != "_internal"}


def influence(model: dict[str, Any], panel: Panel, geo: R.Geography, refit: Callable[[list[str]], dict[str, Any]]) -> dict[str, Any]:
    """Leave-one-NUTS3-out range, dropping Lisbon/Porto metro areas, and dropping the 5% most influential municipalities.

    Influence is the one-step DFBETA of each municipality on the first regressor
    of the specification (the coefficient of interest in every model here).
    """
    internal = model["_internal"]
    sample, fit = internal["sample"], internal["fit"]
    names = model["spec"]["x"]
    regions = sorted({geo.nuts3[u] for u in sample})
    loo: dict[str, dict[str, float]] = {}
    for r in regions:
        keep = [u for u in sample if geo.nuts3[u] != r]
        m = refit(keep)
        loo[r] = {x: m["coefficients"][x]["estimate"] for x in names}
    # One-step influence of each municipality on each coefficient: A * sum_t x_it u_it.
    a = S.inverse(S.crossprod(internal["x"]))
    score: dict[str, list[float]] = defaultdict(lambda: [0.0] * len(names))
    for (u, _t), row, e in zip(internal["keys"], internal["x"], fit.resid, strict=True):
        for c in range(len(names)):
            score[u][c] += row[c] * e
    dfbeta = {u: S.matvec(a, s) for u, s in score.items()}
    first = 0
    ranked = sorted(sample, key=lambda u: -abs(dfbeta[u][first]))
    cut = max(1, round(0.05 * len(sample)))
    drop = set(ranked[:cut])
    no_metro = refit([u for u in sample if geo.nuts3[u] not in METRO])
    trimmed = refit([u for u in sample if u not in drop])
    return {
        "leave_one_nuts3_out": {x: {"min": round(min(v[x] for v in loo.values()), 4), "max": round(max(v[x] for v in loo.values()), 4),
                                    "region_at_min": min(loo, key=lambda r: loo[r][x]), "region_at_max": max(loo, key=lambda r: loo[r][x])}
                                for x in names},
        "without_lisbon_and_porto_metro": {x: no_metro["coefficients"][x] for x in names} | {"municipalities": no_metro["municipalities"]},
        "without_top5pct_influential": {x: trimmed["coefficients"][x] for x in names} | {"dropped": sorted(geo.names.get(geo.municipality[u], u) for u in drop)},
    }


# -------------------------------------------------------- correlations ----

def cross_section(panel: Panel, geo: R.Geography, pairs: dict[str, tuple[Callable[[str], float | None], Callable[[str], float | None]]],
                  units: Sequence[str]) -> dict[str, Any]:
    """Pearson and Spearman correlations across municipalities, cluster-bootstrapped by NUTS 3."""
    out: dict[str, Any] = {}
    for label, (fx, fy) in pairs.items():
        xs, ys, cl = [], [], []
        for u in units:
            vx, vy = fx(u), fy(u)
            if vx is not None and vy is not None:
                xs.append(vx)
                ys.append(vy)
                cl.append(geo.nuts3[u])
        res = S.cluster_bootstrap_corr(xs, ys, cl, reps=BOOT_REPS)
        out[label] = {k: round(v, 3) for k, v in res.items()}
    return out


def nuts3_correlations(pairs: dict[str, tuple[dict[str, float], dict[str, float]]]) -> dict[str, Any]:
    """Correlations across the 26 NUTS 3 regions (published regional medians/totals), Fisher intervals."""
    out = {}
    for label, (a, b) in pairs.items():
        keys = sorted(a.keys() & b.keys())
        x, y = [a[k] for k in keys], [b[k] for k in keys]
        r, rho = S.pearson(x, y), S.spearman(x, y)
        lo, hi = S.fisher_ci(r, len(keys))
        out[label] = {"n": len(keys), "pearson": round(r, 3), "pearson_ci95": [round(lo, 3), round(hi, 3)], "spearman": round(rho, 3)}
    return out


def growth(panel: Panel, name: str, u: str, a: int, b: int) -> float | None:
    """100 * ln(x_b / x_a) for one municipality, or None if either value is missing."""
    va, vb = panel.get((u, a), {}).get(name), panel.get((u, b), {}).get(name)
    return 100 * math.log(vb / va) if va and vb else None


def mean_rate(panel: Panel, name: str, u: str, years: Sequence[int], base_year: int) -> float | None:
    """Mean of 1000 * x_t / population_base over years (None if anything is missing)."""
    pop = panel.get((u, base_year), {}).get("population")
    vals = [panel.get((u, t), {}).get(name) for t in years]
    if not pop or any(v is None for v in vals):
        return None
    return 1000 * sum(v for v in vals if v is not None) / len(vals) / pop


# ------------------------------------------------------------ national ----

def national_series(folder: Path, date: str) -> dict[str, dict[int, float]]:
    """National annual series, each keyed by the calendar year it describes."""
    hpi = parse_hpi(json.loads((folder / f"eurostat_prc_hpi_a_PT_{date}.json").read_text(encoding="utf-8")))
    silc = parse_income(json.loads((folder / f"eurostat_ilc_di03_PT_{date}.json").read_text(encoding="utf-8")))
    completions, _prelim = parse_completed_dwellings(folder / f"ine_pordata_completed_new_dwellings_PT_{date}.csv")
    pop, _flags = parse_demographic_totals(json.loads((folder / f"eurostat_demo_pjan_PT_{date}.json").read_text(encoding="utf-8")),
                                           households=False)
    monthly, _status = parse_mortgage_rates(folder / f"ecb_mir_PT_house_purchase_cost_of_borrowing_{date}.csv")
    rates, _partial = annual_mean_rates(monthly)
    hicp = json.loads((folder / f"eurostat_prc_hicp_aind_PT_rents_{date}.json").read_text(encoding="utf-8"))
    rents = RT.annual(hicp, geo="PT", coicop="CP041")
    cpi = RT.annual(hicp, geo="PT", coicop="CP00")
    return {
        "house_prices_hpi": hpi,
        "income_silc_income_year": {y - 1: v for y, v in silc.items()},   # survey t reports income of t-1
        "completions": {y: float(v) for y, v in completions.items()},
        "population_end_of_year": {y - 1: v for y, v in pop.items()},     # 1 January t = 31 December t-1
        "rents_hicp": rents,
        "consumer_prices_hicp": cpi,
        "mortgage_rate": rates,
    }


def growth_rates(series: dict[int, float], log: bool = True) -> dict[int, float]:
    """Annual log growth in percent (or first differences for rates)."""
    return {y: (100 * math.log(series[y] / series[y - 1]) if log else series[y] - series[y - 1])
            for y in series if y - 1 in series}


def national_correlations(nat: dict[str, dict[int, float]], start: int, end: int) -> dict[str, Any]:
    """Correlation of annual house-price growth with each other series' growth, with an effective-n Fisher interval."""
    g = {name: growth_rates(s, log=name != "mortgage_rate") for name, s in nat.items()}
    out = {}
    for other in ("income_silc_income_year", "completions", "population_end_of_year", "rents_hicp", "mortgage_rate"):
        years = [y for y in range(start, end + 1) if y in g["house_prices_hpi"] and y in g[other]]
        x, y = [g["house_prices_hpi"][t] for t in years], [g[other][t] for t in years]
        r = S.pearson(x, y)
        n_eff = S.effective_n(x, y)
        lo, hi = S.fisher_ci(r, n_eff)
        out[other] = {"years": [years[0], years[-1]], "n": len(years), "pearson": round(r, 3),
                      "effective_n": round(n_eff, 1), "ci95": [round(lo, 3), round(hi, 3)]}
    return out


# ---------------------------------------------------------------- main ----

def main() -> None:  # noqa: C901 - one linear pipeline, kept in one place for auditability
    """Audit, build the panel, estimate and export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--date", default=DATE)
    parser.add_argument("--results", type=Path, default=HERE / "results_integrated.json")
    parser.add_argument("--panel", type=Path, default=HERE / "integrated_panel.csv")
    parser.add_argument("--audit", type=Path, default=HERE / "data_audit.csv")
    args = parser.parse_args()
    inp = load_inputs(args.data, args.date)
    panel, geo, join_checks = build_panel(inp)
    munis = sorted(geo.municipality)
    audit_rows = audit(inp, panel, geo)

    # ---- Analytical samples (compatible observations only) ----
    samples: dict[str, dict[str, Any]] = {
        "A_transactions_2019_2024": {"years": list(range(2019, 2025)), "variables": ["price_m2", "income", "completions", "population"],
                                     "note": "2022 price method; IRS income; population enters only through a fixed 2019 denominator"},
        "A2_transactions_population_2021_2024": {"years": list(range(2021, 2025)), "variables": ["price_m2", "income", "completions", "population"],
                                                 "note": "administrative-based population only (after the 2020/2021 break)"},
        "B_transactions_2018_method_2016_2020": {"years": list(range(2016, 2021)), "variables": ["price_m2_2018m", "income", "completions", "population"],
                                                 "note": "2018 price method; census-based population throughout"},
        "C_appraisal_2015_2024": {"years": list(range(2015, 2025)), "variables": ["appraisal_m2", "income", "completions", "population"],
                                  "note": "bank appraisals, the longest consistent price-like series; selected (larger) municipalities"},
        "D_rents_2020_2024": {"years": list(range(2020, 2025)), "variables": ["price_m2", "rent_m2", "income"],
                              "note": "2026 rent method; 2022 price method"},
    }
    for s in samples.values():
        units = balanced_units(panel, s["variables"], s["years"], munis, positive=[v for v in s["variables"] if v != "completions"])
        s["municipalities"] = len(units)
        s["metro_municipalities"] = sum(1 for u in units if geo.nuts3[u] in METRO)

    # ---- Robust descriptive layer ----
    def g(name: str, a: int, b: int) -> Callable[[str], float | None]:
        return lambda u: growth(panel, name, u, a, b)

    def lv(name: str, t: int) -> Callable[[str], float | None]:
        return lambda u: math.log(panel[(u, t)][name]) if name in panel.get((u, t), {}) else None

    cross = {
        "levels_2024": cross_section(panel, geo, {
            "log_price_vs_log_income": (lv("income", 2024), lv("price_m2", 2024)),
            "log_price_vs_log_rent": (lv("rent_m2", 2024), lv("price_m2", 2024)),
            "log_price_vs_completions_per_1000_2021_2024": (lambda u: mean_rate(panel, "completions", u, range(2021, 2025), 2021), lv("price_m2", 2024)),
            "log_price_vs_population_growth_2021_2024": (g("population", 2021, 2024), lv("price_m2", 2024)),
        }, munis),
        "changes_2019_2024": cross_section(panel, geo, {
            "price_growth_vs_income_growth": (g("income", 2019, 2024), g("price_m2", 2019, 2024)),
            "price_growth_vs_completions_per_1000": (lambda u: mean_rate(panel, "completions", u, range(2019, 2025), 2019), g("price_m2", 2019, 2024)),
            "price_growth_vs_population_growth_2021_2024": (g("population", 2021, 2024), g("price_m2", 2019, 2024)),
            "price_growth_vs_rent_growth_2020_2024": (g("rent_m2", 2020, 2024), g("price_m2", 2020, 2024)),
        }, munis),
        "changes_2016_2020_2018_method": cross_section(panel, geo, {
            "price_growth_vs_income_growth": (g("income", 2016, 2020), g("price_m2_2018m", 2016, 2020)),
            "price_growth_vs_completions_per_1000": (lambda u: mean_rate(panel, "completions", u, range(2016, 2021), 2016), g("price_m2_2018m", 2016, 2020)),
            "price_growth_vs_population_growth": (g("population", 2016, 2020), g("price_m2_2018m", 2016, 2020)),
            "price_growth_vs_rent_growth_2017_2019": (g("rent_m2_2021m", 2017, 2019), g("price_m2_2018m", 2017, 2019)),
        }, munis),
        "changes_2015_2024_appraisal": cross_section(panel, geo, {
            "appraisal_growth_vs_income_growth": (g("income", 2015, 2024), g("appraisal_m2", 2015, 2024)),
            "appraisal_growth_vs_completions_per_1000": (lambda u: mean_rate(panel, "completions", u, range(2015, 2025), 2015), g("appraisal_m2", 2015, 2024)),
            "appraisal_growth_vs_population_growth_2015_2020": (g("population", 2015, 2020), g("appraisal_m2", 2015, 2020)),
        }, munis),
    }
    # Regional (NUTS 3) published values: medians and totals, never aggregated here.
    price3 = F.on_nuts2024(inp["aff"]["price"], geo, {"3": "H.1"})[0]
    income3 = F.on_nuts2024(inp["aff"]["income"], geo)[0]
    rent3 = F.on_nuts2024(inp["rent"], geo)[0]
    reg_pop = {**{k: v for k, v in R.regional_series(inp["reg"]["pop13"], geo)[0].items() if k[1] <= 2020},
               **R.regional_series(inp["reg"]["pop24"], geo)[0]}
    reg_comp = R.regional_series(inp["reg"]["comp24"], geo)[0]
    n3 = sorted(geo.members3)

    def d3(series: Mapping[Key, float], a: int, b: int) -> dict[str, float]:
        return {r: 100 * math.log(series[(r, b)] / series[(r, a)]) for r in n3 if (r, a) in series and (r, b) in series}

    nuts3 = nuts3_correlations({
        "price_growth_vs_income_growth_2019_2024": (d3(price3, 2019, 2024), d3(income3, 2019, 2024)),
        "price_growth_vs_completions_per_1000_2019_2024": (d3(price3, 2019, 2024),
                                                            {r: 1000 * sum(reg_comp[(r, t)] for t in range(2019, 2025)) / 6 / reg_pop[(r, 2019)] for r in n3}),
        "price_growth_2019_2024_vs_population_growth_2021_2024": (d3(price3, 2019, 2024), d3(reg_pop, 2021, 2024)),
        "price_growth_vs_rent_growth_2020_2024": (d3(price3, 2020, 2024), d3(rent3, 2020, 2024)),
        "log_price_vs_log_income_2024": ({r: math.log(price3[(r, 2024)]) for r in n3}, {r: math.log(income3[(r, 2024)]) for r in n3}),
    })

    # Gross rental yield: 12 x monthly new-lease rent per m2 / sale price per m2 (both medians, different dwellings).
    yields = {}
    for r in ["PT", *sorted(geo.members2), *n3]:
        yields[r] = {str(t): round(100 * 12 * rent3[(r, t)] / price3[(r, t)], 2) for t in range(2020, 2026)
                     if (r, t) in rent3 and (r, t) in price3}
    falling = sum(1 for r in n3 if yields[r]["2025"] < yields[r]["2020"])

    # ---- Model-dependent layer ----
    a_years, a2_years = samples["A_transactions_2019_2024"]["years"], samples["A2_transactions_population_2021_2024"]["years"]
    b_years, c_years, d_years = (samples["B_transactions_2018_method_2016_2020"]["years"], samples["C_appraisal_2015_2024"]["years"],
                                 samples["D_rents_2020_2024"]["years"])
    specs: dict[str, dict[str, Any]] = {
        "A1_twoway_income": dict(y="log_price_m2", xs=["log_income"], years=a_years),
        "A1_fd_income": dict(y="log_price_m2", xs=["log_income"], years=a_years, kind="fd"),
        "A1_region_year_income": dict(y="log_price_m2", xs=["log_income"], years=a_years, kind="region_year"),
        "A2_twoway_full": dict(y="log_price_m2", xs=["log_income", "completions_per_1000", "log_population"], years=a2_years, base_year=2021),
        "A2_twoway_income_completions": dict(y="log_price_m2", xs=["log_income", "completions_per_1000"], years=a2_years, base_year=2021),
        "A2_fd_income_completions": dict(y="log_price_m2", xs=["log_income", "completions_per_1000"], years=a2_years, kind="fd", base_year=2021),
        "A2_twoway_existing_prices": dict(y="log_price_m2_existing", xs=["log_income", "completions_per_1000"], years=a2_years, base_year=2021),
        "A2_fd_full": dict(y="log_price_m2", xs=["log_income", "completions_per_1000", "log_population"], years=a2_years, kind="fd", base_year=2021),
        "B_twoway_full": dict(y="log_price_m2_2018m", xs=["log_income", "completions_per_1000", "log_population"], years=b_years),
        "B_fd_full": dict(y="log_price_m2_2018m", xs=["log_income", "completions_per_1000", "log_population"], years=b_years, kind="fd"),
        "C_twoway_income_completions": dict(y="log_appraisal_m2", xs=["log_income", "completions_per_1000"], years=c_years),
        "C_fd_income_completions": dict(y="log_appraisal_m2", xs=["log_income", "completions_per_1000"], years=c_years, kind="fd"),
        "D_twoway_price_on_rent": dict(y="log_price_m2", xs=["log_rent_m2"], years=d_years),
        "D_fd_price_on_rent": dict(y="log_price_m2", xs=["log_rent_m2"], years=d_years, kind="fd"),
        "D_twoway_rent_on_income": dict(y="log_rent_m2", xs=["log_income"], years=d_years),
        "A1_twoway_existing_prices": dict(y="log_price_m2_existing", xs=["log_income"], years=a_years),
        "A1_twoway_net_income": dict(y="log_price_m2", xs=["log_income_net"], years=a_years),
        "A1_twoway_2019_2022": dict(y="log_price_m2", xs=["log_income"], years=list(range(2019, 2023))),
        "A1_twoway_2021_2024": dict(y="log_price_m2", xs=["log_income"], years=list(range(2021, 2025))),
    }
    models: dict[str, dict[str, Any]] = {}
    for name, sp in specs.items():
        models[name] = fit_model(panel, sp["y"], sp["xs"], sp["years"], munis, geo, sp.get("kind", "twoway"),
                                 base_year=sp.get("base_year"))

    # Long differences (one cross-section per window, NUTS 2 effects removed).
    def long_diff(yname: str, xnames: list[str], a: int, b: int) -> dict[str, Any]:
        units = [u for u in munis if all(growth(panel, n, u, a, b) is not None for n in [yname, *xnames])]
        nuts2 = {u: geo.nuts2_of3[geo.nuts3[u]] for u in units}
        vals = {n: {u: growth(panel, n, u, a, b) or 0.0 for u in units} for n in [yname, *xnames]}
        dm = {}
        for n, v in vals.items():
            means: dict[str, list[float]] = defaultdict(list)
            for u in units:
                means[nuts2[u]].append(v[u])
            avg = {r: sum(x) / len(x) for r, x in means.items()}
            dm[n] = {u: v[u] - avg[nuts2[u]] for u in units}
        x_rows = [[dm[n][u] / 100 for n in xnames] for u in units]
        y_vec = [dm[yname][u] / 100 for u in units]
        fit = S.ols(x_rows, y_vec, [geo.nuts3[u] for u in units], [f"dlog_{n}" for n in xnames])
        out = {"spec": {"y": f"dlog_{yname}", "x": [f"dlog_{n}" for n in xnames], "years": [a, b], "transform": "long difference, NUTS 2 effects"},
               "municipalities": len(units), **fit.as_dict()}
        out["wild_cluster_bootstrap_nuts3"] = {
            f"dlog_{n}": {k: round(v, 4) for k, v in S.wild_cluster_bootstrap(x_rows, y_vec, [geo.nuts3[u] for u in units], j, reps=BOOT_REPS).items()}
            for j, n in enumerate(xnames)}
        return out

    long_diffs = {
        "price_on_income_2019_2024": long_diff("price_m2", ["income"], 2019, 2024),
        "price_2018m_on_income_2016_2020": long_diff("price_m2_2018m", ["income"], 2016, 2020),
        "price_on_rent_2020_2024": long_diff("price_m2", ["rent_m2"], 2020, 2024),
    }

    # Diagnostics on the main specifications.
    diagnostics: dict[str, Any] = {}
    for main_name, fd_name in (("A1_twoway_income", "A1_fd_income"), ("A2_twoway_full", "A2_fd_full"),
                               ("A2_twoway_income_completions", "A2_fd_income_completions"),
                               ("B_twoway_full", "B_fd_full"), ("C_twoway_income_completions", "C_fd_income_completions"),
                               ("D_twoway_price_on_rent", "D_fd_price_on_rent")):
        fd = models[fd_name]["_internal"]
        resid = {k: e for k, e in zip(fd["keys"], fd["fit"].resid, strict=True)}
        fd_years = sorted({t for _u, t in fd["keys"]})
        serial = S.wooldridge_serial(resid, fd["sample"], fd_years) if len(fd_years) >= 3 else None
        sp = specs[main_name]
        main_model = models[main_name]

        def refit(units: list[str], sp: dict[str, Any] = sp) -> dict[str, Any]:
            return strip(fit_model(panel, sp["y"], sp["xs"], sp["years"], units, geo, sp.get("kind", "twoway"),
                                   bootstrap=False, base_year=sp.get("base_year")))

        se_ratio = {x: round(main_model["coefficients"][x]["se_cr1"] / main_model["coefficients"][x]["se_conventional"], 2)
                    for x in sp["xs"] if main_model["coefficients"][x]["se_conventional"]}
        diagnostics[main_name] = {
            "wooldridge_fd_serial_correlation": {k: round(v, 3) for k, v in serial.items()} if serial else None,
            "fe_vs_fd": {x: {"fe": main_model["coefficients"][x]["estimate"], "fd": models[fd_name]["coefficients"][x]["estimate"]}
                         for x in sp["xs"]},
            "cluster_robust_over_conventional_se": se_ratio,
            **influence(main_model, panel, geo, refit),
        }

    # Across specifications: does the sign of each association survive changes of sample, definition and transform?
    concepts = {"income": ("log_income", "log_income_net", "dlog_income"), "completions_per_1000": ("completions_per_1000",),
                "population": ("log_population",), "rent": ("log_rent_m2", "dlog_rent_m2")}
    across: dict[str, Any] = {}
    every = {**{k: v for k, v in models.items()}, **{f"LD_{k}": v for k, v in long_diffs.items()}}
    for concept, names in concepts.items():
        rows = []
        for spec_name, m in every.items():
            if spec_name.startswith("D_twoway_rent_on_income"):
                continue  # rent as outcome, not a price regression
            for x in names:
                if x in m["coefficients"]:
                    wb = m["wild_cluster_bootstrap_nuts3"][x] if "wild_cluster_bootstrap_nuts3" in m else {}
                    rows.append({"spec": spec_name, "outcome": m["spec"]["y"], "estimate": m["coefficients"][x]["estimate"],
                                 "ci95_wild_bootstrap": [wb.get("ci95_low"), wb.get("ci95_high")]})
        pos = sum(1 for r in rows if r["estimate"] > 0)
        across[concept] = {"specifications": len(rows), "positive": pos, "negative": len(rows) - pos,
                           "excluding_zero": sum(1 for r in rows if r["ci95_wild_bootstrap"][0] is not None and
                                                 (r["ci95_wild_bootstrap"][0] > 0 or r["ci95_wild_bootstrap"][1] < 0)),
                           "estimates": rows}

    # ---- Verdicts: which within-municipality associations survive the checks ----
    def agreement(x: str, est: float) -> float:
        concept = next(c for c, names in concepts.items() if x in names)
        rows = across[concept]["estimates"]
        return sum(1 for r in rows if r["estimate"] * est > 0) / len(rows)

    def verdict(main_name: str, x: str) -> dict[str, Any]:
        m, d = models[main_name], diagnostics[main_name]
        est = m["coefficients"][x]["estimate"]
        wb = m["wild_cluster_bootstrap_nuts3"][x]
        share = (m["within_variance_share"] or {}).get(x)
        serial = d["wooldridge_fd_serial_correlation"]
        loo = d["leave_one_nuts3_out"][x]
        checks = {
            "regressor_keeps_at_least_1pct_of_variance_after_fixed_effects": share is not None and share >= 0.01,
            "wild_bootstrap_interval_excludes_zero": (wb["ci95_low"] > 0 or wb["ci95_high"] < 0) and wb["p_value_restricted"] < 0.05,
            "same_sign_in_first_differences": est * d["fe_vs_fd"][x]["fd"] > 0,
            "same_sign_leaving_out_any_nuts3": loo["min"] * loo["max"] > 0 and loo["min"] * est > 0,
            "same_sign_without_lisbon_porto_metro": d["without_lisbon_and_porto_metro"][x]["estimate"] * est > 0,
            "sign_shared_by_at_least_80pct_of_specifications": agreement(x, est) >= 0.8,
        }
        cautions = []
        if share is not None and share < 0.05:
            cautions.append("under 5% of the regressor's variance is within-municipality: measurement error in medians attenuates the estimate")
        if serial is not None and abs(serial["slope"] + 0.5) >= 0.15:
            cautions.append("first-difference residuals are serially correlated: errors are persistent, prefer the first-difference estimate")
        label = ("robust" if all(checks.values()) else
                 "fragile" if checks["wild_bootstrap_interval_excludes_zero"] else "not supported")
        return {"estimate": est, "within_variance_share": share, "checks": checks, "cautions": cautions, "verdict": label}

    verdicts = {f"{name}:{x}": verdict(name, x) for name in diagnostics for x in specs[name]["xs"]}

    # ---- National time series (exploratory; few observations, trending series) ----
    nat = national_series(args.data, args.date)
    national = {
        "note": "Correlations of annual growth rates; effective sample size from the lag-1 autocorrelations (Bartlett).",
        "2005_2024": national_correlations(nat, 2005, 2024),
        "2013_2024": national_correlations(nat, 2013, 2024),
        "series_years": {k: [min(v), max(v)] for k, v in nat.items()},
    }

    results: dict[str, Any] = {
        "layers": {
            "robust_descriptive": "levels, growth and rank correlations that hold across samples, definitions and NUTS levels",
            "model_dependent": "within-municipality associations from fixed-effects, first-difference and long-difference models",
            "requires_causal_identification": "any statement about what raises or lowers prices or rents",
        },
        "join_checks": join_checks,
        "audit": audit_rows,
        "samples": samples,
        "cross_section_municipalities": cross,
        "cross_section_nuts3": nuts3,
        "gross_rental_yield_pct": {"note": "12 x median new-lease rent per m2 / median sale price per m2; medians of different dwellings",
                                   "nuts3_falling_2020_2025": falling, "by_region": yields},
        "models": {k: strip(v) for k, v in models.items()},
        "long_differences": long_diffs,
        "diagnostics": diagnostics,
        "verdicts": verdicts,
        "completions_composition_check": {
            "note": "Median price of all sales vs existing dwellings only, same sample and years (2021-2024): if the completions "
                    "association vanishes for existing dwellings, it reflects new dwellings entering the sale mix",
            "all_dwellings": strip(models["A2_twoway_income_completions"])["coefficients"]["completions_per_1000"],
            "existing_dwellings": strip(models["A2_twoway_existing_prices"])["coefficients"]["completions_per_1000"],
            "municipalities": [models["A2_twoway_income_completions"]["municipalities"], models["A2_twoway_existing_prices"]["municipalities"]],
        },
        "across_specifications": across,
        "national_time_series": national,
    }
    args.results.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    with args.panel.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        cols = list(VARIABLES)
        writer.writerow(["dico", "municipality", "nuts3", "nuts2", "year", *cols, "population_basis", "flags"])
        for u in munis:
            for t in YEARS:
                row = panel.get((u, t), {})
                if not row:
                    continue
                writer.writerow([u, geo.names[geo.municipality[u]], geo.nuts3[u], geo.nuts2_of3[geo.nuts3[u]], t,
                                 *[(f"{row[c]:.6g}" if c in row else "") for c in cols],
                                 ("census-based" if t <= 2020 else "administrative-based") if "population" in row else "",
                                 ";".join(R.flags_for([u], t))])
    with args.audit.open("w", newline="", encoding="utf-8") as fh:
        audit_writer = csv.DictWriter(fh, fieldnames=list(audit_rows[0]), lineterminator="\n")
        audit_writer.writeheader()
        audit_writer.writerows(audit_rows)
    print(json.dumps({k: v["coefficients"] for k, v in results["models"].items() if k.startswith(("A1_twoway_income", "A2", "B_tw", "C_tw", "D_tw"))},
                     ensure_ascii=False, indent=1)[:4000])


if __name__ == "__main__":
    main()
