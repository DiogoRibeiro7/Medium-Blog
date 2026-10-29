"""Regional housing affordability: transaction prices against declared income, NUTS 2024.

Reads the pinned INE containers written by ``fetch_regional.py --set affordability``
(every response re-checked against its SHA-256), and builds a region-year panel of

* the median value per m2 of family dwellings sold (transactions; INE local
  housing-price statistics, 2022 method), taken from the 12-month window ending
  in the fourth quarter, i.e. the calendar year;
* the median declared gross income per tax household (IRS returns; INE local
  income statistics);

for Portugal, the 9 NUTS 2 regions, the 26 NUTS 3 regions and the 308
municipalities. Both are medians published by INE for each area; they are never
aggregated or averaged across areas here, so a region missing from INE's tables
is missing from the panel. The headline proxy is the number of square metres of
a median-priced dwelling that one year of median gross household income would
pay for, and its change over time. It is not the number of years needed to buy a
dwelling: it ignores taxes, savings, deposits, financing, dwelling size and the
fact that buyers are not a random draw of tax households.

Run:  python affordability.py      # writes results_affordability.json and affordability_panel.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import regional as R
from affordability_core import (
    Panel,
    annual_growth,
    dispersion,
    growth_comparison,
    m2_per_income_year,
    median,
    pearson,
    period_year,
    price_income_index,
    spearman,
)

HERE = Path(__file__).parent
DATA = HERE / "data"
SNAPSHOT_DATE = "2026-10-29"
CODES = {"0012234": "price", "0009490": "price_2018", "0012256": "appraisal", "0012712": "income",
         "0012741": "income_net", "0012749": "income_taxpayer", "0009763": "income_2013"}
FOCUS = {"1A0": "Grande Lisboa", "1B0": "Península de Setúbal", "11A": "Área Metropolitana do Porto", "PT": "Portugal"}
OLD_LISBON = "170"  # Área Metropolitana de Lisboa in NUTS 2013 = Grande Lisboa + Península de Setúbal
MAIN = (2019, 2024)  # 2022-method prices start in the 12 months to 2019 Q4; income ends in 2024
BASES = (2019, 2020, 2021)
MUNICIPALITY = 5  # INE's level number for municipalities (1 Portugal, 2 NUTS 1, 3 NUTS 2, 4 NUTS 3, 6 parishes)
Key = tuple[str, int, tuple[str, ...]]


@dataclass
class Series:
    """One INE indicator: values by (geographic code, year, codes of the other dimensions)."""

    code: str
    name: str
    title: str
    unit: str
    note: str
    last_update: str
    values: dict[Key, float] = field(default_factory=dict)
    unavailable: dict[Key, str] = field(default_factory=dict)
    levels: dict[str, int] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    names: dict[str, str] = field(default_factory=dict)
    dims: dict[str, dict[str, str]] = field(default_factory=dict)
    retrieved: list[str] = field(default_factory=list)

    def combo(self, dims: dict[str, str] | None = None) -> tuple[str, ...]:
        """The dimension-code tuple for a selection such as {"3": "H.1"} (every extra dimension must be named)."""
        dims = dims or {}
        if set(dims) != set(self.dims):
            raise ValueError(f"{self.code}: select exactly the dimensions {sorted(self.dims)}, got {sorted(dims)}")
        for n, value in dims.items():
            if value not in self.dims[n]:
                raise ValueError(f"{self.code}: dimension {n} has no category {value!r}")
        return tuple(dims[n] for n in sorted(dims))

    def get(self, dims: dict[str, str] | None = None) -> dict[tuple[str, int], float]:
        """(geographic code, year) -> value for one combination of the other dimensions."""
        wanted = self.combo(dims)
        return {(geo, year): v for (geo, year, combo), v in self.values.items() if combo == wanted}

    def missing(self, dims: dict[str, str] | None = None) -> dict[tuple[str, int], str]:
        """(geographic code, year) -> INE's reason, for cells published without a value."""
        wanted = self.combo(dims)
        return {(geo, year): why for (geo, year, combo), why in self.unavailable.items() if combo == wanted}

    def category(self, dim: str, name: str) -> str:
        """The code of the category called ``name`` in dimension ``dim``."""
        matches = [code for code, label in self.dims[dim].items() if label.strip().lower() == name.lower()]
        if len(matches) != 1:
            raise ValueError(f"{self.code}: dimension {dim} has {len(matches)} categories named {name!r}")
        return matches[0]

    def years(self) -> list[int]:
        """Years with at least one observation."""
        return sorted({year for _geo, year, _combo in self.values})


def parse_series(path: Path) -> Series:
    """Decode a container; quarterly series keep only fourth quarters (12 months to December)."""
    header, responses = R.read_container(path)
    meta = json.loads(responses[0]["body"])[0]
    if meta["IndicadorCod"] != header["indicator"]:
        raise ValueError(f"{path.name}: metadata is for {meta['IndicadorCod']}")
    cats = meta["Dimensoes"]["Categoria_Dim"][0]
    geo = [v[0] for k, v in cats.items() if k.startswith("Dim_Num2_")]
    other = sorted(d["dim_num"] for d in meta["Dimensoes"]["Descricao_Dim"] if d["dim_num"] not in ("1", "2"))
    series = Series(code=header["indicator"], name=header["name"], title=meta["IndicadorNome"],
                    unit=meta.get("UnidadeMedida", ""), note=meta.get("Nota", ""),
                    last_update=meta.get("DataUltimaAtualizacao", ""),
                    levels={g["categ_cod"]: int(g["categ_nivel"]) for g in geo},
                    order=[g["categ_cod"] for g in sorted(geo, key=lambda g: int(g["categ_ord"]))],
                    names={g["categ_cod"]: g["categ_dsg"] for g in geo},
                    dims={n: {v[0]["categ_cod"]: v[0]["categ_dsg"] for k, v in cats.items() if k.startswith(f"Dim_Num{n}_")}
                          for n in other})
    # Data rows write category codes without the dots used in the metadata (H1 for H.1, H12 for H.1.2).
    undotted = {n: {code.replace(".", ""): code for code in series.dims[n]} for n in other}
    for n in other:
        if len(undotted[n]) != len(series.dims[n]):
            raise ValueError(f"{path.name}: dimension {n} codes collide without dots")

    def category(n: str, code: str) -> str:
        if code in series.dims[n]:
            return code
        if code in undotted[n]:
            return undotted[n][code]
        raise ValueError(f"{path.name}: unknown category {code!r} in dimension {n}")

    for response in responses[1:]:
        item = json.loads(response["body"])[0]
        if item["IndicadorCod"] != series.code or "Verdadeiro" not in item.get("Sucesso", {}):
            raise ValueError(f"{path.name}: INE reported a failure")
        series.retrieved.append(response["retrieved_at_utc"])
        for label, rows in item["Dados"].items():
            year, quarter = period_year(label)
            if quarter not in (None, 4):
                raise ValueError(f"{path.name}: unexpected quarter in {label}")
            for row in rows:
                if row["geocod"] not in series.levels:
                    raise ValueError(f"{path.name}: unknown geographic code {row['geocod']}")
                combo = tuple(category(n, row[f"dim_{n}"]) for n in other)
                key = (row["geocod"], year, combo)
                if key in series.values or key in series.unavailable:
                    raise ValueError(f"{path.name}: duplicate {key}")
                if "valor" in row and row["valor"] not in ("", None):
                    value = float(row["valor"])
                    if value <= 0:
                        raise ValueError(f"{path.name}: non-positive value {key}")
                    series.values[key] = value
                else:
                    series.unavailable[key] = row.get("sinal_conv_desc") or row.get("sinal_conv") or "no value"
    return series


def load(folder: Path, date: str) -> dict[str, Series]:
    """All pinned affordability containers by short name."""
    return {label: parse_series(folder / f"ine_{code}_{date}.jsonl.gz") for code, label in CODES.items()}


def as_source(series: Series, dims: dict[str, str] | None = None) -> R.Source:
    """One slice of a series in the regional module's format (for its hierarchy and correspondence checks)."""
    values = series.get(dims)
    return R.Source(code=series.code, name=series.name, title=series.title, last_update=series.last_update,
                    note=series.note, values=values, unavailable=series.missing(dims), geo_names=series.names,
                    geo_levels=series.levels, geo_order=series.order, years=sorted({y for _g, y in values}))


def on_nuts2024(series: Series, geo: R.Geography, dims: dict[str, str] | None = None) -> tuple[Panel, dict[tuple[str, int], str]]:
    """Values and unavailable cells keyed by NUTS 2024 code: Portugal, NUTS 2, NUTS 3 and municipalities.

    Regions keep the code INE published; municipalities are matched by their DICO
    code, so a source using another code list for the same municipalities lines
    up. NUTS 1 and parishes are dropped. A regional code that is not in NUTS 2024
    is an error: this function is only for NUTS 2024 sources.
    """
    regions = {"PT", *geo.members2, *geo.members3}

    def key(code: str) -> str | None:
        level = series.levels[code]
        if level == MUNICIPALITY:
            dico = code[-4:]
            if dico not in geo.municipality:
                raise ValueError(f"{series.code}: municipality {code} is not in NUTS 2024")
            return geo.municipality[dico]
        if level in (1, 3, 4):
            if code not in regions:
                raise ValueError(f"{series.code}: region {code} is not in NUTS 2024")
            return code
        return None

    values = {(k, y): v for (code, y), v in series.get(dims).items() if (k := key(code)) is not None}
    missing = {(k, y): why for (code, y), why in series.missing(dims).items() if (k := key(code)) is not None}
    return values, missing


def compare_regions(price: Panel, income: Panel, regions: list[str], start: int, end: int) -> dict[str, Any]:
    """How many regions saw prices outpace incomes, and how the two growth rates co-move across regions."""
    rows = {r: c for r in regions if (c := growth_comparison(price, income, r, start, end))}
    if len(rows) < 3:
        return {"regions": len(rows)}
    gaps = {r: c["gap_log_points"] for r, c in rows.items()}
    p = [rows[r]["price_log_points"] for r in sorted(rows)]
    y = [rows[r]["income_log_points"] for r in sorted(rows)]
    return {
        "regions": len(rows),
        "regions_without_both_years": len(regions) - len(rows),
        "price_outpaced_income": sum(1 for g in gaps.values() if g > 0),
        "share_price_outpaced_income": round(sum(1 for g in gaps.values() if g > 0) / len(gaps), 3),
        "median_price_annual_pct": round(median(c["price_annual_pct"] for c in rows.values()), 2),
        "median_income_annual_pct": round(median(c["income_annual_pct"] for c in rows.values()), 2),
        "median_gap_log_points": round(median(gaps.values()), 1),
        "min_gap": _extreme(gaps, min), "max_gap": _extreme(gaps, max),
        "pearson_price_vs_income_log_points": round(pearson(p, y), 3),
        "spearman_price_vs_income_log_points": round(spearman(dict(zip(sorted(rows), p, strict=True)),
                                                              dict(zip(sorted(rows), y, strict=True))), 3),
    }


def _extreme(values: dict[str, float], pick: Any) -> dict[str, Any]:
    region = pick(values, key=lambda r: values[r])
    return {"region": region, "gap_log_points": round(values[region], 1)}


def rounded(d: dict[str, float] | None, digits: int = 2) -> dict[str, float] | None:
    """Round every value of a flat dictionary (None passes through)."""
    return {k: round(v, digits) for k, v in d.items()} if d else None


def method_overlap(old: Panel, new: Panel, pairs: list[tuple[str, str]], years: list[int]) -> dict[str, Any]:
    """Relative differences between two series over (old code, new code) pairs in shared years."""
    diffs = [(o, n, y, 100 * (new[(n, y)] / old[(o, y)] - 1)) for o, n in pairs for y in years
             if (o, y) in old and (n, y) in new]
    if not diffs:
        return {"cells": 0}
    absolute = [abs(d[3]) for d in diffs]
    worst = max(diffs, key=lambda d: abs(d[3]))
    return {"cells": len(diffs), "identical": sum(1 for a in absolute if a == 0),
            "median_abs_diff_pct": round(median(absolute), 2),
            "share_within_5pct": round(sum(1 for a in absolute if a <= 5) / len(absolute), 3),
            "largest": {"old": worst[0], "new": worst[1], "year": worst[2], "diff_pct": round(worst[3], 1)}}


def main() -> None:  # noqa: C901 - one linear pipeline, kept in one place for auditability
    """Build, check and export the affordability panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--date", default=SNAPSHOT_DATE)
    parser.add_argument("--results", type=Path, default=HERE / "results_affordability.json")
    parser.add_argument("--panel", type=Path, default=HERE / "affordability_panel.csv")
    args = parser.parse_args()
    src = load(args.data, args.date)

    # Categories, read by name so a recoded list fails loudly instead of silently selecting something else.
    cat = {"total": src["price"].category("3", "Total"), "new": src["price"].category("3", "Novos"),
           "existing": src["price"].category("3", "Existentes")}
    appraisal_dims = {n: src["appraisal"].category(n, "Total") for n in src["appraisal"].dims}

    geo = R.nuts2024_geography(as_source(src["price"], {"3": cat["total"]}))
    for label in ("income", "income_net", "income_taxpayer"):
        other = R.nuts2024_geography(as_source(src[label]))
        if other.nuts3 != geo.nuts3:
            raise ValueError(f"{label}: NUTS 2024 municipality assignment differs from the price source")
    nuts2 = [c for c in src["price"].order if c in geo.members2]
    nuts3 = [c for c in src["price"].order if c in geo.members3]
    munis = [geo.municipality[c[-4:]] for c in src["price"].order if src["price"].levels[c] == MUNICIPALITY]
    units = ["PT", *nuts2, *nuts3, *munis]
    level = {"PT": "country", **{r: "NUTS2" for r in nuts2}, **{r: "NUTS3" for r in nuts3}, **{m: "municipality" for m in munis}}
    nuts3_of = {**{geo.municipality[d]: n for d, n in geo.nuts3.items()}, **{r: r for r in nuts3}}
    nuts2_of = {**{m: geo.nuts2_of3[n] for m, n in nuts3_of.items()}, **{r: r for r in nuts2}}

    price, price_na = on_nuts2024(src["price"], geo, {"3": cat["total"]})
    price_new, _ = on_nuts2024(src["price"], geo, {"3": cat["new"]})
    price_existing, _ = on_nuts2024(src["price"], geo, {"3": cat["existing"]})
    appraisal, _ = on_nuts2024(src["appraisal"], geo, appraisal_dims)
    income, income_na = on_nuts2024(src["income"], geo)
    income_net, _ = on_nuts2024(src["income_net"], geo)
    income_taxpayer, _ = on_nuts2024(src["income_taxpayer"], geo)
    m2 = m2_per_income_year(price, income)
    price_years = sorted({y for _r, y in price})
    income_years = sorted({y for _r, y in income})
    years = [y for y in price_years if y in income_years]
    if (years[0], years[-1]) != MAIN or years != list(range(MAIN[0], MAIN[1] + 1)):
        raise ValueError(f"Common price-income years are {years}, expected {MAIN[0]}-{MAIN[1]}")
    a, b = MAIN

    # Checks: NUTS 2 regions with a single NUTS 3 region must carry identical medians, and every
    # NUTS 3 region and Portugal must be complete over the main window.
    singletons = [(n2, n3) for n3, n2 in geo.nuts2_of3.items() if [k for k, v in geo.nuts2_of3.items() if v == n2] == [n3]]
    for n2, n3 in singletons:
        for name, s in (("price", price), ("income", income)):
            for y in years:
                if s.get((n2, y)) != s.get((n3, y)):
                    raise ValueError(f"{name}: {n2} and {n3} differ in {y}")
    for name, s, ys in (("price", price, price_years), ("income", income, years)):
        gaps = [(r, y) for r in ["PT", *nuts2, *nuts3] for y in ys if (r, y) not in s]
        if gaps:
            raise ValueError(f"{name}: regional cells missing {gaps[:5]}")

    # Region summaries.
    summary: dict[str, Any] = {}
    for r in units:
        entry: dict[str, Any] = {"name": geo.names.get(r, r), "level": level[r]}
        if r != "PT":
            entry["nuts3"], entry["nuts2"] = nuts3_of.get(r), nuts2_of.get(r)
        entry[f"price_m2_{a}"], entry[f"price_m2_{b}"] = price.get((r, a)), price.get((r, b))
        entry["price_m2_2025"] = price.get((r, 2025))
        entry[f"income_{a}"], entry[f"income_{b}"] = income.get((r, a)), income.get((r, b))
        entry[f"m2_per_income_year_{a}"] = round(m2[(r, a)], 2) if (r, a) in m2 else None
        entry[f"m2_per_income_year_{b}"] = round(m2[(r, b)], 2) if (r, b) in m2 else None
        entry[f"growth_{a}_{b}"] = rounded(growth_comparison(price, income, r, a, b))
        g25 = annual_growth(price, r, a, 2025)  # price only; income for 2025 is not yet published
        entry[f"price_annual_pct_{a}_2025"] = round(g25, 2) if g25 is not None else None
        entry[f"price_income_index_{b}"] = {str(base): round(price_income_index(price, income, r, base, [b])[b], 1)
                                            for base in BASES if (r, base) in price and (r, base) in income
                                            and (r, b) in price and (r, b) in income}
        summary[r] = entry
    for r in units:
        own, nat = summary[r][f"m2_per_income_year_{b}"], summary["PT"][f"m2_per_income_year_{b}"]
        summary[r][f"m2_per_income_year_{b}_relative_to_portugal"] = round(own / nat, 3) if own and nat else None

    # Price growth against income growth across regions.
    balanced = [m for m in munis if all((m, y) in price and (m, y) in income for y in years)]
    comparison = {
        "nuts3": compare_regions(price, income, nuts3, a, b),
        "nuts2": compare_regions(price, income, nuts2, a, b),
        "municipalities_with_both_end_years": compare_regions(price, income, munis, a, b),
        "municipalities_balanced_every_year": compare_regions(price, income, balanced, a, b),
        "by_year_pair_nuts3": {f"{y0}_{y0 + 1}": compare_regions(price, income, nuts3, y0, y0 + 1) for y0 in years[:-1]},
    }

    def spread(s: Panel, regions: list[str], ys: list[int]) -> dict[str, Any]:
        return {str(y): rounded(dispersion([s[(r, y)] for r in regions]), 3) for y in ys
                if all((r, y) in s for r in regions)}

    disparities = {
        "nuts3_price_m2": spread(price, nuts3, price_years),
        "nuts3_income": spread(income, nuts3, years),
        "nuts3_m2_per_income_year": spread(m2, nuts3, years),
        "balanced_municipalities_m2_per_income_year": spread(m2, balanced, years),
    }

    # Sensitivity 1: base year of the normalised index.
    index_paths = {str(base): {r: {str(y): round(v, 1) for y, v in price_income_index(price, income, r, base, years).items()}
                               for r in FOCUS} for base in BASES}
    at_end = {base: {r: price_income_index(price, income, r, base, [b])[b] for r in nuts3} for base in BASES}
    baseline = {str(base): {"nuts3_index_above_100": sum(1 for v in at_end[base].values() if v > 100),
                            "nuts3_median_index": round(median(at_end[base].values()), 1),
                            "spearman_with_base_2019": round(spearman(at_end[base], at_end[BASES[0]]), 3),
                            "focus": {r: round(price_income_index(price, income, r, base, [b])[b], 1) for r in FOCUS}}
                for base in BASES}

    # Sensitivity 2: which price and which income.
    prices = {"total": price, "existing": price_existing, "new": price_new}
    incomes = {"gross_per_household": income, "net_per_household": income_net, "gross_per_taxpayer": income_taxpayer}
    head_gap = {r: c["gap_log_points"] for r in nuts3 if (c := growth_comparison(price, income, r, a, b))}
    head_m2 = {r: m2[(r, b)] for r in nuts3}
    definitions: dict[str, Any] = {}
    for pn, p_series in prices.items():
        for yn, y_series in incomes.items():
            comp = compare_regions(p_series, y_series, nuts3, a, b)
            gap = {r: c["gap_log_points"] for r in nuts3 if (c := growth_comparison(p_series, y_series, r, a, b))}
            level_b = m2_per_income_year(p_series, y_series)
            m2_b = {r: level_b[(r, b)] for r in nuts3 if (r, b) in level_b}
            definitions[f"{pn}_price__{yn}_income"] = {
                **{k: comp.get(k) for k in ("regions", "price_outpaced_income", "share_price_outpaced_income",
                                            "median_gap_log_points")},
                "spearman_gap_with_headline": round(spearman(gap, head_gap), 3) if len(gap) >= 3 else None,
                "spearman_m2_per_income_year_with_headline": round(spearman(m2_b, head_m2), 3) if len(m2_b) >= 3 else None,
                "focus_gap_log_points": {r: round(c["gap_log_points"], 1) if (c := growth_comparison(p_series, y_series, r, a, b))
                                         else None for r in FOCUS},
            }

    # Sensitivity 3: transactions against bank appraisals (a valuation for mortgage credit, not a price paid).
    appraisal_regions = [r for r in nuts3 if all((r, y) in appraisal and (r, y) in price for y in (a, b))]
    unit_kind = "nuts3"
    if len(appraisal_regions) < 10:  # appraisal geography without NUTS 3 rows: compare municipalities instead
        appraisal_regions = [m for m in munis if all((m, y) in appraisal and (m, y) in price for y in (a, b))]
        unit_kind = "municipalities"
    appraisal_cmp: dict[str, Any] = {"units": unit_kind, "n": len(appraisal_regions)}
    if len(appraisal_regions) >= 3:
        t_growth = {r: annual_growth(price, r, a, b) or 0.0 for r in appraisal_regions}
        v_growth = {r: annual_growth(appraisal, r, a, b) or 0.0 for r in appraisal_regions}
        appraisal_cmp.update({
            "median_transaction_price_annual_pct": round(median(t_growth.values()), 2),
            "median_appraisal_annual_pct": round(median(v_growth.values()), 2),
            "pearson_annual_growth": round(pearson([t_growth[r] for r in appraisal_regions],
                                                   [v_growth[r] for r in appraisal_regions]), 3),
            f"median_appraisal_over_transaction_{b}": round(median(appraisal[(r, b)] / price[(r, b)] for r in appraisal_regions), 3),
            "share_price_outpaced_income_with_appraisal": compare_regions(appraisal, income, appraisal_regions, a, b)
            .get("share_price_outpaced_income"),
            "share_price_outpaced_income_with_transactions_same_units": compare_regions(price, income, appraisal_regions, a, b)
            .get("share_price_outpaced_income"),
            "focus": {r: {"transactions_annual_pct": round(annual_growth(price, r, a, b) or 0.0, 2),
                          "appraisal_annual_pct": round(annual_growth(appraisal, r, a, b) or 0.0, 2)}
                      for r in FOCUS if all((r, y) in appraisal and (r, y) in price for y in (a, b))},
        })

    # Sensitivity 4: price method (2018 vs 2022) and NUTS version (2013 vs 2024), and an earlier window.
    old_price = as_source(src["price_2018"])
    old_income = as_source(src["income_2013"])
    groups = R.correspondence(old_price, geo)
    unchanged = [(o[0], n[0]) for o, n, _m in groups if len(o) == 1 and len(n) == 1]
    old_muni = R.municipalities(old_price)
    muni_pairs = [(code, geo.municipality[d]) for d, code in old_muni.items() if d in geo.municipality]
    overlap_years = sorted(set(old_price.years) & set(price_years))
    old_income_muni = R.municipalities(old_income)
    income_pairs = [(code, geo.municipality[d]) for d, code in old_income_muni.items() if d in geo.municipality]
    income_overlap = sorted(set(old_income.years) & set(income_years))
    early_years = sorted(set(old_price.years) & set(old_income.years))
    e0, e1 = early_years[0], early_years[-1]
    old_nuts3 = [c for c in old_price.geo_order if old_price.geo_levels[c] == MUNICIPALITY - 1]
    early = {
        "window": [e0, e1],
        "note": "2018-method prices with NUTS 2013 income; reported beside the main window, never spliced onto it",
        "nuts3_nuts2013": compare_regions(old_price.values, old_income.values, old_nuts3, e0, e1),
        "portugal": rounded(growth_comparison(old_price.values, old_income.values, "PT", e0, e1)),
        "area_metropolitana_de_lisboa": rounded(growth_comparison(old_price.values, old_income.values, OLD_LISBON, e0, e1)),
        "area_metropolitana_do_porto": rounded(growth_comparison(old_price.values, old_income.values, "11A", e0, e1))
        if ("11A", e0) in old_price.values else None,
    }
    lisbon_old_m2 = m2_per_income_year(old_price.values, old_income.values)
    lisbon = {
        "grande_lisboa": summary["1A0"][f"growth_{a}_{b}"], "peninsula_de_setubal": summary["1B0"][f"growth_{a}_{b}"],
        "area_metropolitana_de_lisboa_nuts2013_m2_per_income_year": {str(y): round(lisbon_old_m2[(OLD_LISBON, y)], 2)
                                                                     for y in early_years if (OLD_LISBON, y) in lisbon_old_m2},
        "grande_lisboa_and_setubal_m2_per_income_year_overlap": {str(y): {"1A0": round(m2[("1A0", y)], 2), "1B0": round(m2[("1B0", y)], 2)}
                                                                 for y in overlap_years if ("1A0", y) in m2},
        "note": "Medians cannot be combined: the former metropolitan area exists only in the NUTS 2013 series (to 2020 for prices, 2021 for income)",
    }
    methods = {
        "price_2018_vs_2022_method_municipalities": method_overlap(old_price.values, price, muni_pairs, overlap_years),
        "price_2018_vs_2022_method_unchanged_nuts3": method_overlap(old_price.values, price, unchanged, overlap_years),
        "price_overlap_years": overlap_years,
        "income_nuts2013_vs_nuts2024_municipalities": method_overlap(old_income.values, income, income_pairs, income_overlap),
        "income_nuts2013_vs_nuts2024_unchanged_nuts3": method_overlap(old_income.values, income, unchanged, income_overlap),
        "income_overlap_years": income_overlap,
        "unchanged_nuts3_regions": [{"nuts2013": o, "nuts2024": n} for o, n in unchanged],
        "changed_groups": [{"nuts2013": o, "nuts2024": n} for o, n, _m in groups if len(o) != 1 or len(n) != 1],
    }

    # Sensitivity 5: geographical coverage.
    def reasons(missing: dict[tuple[str, int], str], y: int) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for (r, yy), why in missing.items():
            if yy == y and level.get(r) == "municipality":
                counts[why] += 1
        return dict(counts)

    coverage = {
        "municipalities": len(munis),
        "by_year": {str(y): {"price_published": sum(1 for m in munis if (m, y) in price),
                             "income_published": sum(1 for m in munis if (m, y) in income),
                             "both": sum(1 for m in munis if (m, y) in price and (m, y) in income),
                             "price_unavailable_reasons": reasons(price_na, y),
                             "income_unavailable_reasons": reasons(income_na, y)} for y in price_years},
        "balanced_municipalities": len(balanced),
        "balanced_by_nuts2": {r: sum(1 for m in balanced if nuts2_of[m] == r) for r in nuts2},
        "municipalities_by_nuts2": {r: sum(1 for m in munis if nuts2_of[m] == r) for r in nuts2},
    }

    sources = {label: {"indicator": s.code, "title": s.title, "unit": s.unit, "last_update": s.last_update,
                       "ine_note": s.note, "retrieved_first": min(s.retrieved), "retrieved_last": max(s.retrieved),
                       "years": s.years(), "dimensions": s.dims,
                       "cells_with_value": len(s.values), "cells_unavailable": len(s.unavailable)}
               for label, s in src.items()}
    results: dict[str, Any] = {
        "definitions": {
            "price": "Median value per m2 of family dwellings sold in the 12 months ending in Q4 of the year (INE, 2022 method)",
            "income": "Median declared gross income per tax household (IRS Modelo 3; INE local income statistics)",
            "m2_per_income_year": "Median income / median price per m2: floor area that one year of median gross "
                                  "household income would pay for at the median price. A ratio of medians from two "
                                  "different populations; not the years needed to buy a dwelling",
            "price_income_index": "(P_t / P_base) / (Y_t / Y_base) * 100: relative change only, not a level",
            "gap_log_points": "100 * ln(P_end/P_start) - 100 * ln(Y_end/Y_start); positive = prices rose faster",
        },
        "main_window": list(MAIN),
        "geography": {"municipalities": len(munis), "nuts3": len(nuts3), "nuts2": len(nuts2),
                      "nuts2_equal_to_one_nuts3": [{"nuts2": n2, "nuts3": n3} for n2, n3 in singletons]},
        "sources": sources,
        "price_vs_income_growth": comparison,
        "disparities": disparities,
        "sensitivity_base_year": {"index_at_end_year": baseline, "focus_index_paths": index_paths},
        "sensitivity_definitions": definitions,
        "sensitivity_transactions_vs_appraisal": appraisal_cmp,
        "sensitivity_method_and_nuts_version": methods,
        "earlier_window_2018_method": early,
        "lisbon": lisbon,
        "coverage": coverage,
        "regions": summary,
    }
    args.results.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    with args.panel.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["region_code", "region_name", "level", "nuts3", "nuts2", "year", "price_m2_total",
                         "price_m2_existing", "price_m2_new", "appraisal_m2", "income_gross_per_household",
                         "income_net_per_household", "income_gross_per_taxpayer", "m2_per_income_year",
                         "price_unavailable", "income_unavailable"])
        for r in units:
            for y in price_years:
                def fmt(s: Panel, digits: int = 0) -> str:
                    return f"{s[(r, y)]:.{digits}f}" if (r, y) in s else ""
                writer.writerow([r, summary[r]["name"], level[r], nuts3_of.get(r, ""), nuts2_of.get(r, ""), y,
                                 fmt(price), fmt(price_existing), fmt(price_new), fmt(appraisal), fmt(income),
                                 fmt(income_net), fmt(income_taxpayer), fmt(m2, 2),
                                 price_na.get((r, y), ""), income_na.get((r, y), "")])
    print(json.dumps({"nuts3": comparison["nuts3"], "focus": {r: summary[r][f"growth_{a}_{b}"] for r in FOCUS}},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
