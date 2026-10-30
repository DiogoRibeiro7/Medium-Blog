"""Regional panel of population, completed and permitted dwellings, and households (NUTS 2024).

Reads the pinned INE containers written by ``fetch_regional.py``, checks every raw
response against its recorded SHA-256, and builds a region-year panel on one
geography: NUTS 2024 as published by INE. Series published under older NUTS
versions are regrouped through municipalities, which are identical across
versions and identified by their four-digit DICO code; the regrouping is
checked against each source's own regional totals. Nothing is downloaded here,
no missing value is filled, and nothing in the output measures housing need.

Run:  python regional.py            # writes results_regional.json and regional_panel.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from analysis import COMPLETIONS_FILE, parse_completed_dwellings, parse_demographic_totals, read_container
from regional_core import (
    coefficient_of_variation,
    coverage,
    growth,
    longest_complete_span,
    per_thousand,
    period_mean,
    regroup,
)

HERE = Path(__file__).parent
DATA = HERE / "data"
SNAPSHOT_DATE = "2026-10-29"

LISBON_UNION = "AML"  # Grande Lisboa + Península de Setúbal = the former Área Metropolitana de Lisboa
FOCUS = {"1A0": "Grande Lisboa", "1B0": "Península de Setúbal", LISBON_UNION: "Grande Lisboa + Península de Setúbal",
         "11A": "Área Metropolitana do Porto", "PT": "Portugal"}

# Municipalities named in INE's notes on under-reported construction data (DICO codes).
AMADORA, LISBOA, SEIA = "1115", "1106", "0912"


@dataclass
class Source:
    """One INE indicator as pinned: values by (geographic code, year) and what INE says about them."""

    code: str
    name: str
    title: str
    last_update: str
    note: str
    values: dict[tuple[str, int], float] = field(default_factory=dict)
    unavailable: dict[tuple[str, int], str] = field(default_factory=dict)
    geo_names: dict[str, str] = field(default_factory=dict)
    geo_levels: dict[str, int] = field(default_factory=dict)
    geo_order: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)


def parse_source(path: Path) -> Source:
    """Decode one container into a ``Source``, validating codes, filters and values."""
    header, responses = read_container(path)
    meta = json.loads(responses[0]["body"])[0]
    code = header["indicator"]
    if meta["IndicadorCod"] != code:
        raise ValueError(f"{path.name}: metadata is for {meta['IndicadorCod']}")
    categories = meta["Dimensoes"]["Categoria_Dim"][0]
    geo = [v[0] for k, v in categories.items() if k.startswith("Dim_Num2_")]
    source = Source(code=code, name=header["name"], title=meta["IndicadorNome"],
                    last_update=meta.get("DataUltimaAtualizacao", ""), note=meta.get("Nota", ""),
                    geo_names={g["categ_cod"]: g["categ_dsg"] for g in geo},
                    geo_levels={g["categ_cod"]: int(g["categ_nivel"]) for g in geo},
                    geo_order=[g["categ_cod"] for g in sorted(geo, key=lambda g: int(g["categ_ord"]))],
                    years=list(header["years"]))
    filters = {f"dim_{k[3:]}": v for k, v in header["filters"].items() if k.startswith("Dim") and k != "Dim2"}
    for response in responses[1:]:
        item = json.loads(response["body"])[0]
        if item["IndicadorCod"] != code or "Verdadeiro" not in item.get("Sucesso", {}):
            raise ValueError(f"{path.name}: INE reported a failure for {response['url'][:120]}")
        source.retrieved.append(response["retrieved_at_utc"])
        for year_text, rows in item["Dados"].items():
            year = int(year_text)
            for row in rows:
                if any(row.get(dim) != value for dim, value in filters.items()):
                    raise ValueError(f"{path.name}: row outside the requested filters: {row}")
                key = (row["geocod"], year)
                if key in source.values or key in source.unavailable:
                    raise ValueError(f"{path.name}: duplicate observation {key}")
                if "valor" in row:
                    value = float(row["valor"])
                    if value < 0:
                        raise ValueError(f"{path.name}: negative value {key}")
                    source.values[key] = value
                else:
                    source.unavailable[key] = row.get("sinal_conv_desc") or row.get("sinal_conv") or "no value"
    return source


def municipalities(source: Source) -> dict[str, str]:
    """DICO code -> the source's own municipality code (level 5, or level 4 in the census series)."""
    level = max(lvl for lvl in source.geo_levels.values() if lvl <= 5)
    if source.code == "0014406":
        level = 4
    result: dict[str, str] = {}
    for geocode, lvl in source.geo_levels.items():
        if lvl == level:
            dico = geocode[-4:]
            if dico in result:
                raise ValueError(f"{source.code}: DICO {dico} appears twice")
            result[dico] = geocode
    return result


@dataclass
class Geography:
    """The NUTS 2024 hierarchy as published by INE, keyed by DICO."""

    municipality: dict[str, str]          # DICO -> NUTS 2024 municipality code
    nuts3: dict[str, str]                 # DICO -> NUTS 3 code
    names: dict[str, str]                 # any code -> name
    members3: dict[str, list[str]]        # NUTS 3 code -> DICO list
    members2: dict[str, list[str]]        # NUTS 2 code -> DICO list
    nuts2_of3: dict[str, str]             # NUTS 3 -> NUTS 2

    def regions(self) -> dict[str, list[str]]:
        """Every reported region (NUTS 3, NUTS 2, the Lisbon union and Portugal) -> DICO members."""
        out: dict[str, list[str]] = {**self.members3, **self.members2}
        out[LISBON_UNION] = sorted(self.members3["1A0"] + self.members3["1B0"])
        out["PT"] = sorted(self.municipality)
        return out


def nuts2024_geography(source: Source) -> Geography:
    """Build the hierarchy from a NUTS 2024 source: 308 municipalities, 26 NUTS 3, 9 NUTS 2."""
    munis = [g for g in source.geo_order if source.geo_levels[g] == 5]
    nuts3 = sorted(g for g, lvl in source.geo_levels.items() if lvl == 4)
    nuts2 = sorted(g for g, lvl in source.geo_levels.items() if lvl == 3)
    if (len(munis), len(nuts3), len(nuts2)) != (308, 26, 9):
        raise ValueError(f"Unexpected NUTS 2024 shape: {len(munis)}, {len(nuts3)}, {len(nuts2)}")
    parent3, parent2 = parents_at_level(source, 4), parents_at_level(source, 3)
    municipality = {g[-4:]: g for g in munis}
    if len(municipality) != 308:
        raise ValueError("Municipality DICO codes are not unique")
    nuts3_of = {g[-4:]: parent3[g] for g in munis}
    nuts2_of3 = {c: parent2[c] for c in nuts3}
    members3: dict[str, list[str]] = defaultdict(list)
    members2: dict[str, list[str]] = defaultdict(list)
    for dico, region in nuts3_of.items():
        members3[region].append(dico)
        members2[nuts2_of3[region]].append(dico)
    if set(members3) != set(nuts3) or set(members2) != set(nuts2):
        raise ValueError("Municipalities do not cover every published NUTS 3 / NUTS 2 region")
    names = dict(source.geo_names)
    names[LISBON_UNION] = FOCUS[LISBON_UNION]
    return Geography(municipality, nuts3_of, names, {k: sorted(v) for k, v in members3.items()},
                     {k: sorted(v) for k, v in members2.items()}, nuts2_of3)


def by_dico(source: Source) -> dict[tuple[str, int], float]:
    """Municipal values of a source keyed by (DICO, year)."""
    codes = municipalities(source)
    return {(dico, year): source.values[(code, year)]
            for dico, code in codes.items() for year in source.years if (code, year) in source.values}


def regional_series(source: Source, geo: Geography) -> tuple[dict[tuple[str, int], float], dict[tuple[str, int], list[str]]]:
    """Regroup a source's municipal values into every NUTS 2024 region; gaps are reported, not filled."""
    values = {key: value for key, value in by_dico(source).items() if key[0] in geo.municipality}
    return regroup(values, geo.regions(), source.years)


def check_dico_universe(source: Source, geo: Geography) -> dict[str, Any]:
    """The source's municipalities against the 308 NUTS 2024 municipalities (by DICO)."""
    codes = municipalities(source)
    extra = sorted(set(codes) - set(geo.municipality))
    extra_with_values = sorted(d for d in extra if any((codes[d], y) in source.values for y in source.years))
    missing = sorted(set(geo.municipality) - set(codes))
    if missing or extra_with_values:
        raise ValueError(f"{source.code}: municipalities missing {missing} or extra with values {extra_with_values}")
    renamed = {d: (source.geo_names[codes[d]], geo.names[geo.municipality[d]]) for d in geo.municipality
               if source.geo_names[codes[d]] != geo.names[geo.municipality[d]]}
    return {"municipalities_matched": len(geo.municipality), "extra_historical_codes_without_values": len(extra),
            "names_that_differ": renamed}


def parents_at_level(source: Source, level: int) -> dict[str, str]:
    """Each category's ancestor at ``level``, read from INE's hierarchical category order."""
    current: dict[int, str] = {}
    result: dict[str, str] = {}
    for code in source.geo_order:
        lvl = source.geo_levels[code]
        current[lvl] = code
        for deeper in [k for k in current if k > lvl]:
            del current[deeper]
        if lvl > level and level in current:
            result[code] = current[level]
    return result


def correspondence(source: Source, geo: Geography) -> list[tuple[list[str], list[str], list[str]]]:
    """Groups of the source's own regions and NUTS 2024 NUTS 3 regions that cover the same municipalities.

    The source regions are the level directly above municipalities (NUTS 3 in the
    NUTS 2002 and 2013 series, districts in the historical census series). Regions
    are linked when they share a municipality; each connected group is returned as
    (source regions, NUTS 2024 regions, DICO members). A one-to-one group means the
    region did not change.
    """
    codes = municipalities(source)
    level = source.geo_levels[next(iter(codes.values()))] - 1
    parent_of = parents_at_level(source, level)
    own_of = {dico: parent_of[code] for dico, code in codes.items() if dico in geo.municipality}
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for dico, own in own_of.items():
        parent[find("old:" + own)] = find("new:" + geo.nuts3[dico])
    groups: dict[str, list[str]] = defaultdict(list)
    for dico in own_of:
        groups[find("new:" + geo.nuts3[dico])].append(dico)
    result = []
    for members in groups.values():
        olds = sorted({own_of[d] for d in members})
        news = sorted({geo.nuts3[d] for d in members})
        result.append((olds, news, sorted(members)))
    return sorted(result, key=lambda g: g[1])


def check_published_totals(source: Source, geo: Geography) -> dict[str, Any]:
    """Regrouped municipal sums against the source's own published regional and national totals."""
    dico_values = by_dico(source)
    groups = correspondence(source, geo)
    mismatches = []
    for olds, news, members in groups:
        for year in source.years:
            published = [source.values.get((o, year)) for o in olds]
            regrouped = [dico_values.get((d, year)) for d in members]
            if any(v is None for v in published) or any(v is None for v in regrouped):
                continue
            p = sum(v for v in published if v is not None)
            r = sum(v for v in regrouped if v is not None)
            if abs(p - r) > 0.5:
                mismatches.append({"source_regions": olds, "year": year, "published": p, "regrouped": r})
    national = []
    for year in source.years:
        if ("PT", year) not in source.values:
            continue
        total = sum(v for (dico, y), v in dico_values.items() if y == year and dico in geo.municipality)
        if abs(source.values[("PT", year)] - total) > 0.5:
            national.append({"year": year, "published": source.values[("PT", year)], "regrouped": total})
    return {
        "groups": [{"source_regions": o, "nuts2024_regions": n, "municipalities": len(m)} for o, n, m in groups],
        "changed_groups": [{"source_regions": o, "nuts2024_regions": n} for o, n, _ in groups if len(o) != 1 or len(n) != 1],
        "region_mismatches": mismatches,
        "national_mismatches": national,
    }


def overlap(a: dict[tuple[str, int], float], b: dict[tuple[str, int], float], regions: list[str],
            years: list[int]) -> dict[str, Any]:
    """Largest absolute and relative difference between two regional series over shared cells."""
    cells = [(r, y) for r in regions for y in years if (r, y) in a and (r, y) in b]
    if not cells:
        return {"cells": 0}
    worst = max(cells, key=lambda c: abs(a[c] - b[c]))
    worst_rel = max(cells, key=lambda c: abs(a[c] - b[c]) / b[c] if b[c] else 0.0)
    return {
        "cells": len(cells),
        "identical_cells": sum(1 for c in cells if a[c] == b[c]),
        "max_abs_diff": {"region": worst[0], "year": worst[1], "a": a[worst], "b": b[worst]},
        "max_rel_diff_pct": round(100 * abs(a[worst_rel] - b[worst_rel]) / b[worst_rel], 4) if b[worst_rel] else None,
    }


def flags_for(region_members: list[str], year: int) -> list[str]:
    """Data-quality flags that INE's notes attach to a region-year (never used to alter values)."""
    flags = []
    if AMADORA in region_members and 2011 <= year <= 2017:
        flags.append("completions_underreported_amadora")
    if (LISBOA in region_members or SEIA in region_members) and 2002 <= year <= 2005:
        flags.append("construction_underreported_lisboa_seia")
    if year >= 2021:
        flags.append("construction_preliminary")
        flags.append("population_administrative_basis")
    if year >= 2024:
        flags.append("completions_estimated")
    return flags


def load_sources(folder: Path, date: str) -> dict[str, Source]:
    """Parse every pinned container."""
    names = {"0012917": "pop24", "0008273": "pop13", "0013248": "comp24", "0013235": "perm24",
             "0014406": "hh_series", "0014627": "hh21"}
    return {label: parse_source(folder / f"ine_{code}_{date}.jsonl.gz") for code, label in names.items()}


def main() -> None:  # noqa: C901 - one linear pipeline, kept in one place for auditability
    """Build, check and export the regional panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--date", default=SNAPSHOT_DATE)
    parser.add_argument("--results", type=Path, default=HERE / "results_regional.json")
    parser.add_argument("--panel", type=Path, default=HERE / "regional_panel.csv")
    args = parser.parse_args()

    src = load_sources(args.data, args.date)
    geo = nuts2024_geography(src["pop24"])
    for label in ("comp24", "perm24", "hh21"):
        other = nuts2024_geography_or_none(src[label])
        if other is not None and other.nuts3 != geo.nuts3:
            raise ValueError(f"{label}: NUTS 2024 municipality assignment differs from the population source")
    regions = geo.regions()
    nuts3 = sorted(geo.members3)

    checks = {label: {"indicator": s.code, "title": s.title, "last_update": s.last_update, "ine_note": s.note,
                      "retrieved_first": min(s.retrieved), "retrieved_last": max(s.retrieved),
                      "municipal_cells_unavailable": unavailable_municipal_cells(s),
                      **check_dico_universe(s, geo), **check_published_totals(s, geo)}
              for label, s in src.items()}

    series = {label: regional_series(s, geo) for label, s in src.items()}
    reg = {label: values for label, (values, _gaps) in series.items()}
    gaps = {label: len(g) for label, (_values, g) in series.items()}

    overlaps = {
        "population_nuts2013_vs_nuts2024_2021_2023": overlap(reg["pop13"], reg["pop24"], list(regions), [2021, 2022, 2023]),
        "households_census_series_vs_census2021_nuts2024": overlap(reg["hh_series"], reg["hh21"], list(regions), [2021]),
    }

    # National cross-checks against series already pinned for the national analysis.
    eurostat_raw = json.loads((args.data / "eurostat_demo_pjan_PT_2026-10-29.json").read_text(encoding="utf-8"))
    eurostat_pop, _flags = parse_demographic_totals(eurostat_raw, households=False)
    national_completions, _prelim = parse_completed_dwellings(args.data / COMPLETIONS_FILE)
    national_checks = {
        "population_ine_year_t_vs_eurostat_1_january_t_plus_1": {
            str(y): {"ine": reg_pop, "eurostat": eurostat_pop.get(y + 1),
                     "difference_pct": round(100 * (reg_pop / eurostat_pop[y + 1] - 1), 3) if y + 1 in eurostat_pop else None}
            for y in range(2011, 2026)
            for reg_pop in [(reg["pop13"] if y <= 2020 else reg["pop24"]).get(("PT", y))] if reg_pop is not None
        },
        "completions_nuts2024_vs_national_series_0000076": {
            "years_compared": sorted(y for y in range(2011, 2026) if y in national_completions),
            "years_differing": [y for y in range(2011, 2026)
                                if y in national_completions and reg["comp24"][("PT", y)] != national_completions[y]],
        },
    }

    # Assemble the panel. Each series switches source at a documented year; nothing is spliced inside a year.
    population: dict[tuple[str, int], float] = {}
    for (r, y), v in reg["pop13"].items():
        if 2011 <= y <= 2020:
            population[(r, y)] = v
    for (r, y), v in reg["pop24"].items():
        population[(r, y)] = v
    completions = dict(reg["comp24"])
    permits = dict(reg["perm24"])
    households = {k: v for k, v in reg["hh_series"].items() if k[1] in (2011, 2021)}
    comp_rate = per_thousand(completions, population)
    perm_rate = per_thousand(permits, population)

    years = list(range(2011, 2026))
    main_years = years
    diag = {
        name: coverage({k: v for k, v in s.items() if k[0] in nuts3}, nuts3, ys)
        for name, s, ys in (("population", population, years), ("completions", completions, years),
                            ("permits", permits, years), ("households_census", households, [2011, 2021]))
    }
    complete_all = set(complete_years(diag["population"])) & set(complete_years(diag["completions"])) \
        & set(complete_years(diag["permits"]))
    span = longest_complete_span(complete_all)

    def g(series_: dict[tuple[str, int], float], r: str, a: int, b: int) -> dict[str, float] | None:
        out = growth(series_, r, a, b)
        return {k: round(v, 3) for k, v in out.items()} if out else None

    region_list = nuts3 + sorted(geo.members2) + [LISBON_UNION, "PT"]
    periods = [(2011, 2015), (2016, 2020), (2021, 2025)]
    summary: dict[str, Any] = {}
    for r in region_list:
        entry: dict[str, Any] = {"name": geo.names.get(r, FOCUS.get(r, r)),
                                 "level": "NUTS3" if r in geo.members3 else "NUTS2" if r in geo.members2
                                 else "country" if r == "PT" else "NUTS2 union"}
        entry["population_growth_2011_2020"] = g(population, r, 2011, 2020)
        entry["population_growth_2021_2025"] = g(population, r, 2021, 2025)
        entry["population_growth_2011_2025_crosses_2021_break"] = g(population, r, 2011, 2025)
        entry["households_growth_2011_2021_census"] = g(households, r, 2011, 2021)
        for a, b in periods:
            cm, pm = period_mean(comp_rate, r, a, b), period_mean(perm_rate, r, a, b)
            entry[f"completions_per_1000_mean_{a}_{b}"] = round(cm, 3) if cm is not None else None
            entry[f"permits_per_1000_mean_{a}_{b}"] = round(pm, 3) if pm is not None else None
        summary[r] = entry
    for r in region_list:
        for a, b in periods:
            for kind in ("completions", "permits"):
                own, nat = summary[r][f"{kind}_per_1000_mean_{a}_{b}"], summary["PT"][f"{kind}_per_1000_mean_{a}_{b}"]
                summary[r][f"{kind}_per_1000_ratio_to_portugal_{a}_{b}"] = round(own / nat, 3) if own and nat else None

    def ranked(key: str) -> list[dict[str, Any]]:
        rows = [(r, summary[r][key]["annual_pct"]) for r in nuts3 if summary[r][key]]
        return [{"region": r, "name": geo.names[r], "annual_pct": v} for r, v in sorted(rows, key=lambda t: -t[1])]

    dispersion = {
        kind: {str(y): round(coefficient_of_variation([s[(r, y)] for r in nuts3]), 3)
               for y in years if all((r, y) in s for r in nuts3)}
        for kind, s in (("completions_per_1000", comp_rate), ("permits_per_1000", perm_rate))
    }
    results: dict[str, Any] = {
        "geography": {
            "version": "NUTS 2024 as published by INE (Local de residência / Localização geográfica, NUTS - 2024)",
            "municipalities": len(geo.municipality), "nuts3": len(geo.members3), "nuts2": len(geo.members2),
            "nuts3_regions": {r: {"name": geo.names[r], "nuts2": geo.nuts2_of3[r], "municipalities": len(geo.members3[r])} for r in nuts3},
            "lisbon_union": {"code": LISBON_UNION, "members": ["1A0", "1B0"],
                             "note": "Equal to the NUTS 2013 Área Metropolitana de Lisboa (checked below)"},
        },
        "source_checks": checks,
        "regrouping_gaps": gaps,
        "cross_source_overlaps": overlaps,
        "national_cross_checks": national_checks,
        "coverage_nuts3": diag,
        "longest_common_complete_span": list(span),
        "main_panel_years": [main_years[0], main_years[-1]],
        "rankings_population_growth_2011_2020": ranked("population_growth_2011_2020"),
        "rankings_population_growth_2021_2025": ranked("population_growth_2021_2025"),
        "dispersion_across_nuts3_cv": dispersion,
        "regions": summary,
    }
    args.results.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    with args.panel.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["region_code", "region_name", "level", "year", "population", "completed_dwellings",
                         "permitted_dwellings", "private_households_census", "completed_per_1000_residents",
                         "permitted_per_1000_residents", "flags"])
        for r in region_list:
            for y in years:
                def fmt(s: dict[tuple[str, int], float], digits: int = 0) -> str:
                    if (r, y) not in s:
                        return ""
                    return f"{s[(r, y)]:.{digits}f}"
                writer.writerow([r, summary[r]["name"], summary[r]["level"], y, fmt(population), fmt(completions),
                                 fmt(permits), fmt(households), fmt(comp_rate, 3), fmt(perm_rate, 3),
                                 ";".join(flags_for(regions[r], y))])
    print(json.dumps({"span": span, "rank_2021_2025": ranked("population_growth_2021_2025")[:5]}, ensure_ascii=False, indent=1))


def unavailable_municipal_cells(source: Source) -> dict[str, int]:
    """Municipality-year cells that INE published without a value, counted by INE's reason."""
    codes = set(municipalities(source).values())
    counts: dict[str, int] = defaultdict(int)
    for (geocode, _year), reason in source.unavailable.items():
        if geocode in codes:
            counts[reason] += 1
    return dict(counts)


def complete_years(report: dict[str, object]) -> list[int]:
    """The complete years listed in a coverage report."""
    years = report["complete_years"]
    if not isinstance(years, list):
        raise TypeError("complete_years must be a list")
    return [int(y) for y in years]


def nuts2024_geography_or_none(source: Source) -> Geography | None:
    """The NUTS 2024 hierarchy of a source, or None if it is not a five-level NUTS 2024 geography."""
    try:
        return nuts2024_geography(source)
    except ValueError:
        return None


if __name__ == "__main__":
    main()
