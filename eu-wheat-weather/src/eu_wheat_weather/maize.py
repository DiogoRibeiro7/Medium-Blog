"""Grain maize (METHODS.md, section 9): the region-year panel and its data report.

No estimate here: the report describes coverage only. The sample rules apply in the order
section 9.1 fixes, and every excluded region-year is recorded with its reason.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from eu_wheat_weather.contract import LAST_YEAR
from eu_wheat_weather.crops import WheatRow, audit_snapshot
from eu_wheat_weather.panel import (
    Exclusion,
    read_region_map,
    read_weather,
    read_wheat,
    write_csv,
)
from eu_wheat_weather.quality import (
    Flag,
    q1_one_figure,
    q2_pro_rated,
    q3_carried_forward,
    q4_sums_disagree,
    q5_level_shifts,
)
from eu_wheat_weather.regions import CodeMatch, Conflict, HarmonisedRow, harmonise

FIRST_YEAR = 2010               # coverage before 2010 misses most of the EU's maize (9.1)
AREA_MIN = 1.0                  # thousand hectares
YIELD_MIN, YIELD_MAX = 1.0, 20.0
MIN_YEARS = 10
SEASON = ("season", "all")
TERMS = ("gdd_10_30", "edd_30", "prec")
# The exclusion reasons, in the order the rules apply (section 9.1).
REASONS = ("no_area_or_production", "area_below_1kha", "implausible_yield", "data_rule",
           "weather_missing", "region_below_10_years")


@dataclass(frozen=True)
class MaizeRow:
    geo: str
    country: str
    year: int
    area_kha: float
    production_kt: float
    yield_t_ha: float
    log_yield: float
    source: str
    rule: str
    area_flag: str
    production_flag: str
    gdd_10_30: float
    edd_30: float
    prec: float


def first_reason(h: HarmonisedRow) -> str | None:
    if h.area_kha is None or h.production_kt is None:
        return "no_area_or_production"
    if h.area_kha < AREA_MIN:
        return "area_below_1kha"
    if h.yield_t_ha is None or not YIELD_MIN <= h.yield_t_ha <= YIELD_MAX:
        return "implausible_yield"
    return None


def _as_panel(h: HarmonisedRow) -> dict[str, str]:
    """A row in the form the data rule's functions read."""
    assert h.yield_t_ha is not None
    return {"geo": h.geo, "country": h.geo[:2], "year": str(h.year),
            "yield_t_ha": str(h.yield_t_ha), "log_yield": str(math.log(h.yield_t_ha)),
            "area_kha": str(h.area_kha), "production_kt": str(h.production_kt)}


def build_panel(harmonised: Sequence[HarmonisedRow], published: Sequence[WheatRow],
                matches: Sequence[CodeMatch],
                weather: Mapping[tuple[str, int, str, str], Mapping[str, str]],
                ) -> tuple[list[MaizeRow], list[Exclusion], list[Flag]]:
    rows = [h for h in harmonised if FIRST_YEAR <= h.year <= LAST_YEAR]
    excluded: list[Exclusion] = []
    passing = []
    for h in rows:
        reason = first_reason(h)
        if reason is None:
            passing.append(h)
        else:
            excluded.append(Exclusion(h.geo, h.year, h.source, reason, h.yield_t_ha))
    # The data rule of section 8.2, on the region-years that pass the first two rules.
    candidate = [_as_panel(h) for h in passing]
    flags = (q1_one_figure(candidate) + q2_pro_rated(candidate)
             + q3_carried_forward(candidate, harmonised)
             + q4_sums_disagree(candidate, published, matches) + q5_level_shifts(candidate))
    flagged = {(f.geo, f.year) for f in flags}
    kept: list[MaizeRow] = []
    for h in passing:
        if (h.geo, h.year) in flagged:
            excluded.append(Exclusion(h.geo, h.year, h.source, "data_rule", h.yield_t_ha))
            continue
        w = weather.get((h.geo, h.year, *SEASON))
        if w is None or any(w[v] == "" for v in TERMS):
            excluded.append(Exclusion(h.geo, h.year, h.source, "weather_missing",
                                      h.yield_t_ha))
            continue
        assert h.area_kha is not None and h.production_kt is not None
        assert h.yield_t_ha is not None
        kept.append(MaizeRow(h.geo, h.geo[:2], h.year, h.area_kha, h.production_kt,
                             h.yield_t_ha, math.log(h.yield_t_ha), h.source, h.rule,
                             h.area_flag, h.production_flag, float(w["gdd_10_30"]),
                             float(w["edd_30"]), float(w["prec"])))
    years = Counter(r.geo for r in kept)
    short = {geo for geo, n in years.items() if n < MIN_YEARS}
    excluded += [Exclusion(r.geo, r.year, r.source, "region_below_10_years", r.yield_t_ha)
                 for r in kept if r.geo in short]
    return [r for r in kept if r.geo not in short], excluded, flags


def read_seasons(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return {r["geo"]: r for r in csv.DictReader(stream)}


def _date(month_day: str) -> str:
    months = ("January", "February", "March", "April", "May", "June", "July", "August",
              "September", "October", "November", "December")
    return f"{int(month_day[3:])} {months[int(month_day[:2]) - 1]}"


def summary_document(snapshot: Path, matches: Sequence[CodeMatch],
                     harmonised: Sequence[HarmonisedRow], conflicts: Sequence[Conflict],
                     panel: Sequence[MaizeRow], excluded: Sequence[Exclusion],
                     flags: Sequence[Flag],
                     weather: Mapping[tuple[str, int, str, str], Mapping[str, str]],
                     seasons: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    by_country: dict[str, dict[str, int]] = {}
    for country in sorted({r.country for r in panel}):
        rows = [r for r in panel if r.country == country]
        by_country[country] = {"regions": len({r.geo for r in rows}), "region_years": len(rows)}
    windows: dict[str, dict[str, int]] = {}
    for (_, _, window, weighting), w in weather.items():
        cell = windows.setdefault(f"{window}/{weighting}", {"region_seasons": 0, "missing": 0})
        cell["region_seasons"] += 1
        cell["missing"] += int(any(w[v] == "" for v in TERMS))
    regions = sorted({r.geo for r in panel})
    planting = sorted(int(seasons[g]["planting_day"]) for g in regions)
    maturity = sorted(int(seasons[g]["maturity_day"]) for g in regions)
    by_day = {int(s["planting_day"]): s["planting_date"] for s in seasons.values()
              if s["planting_day"]}
    mat_day = {int(s["maturity_day"]): s["maturity_date"] for s in seasons.values()
               if s["maturity_day"]}
    flagged = {(f.geo, f.year) for f in flags}
    in_period = [h for h in harmonised if FIRST_YEAR <= h.year <= LAST_YEAR]
    return {
        "snapshot": snapshot.name, "snapshot_audit": audit_snapshot(snapshot),
        "years": [FIRST_YEAR, LAST_YEAR],
        "codes_by_rule": dict(sorted(Counter(m.rule for m in matches).items())),
        "harmonised_region_years": len(in_period),
        "harmonised_with_yield": sum(h.yield_t_ha is not None for h in in_period),
        "conflicts": sum(FIRST_YEAR <= c.year <= LAST_YEAR for c in conflicts),
        "excluded_by_reason": {reason: n for reason in REASONS
                               if (n := sum(e.reason == reason for e in excluded))},
        "data_rule_flags": dict(sorted(Counter(f.rule for f in flags).items())),
        "data_rule_by_country": dict(sorted(Counter(g[:2] for g, _ in flagged).items())),
        "panel_region_years": len(panel), "panel_regions": len(regions),
        "panel_countries": len(by_country),
        "single_region_countries": [c for c, v in by_country.items() if v["regions"] == 1],
        "by_country": by_country,
        "season": {"planting_first": by_day[planting[0]], "planting_last": by_day[planting[-1]],
                   "planting_median_day": statistics.median(planting),
                   "maturity_first": mat_day[maturity[0]],
                   "maturity_last": mat_day[maturity[-1]],
                   "maturity_median_day": statistics.median(maturity),
                   "planting_before_march": sum(d < 60 for d in planting)},
        "flags": dict(sorted(Counter(f for r in panel for f in (r.area_flag, r.production_flag)
                                     if f).items())),
        "weather_missing": windows,
    }


def render(s: Mapping[str, Any], excluded: Sequence[Exclusion], flags: Sequence[Flag],
           command: str) -> str:
    season = s["season"]
    lines = [
        "# Data: the grain maize panel", "",
        f"Generated by `eu-wheat maize-panel` from the crop snapshot `{s['snapshot']}`, the "
        "harmonised region codes and the E-OBS weather over each region's maize season. It "
        "describes coverage only: no yield is related to any weather variable here "
        "([METHODS.md](../METHODS.md), section 9).", "",
        "## Summary", "",
        f"- **Panel:** {s['panel_region_years']} region-years in {s['panel_regions']} "
        f"regions of {s['panel_countries']} countries, harvest years {s['years'][0]} to "
        f"{s['years'][1]}.",
        "- **Codes:** " + ", ".join(f"{n} {rule}" for rule, n in s["codes_by_rule"].items())
        + ".",
        f"- **Harmonised region-years, {s['years'][0]}-{s['years'][1]}:** "
        f"{s['harmonised_region_years']}, of which {s['harmonised_with_yield']} have a yield. "
        f"Sources that disagree by more than 1%: {s['conflicts']}.",
        "- **Excluded region-years, in the order the rules apply (9.1):** "
        + "; ".join(f"{reason} {n}" for reason, n in s["excluded_by_reason"].items()) + ".",
        "- **The data rule (8.2), by rule:** "
        + (", ".join(f"{k} {v}" for k, v in s["data_rule_flags"].items()) or "none")
        + "; by country: "
        + (", ".join(f"{k} {v}" for k, v in s["data_rule_by_country"].items()) or "none")
        + ". Every flag is in [maize/quality_flags.csv](maize/quality_flags.csv).",
        "- **Countries with a single region in the panel** (no within-country variation for "
        "M1): " + (", ".join(s["single_region_countries"]) or "none") + ".",
        f"- **The season (GGCMI rainfed maize):** in the panel's regions, planting runs from "
        f"{_date(season['planting_first'])} to {_date(season['planting_last'])} and maturity "
        f"from {_date(season['maturity_first'])} to {_date(season['maturity_last'])}; "
        f"{season['planting_before_march']} regions would plant before 1 March and start on "
        "it.", "",
        "## By country", "", "| Country | Regions | Region-years |", "| --- | ---: | ---: |",
    ]
    lines += [f"| {c} | {v['regions']} | {v['region_years']} |"
              for c, v in s["by_country"].items()]
    implausible = [e for e in excluded if e.reason == "implausible_yield"]
    lines += ["", "## Implausible yields", "",
              f"Outside {YIELD_MIN:g}-{YIELD_MAX:g} t/ha, a bound fixed from agronomy (9.1).",
              ""]
    if implausible:
        lines += ["| Region | Year | Source | Yield (t/ha) |", "| --- | ---: | --- | ---: |"]
        lines += [f"| {e.geo} | {e.year} | {e.source} | "
                  f"{'' if e.yield_t_ha is None else f'{e.yield_t_ha:.2f}'} |"
                  for e in implausible]
    else:
        lines.append("None.")
    details: dict[tuple[str, str], list[Flag]] = {}
    for f in flags:
        details.setdefault((f.rule, f.detail), []).append(f)
    lines += ["", "## The data rule's flags", "", "| Rule | Regions | Detail | Region-years |",
              "| --- | --- | --- | ---: |"]
    lines += [f"| {rule} | {', '.join(sorted({f.geo for f in fs}))} | {detail} | {len(fs)} |"
              for (rule, detail), fs in sorted(details.items())]
    if not details:
        lines.append("| - | - | none | 0 |")
    lines += ["", "## Weather coverage", "",
              "Region-seasons of every EU-27 NUTS-2021 region on the E-OBS grid, and those "
              "with a missing day.", "", "| Window / weighting | Region-seasons | Missing |",
              "| --- | ---: | ---: |"]
    lines += [f"| {k} | {v['region_seasons']} | {v['missing']} |"
              for k, v in sorted(s["weather_missing"].items())]
    lines += ["", "## Flags in the panel", "",
              (", ".join(f"`{f}` {n}" for f, n in s["flags"].items()) or "None") + ".", "",
              "## Reproduce", "", "```", command, "```", ""]
    return "\n".join(lines)


def write_panel(snapshot: Path, region_map: Path, weather_path: Path, seasons_path: Path,
                output_dir: Path, report: Path, command: str) -> dict[str, Any]:
    matches = read_region_map(region_map)
    published = read_wheat(snapshot)
    harmonised, conflicts = harmonise(published, matches)
    weather = read_weather(weather_path)
    seasons = read_seasons(seasons_path)
    panel, excluded, flags = build_panel(harmonised, published, matches, weather)
    write_csv(output_dir / "panel.csv", [asdict(r) for r in panel],
              list(MaizeRow.__annotations__))
    write_csv(output_dir / "exclusions.csv", [asdict(e) for e in excluded],
              list(Exclusion.__annotations__))
    write_csv(output_dir / "conflicts.csv",
              [asdict(c) for c in conflicts if FIRST_YEAR <= c.year <= LAST_YEAR],
              list(Conflict.__annotations__))
    write_csv(output_dir / "quality_flags.csv", [asdict(f) for f in flags],
              list(Flag.__annotations__))
    summary = summary_document(snapshot, matches, harmonised, conflicts, panel, excluded,
                               flags, weather, seasons)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n",
                                             encoding="utf-8", newline="\n")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render(summary, excluded, flags, command), encoding="utf-8",
                      newline="\n")
    return summary
