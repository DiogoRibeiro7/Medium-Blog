"""The region-year panel and its coverage report (METHODS.md, section 3). No estimate.

The sample rules apply in a fixed order, and every excluded region-year is recorded with
its reason: no usable area or production, an implausible yield, missing weather, and
last the minimum number of years per region.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from eu_wheat_weather.crops import WheatRow, audit_snapshot, snapshot_crop, table_name
from eu_wheat_weather.regions import CodeMatch, Conflict, HarmonisedRow, harmonise

AREA_MIN = 1.0                 # thousand hectares
YIELD_MIN, YIELD_MAX = 0.5, 15.0
MIN_YEARS = 10
PRIMARY = ("mar_jul", "all")
WEATHER_TERMS = ("gdd_0_30", "edd_30", "prec")
# The exclusion reasons, in the order the rules apply.
REASONS = ("no_area_or_production", "area_below_1kha", "implausible_yield", "weather_missing",
           "region_below_10_years")


def read_wheat(snapshot: Path) -> list[WheatRow]:
    """The snapshot's processed table, for whichever crop its manifest names."""
    def number(x: str) -> float | None:
        return None if x == "" else float(x)

    with (snapshot / table_name(snapshot_crop(snapshot))).open(encoding="utf-8") as stream:
        return [WheatRow(r["geo"], int(r["year"]), number(r["area_kha"]),
                         number(r["production_kt"]), r["area_flag"], r["production_flag"])
                for r in csv.DictReader(stream)]


def read_region_map(path: Path) -> list[CodeMatch]:
    with path.open(encoding="utf-8") as stream:
        return [CodeMatch(r["code"], r["target"] or None, r["rule"],
                          int(r["version"]) if r["version"] else None,
                          float(r["share"]) if r["share"] else None, r["candidate"])
                for r in csv.DictReader(stream)]


def read_weather(path: Path) -> dict[tuple[str, int, str, str], dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return {(r["geo"], int(r["year"]), r["window"], r["weighting"]): r
                for r in csv.DictReader(stream)}


@dataclass(frozen=True)
class PanelRow:
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
    gdd_0_30: float
    edd_30: float
    prec: float


@dataclass(frozen=True)
class Exclusion:
    geo: str
    year: int
    source: str
    reason: str
    yield_t_ha: float | None


def first_reason(h: HarmonisedRow, weather: Mapping[str, str] | None) -> str | None:
    if h.area_kha is None or h.production_kt is None:
        return "no_area_or_production"
    if h.area_kha < AREA_MIN:
        return "area_below_1kha"
    if h.yield_t_ha is None or not YIELD_MIN <= h.yield_t_ha <= YIELD_MAX:
        return "implausible_yield"
    if weather is None or any(weather[v] == "" for v in WEATHER_TERMS):
        return "weather_missing"
    return None


def build_panel(harmonised: Sequence[HarmonisedRow],
                weather: Mapping[tuple[str, int, str, str], Mapping[str, str]],
                ) -> tuple[list[PanelRow], list[Exclusion]]:
    kept: list[PanelRow] = []
    excluded: list[Exclusion] = []
    for h in harmonised:
        w = weather.get((h.geo, h.year, *PRIMARY))
        reason = first_reason(h, w)
        if reason is not None:
            excluded.append(Exclusion(h.geo, h.year, h.source, reason, h.yield_t_ha))
            continue
        assert w is not None and h.area_kha is not None and h.production_kt is not None
        assert h.yield_t_ha is not None
        kept.append(PanelRow(h.geo, h.geo[:2], h.year, h.area_kha, h.production_kt,
                             h.yield_t_ha, math.log(h.yield_t_ha), h.source, h.rule,
                             h.area_flag, h.production_flag, float(w["gdd_0_30"]),
                             float(w["edd_30"]), float(w["prec"])))
    years = Counter(r.geo for r in kept)
    short = {geo for geo, n in years.items() if n < MIN_YEARS}
    excluded += [Exclusion(r.geo, r.year, r.source, "region_below_10_years", r.yield_t_ha)
                 for r in kept if r.geo in short]
    return [r for r in kept if r.geo not in short], excluded


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                             for k, v in row.items()})


def summary_document(snapshot: Path, matches: Sequence[CodeMatch],
                     harmonised: Sequence[HarmonisedRow], conflicts: Sequence[Conflict],
                     panel: Sequence[PanelRow], excluded: Sequence[Exclusion],
                     weather: Mapping[tuple[str, int, str, str], Mapping[str, str]],
                     ) -> dict[str, Any]:
    by_country: dict[str, dict[str, int]] = {}
    for country in sorted({r.country for r in panel}):
        rows = [r for r in panel if r.country == country]
        by_country[country] = {"regions": len({r.geo for r in rows}), "region_years": len(rows)}
    windows: dict[str, dict[str, int]] = {}
    for (_, _, window, weighting), w in weather.items():
        cell = windows.setdefault(f"{window}/{weighting}", {"region_seasons": 0, "missing": 0})
        cell["region_seasons"] += 1
        cell["missing"] += int(any(w[v] == "" for v in WEATHER_TERMS))
    return {
        "snapshot": snapshot.name, "snapshot_audit": audit_snapshot(snapshot),
        "codes_by_rule": dict(sorted(Counter(m.rule for m in matches).items())),
        "harmonised_region_years": len(harmonised),
        "harmonised_with_yield": sum(h.yield_t_ha is not None for h in harmonised),
        "conflicts": len(conflicts),
        "excluded_by_reason": {reason: n for reason in REASONS
                               if (n := sum(e.reason == reason for e in excluded))},
        "panel_region_years": len(panel), "panel_regions": len({r.geo for r in panel}),
        "panel_countries": len(by_country),
        "years": [min((r.year for r in panel), default=0), max((r.year for r in panel),
                                                                default=0)],
        "single_region_countries": [c for c, v in by_country.items() if v["regions"] == 1],
        "by_country": by_country,
        "flags": dict(sorted(Counter(f for r in panel for f in (r.area_flag, r.production_flag)
                                     if f).items())),
        "weather_missing": windows,
    }


def render(summary: Mapping[str, Any], matches: Sequence[CodeMatch],
           conflicts: Sequence[Conflict], excluded: Sequence[Exclusion], command: str) -> str:
    s = summary
    lines = [
        "# Data: the region-year panel", "",
        f"Generated by `eu-wheat panel` from the crop snapshot `{s['snapshot']}`, the "
        "harmonised region codes and the E-OBS weather aggregated to NUTS-2021 regions. "
        "It describes coverage only: no yield is related to any weather variable here "
        "([METHODS.md](../METHODS.md), section 3).", "",
        "## Summary", "",
        f"- **Panel:** {s['panel_region_years']} region-years in {s['panel_regions']} "
        f"regions of {s['panel_countries']} countries, harvest years {s['years'][0]} to "
        f"{s['years'][1]}.",
        "- **Codes:** " + ", ".join(f"{n} {rule}" for rule, n in s["codes_by_rule"].items())
        + ".",
        f"- **Harmonised region-years:** {s['harmonised_region_years']}, of which "
        f"{s['harmonised_with_yield']} have a yield. Sources that disagree by more than 1%: "
        f"{s['conflicts']}.",
        "- **Excluded region-years, in the order the rules apply:** "
        + "; ".join(f"{reason} {n}" for reason, n in s["excluded_by_reason"].items()) + ".",
        "- **Countries with a single region in the panel** (no within-country variation for "
        "M1): " + (", ".join(s["single_region_countries"]) or "none") + ".", "",
        "## Region codes", "",
        "| Code | Target | Rule | NUTS version | Symmetric difference | Closest region |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for m in matches:
        if m.rule == "nuts2021":
            continue
        share = "" if m.share is None else f"{m.share:.2%}"
        lines.append(f"| {m.code} | {m.target or '-'} | {m.rule} | {m.version or ''} | "
                     f"{share} | {m.candidate} |")
    lines += ["", "Codes that are NUTS-2021 regions themselves are not listed.", "",
              "## Sources that disagree", ""]
    if conflicts:
        lines += ["| Region | Year | Used | Yield | Other | Yield |",
                  "| --- | ---: | --- | ---: | --- | ---: |"]
        lines += [f"| {c.geo} | {c.year} | {c.used} | {c.used_yield:.2f} | {c.other} | "
                  f"{c.other_yield:.2f} |" for c in conflicts]
    else:
        lines.append("None.")
    lines += ["", "## By country", "", "| Country | Regions | Region-years |",
              "| --- | ---: | ---: |"]
    lines += [f"| {c} | {v['regions']} | {v['region_years']} |"
              for c, v in s["by_country"].items()]
    implausible = [e for e in excluded if e.reason == "implausible_yield"]
    lines += ["", "## Implausible yields", ""]
    if implausible:
        lines += ["| Region | Year | Source | Yield (t/ha) |", "| --- | ---: | --- | ---: |"]
        lines += [f"| {e.geo} | {e.year} | {e.source} | "
                  f"{'' if e.yield_t_ha is None else f'{e.yield_t_ha:.2f}'} |"
                  for e in implausible]
    else:
        lines.append("None.")
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


def write_panel(snapshot: Path, region_map: Path, weather_path: Path, output_dir: Path,
                report: Path, command: str) -> dict[str, Any]:
    matches = read_region_map(region_map)
    harmonised, conflicts = harmonise(read_wheat(snapshot), matches)
    weather = read_weather(weather_path)
    panel, excluded = build_panel(harmonised, weather)
    write_csv(output_dir / "panel.csv", [asdict(r) for r in panel], list(PanelRow.__annotations__))
    write_csv(output_dir / "exclusions.csv", [asdict(e) for e in excluded],
           list(Exclusion.__annotations__))
    write_csv(output_dir / "conflicts.csv", [asdict(c) for c in conflicts],
           list(Conflict.__annotations__))
    summary = summary_document(snapshot, matches, harmonised, conflicts, panel, excluded,
                               weather)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n",
                                             encoding="utf-8", newline="\n")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render(summary, matches, conflicts, excluded, command), encoding="utf-8",
                      newline="\n")
    return summary
