"""Grain maize (METHODS.md, section 9): the sample rules and the committed panel."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from eu_wheat_weather.maize import FIRST_YEAR, build_panel, write_panel
from eu_wheat_weather.regions import HarmonisedRow

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "maize"
WEATHER = {"gdd_10_30": "1500.0", "edd_30": "12.5", "prec": "350.0"}


def row(geo: str, year: int, area: float | None, production: float | None,
        ) -> HarmonisedRow:
    y = None if area is None or production is None or area <= 0 else production / area
    return HarmonisedRow(geo, year, geo, "nuts2021", area, production, y, "", "")


def test_rules_apply_in_the_registered_order() -> None:
    """Section 9.1: area and production, the 1-20 t/ha bound, the data rule, weather,
    then 10 years; nothing before 2010."""
    years = range(2008, 2025)

    def yields(geo: str, t: int, base: float, step: float) -> HarmonisedRow:
        y = 8.0 if t == 2016 else base + step * (t % 4)      # one figure for all in 2016
        return row(geo, t, 50.0, 50.0 * y)

    harmonised = [
        *(yields("AA11", t, 7.0, 0.1) for t in years),
        *(yields("AA12", t, 8.5, 0.2) for t in years if t != 2015),
        row("AA12", 2015, 40.0, 40.0 * 0.5),                        # below 1 t/ha
        *(yields("AA13", t, 6.0, 0.3) for t in range(2010, 2018)),
    ]
    weather = {(h.geo, h.year, "season", "all"): WEATHER for h in harmonised}
    del weather[("AA11", 2020, "season", "all")]
    panel, excluded, flags = build_panel(harmonised, [], [], weather)
    reasons = {(e.geo, e.year): e.reason for e in excluded}
    assert reasons[("AA12", 2015)] == "implausible_yield"
    # the data rule comes before the weather and the 10-year rule
    assert {g for (g, t), why in reasons.items() if why == "data_rule"} == {
        "AA11", "AA12", "AA13"} and {f.rule for f in flags} == {"Q1"}
    assert reasons[("AA11", 2020)] == "weather_missing"
    assert all(reasons[("AA13", t)] == "region_below_10_years"
               for t in range(2010, 2018) if t != 2016)
    assert min(r.year for r in panel) == FIRST_YEAR
    assert {r.geo for r in panel} == {"AA11", "AA12"}


def test_committed_maize_panel_regenerates(tmp_path: Path) -> None:
    if not (DOCS / "panel.csv").exists():
        pytest.skip("No committed maize panel.")
    report = (ROOT / "docs" / "MAIZE_DATA.md").read_text(encoding="utf-8")
    command = re.search(r"```\n(poetry run eu-wheat maize-panel [^\n]+)\n```", report)
    assert command is not None
    snapshot = re.search(r"--crops (\S+)", command.group(1))
    assert snapshot is not None
    write_panel(ROOT / snapshot.group(1), DOCS / "region_map.csv", DOCS / "weather.csv",
                DOCS / "season.csv", tmp_path, tmp_path / "MAIZE_DATA.md", command.group(1))
    for name in ("panel.csv", "exclusions.csv", "conflicts.csv", "quality_flags.csv",
                 "summary.json"):
        assert (tmp_path / name).read_bytes() == (DOCS / name).read_bytes(), name
    assert (tmp_path / "MAIZE_DATA.md").read_text(encoding="utf-8") == report


def test_committed_maize_region_map_regenerates() -> None:
    boundaries = ROOT / "data" / "boundaries"
    if not (DOCS / "region_map.csv").exists() or not boundaries.exists():
        pytest.skip("Needs the committed region map and the local GISCO boundaries.")
    from eu_wheat_weather.nuts_changes import audit_nuts_changes
    from eu_wheat_weather.panel import read_region_map, read_wheat
    from eu_wheat_weather.regions import load_all, match_codes

    crops = sorted((ROOT / "snapshots").glob("eurostat-maize-*"))[-1]
    changes = audit_nuts_changes(sorted((ROOT / "snapshots").glob("nuts-changes-*"))[-1])
    matches = match_codes({r.geo for r in read_wheat(crops)}, load_all(boundaries), changes)
    committed = read_region_map(DOCS / "region_map.csv")
    assert [(m.code, m.target, m.rule) for m in matches] == [
        (m.code, m.target, m.rule) for m in committed]
