"""Tests of the sample rules, and regeneration of the committed panel and region map."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import pytest

from eu_wheat_weather.panel import (
    MIN_YEARS,
    PRIMARY,
    build_panel,
    read_region_map,
    read_wheat,
    write_panel,
)
from eu_wheat_weather.regions import HarmonisedRow

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "panel"
WEATHER = {"gdd_0_30": "1500.0", "edd_30": "12.5", "prec": "250.0"}


def row(geo: str, year: int, area: float | None, production: float | None,
        ) -> HarmonisedRow:
    y = None if area is None or production is None or area <= 0 else production / area
    return HarmonisedRow(geo, year, geo, "nuts2021", area, production, y, "", "")


def test_sample_rules_apply_in_order_and_keep_reasons() -> None:
    rows = [row("AA11", 2000 + k, 10.0, 60.0) for k in range(MIN_YEARS)]
    rows += [row("AA11", 2010, None, 60.0), row("AA11", 2011, 0.5, 3.0),
             row("AA11", 2012, 10.0, 200.0), row("AA11", 2013, 10.0, 60.0)]
    rows += [row("BB22", 2000 + k, 10.0, 50.0) for k in range(MIN_YEARS - 1)]
    weather = {(r.geo, r.year, *PRIMARY): dict(WEATHER) for r in rows}
    weather[("AA11", 2013, *PRIMARY)]["edd_30"] = ""
    panel, excluded = build_panel(rows, weather)
    reasons = {(e.geo, e.year): e.reason for e in excluded}
    assert reasons[("AA11", 2010)] == "no_area_or_production"
    assert reasons[("AA11", 2011)] == "area_below_1kha"
    assert reasons[("AA11", 2012)] == "implausible_yield"           # 20 t/ha
    assert reasons[("AA11", 2013)] == "weather_missing"
    assert {reasons[("BB22", 2000 + k)] for k in range(MIN_YEARS - 1)} == {
        "region_below_10_years"}
    assert {(r.geo, r.year) for r in panel} == {("AA11", 2000 + k) for k in range(MIN_YEARS)}
    assert panel[0].log_yield == pytest.approx(math.log(6.0)) and panel[0].country == "AA"


def test_committed_panel_regenerates(tmp_path: Path) -> None:
    if not (DOCS / "panel.csv").exists():
        pytest.skip("No committed panel.")
    report = (ROOT / "docs" / "DATA.md").read_text(encoding="utf-8")
    command = re.search(r"```\n(poetry run eu-wheat panel [^\n]+)\n```", report)
    assert command is not None
    snapshot = re.search(r"--crops (\S+)", command.group(1))
    assert snapshot is not None
    write_panel(ROOT / snapshot.group(1), DOCS / "region_map.csv", DOCS / "weather.csv",
                tmp_path, tmp_path / "DATA.md", command.group(1))
    for name in ("panel.csv", "exclusions.csv", "conflicts.csv", "summary.json"):
        assert (tmp_path / name).read_bytes() == (DOCS / name).read_bytes(), name
    assert (tmp_path / "DATA.md").read_text(encoding="utf-8") == report


def test_committed_region_map_regenerates() -> None:
    boundaries = ROOT / "data" / "boundaries"
    if not (DOCS / "region_map.csv").exists() or not boundaries.exists():
        pytest.skip("Needs the committed region map and the local GISCO boundaries.")
    from eu_wheat_weather.nuts_changes import audit_nuts_changes
    from eu_wheat_weather.regions import load_all, match_codes

    crops = sorted((ROOT / "snapshots").glob("eurostat-crops-*"))[-1]
    changes = audit_nuts_changes(sorted((ROOT / "snapshots").glob("nuts-changes-*"))[-1])
    matches = match_codes({r.geo for r in read_wheat(crops)}, load_all(boundaries), changes)
    committed = read_region_map(DOCS / "region_map.csv")
    assert [(m.code, m.target, m.rule) for m in matches] == [
        (m.code, m.target, m.rule) for m in committed]


@pytest.mark.skipif(not (ROOT / "data" / "boundaries").is_dir(),
                    reason="No local GISCO boundaries in data/boundaries.")
def test_committed_latitudes_regenerate(tmp_path: Path) -> None:
    from eu_wheat_weather.regions import load_all, target_regions
    from eu_wheat_weather.weather import region_latitudes, write_latitudes

    rows = region_latitudes(target_regions(load_all(ROOT / "data" / "boundaries")))
    write_latitudes(rows, tmp_path / "region_latitude.csv")
    assert ((tmp_path / "region_latitude.csv").read_bytes()
            == (DOCS / "region_latitude.csv").read_bytes())


@pytest.mark.skipif(os.environ.get("EU_WHEAT_REPRODUCE_WEATHER") != "1",
                    reason="Set EU_WHEAT_REPRODUCE_WEATHER=1 with E-OBS in data/eobs (minutes).")
def test_committed_weather_regenerates(tmp_path: Path) -> None:
    from eu_wheat_weather.contract import (
        CALENDAR_FILE,
        CALENDAR_SNAPSHOT,
        FIRST_YEAR,
        LAST_YEAR,
    )
    from eu_wheat_weather.regions import load_all, target_regions
    from eu_wheat_weather.weather import build_weather, write_maturity, write_weather

    regions = target_regions(load_all(ROOT / "data" / "boundaries"))
    rows, maturity = build_weather(ROOT / "data" / "eobs", regions,
                                   range(FIRST_YEAR, LAST_YEAR + 1),
                                   ROOT / CALENDAR_SNAPSHOT / CALENDAR_FILE)
    write_weather(rows, tmp_path / "weather.csv")
    write_maturity(maturity, tmp_path / "maturity.csv")
    assert (tmp_path / "weather.csv").read_bytes() == (DOCS / "weather.csv").read_bytes()
    assert (tmp_path / "maturity.csv").read_bytes() == (DOCS / "maturity.csv").read_bytes()
