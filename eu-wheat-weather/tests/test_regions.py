"""Tests of the harmonisation of region codes to NUTS 2021, on synthetic boundaries."""

from __future__ import annotations

import pytest
import shapely

from eu_wheat_weather.contract import NUTS_VERSIONS
from eu_wheat_weather.crops import WheatRow
from eu_wheat_weather.nuts_changes import Change
from eu_wheat_weather.regions import harmonise, match_codes, symmetric_share

CHANGES = [Change(2006, 2010, "ITD3", "ITH3", "Code change"),
           Change(2010, 2013, "SI01", "", "Boundary shift"),
           Change(2010, 2013, "", "SI03", "New region")]


def boundaries() -> dict[int, dict[str, shapely.Geometry]]:
    out: dict[int, dict[str, shapely.Geometry]] = {v: {} for v in NUTS_VERSIONS}
    out[2021] = {"FRB0": shapely.box(0, 0, 1, 1), "ITH3": shapely.box(2, 0, 3, 1),
                 "DE40": shapely.box(4, 0, 5, 1), "DED2": shapely.box(6, 0, 7, 1),
                 "DED4": shapely.box(7, 0, 8, 1), "TRC1": shapely.box(9, 0, 10, 1)}
    out[2021]["SI03"] = shapely.box(11, 0, 12, 1)
    # A recode whose generalised boundary differs by 5%: the official table decides.
    out[2006] = {"ITD3": shapely.box(2, 0, 3.05, 1),
                 "PL12": shapely.box(0, 0, 1.001, 1)}         # same shape, no recode
    out[2010] = {"SI01": shapely.box(11, 0, 12.001, 1)}       # a recorded boundary shift
    return out


def by_code(codes: list[str]) -> dict[str, tuple[str | None, str]]:
    return {m.code: (m.target, m.rule) for m in match_codes(codes, boundaries(), CHANGES)}


def test_each_rule() -> None:
    got = by_code(["FRB0", "ITD3", "DE4", "DED", "PL12", "SI01", "XX99", "TRC1"])
    assert got["FRB0"] == ("FRB0", "nuts2021")
    assert got["ITD3"] == ("ITH3", "recode")         # despite a 5% geometric difference
    assert got["DE4"] == ("DE40", "nuts1_single")
    assert got["DED"] == (None, "no_match")          # two NUTS-2 regions
    assert got["PL12"] == (None, "no_match")         # near-identical shape, no recode
    assert got["SI01"] == (None, "no_match")         # recorded boundary shift
    assert got["XX99"] == (None, "no_boundary")
    assert got["TRC1"] == (None, "no_match")         # not an EU-27 target


def test_geometric_shares_are_reported_but_decide_nothing() -> None:
    matches = {m.code: m for m in match_codes(["ITD3", "PL12", "SI01"], boundaries(), CHANGES)}
    assert matches["ITD3"].version == 2006 and matches["ITD3"].share == pytest.approx(0.05)
    assert matches["PL12"].candidate == "FRB0" and matches["PL12"].share == pytest.approx(0.001)
    assert "Boundary shift" in matches["SI01"].candidate
    assert symmetric_share(shapely.box(0, 0, 1, 1), shapely.box(0, 0, 1, 1)) == 0.0


def row(geo: str, year: int, area: float | None, production: float | None) -> WheatRow:
    return WheatRow(geo, year, area, production, "", "")


def test_harmonise_prefers_own_code_with_a_yield_and_reports_conflicts() -> None:
    matches = match_codes(["ITH3", "ITD3", "DE4", "PL12"], boundaries(), CHANGES)
    rows = [row("ITH3", 2000, 10.0, 50.0), row("ITD3", 2000, 10.0, 52.0),   # 5.0 vs 5.2
            row("ITH3", 2001, 10.0, None), row("ITD3", 2001, 10.0, 51.0),   # only ITD3
            row("DE4", 2000, 100.0, 700.0), row("PL12", 2000, 5.0, 25.0)]
    harmonised, conflicts = harmonise(rows, matches)
    got = {(h.geo, h.year): (h.source, h.rule, h.yield_t_ha) for h in harmonised}
    assert got[("ITH3", 2000)] == ("ITH3", "nuts2021", 5.0)
    assert got[("ITH3", 2001)] == ("ITD3", "recode", 5.1)
    assert got[("DE40", 2000)] == ("DE4", "nuts1_single", 7.0)
    assert ("FRB0", 2000) not in got                      # PL12 is dropped
    assert [(c.geo, c.year, c.used, c.other) for c in conflicts] == [
        ("ITH3", 2000, "ITH3", "ITD3")]
