"""Tests of the data rule of Amendment A1 (METHODS.md, section 8.2) on synthetic panels."""

from __future__ import annotations

import math

from eu_wheat_weather.crops import WheatRow
from eu_wheat_weather.quality import (
    apply_rule,
    q1_one_figure,
    q2_pro_rated,
    q3_carried_forward,
    q4_sums_disagree,
    q5_level_shifts,
    q6_repeated_yields,
    q7_join_breaks,
    q8_other_codes_figures,
)
from eu_wheat_weather.regions import CodeMatch, HarmonisedRow


def row(geo: str, year: int, y: float, area: float = 10.0) -> dict[str, str]:
    return {"geo": geo, "country": geo[:2], "year": str(year), "yield_t_ha": str(y),
            "log_yield": str(math.log(y)), "area_kha": str(area),
            "production_kt": str(area * y)}


def varied(geos: list[str], years: range) -> list[dict[str, str]]:
    """Yields that differ by region and move differently over time."""
    return [row(g, t, 5 + 0.3 * i + 0.4 * ((t * (i + 2)) % 5)) for i, g in enumerate(geos)
            for t in years]


def test_q1_one_figure_for_every_region() -> None:
    panel = varied(["AA11", "AA12", "AA13"], range(2000, 2005))
    panel = [r for r in panel if r["year"] != "2002"] + [
        row("AA11", 2002, 8.39), row("AA12", 2002, 8.40), row("AA13", 2002, 8.395)]
    assert {(f.geo, f.year) for f in q1_one_figure(panel)} == {
        ("AA11", 2002), ("AA12", 2002), ("AA13", 2002)}
    assert q1_one_figure(varied(["AA11", "AA12", "AA13"], range(2000, 2005))) == []


def test_q2_pro_rated_series() -> None:
    geos = ["AA11", "AA12", "AA13", "AA14"]
    level = {g: 4 + i for i, g in enumerate(geos)}
    growth = {2000: 1.0, 2001: 1.1, 2002: 0.9}
    panel = [row(g, t, level[g] * growth[t]) for g in geos for t in growth]  # one series
    flagged = {(f.geo, f.year) for f in q2_pro_rated(panel)}
    assert flagged == {(g, t) for g in geos for t in growth}
    assert q2_pro_rated(varied(geos, range(2000, 2006))) == []
    assert q2_pro_rated(panel[:9]) == []             # only 3 regions: no test


def test_q3_carried_forward() -> None:
    panel = [row("AA11", 2001, 5.0), row("AA11", 2002, 5.0)]
    harmonised = [HarmonisedRow("AA11", t, "AA11", "nuts2021", 10.0, 50.0, 5.0, "", "")
                  for t in (2000, 2001)] + [
        HarmonisedRow("AA11", 2002, "AA11", "nuts2021", 10.0, 50.5, 5.05, "", "")]
    assert [(f.geo, f.year) for f in q3_carried_forward(panel, harmonised)] == [("AA11", 2001)]


def test_q4_sums_that_disagree_with_nuts1() -> None:
    published = [WheatRow("AA1", 2005, 100.0, 600.0, "", ""),
                 WheatRow("AA11", 2005, 60.0, 400.0, "", ""),
                 WheatRow("AA12", 2005, 20.0, 200.0, "", ""),      # area sums to 80
                 WheatRow("AA1", 2006, 100.0, 600.0, "", ""),
                 WheatRow("AA11", 2006, 60.0, 400.0, "", ""),
                 WheatRow("AA12", 2006, 40.0, 200.0, "", "")]      # both sums agree
    matches = [CodeMatch("AA11", "AA11", "nuts2021", 2021, None, ""),
               CodeMatch("AA12", "AA12", "nuts2021", 2021, None, "")]
    panel = [row(g, t, 5.0) for g in ("AA11", "AA12") for t in (2005, 2006)]
    assert {(f.geo, f.year) for f in q4_sums_disagree(panel, published, matches)} == {
        ("AA11", 2005), ("AA12", 2005)}


def test_q5_level_shifts_of_three_years_or_more() -> None:
    yields = {t: 3.0 for t in range(2000, 2016)} | {2012: 1.0, 2013: 1.1, 2014: 0.9}
    flagged = [f.year for f in q5_level_shifts([row("AA11", t, y) for t, y in yields.items()])]
    assert flagged == [2012, 2013, 2014]
    single = {t: 3.0 for t in range(2000, 2016)} | {2003: 0.9}   # a one-year failure
    assert q5_level_shifts([row("AA11", t, y) for t, y in single.items()]) == []


def test_the_ten_year_rule_is_checked_again() -> None:
    panel = varied(["AA11", "AA12", "AA13"], range(2000, 2011))   # 11 years each
    yields = {t: 3.0 for t in range(2000, 2011)} | {2004: 1.0, 2005: 1.0, 2006: 1.0}
    panel = [r for r in panel if r["geo"] != "AA13"] + [row("AA13", t, y)
                                                       for t, y in yields.items()]
    kept, flags, short = apply_rule(panel, [], [], [])
    assert short == ["AA13"] and {f.rule for f in flags} == {"Q5"}
    assert {r["geo"] for r in kept} == {"AA11", "AA12"}


def test_q5_skips_a_region_with_a_single_year() -> None:
    """Maize applies the rule before the 10-year rule (section 9.1), so a region can have
    one year; with no other year there is nothing to compare."""
    assert q5_level_shifts([row("AA11", 2015, 7.0)]) == []


# Amendment A2 (section 10.2), for maize.


def test_q6_repeated_yields_keep_the_first_year_of_a_run() -> None:
    panel = [row("AA11", 2010, 9.0), row("AA11", 2011, 11.28), row("AA11", 2012, 11.29),
             row("AA11", 2013, 11.29), row("AA11", 2014, 9.5)]
    assert {(f.geo, f.year) for f in q6_repeated_yields(panel)} == {("AA11", 2012),
                                                                    ("AA11", 2013)}
    two = [row("AA12", 2010, 8.0), row("AA12", 2011, 8.05), row("AA12", 2012, 9.0)]
    gap = [row("AA13", 2010, 8.0), row("AA13", 2011, 8.05), row("AA13", 2013, 8.06)]
    assert q6_repeated_yields(two + gap) == []        # a run of two, or a gap


def test_q7_a_break_where_two_codes_join() -> None:
    geos = ["AA11", "AA12", "AA13"]
    panel = varied(geos, range(2010, 2016))
    jumped = [r if not (r["geo"] == "AA11" and int(r["year"]) >= 2014)
              else row("AA11", int(r["year"]), float(r["yield_t_ha"]) * 1.6) for r in panel]
    sources = {(g, t): ("AA99" if g == "AA11" and t < 2014 else g)
               for g in geos for t in range(2010, 2016)}
    flags = q7_join_breaks(jumped, sources, {"AA99"})
    assert {(f.geo, f.year) for f in flags} == {("AA11", t) for t in range(2010, 2014)}
    assert q7_join_breaks(panel, sources, {"AA99"}) == []        # no break
    assert q7_join_breaks(jumped, sources, set()) == []          # not an official recode
    assert q7_join_breaks([r for r in jumped if r["geo"] != "AA13"], sources, {"AA99"}) == []


def test_q8_figures_that_belong_to_another_code() -> None:
    panel = [row("AA11", 2013, 6.0, area=30.0), row("AA12", 2013, 7.0, area=20.0)]
    published = [WheatRow("AA11", 2013, 30.0, 180.0, "", ""),
                 WheatRow("AA14", 2013, 30.0, 180.0, "", ""),      # another region's code
                 WheatRow("AA2", 2013, 20.0, 140.0, "", "")]       # an older parent
    sources = {("AA11", 2013): "AA11", ("AA12", 2013): "AA12"}
    matches = [CodeMatch("AA11", "AA11", "nuts2021", 2021, 0.0, "AA11"),
               CodeMatch("AA12", "AA12", "nuts2021", 2021, 0.0, "AA12")]
    flags = q8_other_codes_figures(panel, sources, published, matches)
    assert {(f.geo, f.year) for f in flags} == {("AA11", 2013), ("AA12", 2013)}
    with_parent = q8_other_codes_figures(panel, sources, published, matches, {"AA2": "AA12"})
    assert {(f.geo, f.year) for f in with_parent} == {("AA11", 2013)}
