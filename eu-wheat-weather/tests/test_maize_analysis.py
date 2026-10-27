"""Maize estimates (METHODS.md, section 9.2): units, decisions and regeneration."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest
from tests_support import assert_same

from eu_wheat_weather.maize_analysis import PLACEBO_TERMS, decide, observations, render
from eu_wheat_weather.weather import bin_names

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "maize" / "results" / "results.json"
REPORT = ROOT / "docs" / "MAIZE.md"
WHEAT = ROOT / "docs" / "amendment" / "results.json"


def weather_row(**values: float) -> dict[str, str]:
    row = {"gdd_10_27": "1200", "gdd_10_30": "1500", "gdd_10_33": "1600", "edd_27": "40",
           "edd_30": "25", "edd_33": "9", "prec": "350"}
    row.update({b: "1.0" for b in bin_names()})
    row.update({k: str(v) for k, v in values.items()})
    return row


def test_observations_use_the_registered_units() -> None:
    """GDD from 10 °C in hundreds, EDD in tens, precipitation in hundreds of mm."""
    panel = [{"geo": "AA11", "country": "AA", "year": "2015", "log_yield": "2.0",
              "area_kha": "30"},
             {"geo": "BB11", "country": "BB", "year": "2015", "log_yield": "2.0",
              "area_kha": "30"}]
    weather = {("AA11", 2015, "season"): weather_row(),
               ("AA11", 2015, "post_maturity"): weather_row(edd_30=5, prec=120),
               ("BB11", 2015, "season"): weather_row(prec="")}       # type: ignore[arg-type]
    obs = observations(panel, weather)
    assert len(obs) == 1                       # BB11 has missing season weather
    x = obs[0].x
    assert (x["gdd"], x["edd"], x["p"]) == (15.0, 2.5, 3.5) and math.isclose(x["p2"], 12.25)
    assert (x[PLACEBO_TERMS[0]], x[PLACEBO_TERMS[1]]) == (0.5, 1.2)
    assert observations(panel, weather, threshold=27)[0].x["gdd"] == 12.0
    assert observations(panel, weather, drop=("AA",)) == []


def test_decisions_need_both_methods_and_a_negative_sign() -> None:
    e1 = {"estimate": -0.05, "p_twoway": 0.001, "p_bootstrap": 0.02}
    e2 = {"estimate": 0.03, "p_twoway": 0.001, "p_bootstrap": 0.001}
    d = decide(e1, e2)
    assert d["E1"]["verdict"] == "established negative"
    assert d["E2"]["verdict"] == "rejected but not negative"
    assert decide({**e1, "p_bootstrap": 0.03}, {**e2, "p_bootstrap": 0.04})["E1"][
        "verdict"] == "not determined"                          # 0.03 x 2 > 0.05 after Holm


@pytest.mark.skipif(not RESULTS.exists(), reason="No committed maize results.")
def test_report_regenerates_from_the_committed_results() -> None:
    command = re.search(r"```\n(poetry run eu-wheat maize-estimate [^\n]+)\n```",
                        REPORT.read_text(encoding="utf-8"))
    assert command is not None
    document = json.loads(RESULTS.read_text(encoding="utf-8"))
    wheat = json.loads(WHEAT.read_text(encoding="utf-8"))
    assert render(document, wheat, command.group(1)) == REPORT.read_text(encoding="utf-8")


@pytest.mark.skipif(not RESULTS.exists(), reason="No committed maize results.")
def test_committed_results_regenerate() -> None:
    """The whole maize estimation, bootstraps included, reproduces results.json."""
    from eu_wheat_weather.analysis import _round
    from eu_wheat_weather.maize_analysis import run

    regenerated = _round(run(ROOT / "docs" / "maize" / "panel.csv",
                             ROOT / "docs" / "maize" / "weather.csv"))
    assert_same(json.loads(RESULTS.read_text(encoding="utf-8")),
                json.loads(json.dumps(regenerated)))


@pytest.mark.skipif(not RESULTS.exists(), reason="No committed maize results.")
def test_the_parts_agree() -> None:
    r = json.loads(RESULTS.read_text(encoding="utf-8"))
    edd = next(c for c in r["coefficients"] if c["term"] == "edd")
    assert math.isclose(edd["estimate"], r["E1"]["estimate"], rel_tol=1e-12)
    for test in (r["E1"], r["E2"], r["E5"]):
        assert test["p_bootstrap"] == max(test["p_bootstrap_country"],
                                          test["p_bootstrap_year"])
