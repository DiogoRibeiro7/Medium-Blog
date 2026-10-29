"""Amendment A2 to the maize study (METHODS.md, section 10): pieces and the report."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from tests_support import assert_same

from eu_wheat_weather.analysis import BASE, Obs, estimate
from eu_wheat_weather.crops import WheatRow
from eu_wheat_weather.estimation import t_test, twoway_vcov
from eu_wheat_weather.maize_amendment import (
    EFFECTS,
    _TwoWay,
    equivalents,
    randomization,
    render,
)
from eu_wheat_weather.maize_analysis import decide
from eu_wheat_weather.nuts_changes import Change
from eu_wheat_weather.regions import CodeMatch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "maize" / "amendment" / "results.json"
REGISTERED = ROOT / "docs" / "maize" / "results" / "results.json"
REPORT = ROOT / "docs" / "MAIZE_AMENDMENT.md"


def synthetic(seed: int = 0) -> tuple[list[Obs], dict[tuple[str, int, str], dict[str, str]]]:
    rng = np.random.default_rng(seed)
    obs, weather = [], {}
    for c in ("AA", "BB", "CC", "DD"):
        for r in range(3):
            geo = f"{c}{r + 1}1"
            for year in range(2010, 2018):
                gdd, edd, prec = rng.normal(14, 1), rng.gamma(2, 1), rng.normal(400, 80)
                weather[(geo, year, "season")] = {"gdd_10_30": str(gdd * 100),
                                                  "edd_30": str(edd * 10), "prec": str(prec)}
                p = prec / 100
                x = {"gdd": gdd, "edd": edd, "p": p, "p2": p * p}
                obs.append(Obs(geo, c, year, 2.0 - 0.05 * edd + rng.normal(0, 0.1), 10.0, x))
    return obs, weather


def test_the_fast_two_way_t_is_the_registered_one() -> None:
    obs, _ = synthetic()
    f = estimate(obs, BASE, EFFECTS)
    expected = t_test(f.beta, twoway_vcov(f)[0], 1, 1)[0]
    tw = _TwoWay(f.country, f.year)
    assert np.isclose(tw.t(f.x, f.y, 1), expected, rtol=1e-10)


def test_randomization_is_seeded_and_counts_the_observed_draw() -> None:
    obs, weather = synthetic()
    a = randomization(obs, weather, draws=19, seed=7)
    assert a == randomization(obs, weather, draws=19, seed=7)
    for key in ("E1", "E2"):
        assert 1 / 20 <= a[key]["p"] <= 1
        assert a[key]["p"] * 20 == round(a[key]["p"] * 20)
    assert a["permutations_with_missing_weather"] == 0
    # A region with no weather in a year it is not observed in: permutations that give it
    # that year's weather leave it out.
    del weather[("AA11", 2015, "season")]
    b = randomization([o for o in obs if (o.geo, o.year) != ("AA11", 2015)], weather,
                      draws=19, seed=7)
    assert b["permutations_with_missing_weather"] > 0
    assert all(1 / 20 <= b[k]["p"] <= 1 for k in ("E1", "E2"))


def test_an_older_parent_counts_as_the_region_its_only_child_became() -> None:
    published = [WheatRow(code, 2012, 1.0, 1.0, "", "") for code in ("FR3", "FRE1", "HU1",
                                                                     "HU11", "HU12")]
    matches = [CodeMatch(code, code, "nuts2021", 2021, 0.0, code)
               for code in ("FRE1", "HU11", "HU12")]
    changes = [Change(2013, 2016, "FR30", "FRE1", "recoded")]
    assert equivalents(published, matches, changes) == {"FR3": "FRE1"}


@pytest.mark.skipif(not RESULTS.exists(), reason="No amended maize results.")
def test_the_decisions_and_the_report_regenerate() -> None:
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    registered = json.loads(REGISTERED.read_text(encoding="utf-8"))
    assert_same(results["decisions"], decide(results["E1"], results["E2"]))
    text = REPORT.read_text(encoding="utf-8")
    command = re.search(r"```\n(poetry run eu-wheat maize-amend [^\n]+)\n```", text)
    assert command is not None
    assert render(results, registered, command.group(1)) == text
