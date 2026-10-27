"""Amendment A1 (METHODS.md, section 8): the committed results and report regenerate."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from tests_support import assert_same

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "amendment" / "results.json"
REPORT = ROOT / "docs" / "AMENDMENT.md"

pytestmark = pytest.mark.skipif(not RESULTS.exists(), reason="No committed amendment results.")


def _command() -> str:
    found = re.search(r"```\n(poetry run eu-wheat amend [^\n]+)\n```",
                      REPORT.read_text(encoding="utf-8"))
    assert found is not None
    return found.group(1)


def test_report_regenerates_from_the_committed_results() -> None:
    from eu_wheat_weather.amendment import render

    registered = json.loads((ROOT / "docs" / "results" / "results.json").read_text(
        encoding="utf-8"))
    document = json.loads(RESULTS.read_text(encoding="utf-8"))
    assert render(document, registered, _command()) == REPORT.read_text(encoding="utf-8")


def test_committed_results_regenerate() -> None:
    """The whole amendment, bootstraps included, reproduces results.json."""
    from eu_wheat_weather.amendment import run
    from eu_wheat_weather.analysis import _round

    args = dict(re.findall(r"--([a-z-]+) (\S+)", _command()))
    regenerated = _round(run(ROOT / args["panel"], ROOT / args["weather"],
                             ROOT / args["maturity"], ROOT / args["crops"],
                             ROOT / args["region-map"]))
    assert_same(json.loads(RESULTS.read_text(encoding="utf-8")),
                json.loads(json.dumps(regenerated)))


def test_the_parts_agree() -> None:
    r = json.loads(RESULTS.read_text(encoding="utf-8"))
    both = next(x for x in r["decomposition"] if x["combination"] == "all")
    assert both["E1"] == r["E1"]
    assert {k: v for k, v in r["E2"].items() if k != "mean_p10_minus_p50_mm"} == both["E2"]
    for test in (r["E1"], r["E2"], r["E3"], r["E5"]):
        assert test["p_bootstrap"] == max(test["p_bootstrap_country"],
                                          test["p_bootstrap_year"])
    flagged = {(f["geo"], f["year"]) for f in r["quality_flags"]}
    assert r["sample"]["excluded_region_years"] == len(flagged)
