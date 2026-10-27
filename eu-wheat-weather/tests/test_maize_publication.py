"""Tests of the maize article: generated values, no hand-typed results, true claims."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pytest

from eu_wheat_weather.publication import PLACEHOLDER

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "publication" / "maize"
TEMPLATE = PUB / "article.template.md"
RESULTS = ROOT / "docs" / "maize" / "results" / "results.json"
WHEAT_A1 = ROOT / "docs" / "amendment" / "results.json"
ACKNOWLEDGEMENT = (
    "We acknowledge the E-OBS dataset from the EU-FP6 project UERRA (https://www.uerra.eu) "
    "and the Copernicus Climate Change Service, and the data providers in the ECA&D project "
    "(https://www.ecad.eu).")
CITATION = ("Cornes, R., G. van der Schrier, E.J.M. van den Besselaar, and P.D. Jones. 2018: "
            "An Ensemble Version of the E-OBS Temperature and Precipitation Datasets, "
            "J. Geophys. Res. Atmos., 123. doi:10.1029/2017JD028200")
# Design, not results: the thresholds, the temperature bands, the test years and the
# interval level.
DESIGN_CONSTANTS = {"30", "36", "27", "33", "15", "18", "12", "2012", "2015", "2022", "95"}
NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?")

pytestmark = pytest.mark.skipif(not (RESULTS.exists() and TEMPLATE.exists()),
                                reason="No maize results or article.")


def test_no_result_is_typed_by_hand() -> None:
    text = TEMPLATE.read_text(encoding="utf-8").split("## Data, code and acknowledgements")[0]
    text = re.sub(r"\]\([^)]*\)", "]", PLACEHOLDER.sub(" ", text))
    found = {m.group(0) for m in NUMBER.finditer(text)}
    assert sorted(found - DESIGN_CONSTANTS) == [], "use {{placeholders}} for results"


def test_article_carries_the_notices_acknowledgement_and_citations() -> None:
    article = re.sub(r"\s+", " ", (PUB / "article.md").read_text(encoding="utf-8"))
    assert ACKNOWLEDGEMENT in article and CITATION in article
    intro = article[:article.index("## The data")]
    assert "© European Union, Eurostat" in intro and "© EuroGeographics" in intro
    assert "for non-commercial use" in intro and "E-OBS v33.0e" in intro
    assert ("GGCMI Phase 3 crop calendar v1.01, CC BY 4.0" in article
            and "doi:10.5281/zenodo.5062513" in article)


def test_words_in_the_article_match_the_results() -> None:
    r = json.loads(RESULTS.read_text(encoding="utf-8"))
    a1 = json.loads(WHEAT_A1.read_text(encoding="utf-8"))
    text = re.sub(r"\s+", " ", TEMPLATE.read_text(encoding="utf-8"))
    d, e1, e2, e5 = r["decisions"], r["E1"], r["E2"], r["E5"]
    bins = {b["term"]: b for b in r["M3"]}
    clear_two_way = {t for t, b in bins.items() if b["high"] < 0 or b["low"] > 0}
    clear_boot = {t for t, b in bins.items() if b["p_bootstrap"] < 0.05}
    sens = {(x["variant"], x["estimand"]): x["estimate"] for x in r["sensitivity"]}
    e4 = {x["year"]: x for x in r["E4"]}
    with (ROOT / "docs" / "maize" / "panel.csv").open(encoding="utf-8") as stream:
        edd = np.quantile([float(x["edd_30"]) for x in csv.DictReader(stream)], [0.25, 0.75])
    with (ROOT / "docs" / "maize" / "quality_flags.csv").open(encoding="utf-8") as stream:
        flags = list(csv.DictReader(stream))
    flagged = {(f["geo"], f["year"]) for f in flags}
    prorated_or_carried = {(f["geo"], f["year"]) for f in flags if f["rule"] in ("Q1", "Q2", "Q3")}
    claims = {
        "so ten more is well within the usual swing": edd[1] - edd[0] > 10,
        "mostly pro-rated or carried forward": len(prorated_or_carried) > len(flagged) / 2,
        "One of the two methods finds a link":
            (e5["p_twoway"] < 0.05) != (e5["p_bootstrap"] < 0.05),
        "so it passes, but only narrowly": not e5["significant_both"],
        "that is established, but only just": d["E1"]["verdict"] == "established negative"
        and max(d["E1"]["p_twoway_holm"], d["E1"]["p_bootstrap_holm"]) >= 0.04,
        "only time at 27–30 °C and at 33–36 °C differs from zero":
            clear_two_way & clear_boot == {"t_27_30", "t_33_36"},
        "One cool band, 12–15 °C, differs under the first method only":
            clear_two_way - clear_boot == {"t_12_15"},
        "Time at 36 °C and above has the largest estimate":
            min(bins, key=lambda t: bins[t]["estimate"]) == "t_ge36",
        "the second method puts it right at the line":
            0.045 <= bins["t_ge36"]["p_bootstrap"] <= 0.055,
        "that is established: corrected p": d["E2"]["verdict"] == "established negative",
        "and neither is established": all(
            x["verdict"] == "not determined" for k, x in a1["decisions"].items()
            if k in ("E1", "E2")),
        "The drought estimates are close, but only maize's is precise enough to count, and "
        "maize's heat estimate is larger": abs(e2["estimate"] - a1["E2"]["estimate"]) < 0.01
        and e1["estimate"] < a1["E1"]["estimate"],
        "None reverses a sign, and the effects grow in two of them":
            not any(x["sign_reversed"] for x in r["sensitivity"])
            and sens[("not_irrigated", "E1")] < e1["estimate"]
            and sens[("area_weighted", "E1")] < e1["estimate"]
            and sens[("area_weighted", "E2")] < e2["estimate"],
        "2015 is predicted well, but in 2012 and 2022 the losses were far larger":
            abs(e4[2015]["eu_yield_observed_pct"] - e4[2015]["eu_yield_predicted_pct"]) < 1
            and all(e4[y]["eu_yield_observed_pct"] < e4[y]["eu_yield_predicted_pct"] - 5
                    for y in (2012, 2022)),
    }
    for phrase, holds in claims.items():
        assert phrase in text, phrase
        assert holds, phrase
