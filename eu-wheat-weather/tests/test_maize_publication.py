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
# Design, not results: the thresholds, the temperature bands, the test years, the interval
# level and the unit of precipitation.
DESIGN_CONSTANTS = {"30", "36", "27", "33", "15", "18", "12", "2012", "2015", "2022", "95",
                    "100"}
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


def _first(shares: dict[str, float], n: int) -> list[str]:
    return list(shares)[:n]


def test_words_in_the_article_match_the_results() -> None:
    from eu_wheat_weather.maize_analysis import heat_difference
    from eu_wheat_weather.publication import _wheat_harvest_edd_median

    r = json.loads(RESULTS.read_text(encoding="utf-8"))
    a1 = json.loads(WHEAT_A1.read_text(encoding="utf-8"))
    text = re.sub(r"\s+", " ", TEMPLATE.read_text(encoding="utf-8"))
    d, e1, e2, e5, ph = r["decisions"], r["E1"], r["E2"], r["E5"], r["post_hoc"]
    bins = {b["term"]: b for b in r["M3"]}
    clear_two_way = {t for t, b in bins.items() if b["high"] < 0 or b["low"] > 0}
    clear_boot = {t for t, b in bins.items() if b["p_bootstrap"] < 0.05}
    sens = {(x["variant"], x["estimand"]): x["estimate"] for x in r["sensitivity"]}
    e4 = {x["year"]: x for x in r["E4"]}
    with (ROOT / "docs" / "maize" / "panel.csv").open(encoding="utf-8") as stream:
        panel = list(csv.DictReader(stream))
    edd_all = np.array([float(x["edd_30"]) for x in panel])
    by_region: dict[str, list[float]] = {}
    for x in panel:
        by_region.setdefault(x["geo"], []).append(float(x["edd_30"]))
    between = sum(len(v) * (np.mean(v) - edd_all.mean()) ** 2 for v in by_region.values())
    area = {g: sum(float(x["area_kha"]) for x in panel if x["geo"] == g) for g in by_region}
    with (ROOT / "docs" / "maize" / "quality_flags.csv").open(encoding="utf-8") as stream:
        flags = list(csv.DictReader(stream))
    with (ROOT / "docs" / "maize" / "season.csv").open(encoding="utf-8") as stream:
        season = list(csv.DictReader(stream))
    flagged = {(f["geo"], f["year"]) for f in flags}
    prorated_or_carried = {(f["geo"], f["year"]) for f in flags if f["rule"] in ("Q1", "Q2", "Q3")}
    swing, pl, south = ph["heat_swing"], ph["placebo"], ph["south"]
    var, contrib, iv = ph["E1_variance"], ph["E1_contributions"], ph["identifying_variation"]
    loo, cy, ni = ph["leave_one_country_out"], ph["without_largest_country_year"], \
        ph["not_irrigated_E1_variance"]
    lost = {k: [x["country"] for x in loo if x[k]["verdict"] != "established negative"]
            for k in ("E1", "E2")}
    without_es = next(x for x in loo if x["country"] == "ES")["E2"]
    hours = list(ph["hours"].values())
    gap, margin = heat_difference(e1, a1["E1"])
    south_var = sum(s for c, s in iv["edd"].items() if c in ("ES", "PT", "IT", "EL"))
    south_area = sum(s for c, s in ph["maize_area"].items() if c in ("ES", "PT", "IT", "EL"))
    greek_later = [x for x in panel if x["country"] == "EL"
                   and int(x["year"]) >= ph["estimated_rows"]["years"][0]]
    methods = (ROOT / "METHODS.md").read_text(encoding="utf-8")
    claims = {
        # the data and the comparison
        "with the same dates every year": "year" not in season[0]
        and len({s["geo"] for s in season}) == len(season),
        "mostly pro-rated or carried forward": len(prorated_or_carried) > len(flagged) / 2,
        "But most of that spread is between regions":
            between / float(np.sum((edd_all - edd_all.mean()) ** 2)) > 0.5,
        "So ten more degree days make a hot season for that region":
            swing["median_change"] < 10 and swing["share_changes_10"] < 0.5,
        "Two later entries in the plan's change log settle details it had left open":
            "- **2026-10-26, maize data, before any maize estimate" in methods
            and "- **2026-10-27, maize estimates" in methods,
        # the placebo
        "The second method does not": e5["p_bootstrap"] >= 0.05,
        "so the check passes": not e5["significant_both"],
        "The first method does find one": e5["p_twoway"] < 0.05,
        "What it finds is rain after maturity going with higher yields":
            pl["terms"]["p_post_maturity"]["estimate"] > 0
            and pl["terms"]["p_post_maturity"]["p_twoway"] < 0.05
            and pl["terms"]["edd_post_maturity"]["p_twoway"] >= 0.05,
        "mostly in France: without France, its p rises":
            pl["without_country"]["country"] == "FR"
            and pl["without_country"]["p_twoway"] >= 0.05,
        "That rain is unrelated to the season's own rain": abs(pl["rain_vs_season_rain"]) < 0.1,
        "adding it to the model leaves the heat estimate at":
            abs(pl["E1_with_placebo_terms"]["estimate"] - e1["estimate"]) < 0.005,
        # heat
        "that clears the bar: corrected p": d["E1"]["verdict"] == "established negative",
        "But it clears it by a hair":
            max(d["E1"]["p_twoway_holm"], d["E1"]["p_bootstrap_holm"]) >= 0.04,
        "slightly smaller than with errors grouped by country alone":
            var["country"]["se"] / 1.05 < var["twoway"]["se"] < var["country"]["se"]
            and var["twoway"]["p"] < 0.05 <= var["country"]["p"],
        "Single countries decide it": len(lost["E1"]) >= 3,
        "Romania among them": "RO" in lost["E1"],
        "So does leaving out only the": cy["country_year"] == "RO 2024"
        and cy["E1"]["verdict"] != "established negative",
        "Romania supplies": _first(contrib["country"], 1) == ["RO"],
        "one small region, Bucharest and its surroundings":
            _first(contrib["region"], 1) == ["RO32"]
            and area["RO32"] / sum(area.values()) < 0.01,
        "The pooled number describes neither group":
            south["south"]["low"] < 0 < south["south"]["high"]
            and south["elsewhere"]["high"] < e1["estimate"],
        # the bands
        "An average season has": hours == sorted(hours, reverse=True),
        "hints that the harm begins below 30 °C, though the pattern is irregular":
            "t_27_30" in clear_two_way & clear_boot
            and "t_30_33" not in clear_two_way & clear_boot,
        "only time at 27–30 °C and at 33–36 °C differs from zero":
            clear_two_way & clear_boot == {"t_27_30", "t_33_36"},
        "One cool band, 12–15 °C, differs under the first method only":
            clear_two_way - clear_boot == {"t_12_15"},
        "Time at 36 °C and above has the largest estimate":
            min(bins, key=lambda t: bins[t]["estimate"]) == "t_ge36",
        "but it is rare, Spain supplies": _first(iv["t_ge36"], 1) == ["ES"]
        and iv["t_ge36"]["ES"] > 0.5,
        "the second method puts it right at the line":
            0.045 <= bins["t_ge36"]["p_bootstrap"] <= 0.055,
        # drought
        "that is established: corrected p": d["E2"]["verdict"] == "established negative",
        "This result is much sturdier than heat's": len(lost["E2"]) < len(lost["E1"]),
        "Under the first method it stays clear whichever single country is left out":
            all(x["E2"]["p_twoway_holm"] < 0.05 for x in loo),
        "Under the full rule it fails in one case: without Spain": lost["E2"] == ["ES"],
        "that result survives leaving out any single country except Spain":
            lost["E2"] == ["ES"],
        "the second method no longer confirms it once the correction for two questions is "
        "applied": without_es["p_bootstrap"] < 0.05 <= without_es["p_bootstrap_holm"],
        # beside wheat
        "and neither is established": all(
            x["verdict"] == "not determined" for k, x in a1["decisions"].items()
            if k in ("E1", "E2")),
        "The drought estimates are close, but only maize's is precise enough to count":
            abs(e2["estimate"] - a1["E2"]["estimate"]) < 0.01,
        "Maize's heat estimate is larger, but the two cannot be told apart":
            e1["estimate"] < a1["E1"]["estimate"] and abs(gap) < margin,
        "Ten more degree days are also a far bigger shock for wheat":
            3 * _wheat_harvest_edd_median(ROOT) < float(np.median(edd_all)),
        # robustness
        "None reverses a sign": not any(x["sign_reversed"] for x in r["sensitivity"]),
        "Heat is larger without the irrigated south":
            sens[("not_irrigated", "E1")] < e1["estimate"]
            and sens[("area_weighted", "E1")] < e1["estimate"],
        "Drought is larger with area weights":
            sens[("area_weighted", "E2")] < e2["estimate"] < sens[("not_irrigated", "E2")],
        "Both heat variations lean on Romania": _first(iv["edd_not_irrigated"], 1)
        == _first(iv["edd_area_weighted"], 1) == ["RO"],
        "the narrow interval without the south comes from the first method's arithmetic":
            ni["twoway"]["se"] < min(ni[k]["se"] for k in ("country", "year", "country_year")),
        # prediction
        "The 2015 total is predicted well, but in 2012 and 2022 the losses were far larger":
            abs(e4[2015]["eu_yield_observed_pct"] - e4[2015]["eu_yield_predicted_pct"]) < 1
            and all(e4[y]["eu_yield_observed_pct"] < e4[y]["eu_yield_predicted_pct"] - 5
                    for y in (2012, 2022)),
        "Romania and Hungary account for": all(
            set(_first(ph["E4_gap_by_country"][y], 2)) == {"RO", "HU"} for y in ("2012", "2022")),
        "in 2022 the predictions explain none of the differences between regions":
            e4[2022]["variance_explained"] < 0,
        # what this does not say
        "Not every country equally.** Spain supplies":
            _first(iv["edd"], 3) == ["ES", "IT", "RO"],
        "together supply": south_var > 2 * south_area,
        "Probably too small.** Only": swing["variance_left"] < 0.1,
        "Greek regional harvests from": ph["estimated_rows"]["countries"] == ["EL"]
        and len(greek_later) == ph["estimated_rows"]["region_years"],
    }
    for phrase, holds in claims.items():
        assert phrase in text, phrase
        assert holds, phrase
