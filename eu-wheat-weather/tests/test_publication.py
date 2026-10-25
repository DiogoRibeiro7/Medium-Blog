"""Tests of the article: generated values, no hand-typed results, true claims."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from eu_wheat_weather.publication import PLACEHOLDER, check, render, write_publication

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "publication"
TEMPLATE = PUB / "article.template.md"
RESULTS = ROOT / "docs" / "results" / "results.json"
ACKNOWLEDGEMENT = (
    "We acknowledge the E-OBS dataset from the EU-FP6 project UERRA (https://www.uerra.eu) "
    "and the Copernicus Climate Change Service, and the data providers in the ECA&D project "
    "(https://www.ecad.eu).")
CITATION = ("Cornes, R., G. van der Schrier, E.J.M. van den Besselaar, and P.D. Jones. 2018: "
            "An Ensemble Version of the E-OBS Temperature and Precipitation Datasets, "
            "J. Geophys. Res. Atmos., 123. doi:10.1029/2017JD028200")
# Numbers that are design, not results: thresholds, the season, the test years, the period
# split, the interval level and the elevation cut-off.
DESIGN_CONSTANTS = {"30", "36", "15", "18", "1", "31", "2003", "2018", "2022", "2000", "2011",
                    "2012", "2024", "95", "800"}
NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?")

pytestmark = pytest.mark.skipif(not RESULTS.exists(), reason="No committed results.")


def body_without_acknowledgements() -> str:
    text = TEMPLATE.read_text(encoding="utf-8")
    text = text.split("## Data, code and acknowledgements")[0]
    text = PLACEHOLDER.sub(" ", text)
    return re.sub(r"\]\([^)]*\)", "]", text)          # image and link targets


def test_no_result_is_typed_by_hand() -> None:
    found = {m.group(0) for m in NUMBER.finditer(body_without_acknowledgements())}
    assert sorted(found - DESIGN_CONSTANTS) == [], "use {{placeholders}} for results"


def test_generated_article_and_values_are_current() -> None:
    assert check(ROOT) == []


def test_unknown_and_unused_values_are_errors() -> None:
    with pytest.raises(KeyError, match="Unknown"):
        render("{{a}} {{b}}", {"a": "1"})
    with pytest.raises(KeyError, match="does not use"):
        render("{{a}}", {"a": "1", "b": "2"})


def test_article_carries_the_required_acknowledgement_and_citation() -> None:
    article = re.sub(r"\s+", " ", (PUB / "article.md").read_text(encoding="utf-8"))
    assert ACKNOWLEDGEMENT in article and CITATION in article
    assert "© European Union, Eurostat" in article and "© EuroGeographics" in article


def test_words_in_the_article_match_the_results() -> None:
    r = json.loads(RESULTS.read_text(encoding="utf-8"))
    text = re.sub(r"\s+", " ", TEMPLATE.read_text(encoding="utf-8"))
    d = r["decisions"]
    claims = {
        "And it shows no link": not r["E5"]["significant_both"],
        "this counts as an established negative effect": d["E2"]["verdict"]
        == "established negative",
        "that is not established": d["E1"]["verdict"] == "not determined",
        "there is no detectable change": d["E3"]["verdict"] == "not determined",
        "Rainfall clearly matters": max(d["E2"]["p_twoway_holm"],
                                        d["E2"]["p_bootstrap_holm"]) < 0.05,
        "only days at 36 °C and above clearly lower yields":
            [b["term"] for b in r["M3"] if b["high"] < 0] == ["t_ge36"],
        "None reverses a sign": not any(x["sign_reversed"] for x in r["sensitivity"]),
        "Heat stays harmful in every one": all(
            x["estimate"] < 0 for x in r["sensitivity"] if x["estimand"] == "E1"),
        "which is larger than the heat effect itself": r["E3"]["minimum_detectable"] > abs(
            next(c["estimate"] for c in r["coefficients"]
                 if c["model"] == "M1" and c["term"] == "edd")),
        "The 2003 heatwave did far more damage than": next(
            x for x in r["E4"] if x["year"] == 2003)["eu_yield_observed_pct"]
        < next(x for x in r["E4"] if x["year"] == 2003)["eu_yield_predicted_pct"],
        "so its size is not pinned down": r["E2"]["dry_season"]["high"] > 0,
    }
    for phrase, holds in claims.items():
        assert phrase in text, phrase
        assert holds, phrase


def test_figures_render(tmp_path: Path) -> None:
    import shutil

    for part in ("docs/results", "docs/panel", "publication"):
        shutil.copytree(ROOT / part, tmp_path / part)
    written = write_publication(tmp_path)
    figures = [tmp_path / w for w in written if w.endswith(".png")]
    assert len(figures) == 4 and all(f.stat().st_size > 10_000 for f in figures)
