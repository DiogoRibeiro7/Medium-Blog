"""Reproducible wheat results, provenance, publication notices, and figure checks."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from eu_wheat_weather.publication import check, render, write_publication

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "publication"
RESULTS = ROOT / "docs" / "results" / "results.json"
AMENDED = ROOT / "docs" / "amendment" / "results.json"

pytestmark = pytest.mark.skipif(not RESULTS.exists(), reason="No committed results.")


def test_public_articles_are_short_compatibility_notices() -> None:
    """Keep editorial prose and templates in the private archive."""
    wheat = (PUB / "article.md").read_text(encoding="utf-8")
    maize = (PUB / "maize" / "article.md").read_text(encoding="utf-8")
    assert all(len(notice.split()) < 125 for notice in (wheat, maize))
    assert all("research-articles-medium" in notice for notice in (wheat, maize))
    assert all("values.json" in notice for notice in (wheat, maize))
    assert not (PUB / "article.template.md").exists()
    assert not (PUB / "maize" / "article.template.md").exists()


def test_generated_article_and_values_are_current() -> None:
    assert check(ROOT) == []


def test_unknown_and_unused_values_are_errors() -> None:
    with pytest.raises(KeyError, match="Unknown"):
        render("{{a}} {{b}}", {"a": "1"})
    with pytest.raises(KeyError, match="does not use"):
        render("{{a}}", {"a": "1", "b": "2"})


def test_required_data_attribution_remains_in_the_public_readme() -> None:
    """Check scientific citation and licensing without reading editorial prose."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    required = (
        "We acknowledge the E-OBS dataset",
        "Copernicus Climate Change Service",
        "Cornes, R.",
        "10.1029/2017JD028200",
        "© European Union, Eurostat",
        "© EuroGeographics",
        "non-commercial",
        "E-OBS v33.0e",
        "GGCMI Phase 3 crop calendar v1.01",
        "doi:10.5281/zenodo.5062513",
        "MIRCA-OS v0.1",
        "doi:10.1038/s41597-024-04313-w",
        "doi:10.4211/hs.60a890eb841c460192c03bb590687145",
    )
    for citation in required:
        assert citation in readme, citation


def _pct(log_points: float) -> float:
    return 100 * (math.exp(log_points) - 1)


def _ends_only(flags: list[bool]) -> bool:
    """True where the flagged points are a run at the start and/or a run at the end."""
    inner = flags[next((i for i, f in enumerate(flags) if not f), len(flags)):]
    inner = inner[:len(inner) - next((i for i, f in enumerate(inner[::-1]) if not f),
                                     len(inner))]
    return not any(inner)


def test_registered_wheat_claims_match_the_results() -> None:
    r = json.loads(RESULTS.read_text(encoding="utf-8"))
    d, ph, dry = r["decisions"], r["post_hoc"], r["E2"]["dry_season"]
    edd = next(c for c in r["coefficients"] if c["model"] == "M1" and c["term"] == "edd")
    sens = {(x["variant"], x["estimand"]): x for x in r["sensitivity"]}
    heat_e1 = {x["variant"]: x for x in r["sensitivity"] if x["estimand"] == "E1"}
    curve = r["precipitation_curve"]
    excluded = [p["low"] > 0 or p["high"] < 0 for p in curve["points"]]
    excluded_mm = [p["mm"] for p, e in zip(curve["points"], excluded, strict=True) if e]
    median, dry_mm = curve["median_mm"], abs(dry["mean_p10_minus_p50"])
    opt, se, boot = ph["optimum_mm"], ph["ge36_se"], ph["M3_p_bootstrap"]
    y2003 = ph["E4_2003"]
    gap, heat, worst = y2003["gap_by_country"], y2003["heat_by_country"], y2003["worst_regions"]
    e4_2003 = next(x for x in r["E4"] if x["year"] == 2003)
    ratio = {c: heat[c]["edd_30"] / heat[c]["edd_30_other_years"] for c in ("FR", "HU")}
    areas = [w["area_kha"] for w in worst]
    with (ROOT / "docs" / "panel" / "panel.csv").open(encoding="utf-8") as stream:
        quartiles = np.quantile([float(x["edd_30"]) for x in csv.DictReader(stream)],
                                [0.25, 0.75])
    claims = {
        # the placebo: neither method detects a link
        "The test finds no detectable link": min(r["E5"]["p_twoway"],
                                                 r["E5"]["p_bootstrap"]) >= 0.05,
        # drought
        "Rainfall as a whole matters": max(d["E2"]["p_twoway_holm"],
                                           d["E2"]["p_bootstrap_holm"]) < 0.05,
        "and it includes the": opt["low"] < median < opt["high"],
        "a loss or a small gain": dry["low"] < 0 < dry["high"],
        "The rule I fixed in advance called this an established negative effect":
            d["E2"]["verdict"] == "established negative",
        "the curve differs from the median only at the extremes, mostly at the wet end, not "
        "where a typical dry season falls": _ends_only(excluded)
        and sum(m > median for m in excluded_mm) > sum(m < median for m in excluded_mm)
        and not any(median - dry_mm <= m <= median for m in excluded_mm),
        "So the cost of a dry spring is not established": not (
            ph["E2_direct"]["p_twoway_holm"] < 0.05
            and ph["E2_direct"]["p_bootstrap_holm"] < 0.05),
        # heat
        "is a large step": 10 > quartiles[1] - quartiles[0],
        "After the correction for three questions, that is not established":
            d["E1"]["verdict"] == "not determined",
        "of the variation in heat that it uses, Spain supplies":
            list(ph["identifying_variation"]["edd"])[:3] == ["ES", "IT", "RO"],
        "Time at 36 °C and above is the only band whose interval lies wholly below zero":
            [b["term"] for b in r["M3"] if b["high"] < 0] == ["t_ge36"],
        "**One country.** Spain supplies":
            list(ph["identifying_variation"]["t_ge36"])[0] == "ES",
        "gives a larger error every time": se["twoway"] < min(
            se["country"], se["year"], se["country_year"]),
        "but so does time at": boot["t_ge36"] < 0.05
        and any(p < 0.05 for b, p in boot.items() if b != "t_ge36"),
        # E3
        "there is no detectable change": d["E3"]["verdict"] == "not determined",
        "which is larger than the average heat effect itself":
            r["E3"]["minimum_detectable"] > abs(edd["estimate"]),
        # E4, 2003
        "The model missed 2003 badly": e4_2003["eu_yield_observed_pct"]
        < e4_2003["eu_yield_predicted_pct"] - 5,
        "France, the largest single part of the miss, was hot that year":
            list(gap)[0] == "FR" and ratio["FR"] > 1,
        "Romania and Hungary together account for more of the miss than France":
            gap["RO"] + gap["HU"] < gap["FR"],
        "Romania was cooler than usual": heat["RO"]["edd_30"] < heat["RO"]["edd_30_other_years"],
        "Hungary only a little hotter": 1 < ratio["HU"] < ratio["FR"],
        "In Romania's three worst-hit regions": all(w["geo"].startswith("RO") for w in worst)
        and all(w["observed"] < 0 for w in worst),
        "while the model predicted almost no change": all(
            abs(_pct(w["predicted"])) < 5 for w in worst),
        "in two of them the area harvested also fell by about a third that year and "
        "recovered the next": sum(
            1 for a in areas if {"2002", "2003", "2004"} <= set(a)
            and 0.6 < a["2003"] / a["2002"] < 0.75 and a["2004"] / a["2002"] > 0.9) == 2,
        # sensitivity
        "None reverses the sign": not any(x["sign_reversed"] for x in r["sensitivity"]),
        "excludes zero in only": {v for v, x in heat_e1.items() if x["high"] < 0}
        == {"trend_year", "area_weighted"},
        "The heat estimate is smallest with the water balance": max(
            heat_e1.values(), key=lambda x: x["estimate"])["variant"] == "water_balance",
        "The dry-season effect shrinks to": abs(sens[("apr_jul", "E2")]["estimate"])
        < abs(dry["effect"]) and abs(sens[("area_weighted", "E2")]["estimate"])
        < abs(dry["effect"]),
    }
    for claim, holds in claims.items():
        assert holds, claim


def test_amended_wheat_claims_match_the_results() -> None:
    """Amendment A1 (METHODS.md, section 8), as the article reports it."""
    r = json.loads(RESULTS.read_text(encoding="utf-8"))
    a = json.loads(AMENDED.read_text(encoding="utf-8"))
    dry = r["E2"]["dry_season"]["effect"]
    edd = next(c["estimate"] for c in r["coefficients"]
               if c["model"] == "M1" and c["term"] == "edd")
    ge36 = next(b for b in a["M3"] if b["term"] == "t_ge36")
    loo = a["leave_one_out"]["summary"]["E1"]
    flags = {(f["geo"][:2], f["rule"]) for f in a["quality_flags"]}
    swapped = next(c for c in a["review_cases"] if c["case"].startswith("NL22"))
    data = next(x for x in a["decomposition"] if x["combination"] == "data")["E1"]
    claims = {
        "The estimates change little, and none is established":
            all(x["verdict"] == "not determined" for x in a["decisions"].values())
            and abs(a["E1"]["estimate"] - edd) < 0.01 and abs(a["E2"]["estimate"] - dry) < 0.01,
        "still finds no detectable link": min(a["E5"]["p_twoway"], a["E5"]["p_bootstrap"])
        >= 0.05,
        "no longer differs from zero once the season ends at harvest":
            ge36["low"] < 0 < ge36["high"] and ge36["p_bootstrap"] >= 0.05,
        "the band no longer differs from zero": ge36["low"] < 0 < ge36["high"],
        "Spain now pulls the heat estimate the other way": loo["most_influential"] == "ES"
        and loo["without_it"] < a["E1"]["estimate"]
        and r["post_hoc"]["E1_without_spain"]["estimate"] > edd,
        "among them Greece's pro-rated early years and what look like pro-rated series in "
        "Czechia": {("EL", "Q2"), ("CZ", "Q2")} <= flags,
        "but it misses at least one case the review found": swapped["in_panel"] > 0
        and swapped["excluded"] == 0,
        "Dropping the suspect yields on its own gives heat an interval that only just "
        "excludes zero": data["high"] < 0 < data["high"] + 0.005,
        "But the second way of measuring uncertainty does not confirm it":
            data["p_bootstrap"] >= 0.05,
        "the interval widens again once the season also ends at harvest":
            a["E1"]["high"] - a["E1"]["low"] > data["high"] - data["low"] and a["E1"]["high"] > 0,
    }
    for claim, holds in claims.items():
        assert holds, claim


def test_figures_render(tmp_path: Path) -> None:
    import shutil

    for part in ("docs/results", "docs/panel", "docs/amendment", "docs/maize", "publication"):
        shutil.copytree(ROOT / part, tmp_path / part)
    written = write_publication(tmp_path)
    figures = [tmp_path / w for w in written if w.endswith(".png")]
    # five for the wheat article, five for the maize article
    assert len(figures) == 10 and all(f.stat().st_size > 10_000 for f in figures)


def test_maize_summary_matches_its_results() -> None:
    maize = ROOT / "docs" / "maize" / "results" / "results.json"
    if not maize.exists():
        pytest.skip("No maize results.")
    r = json.loads(maize.read_text(encoding="utf-8"))
    m, ph, e5 = r["decisions"], r["post_hoc"], r["E5"]
    lost = {key: sum(1 for x in ph["leave_one_country_out"]
                     if x[key]["verdict"] != "established negative") for key in ("E1", "E2")}
    claims = {
        "By the study's rules, a dry season lowers maize yields":
            m["E2"]["verdict"] == "established negative",
        "and that result is the sturdier of the two": lost["E2"] < lost["E1"],
        "clears the bar only just":
            m["E1"]["verdict"] == "established negative"
            and max(m["E1"]["p_twoway_holm"], m["E1"]["p_bootstrap_holm"]) >= 0.04,
        "it rests largely on Romania":
            next(iter(ph["E1_contributions"]["country"])) == "RO",
        "its check on the weather after maturity is split between the two methods":
            (e5["p_twoway"] < 0.05) != (e5["p_bootstrap"] < 0.05),
        "maize compatibility notice": (ROOT / "publication" / "maize" / "article.md").exists(),
    }
    amended = ROOT / "docs" / "maize" / "amendment" / "results.json"
    if amended.exists():
        a2 = json.loads(amended.read_text(encoding="utf-8"))
        ro = next(x for x in a2["leave_one_out"]["rows"] if x["country"] == "RO")
        methods = (ROOT / "METHODS.md").read_text(encoding="utf-8")
        claims |= {
            "A second maize analysis, registered after a critical review":
                "## 10. Amendment A2 to the maize study" in methods,
            "both results hold": all(a2["decisions"][k]["verdict"] == "established negative"
                                     for k in ("E1", "E2")),
            "without Romania heat is smaller":
                abs(ro["E1"]) < abs(a2["E1"]["estimate"]),
            "but still clear with two-way clustering": ro["E1_p_twoway"] < 0.05,
        }
    for claim, holds in claims.items():
        assert holds, claim
