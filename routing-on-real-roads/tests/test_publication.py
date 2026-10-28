"""The article: generated values, no hand-typed results, true claims."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from routing_roads.contract import PAIRS
from routing_roads.publication import PLACEHOLDER, check, per_day, render
from routing_roads.report import area_indices, load, matrix_summary

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "publication"
TEMPLATE = PUB / "article.template.md"
ESTIMATES = ROOT / "results" / "estimates.json"
REPOSITORY = "https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/routing-on-real-roads"
# Not results: the 1 in the formulas, a bridge's name (25 de Abril), the interval level and
# a citation's year.
DESIGN_CONSTANTS = {"1", "25", "95", "2006"}
NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?")

pytestmark = pytest.mark.skipif(not (ESTIMATES.exists() and TEMPLATE.exists()),
                                reason="No results or article.")


def test_no_result_is_typed_by_hand() -> None:
    text = TEMPLATE.read_text(encoding="utf-8").split("## Data, code and acknowledgements")[0]
    text = re.sub(r"\]\([^)]*\)", "]", PLACEHOLDER.sub(" ", text))     # link targets
    found = {m.group(0) for m in NUMBER.finditer(text)}
    assert sorted(found - DESIGN_CONSTANTS) == [], "use {{placeholders}} for results"


def test_generated_article_and_values_are_current() -> None:
    assert check(ROOT) == []


def test_unknown_and_unused_values_are_errors() -> None:
    with pytest.raises(KeyError, match="Unknown"):
        render("{{a}} {{b}}", {"a": "1"})
    with pytest.raises(KeyError, match="does not use"):
        render("{{a}}", {"a": "1", "b": "2"})


def test_article_carries_the_licence_notice_and_citations() -> None:
    article = re.sub(r"\s+", " ", (PUB / "article.md").read_text(encoding="utf-8"))
    notice = "© OpenStreetMap contributors, available under the Open Database Licence"
    assert notice in article[:article.index("## Two plans")] and REPOSITORY in article
    assert "https://www.openstreetmap.org/copyright" in article
    for doi in ("10.1111/gean.70009", "10.1287/ijoc.2023.0055", "10.1287/mnsc.1050.0451"):
        assert doi in article


def test_words_in_the_article_match_the_results() -> None:
    e = json.loads(ESTIMATES.read_text(encoding="utf-8"))
    d = load(ROOT / "data" / "derived")
    days = per_day(ROOT)
    text = re.sub(r"\s+", " ", TEMPLATE.read_text(encoding="utf-8"))
    areas, water, ph = e["areas"], e["water"], e["post_hoc"]
    lisbon, porto = (p.key for p in PAIRS)
    summary = {(p.key, a): matrix_summary(d.road[p.key], d.straight[p.key],
                                          area_indices(d, p, a == "both"))
               for p in PAIRS for a in ("city", "both")}
    ratio, asym = ({k: s[f] for k, s in summary.items()} for f in ("ratio", "asymmetry"))
    sens = {(s["variant"], s["pair"]): s for s in e["sensitivity"]}
    variations = [k for k in dict.fromkeys(s["variant"] for s in e["sensitivity"])
                  if k != "main"]
    bridges = d.manifest["pairs"][lisbon]["crossings"]["bridges"]
    claims = {
        # the opening and the fact
        "in Lisbon a rise is not established": not water[lisbon]["raises_regret"],
        "in Porto adding the far bank narrowed it": water[porto]["high"] < 0,
        "The other three areas have similar medians and longer upper tails": all(
            abs(ratio[k]["p50"] - ratio[(lisbon, "city")]["p50"]) < 0.05
            and ratio[k]["p95"] > ratio[(lisbon, "city")]["p95"]
            for k in ratio if k != (lisbon, "city")),
        # E1 and E2
        "By the registered rule, the regret exceeds the noise in all four areas":
            all(a["E2"]["exceeds_noise"] for a in areas.values()),
        "The two plans use almost the same number of routes":
            all(abs(a["routes_straight"] - a["routes_road"]) < 0.05 for a in areas.values()),
        # E3
        "every shortest road path between the two banks crosses the 25 de Abril bridge":
            list(bridges.values()) == [1.0] and "25 de Abril" in next(iter(bridges)),
        "The interval includes zero, so a rise is not established":
            water[lisbon]["low"] < 0 < water[lisbon]["high"],
        "The whole interval lies below zero: the far bank lowered the regret":
            water[porto]["high"] < 0,
        "The expectation failed in both cities":
            not any(w["raises_regret"] for w in water.values()),
        "they send more routes across it than the road plans": all(
            areas[f"{p.key}_both"]["cross_bank_straight"]
            > areas[f"{p.key}_both"]["cross_bank_road"] for p in PAIRS),
        "in Lisbon the figure barely moves":
            abs(asym[(lisbon, "both")]["p50"] - asym[(lisbon, "city")]["p50"]) < 0.005
            and asym[(porto, "city")]["p50"] - asym[(porto, "both")]["p50"] > 0.02,
        # E4 and the legs (post hoc)
        "It falls short everywhere": all(
            a["E4"][k]["high"] < 0 for a in areas.values() for k in ("own", "best")),
        "In every area it falls short on every one of the": all(
            x["e_own"].max() < 0 and x["e_best"].max() < 0 for x in days.values()),
        "The estimate's error against that plan's road distance is then exactly": all(
            np.isclose(np.mean(areas[k]["alpha"] / x["kappa_straight"] - 1),
                       areas[k]["E4"]["own"]["estimate"], rtol=1e-8)
            for k, x in days.items()),
        "already above α": all(ph[k]["road"]["circuity"] > a["alpha"]
                               for k, a in areas.items()),
        "and the straight-line plan's most of all": all(
            p["straight"]["circuity"] > p["road"]["circuity"] for p in ph.values()),
        "Routes are chains of short hops":
            all(p["road"]["leg_straight_m"] < p["pair_straight_m"] / 2 for p in ph.values()),
        # sensitivity
        "The regret persists.": all(s[k]["low"] > 0 for s in e["sensitivity"]
                                    for k in ("E1_city", "E1_both")),
        "It grows with the vans' capacity in both cities alone": all(
            sens[("capacity_6", p.key)]["E1_city"]["estimate"]
            < sens[("main", p.key)]["E1_city"]["estimate"]
            < sens[("capacity_24", p.key)]["E1_city"]["estimate"] for p in PAIRS),
        "Porto's change from the far bank is negative in every variation":
            all(s["E3"]["estimate"] < 0 for s in e["sensitivity"] if s["pair"] == porto),
        "negative with the largest vans": [
            k for k in variations if sens[(k, lisbon)]["E3"]["estimate"] < 0]
        == ["capacity_24"],
    }
    for phrase, holds in claims.items():
        assert phrase in text, phrase
        assert holds, phrase


def test_figures_draw(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from routing_roads.estimates import load_solves
    from routing_roads.figures import draw_all

    e = json.loads(ESTIMATES.read_text(encoding="utf-8"))
    paths = draw_all(load(ROOT / "data" / "derived"),
                     load_solves(ROOT / "results" / "solves.csv"), e, tmp_path)
    assert [p.name for p in paths] == sorted(
        p.name for p in (PUB / "figures").glob("fig*.png"))
    assert all(p.stat().st_size > 10_000 for p in paths)
