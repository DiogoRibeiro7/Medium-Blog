"""The article: generated values, no hand-typed results, true claims."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from iberian_lp.publication import PLACEHOLDER, check, daily, render, table

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "publication"
TEMPLATE = PUB / "article.template.md"
ESTIMATES = ROOT / "results" / "estimates.json"
REPOSITORY = "https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/iberian-electricity-lp"
# Not results: the bounds in the programme (0 ≤ s), the 95% level and citations' years.
DESIGN_CONSTANTS = {"0", "95", "2005", "2008", "2014"}
NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,:]\d+)?")

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


def test_article_cites_the_source_and_the_software() -> None:
    article = re.sub(r"\s+", " ", (PUB / "article.md").read_text(encoding="utf-8"))
    source = "Data: OMIE (OMI-Polo Español, S.A.), [omie.es](https://www.omie.es)"
    assert source in article[:article.index("## A market is a linear programme")]
    assert "keeping its original content intact and citing the source" in article
    assert REPOSITORY in article
    for doi in ("10.1007/s12532-017-0130-5", "10.1038/s41592-019-0686-2",
                "10.1016/j.enpol.2008.03.035"):
        assert doi in article


def _accepted_never_above_offered_below_price() -> bool:
    """Scalable complex orders: in every Spanish or joint quarter-hour, the accepted volume
    is at most the offered volume priced at or below the published price."""
    from iberian_lp.experiment import SPANISH, fixed, read_steps
    from iberian_lp.report import load

    d = load(ROOT / "data" / "derived")
    for day in d.days:
        steps = read_steps(ROOT / "data" / "derived" / "steps" / f"{day.replace('-', '')}.csv.gz")
        for r in (x for x in d.periods if x["date"] == day and x["zone"] in SPANISH):
            prices, qty = steps[(int(r["period"]), r["zone"])].get(("V", "C02"), ([], []))
            price = fixed(r["price_es"], 2)
            if fixed(r["matched_sell_C02"], 1) > sum(q for p, q in zip(prices, qty, strict=True)
                                                     if p <= price):
                return False
    return True


def test_words_in_the_article_match_the_results() -> None:
    e = json.loads(ESTIMATES.read_text(encoding="utf-8"))
    e1, e2, e3, e4, ph = (e[k] for k in ("E1", "E2", "E3", "E4", "post_hoc"))
    desc = e["descriptive"]
    text = re.sub(r"\s+", " ", TEMPLATE.read_text(encoding="utf-8"))
    methods = re.sub(r"\s+", " ", (ROOT / "METHODS.md").read_text(encoding="utf-8"))
    sens = e3["sensitivity"]
    ratio_small = e3["mean"]["estimate"] / sens["500 MW"]["mean"]["estimate"]
    ratio_large = sens["2000 MW"]["mean"]["estimate"] / e3["mean"]["estimate"]
    floor, line, blocks, zones = ph["floor"], ph["line"], ph["blocks"], ph["zones"]
    bands = [b for b in floor["bands"] if b["midday"]["periods"] and b["outside"]["periods"]]
    larger = sum(1 for b in bands if b["midday"]["mean_abs"] > b["outside"]["mean_abs"])
    all_zones_inside = e1["misses"]["count"] == 0 and all(
        z["inside"] == z["zone_periods"] for z in e1["by_zone"].values())
    periods = desc["periods"]
    mostly_down = e2["all"]["down"] > periods / 2
    nonpos = ph["nonpositive"]
    share_mid = nonpos["midday"]["count"] / nonpos["midday"]["periods"]
    share_out = nonpos["outside"]["count"] / nonpos["outside"]["periods"]
    hours = {int(h): v["estimate"] for h, v in e3["by_hour"].items()}
    t, _ = table(ROOT)
    days = daily(t)
    claims = {
        # the summary
        "quarter-hours, in Spain and in Portugal alike": all_zones_inside,
        "it would have lowered the price in most quarter-hours, by":
            e2["all"]["mean"]["high"] < 0 and mostly_down,
        "even when the quarter-hours are restricted to those in which the line to France was full":
            line["full"]["delta"]["high"] < 0,
        "MW more at zero price would have lowered the price by": e3["mean"]["high"] < 0,
        "and by less around midday than at other hours, as I had expected":
            e3["midday_smaller"],
        "The gap is explained by the first half of the reason I gave":
            e3["midday_smaller"] and abs(ph["positive"]["G"]["estimate"])
            < 0.25 * e3["G"]["estimate"],
        "The second half, a flatter supply curve at midday, finds no support in the data":
            larger > len(bands) / 2,
        # the order types
        "a large part of what Iberian generators offer is submitted through orders":
            _nonconvex_share() > 0.25,
        "only a small part of what was offered through them was accepted":
            max(desc["accepted_share"].values()) < 0.25,
        "the rejected part, as the results will show, was not all priced above":
            e2["sco"]["down"] > 0,
        "each with its own supply and demand and, usually, its own price":
            _split_prices_differ() > 0.5,
        # E1
        "The published price lay inside the interval in every one of the": all_zones_inside,
        "matched the published accepted volume within": e1["volume"]["mismatched"] == 0,
        "the shadow prices formed a genuine interval rather than a single point":
            desc["wide"]["count"] > 0,
        # E2
        "That expectation failed": e2["excludes_zero"] or e2["all"]["moved"]["estimate"] > 0.5,
        "relaxing the non-convex orders lowered the price in most quarter-hours. The plan's":
            "relaxing the non-convex orders lowered the price in most periods" in methods,
        "The interval excludes zero, so by the rule fixed in advance the non-convex orders move":
            e2["excludes_zero"],
        "The effect keeps most of its size under that restriction":
            line["full"]["delta"]["high"] < 0
            and line["full"]["delta"]["estimate"] / e2["all"]["mean"]["estimate"] > 0.5,
        "For the relaxation of every order the overstatement is small":
            abs(blocks["all_dedup"]["estimate"] - e2["all"]["mean"]["estimate"])
            < 0.1 * abs(e2["all"]["mean"]["estimate"]),
        "so the effect is larger where the relaxation reaches the whole market":
            zones["joint"]["estimate"] < zones["split"]["estimate"],
        "Split quarter-hours are a minority, and the pooled mean stays close to the joint one":
            desc["split"] < periods / 2
            and abs(e2["all"]["mean"]["estimate"] - zones["joint"]["estimate"])
            < abs(e2["all"]["mean"]["estimate"] - zones["split"]["estimate"]),
        "although the parts do not add up to the whole": abs(e2["sco"]["mean"]["estimate"]
                                                         + e2["blocks"]["mean"]["estimate"]
                                                         - e2["all"]["mean"]["estimate"]) > 0.5,
        "as much as relaxing every order": abs(e2["sco"]["mean"]["estimate"])
        >= 0.95 * abs(e2["all"]["mean"]["estimate"]),
        "It never raised the price in any quarter-hour": e2["sco"]["up"] == 0,
        "never exceeded their offered volume priced at or below the price":
            _accepted_never_above_offered_below_price(),
        "With the exclusive blocks left as they were and only the simple blocks relaxed":
            blocks["c01_only"]["estimate"] > e2["blocks"]["mean"]["estimate"],
        "The blocks also move the price in both directions":
            e2["blocks"]["up"] > 0 and e2["blocks"]["down"] > 0,
        "the scalable complex orders alone move the price slightly more than every order":
            e2["sco"]["mean"]["estimate"] < e2["all"]["mean"]["estimate"],
        "every day's mean lies below zero in both rows":
            bool(np.all(days["all"] < 0) and np.all(days["sco"] < 0)),
        "The block row sits close to zero on most days, with a tail of days":
            float(np.median(days["blocks"])) > -1 and float(np.min(days["blocks"])) < -3,
        "the bias is not confined to a few hours but appears in most quarter-hours": mostly_down,
        # E3
        "The effect grows roughly in proportion to the size":
            1.8 < ratio_small < 2.2 and 1.8 < ratio_large < 2.2,
        "MW lowers the price by": sens["500 MW"]["mean"]["high"] < 0
        and sens["2000 MW"]["mean"]["high"] < 0,
        "By that rule the effect is smaller at midday": e3["midday_smaller"],
        "The interval lies above zero, so the expectation holds as a statement about G":
            e3["G"]["low"] > 0,
        "only half of the reason holds": larger > len(bands) / 2,
        "prices at or below zero were a midday phenomenon": share_mid > 5 * share_out,
        "and in every one of them m was zero": ph["nonpositive_max_abs_m"] == 0,
        "Those zeros alone account for the gap": abs(ph["positive"]["G"]["estimate"])
        < 0.25 * e3["G"]["estimate"],
        "The bound that m cannot exceed the price in size holds at every price":
            floor["above_price"] == 0 and floor["below_zero"] == 0,
        "midday prices above zero were lower too":
            floor["midday"]["median_price"] < floor["outside"]["median_price"],
        "it gives the flatter-curve explanation no support": larger > len(bands) / 2,
        "The effect is smallest in the solar hours and largest in the evening":
            max(hours, key=hours.get) in range(11, 18)
            and min(hours, key=hours.get) in range(19, 24),
        "In the quarter-hours with the line full, the effect is smaller":
            line["full"]["m"]["estimate"] > line["room"]["m"]["estimate"],
        # E4
        "Without the imports, the price rises; without the exports, it falls":
            e4["imports"]["low"] > 0 and e4["exports"]["high"] < 0,
        "the response to losing imports was about twice the response to losing exports":
            1.7 < abs(e4["imports"]["estimate"] / e4["exports"]["estimate"]) < 2.3,
        "the market imports when Iberian prices are high":
            "A plausible reading, which the study did not test" in text,
    }
    for phrase, holds in claims.items():
        assert phrase in text, phrase
        assert holds, phrase


def _split_prices_differ() -> float:
    """Of the split quarter-hours, the share in which the two published prices differ."""
    from iberian_lp.report import load

    d = load(ROOT / "data" / "derived")
    split = [r for r in d.periods if r["zone"] == "ES"]
    return sum(1 for r in split if r["price_pt"] != r["price_es"]) / len(split)


def _nonconvex_share() -> float:
    from iberian_lp.contract import NON_CONVEX, SIMPLE
    from iberian_lp.report import load

    d = load(ROOT / "data" / "derived")
    sell = {k: sum(float(r[f"offered_sell_{k}"]) for r in d.periods) for k in
            (SIMPLE, *NON_CONVEX)}
    return sum(sell[k] for k in NON_CONVEX) / sum(sell.values())


def test_the_example_period_is_the_one_nearest_the_mean() -> None:
    from iberian_lp.figures import example

    e = json.loads(ESTIMATES.read_text(encoding="utf-8"))
    t, d = table(ROOT)
    i = example(d, t, e)
    assert t.rows[i]["date"] == d.days[0] and t.rows[i]["zone"] == "MI"
    assert abs(t.nearest("X")[i] - e["E2"]["all"]["mean"]["estimate"]) < 5


def test_figures_draw(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from iberian_lp.figures import draw_all

    e = json.loads(ESTIMATES.read_text(encoding="utf-8"))
    t, d = table(ROOT)
    paths = draw_all(ROOT / "data" / "derived", d, t, e, tmp_path)
    assert [p.name for p in paths] == sorted(
        p.name for p in (PUB / "figures").glob("fig*.png"))
    assert all(p.stat().st_size > 10_000 for p in paths)
