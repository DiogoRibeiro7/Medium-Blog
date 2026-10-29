"""The article: generated values, no hand-typed results, true claims."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from iberian_lp.publication import PLACEHOLDER, check, render, table

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "publication"
TEMPLATE = PUB / "article.template.md"
ESTIMATES = ROOT / "results" / "estimates.json"
REPOSITORY = "https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/iberian-electricity-lp"
# Not results: the bounds in the programme (0 ≤ s), the 95% level and a citation's year.
DESIGN_CONSTANTS = {"0", "95", "2008"}
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


def test_words_in_the_article_match_the_results() -> None:
    e = json.loads(ESTIMATES.read_text(encoding="utf-8"))
    e1, e2, e3, e4, ph = (e[k] for k in ("E1", "E2", "E3", "E4", "post_hoc"))
    text = re.sub(r"\s+", " ", TEMPLATE.read_text(encoding="utf-8"))
    sens = e3["sensitivity"]
    ratio_small = e3["mean"]["estimate"] / sens["500 MW"]["mean"]["estimate"]
    ratio_large = sens["2000 MW"]["mean"]["estimate"] / e3["mean"]["estimate"]
    nonpos = ph["nonpositive"]
    claims = {
        # the opening
        "quarter-hours tested, in Spain and in Portugal alike": e1["misses"]["count"] == 0
        and all(z["inside"] == z["zone_periods"] for z in e1["by_zone"].values()),
        "Treating them as ordinary divisible bids would have lowered the price by":
            e2["all"]["mean"]["high"] < 0,
        "against my expectation of no change": e2["excludes_zero"],
        "Almost all of that comes from one kind of order":
            e2["sco"]["mean"]["estimate"] / e2["all"]["mean"]["estimate"] > 0.9
            and e2["blocks"]["mean"]["estimate"] / e2["all"]["mean"]["estimate"] < 0.2,
        "MW more supply at zero price would have lowered the price by":
            e3["mean"]["high"] < 0,
        "The effect is smaller around midday because": e3["midday_smaller"],
        "among prices above zero, no midday difference was established":
            ph["positive"]["G"]["low"] < 0 < ph["positive"]["G"]["high"],
        # E1
        "It lay inside them in every one of the": e1["misses"]["count"] == 0,
        "The programme's accepted volume matched the published one within":
            e1["volume"]["mismatched"] == 0,
        "no step was partly accepted, so the shadow prices formed an interval":
            e["descriptive"]["wide"]["count"] > 0,
        # E2
        "That expectation failed.": e2["excludes_zero"]
        or e2["all"]["moved"]["estimate"] > 0.5,
        "The interval excludes zero, so by the registered rule the indivisible orders move":
            e2["excludes_zero"],
        "Relaxing them alone never raised the price in any quarter-hour":
            e2["sco"]["up"] == 0,
        "and in both directions": e2["blocks"]["up"] > 0 and e2["blocks"]["down"] > 0,
        "almost all of it from the scalable complex orders":
            e2["sco"]["mean"]["estimate"] / e2["all"]["mean"]["estimate"] > 0.9,
        # E3
        "The effect grows roughly in proportion to the size":
            1.8 < ratio_small < 2.2 and 1.8 < ratio_large < 2.2,
        "MW lowers the price by": sens["500 MW"]["mean"]["high"] < 0
        and sens["2000 MW"]["mean"]["high"] < 0,
        "By the registered rule it is": e3["midday_smaller"],
        "The interval lies above zero.": e3["G"]["low"] > 0,
        "midday quarter-hours were priced there, against":
            nonpos["midday"]["count"] / nonpos["midday"]["periods"]
            > nonpos["outside"]["count"] / nonpos["outside"]["periods"],
        "In those quarter-hours m is zero.": ph["nonpositive_max_abs_m"] == 0,
        "no difference is established": ph["positive"]["G"]["low"] < 0
        < ph["positive"]["G"]["high"],
        # E4
        "without the imports, the price rises": e4["imports"]["low"] > 0,
        "without the exports, it falls": e4["exports"]["high"] < 0,
    }
    for phrase, holds in claims.items():
        assert phrase in text, phrase
        assert holds, phrase


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
