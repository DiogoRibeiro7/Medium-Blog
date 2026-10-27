"""Grain maize estimates (METHODS.md, section 9.2), reported beside wheat.

The models, tests and decision rules are wheat's: M1 with region and country-by-year
effects, two-way clustering and Amendment A1's bootstrap of the two-way statistic, E2
tested as the dry-season contrast. E1 and E2 form the Holm family; there is no E3.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from eu_wheat_weather.amendment import _ci, amended_bootstrap, both, contrast_fit, two_way
from eu_wheat_weather.analysis import (
    ALPHA,
    BASE,
    OMITTED_BIN,
    Obs,
    _bin,
    _e4_year,
    _p,
    _pct,
    _round,
    coefficient_rows,
    dry_season_effect,
    estimate,
    without_singletons,
)
from eu_wheat_weather.estimation import holm
from eu_wheat_weather.weather import bin_names

EFFECTS = "region+country_year"
E4_YEARS = (2012, 2015, 2022)             # widespread heat or drought in maize (9.2)
PLACEBO = "post_maturity"
PLACEBO_TERMS = ("edd_post_maturity", "p_post_maturity")
IRRIGATED = ("ES", "PT", "IT", "EL")      # where much grain maize is irrigated (9.2)
SENSITIVITIES: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("edd27", "EDD and GDD at 27 °C", {"threshold": 27}),
    ("edd33", "EDD and GDD at 33 °C", {"threshold": 33}),
    ("may_sep", "Fixed season, 1 May to 30 September", {"window": "may_sep"}),
    ("not_irrigated", "Without Spain, Portugal, Italy and Greece", {"drop": IRRIGATED}),
    ("area_weighted", "Weighted by maize area", {"weighted": True}),
)


def load(panel_path: Path, weather_path: Path,
         ) -> tuple[list[dict[str, str]], dict[tuple[str, int, str], dict[str, str]]]:
    with panel_path.open(encoding="utf-8") as stream:
        panel = list(csv.DictReader(stream))
    with weather_path.open(encoding="utf-8") as stream:
        weather = {(r["geo"], int(r["year"]), r["window"]): r for r in csv.DictReader(stream)}
    return panel, weather


def observations(panel: Sequence[Mapping[str, str]],
                 weather: Mapping[tuple[str, int, str], Mapping[str, str]],
                 window: str = "season", threshold: int = 30,
                 drop: Sequence[str] = ()) -> list[Obs]:
    """The sample with section 9.1's variables: GDD (10 °C to the threshold) in hundreds,
    EDD above the threshold in tens, precipitation in hundreds of mm."""
    out = []
    for r in panel:
        if r["country"] in drop:
            continue
        geo, year = r["geo"], int(r["year"])
        w = weather.get((geo, year, window))
        if w is None or any(w[v] == "" for v in (f"gdd_10_{threshold}",
                                                 f"edd_{threshold}", "prec")):
            continue
        p = float(w["prec"]) / 100
        x = {"gdd": float(w[f"gdd_10_{threshold}"]) / 100,
             "edd": float(w[f"edd_{threshold}"]) / 10, "p": p, "p2": p * p}
        placebo = weather.get((geo, year, PLACEBO))
        if placebo is not None and placebo["edd_30"] != "" and placebo["prec"] != "":
            x[PLACEBO_TERMS[0]] = float(placebo["edd_30"]) / 10
            x[PLACEBO_TERMS[1]] = float(placebo["prec"]) / 100
        if all(w[b] != "" for b in bin_names()):
            x.update({b: float(w[b]) for b in bin_names()})
        out.append(Obs(geo, r["country"], year, float(r["log_yield"]), float(r["area_kha"]),
                       x))
    return out


def decide(e1: Mapping[str, Any], e2: Mapping[str, Any]) -> dict[str, Any]:
    """Holm over E1 and E2 under each method; established only if rejected under both
    and negative (section 9.2)."""
    pt = holm([e1["p_twoway"], e2["p_twoway"]])
    pb = holm([e1["p_bootstrap"], e2["p_bootstrap"]])
    out: dict[str, Any] = {}
    for key, est, a, b in (("E1", e1["estimate"], pt[0], pb[0]),
                           ("E2", e2["estimate"], pt[1], pb[1])):
        rejected = a < ALPHA and b < ALPHA
        verdict = ("established negative" if rejected and est < 0 else
                   "rejected but not negative" if rejected else "not determined")
        out[key] = {"p_twoway_holm": a, "p_bootstrap_holm": b, "verdict": verdict}
    return out


def sensitivity(panel: Sequence[Mapping[str, str]],
                weather: Mapping[tuple[str, int, str], Mapping[str, str]],
                primary: Mapping[str, float]) -> list[dict[str, Any]]:
    rows = []
    for name, label, options in SENSITIVITIES:
        obs, _ = without_singletons(observations(
            panel, weather, options.get("window", "season"), options.get("threshold", 30),
            options.get("drop", ())), EFFECTS)
        weighted = bool(options.get("weighted", False))
        for key, fit, term in (("E1", estimate(obs, BASE, EFFECTS, weighted), "edd"),
                               ("E2", contrast_fit(obs, weighted), "contrast")):
            t = two_way(fit, [term])
            rows.append({"variant": name, "label": label, "estimand": key, **t,
                         "sign_reversed": bool((t["estimate"] < 0) != (primary[key] < 0))})
    return rows


def run(panel_path: Path, weather_path: Path) -> dict[str, Any]:
    panel, weather = load(panel_path, weather_path)
    all_obs = observations(panel, weather)
    obs, singletons = without_singletons(all_obs, EFFECTS)
    m1, c = estimate(obs, BASE, EFFECTS), contrast_fit(obs)
    e1, e2 = both(m1, ["edd"]), both(c, ["contrast"])
    delta = dry_season_effect(m1, obs)
    if abs(delta["effect"] - e2["estimate"]) > 1e-8:
        raise ValueError("The reparametrised contrast differs from the delta-method effect.")
    e2["mean_p10_minus_p50_mm"] = delta["mean_p10_minus_p50"]
    placebo_obs, _ = without_singletons([o for o in obs if PLACEBO_TERMS[0] in o.x], EFFECTS)
    e5 = both(estimate(placebo_obs, [*BASE, *PLACEBO_TERMS], EFFECTS), PLACEBO_TERMS)
    e5["significant_both"] = bool(e5["p_twoway"] < ALPHA and e5["p_bootstrap"] < ALPHA)
    with_bins, _ = without_singletons([o for o in obs if OMITTED_BIN in o.x], EFFECTS)
    names = [b for b in bin_names() if b != OMITTED_BIN]
    m3 = estimate(with_bins, [*names, "p", "p2"], EFFECTS)
    bins = [{**r, **amended_bootstrap(m3, [r["term"]])}
            for r in coefficient_rows(m3, "M3-maize") if r["term"].startswith("t_")]
    e4, e4_regions = [], []
    for year in E4_YEARS:
        result = _e4_year(all_obs, year)
        if result is not None:
            e4.append(result[0])
            e4_regions += result[1]
    return {
        "sample": {"n": m1.n, "regions": len({o.geo for o in obs}),
                   "countries": len({o.country for o in obs}), "singletons_dropped": singletons,
                   "years": [min(o.year for o in obs), max(o.year for o in obs)]},
        "coefficients": coefficient_rows(m1, "M1-maize"),
        "E1": e1, "E2": e2, "E5": e5, "decisions": decide(e1, e2),
        "M3": bins, "E4": e4, "E4_regions": e4_regions,
        "sensitivity": sensitivity(panel, weather, {"E1": e1["estimate"],
                                                    "E2": e2["estimate"]}),
    }


# ----------------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------------


def write_results(results: Mapping[str, Any], wheat: Mapping[str, Any], output_dir: Path,
                  report: Path, command: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    document = _round(dict(results))
    (output_dir / "results.json").write_text(json.dumps(document, indent=1) + "\n",
                                             encoding="utf-8", newline="\n")
    for name, rows in (("coefficients.csv", document["coefficients"]),
                       ("bins.csv", document["M3"]), ("out_of_sample.csv", document["E4"]),
                       ("out_of_sample_regions.csv", document["E4_regions"]),
                       ("sensitivity.csv", document["sensitivity"])):
        with (output_dir / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render(document, wheat, command), encoding="utf-8", newline="\n")


def render(r: Mapping[str, Any], wheat: Mapping[str, Any], command: str) -> str:
    s, d, e1, e2, e5 = r["sample"], r["decisions"], r["E1"], r["E2"], r["E5"]
    one_rejects = (e5["p_twoway"] < ALPHA) != (e5["p_bootstrap"] < ALPHA)
    placebo = (
        "**The placebo fails: weather in the 61 days after maturity is significant under "
        "both methods. The identification is in doubt.**" if e5["significant_both"] else
        "**The placebo passes by the registered rule, which needs both methods to reject, "
        "but only narrowly: one method rejects and the other does not.** Grain maize often "
        "stands in the field for weeks after maturity before it is harvested, so weather "
        "after maturity can also act on the harvested yield directly." if one_rejects else
        "The placebo passes: neither method rejects.")
    lines = [
        "# Grain maize: results", "",
        "Generated by `eu-wheat maize-estimate` from the maize panel "
        "([MAIZE_DATA.md](MAIZE_DATA.md)). Every model, test and decision rule was fixed in "
        "[METHODS.md](../METHODS.md), section 9, before any maize yield was looked at. The "
        "coefficients are effects of weather shocks given each region's normal and each "
        "country's year.", "",
        "## Placebo (stated first)", "",
        f"{placebo} Joint test of degree days above 30 °C and precipitation over the 61 days "
        f"after each region's maturity date: two-way p = {_p(e5['p_twoway'])}, bootstrap "
        f"p = {_p(e5['p_bootstrap'])} (by country {_p(e5['p_bootstrap_country'])}, by year "
        f"{_p(e5['p_bootstrap_year'])}).", "",
        "## Decisions", "",
        "Holm over E1 and E2, separately under the two-way method and the bootstrap of the "
        "two-way statistic (the larger of its runs by country and by year); established "
        "only if rejected under both, and negative.", "",
        "| Estimand | Estimate | p, two-way (Holm) | p, bootstrap (Holm) | Verdict |",
        "| --- | ---: | ---: | ---: | --- |",
        f"| E1, 10 more degree days above 30 °C | {e1['estimate']:+.4f} "
        f"({_pct(e1['estimate'])}) | {_p(d['E1']['p_twoway_holm'])} | "
        f"{_p(d['E1']['p_bootstrap_holm'])} | {d['E1']['verdict']} |",
        f"| E2, a dry season (P10 against median) | {e2['estimate']:+.4f} "
        f"({_pct(e2['estimate'])}) | {_p(d['E2']['p_twoway_holm'])} | "
        f"{_p(d['E2']['p_bootstrap_holm'])} | {d['E2']['verdict']} |", "",
        "## Sample", "",
        f"{s['n']} region-years in {s['regions']} regions of {s['countries']} countries, "
        f"harvest years {s['years'][0]} to {s['years'][1]}; {s['singletons_dropped']} "
        "region-years alone in their region or country-year were dropped.", "",
        "## M1 for maize in detail", "",
        "Two-way clustered 95% intervals. Units: GDD (10-30 °C) in hundreds of degree days, "
        "EDD in tens, precipitation in hundreds of mm.", "",
        "| Term | Estimate | 95% interval | p |", "| --- | ---: | ---: | ---: |",
    ]
    lines += [f"| `{c['term']}` | {c['estimate']:+.4f} | [{c['low']:+.4f}, {c['high']:+.4f}] | "
              f"{_p(c['p'])} |" for c in r["coefficients"]]
    lines += [
        "",
        f"- **E1.** {_ci(e1)}: two-way t = {e1['t']:+.2f}, p = {_p(e1['p_twoway'])}; "
        f"bootstrap p = {_p(e1['p_bootstrap'])} (by country {_p(e1['p_bootstrap_country'])}, "
        f"by year {_p(e1['p_bootstrap_year'])}).",
        f"- **E2.** A dry season is on average {abs(e2['mean_p10_minus_p50_mm']):.0f} mm below "
        f"the region's median. The contrast is {_ci(e2)}: two-way p = {_p(e2['p_twoway'])}; "
        f"bootstrap p = {_p(e2['p_bootstrap'])} (by country {_p(e2['p_bootstrap_country'])}, "
        f"by year {_p(e2['p_bootstrap_year'])}).", "",
        "## Beside wheat (descriptive)", "",
        "Wheat under Amendment A1, the comparable design (harvest window, data rule, the "
        "same inference). Different seasons and samples, so not a test of a difference.", "",
        "| Estimand | Maize | Wheat (Amendment A1) |", "| --- | ---: | ---: |",
        f"| E1, 10 more degree days above 30 °C | {_ci(e1)} | {_ci(wheat['E1'])} |",
        f"| E2, a dry season | {_ci(e2)} | {_ci(wheat['E2'])} |", "",
        "## M3, the temperature bins over the season (descriptive)", "",
        "Change in log yield for 24 more hours in each 3 °C bin, instead of 24 hours at "
        "15-18 °C; divide by 24 for one hour. No bin is adjusted for multiplicity.", "",
        "| Bin | Estimate | 95% interval, two-way | p, bootstrap |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines += [f"| {_bin(b['term'])} | {b['estimate']:+.4f} | [{b['low']:+.4f}, "
              f"{b['high']:+.4f}] | {_p(b['p_bootstrap'])} |" for b in r["M3"]]
    lines += ["", "## Out of sample (descriptive)", "",
              "M2 (region effects and trends) fitted without the year, then used to predict "
              "it. Maize-area-weighted changes for the panel's regions, against the normal.",
              "", "| Year | Regions | Correlation | Variance explained | Observed | "
              "Predicted |", "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    lines += [f"| {x['year']} | {x['regions']} | {x['correlation']:+.2f} | "
              f"{x['variance_explained']:+.2f} | {x['eu_yield_observed_pct']:+.1f}% | "
              f"{x['eu_yield_predicted_pct']:+.1f}% |" for x in r["E4"]]
    lines += ["", "## Sensitivity analyses (fixed in advance)", "",
              "E1 and E2 with two-way 95% intervals. A variant changes a decision only if it "
              "reverses the sign.", "",
              "| Variant | Estimand | N | Estimate [95% interval] | Sign reversed |",
              "| --- | --- | ---: | ---: | --- |"]
    lines += [f"| {x['label']} | {x['estimand']} | {x['n']} | {_ci(x)} | "
              f"{'**yes**' if x['sign_reversed'] else 'no'} |" for x in r["sensitivity"]]
    lines += ["", "## Limits fixed in advance", "",
              "- Much of southern Europe's maize is irrigated; the pooled effect averages "
              "over irrigated and rainfed maize, and the rainfed calendar is used everywhere.",
              "- Grain maize only; silage maize is not included.",
              "- Fifteen harvests: no test of a change in sensitivity.",
              "", "## Reproduce", "", "```", command, "```", ""]
    return "\n".join(lines)
