"""Grain maize estimates (METHODS.md, section 9.2), reported beside wheat.

The models, tests and decision rules are wheat's: M1 with region and country-by-year
effects, two-way clustering and Amendment A1's bootstrap of the two-way statistic, E2
tested as the dry-season contrast. E1 and E2 form the Holm family; there is no E3.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

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
    _shares,
    _tcrit,
    _top,
    coefficient_rows,
    dry_season_effect,
    estimate,
    without_singletons,
)
from eu_wheat_weather.estimation import Fit, cluster_vcov, holm, t_test, twoway_vcov
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
        "post_hoc": post_hoc(panel, weather, obs, all_obs, m1, placebo_obs, with_bins, m3),
    }


# ----------------------------------------------------------------------------------
# After the review (post hoc)
# ----------------------------------------------------------------------------------


def _full_rule(obs: Sequence[Obs]) -> dict[str, Any]:
    """E1 and E2 on a subsample under the whole registered rule: both methods, Holm, sign."""
    sub, _ = without_singletons(obs, EFFECTS)
    e1, e2 = both(estimate(sub, BASE, EFFECTS), ["edd"]), both(contrast_fit(sub), ["contrast"])
    d = decide(e1, e2)
    return {"n": e1["n"], **{key: {"estimate": x["estimate"], "p_twoway": x["p_twoway"],
                                   "p_bootstrap": x["p_bootstrap"], **d[key]}
                             for key, x in (("E1", e1), ("E2", e2))}}


def _variances(f: Fit, term: str) -> dict[str, dict[str, float]]:
    """The term's standard error and p-value (registered df) under the two-way variance and
    under each clustering it combines."""
    j = f.index([term])[0]
    v2, df = twoway_vcov(f)
    cell = np.char.add(np.char.add(f.country, "|"), f.year)
    vc, vy, vcy = (cluster_vcov(f.x, f.resid, codes, f.k)[0]
                   for codes in (f.country, f.year, cell))
    out = {}
    for name, v in (("twoway", v2), ("country", vc), ("year", vy), ("country_year", vcy),
                    ("country_plus_year", vc + vy)):
        _, p, se = t_test(f.beta, v, j, df)
        out[name] = {"se": se, "p": p}
    return out


def _contributions(f: Fit, term: str, obs: Sequence[Obs]) -> dict[str, dict[str, float]]:
    """Each country's, region's and country-year's part of a coefficient (Frisch-Waugh:
    the term's residual on the other regressors, times log yield, over its sum of
    squares). Each set of parts sums to the estimate; most negative first."""
    j = f.index([term])[0]
    others = np.delete(f.x, j, axis=1)
    coef, *_ = np.linalg.lstsq(others, f.x[:, j], rcond=None)
    r = f.x[:, j] - others @ coef
    part = r * f.y / float(np.sum(r ** 2))
    out: dict[str, dict[str, float]] = {"country": {}, "region": {}, "country_year": {}}
    for o, value in zip(obs, part, strict=True):
        for level, key in (("country", o.country), ("region", o.geo),
                           ("country_year", f"{o.country} {o.year}")):
            out[level][key] = out[level].get(key, 0.0) + float(value)
    return {level: dict(sorted(parts.items(), key=lambda kv: kv[1]))
            for level, parts in out.items()}


def _swing(obs: Sequence[Obs], m1: Fit) -> dict[str, float]:
    """How far heat moves within a region: changes between consecutive seasons, and the
    spread left after the fixed effects (degree days above 30 °C)."""
    by_region: dict[str, dict[int, float]] = {}
    for o in obs:
        by_region.setdefault(o.geo, {})[o.year] = 10 * o.x["edd"]
    changes = [abs(v[y] - v[y - 1]) for v in by_region.values() for y in v if y - 1 in v]
    left = 10 * m1.x[:, m1.index(["edd"])[0]]
    raw = np.array([o.x["edd"] for o in obs])
    return {"median_change": statistics.median(changes),
            "share_changes_10": float(np.mean([c >= 10 for c in changes])),
            "sd_after_effects": float(np.sqrt(np.mean(left ** 2))),
            "share_after_effects_10": float(np.mean(np.abs(left) >= 10)),
            "variance_left": float(np.sum(left ** 2) / np.sum((10 * (raw - raw.mean())) ** 2))}


def _area_shares(obs: Sequence[Obs]) -> dict[str, float]:
    total = sum(o.area for o in obs)
    out: dict[str, float] = {}
    for o in obs:
        out[o.country] = out.get(o.country, 0.0) + o.area / total
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _south(obs: Sequence[Obs]) -> dict[str, Any]:
    """E1 in the irrigated south and elsewhere: one model with EDD × south, and the south
    on its own."""
    f = estimate(obs, BASE, EFFECTS,
                 extra={"edd_x_south": lambda o: o.x["edd"] * (o.country in IRRIGATED)})
    v, df = twoway_vcov(f)
    i, j = f.index(["edd", "edd_x_south"])
    grad = np.zeros(f.k)
    grad[i] = grad[j] = 1.0
    south = float(f.beta[i] + f.beta[j])
    half = _tcrit(df) * float(np.sqrt(grad @ v @ grad))
    alone, _ = without_singletons([o for o in obs if o.country in IRRIGATED], EFFECTS)
    return {"elsewhere": two_way(f, ["edd"]), "interaction": two_way(f, ["edd_x_south"]),
            "south": {"estimate": south, "low": south - half, "high": south + half},
            "south_alone": two_way(estimate(alone, BASE, EFFECTS), ["edd"])}


def _placebo(placebo_obs: Sequence[Obs]) -> dict[str, Any]:
    """What drives the placebo's two-way rejection: each term, the country whose removal
    weakens it most, its link to the season's own rain, and E1 with both terms."""
    f = estimate(placebo_obs, [*BASE, *PLACEBO_TERMS], EFFECTS)
    without = {}
    for country in sorted({o.country for o in placebo_obs}):
        sub, _ = without_singletons([o for o in placebo_obs if o.country != country], EFFECTS)
        without[country] = two_way(estimate(sub, [*BASE, *PLACEBO_TERMS], EFFECTS),
                                   PLACEBO_TERMS)["p_twoway"]
    most = max(without, key=lambda c: without[c])
    rain = estimate(placebo_obs, [PLACEBO_TERMS[1], "p"], EFFECTS)
    return {"terms": {t: two_way(f, [t]) for t in PLACEBO_TERMS},
            "edd_zero_share": float(np.mean([o.x[PLACEBO_TERMS[0]] == 0
                                             for o in placebo_obs])),
            "without_country": {"country": most, "p_twoway": without[most]},
            "rain_vs_season_rain": float(np.corrcoef(rain.x[:, 0], rain.x[:, 1])[0, 1]),
            "E1_with_placebo_terms": two_way(f, ["edd"])}


def _hours(with_bins: Sequence[Obs]) -> dict[str, float]:
    """Mean hours per season in the bins above 27 °C (the bins are in units of 24 hours)."""
    names = list(bin_names())
    return {b: 24 * float(np.mean([o.x[b] for o in with_bins]))
            for b in names[names.index("t_27_30"):]}


def post_hoc(panel: Sequence[Mapping[str, str]],
             weather: Mapping[tuple[str, int, str], Mapping[str, str]],
             obs: Sequence[Obs], all_obs: Sequence[Obs], m1: Fit,
             placebo_obs: Sequence[Obs], with_bins: Sequence[Obs], m3: Fit) -> dict[str, Any]:
    """Checks added after the review (docs/MAIZE_REVIEW.md); each is reported as post hoc,
    never as a verdict."""
    contributions = _contributions(m1, "edd", obs)
    largest = next(iter(contributions["country_year"]))
    country, year = largest.split()
    leave_out = [{"country": c, **_full_rule([o for o in obs if o.country != c])}
                 for c in sorted({o.country for o in obs})]
    no_south, _ = without_singletons(observations(panel, weather, drop=IRRIGATED), EFFECTS)
    no_south_fit = estimate(no_south, BASE, EFFECTS)
    estimated = {(r["geo"], int(r["year"])): r["country"] for r in panel
                 if "e" in (r["area_flag"], r["production_flag"])}
    in_sample = [o for o in obs if (o.geo, o.year) in estimated]
    e4_gaps = {}
    for y in E4_YEARS:
        result = _e4_year(all_obs, y)
        if result is not None:
            e4_gaps[str(y)] = result[2]
    return {
        "E1_variance": _variances(m1, "edd"),
        "E1_contributions": contributions,
        "leave_one_country_out": leave_out,
        "without_largest_country_year": {
            "country_year": largest,
            "region_years": sum(1 for o in obs if o.country == country and o.year == int(year)),
            **_full_rule([o for o in obs
                          if not (o.country == country and o.year == int(year))])},
        "identifying_variation": {
            "edd": _shares(m1, "edd", obs), "t_ge36": _shares(m3, "t_ge36", with_bins),
            "edd_not_irrigated": _shares(no_south_fit, "edd", no_south),
            "edd_area_weighted": _shares(estimate(obs, BASE, EFFECTS, True), "edd", obs)},
        "maize_area": _area_shares(obs),
        "not_irrigated_E1_variance": _variances(no_south_fit, "edd"),
        "south": _south(obs),
        "placebo": _placebo(placebo_obs),
        "heat_swing": _swing(obs, m1),
        "hours": _hours(with_bins),
        "E4_gap_by_country": e4_gaps,
        "estimated_rows": {"region_years": len(in_sample),
                           "countries": sorted({o.country for o in in_sample}),
                           "years": [min(o.year for o in in_sample),
                                     max(o.year for o in in_sample)]},
    }


def heat_difference(maize: Mapping[str, float], wheat: Mapping[str, float],
                    ) -> tuple[float, float]:
    """Maize's E1 minus wheat's as percentage changes in yield, in percentage points, and
    its 95% margin if the two estimates were independent (delta method, normal
    approximation). Descriptive, not a test."""
    m, w = np.exp(maize["estimate"]), np.exp(wheat["estimate"])
    margin = 1.96 * float(np.hypot(m * maize["se"], w * wheat["se"]))
    return float(100 * (m - w)), 100 * margin


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


def _verdicts(rows: Sequence[Mapping[str, Any]], key: str) -> str:
    lost = [x["country"] for x in rows if x[key]["verdict"] != "established negative"]
    return (f"not established without {', '.join(lost)} ({len(lost)} of {len(rows)})"
            if lost else f"established in all {len(rows)}")


def _after_review(r: Mapping[str, Any]) -> list[str]:
    """The post-hoc checks added after the review (docs/MAIZE_REVIEW.md)."""
    ph = r["post_hoc"]
    e1 = r["E1"]["estimate"]
    var, contrib, iv = ph["E1_variance"], ph["E1_contributions"], ph["identifying_variation"]
    loo, cy, south = ph["leave_one_country_out"], ph["without_largest_country_year"], ph["south"]
    pl, swing, est = ph["placebo"], ph["heat_swing"], ph["estimated_rows"]
    ni = ph["not_irrigated_E1_variance"]
    labels = {"twoway": "Two-way (registered)", "country": "Country", "year": "Year",
              "country_year": "Country-year", "country_plus_year": "Country plus year"}
    region, region_part = next(iter(contrib["region"].items()))
    south_var = sum(v for c, v in iv["edd"].items() if c in IRRIGATED)
    south_area = sum(v for c, v in ph["maize_area"].items() if c in IRRIGATED)

    def holm_cells(x: Mapping[str, Any]) -> str:
        return (f"{x['estimate']:+.4f} | {_p(x['p_twoway_holm'])} | "
                f"{_p(x['p_bootstrap_holm'])} | {x['verdict']}")

    def gap(year: str) -> str:
        parts = list(ph["E4_gap_by_country"][year].items())
        total = sum(v for _, v in parts)
        return (f"{year}: " + ", ".join(f"{c} {v:+.1f}" for c, v in parts[:2])
                + f" of {total:+.1f} points")

    lines = [
        "", "## After review (post hoc)", "",
        "Checks added after the review ([MAIZE_REVIEW.md](MAIZE_REVIEW.md)); none is a "
        "registered result and none changes a registered verdict.", "",
        f"**E1's standard error under each clustering** (df {r['E1']['df']}, as registered). "
        "The two-way variance adds the country and year variances and subtracts the "
        "country-year one.", "",
        "| Variance | SE | p |", "| --- | ---: | ---: |",
        *[f"| {labels[k]} | {v['se']:.5f} | {v['p']:.4f} |" for k, v in var.items()], "",
        "**Each country's part of E1** (Frisch-Waugh: the parts sum to the estimate, "
        f"{e1:+.4f}): "
        + ", ".join(f"{c} {v:+.4f}" for c, v in list(contrib["country"].items())[:4])
        + f". The largest region's part: {region} {region_part:+.4f}. The largest "
        f"country-year's part: {cy['country_year']} "
        f"{contrib['country_year'][cy['country_year']]:+.4f}.", "",
        "**Leave one country out, under the full registered rule** (both methods, Holm over "
        "E1 and E2; p-values after Holm).", "",
        "| Left out | N | E1 | p, two-way | p, bootstrap | Verdict | E2 | p, two-way | "
        "p, bootstrap | Verdict |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
        *[f"| {x['country']} | {x['n']} | {holm_cells(x['E1'])} | {holm_cells(x['E2'])} |"
          for x in loo], "",
        f"E1 is {_verdicts(loo, 'E1')}. E2 is {_verdicts(loo, 'E2')}; its two-way p before "
        f"Holm stays at or below {max(x['E2']['p_twoway'] for x in loo):.4f}.", "",
        f"**Without {cy['country_year']}** ({cy['region_years']} region-years, the "
        f"country-year with the largest part of E1): "
        + "; ".join(f"{key} {cy[key]['estimate']:+.4f}, p after Holm "
                    f"{_p(cy[key]['p_twoway_holm'])} (two-way) and "
                    f"{_p(cy[key]['p_bootstrap_holm'])} (bootstrap), {cy[key]['verdict']}"
                    for key in ("E1", "E2")) + ".", "",
        "**The irrigated south and elsewhere.** One model with EDD × south (Spain, Portugal, "
        f"Italy and Greece): {_ci(south['elsewhere'])} elsewhere and "
        f"{_ci(south['south'])} in the south. The south on its own: "
        f"{south['south_alone']['estimate']:+.4f}, two-way p "
        f"{_p(south['south_alone']['p_twoway'])}. The four countries supply "
        f"{south_var:.0%} of the identifying variation of `edd` and grow {south_area:.0%} "
        "of the sample's maize.", "",
        "**Where the identifying variation comes from** (each country's share of the "
        "variable's sum of squares once the fixed effects are removed): `edd` "
        f"{_top(iv['edd'])}; time at 36 °C and above {_top(iv['t_ge36'])}; `edd` without "
        f"the irrigated south {_top(iv['edd_not_irrigated'])}; `edd` with area weights "
        f"{_top(iv['edd_area_weighted'])}.", "",
        "**The interval without the irrigated south.** E1's two-way SE there is "
        f"{ni['twoway']['se']:.4f}, against {ni['country']['se']:.4f} (country), "
        f"{ni['year']['se']:.4f} (year) and {ni['country_year']['se']:.4f} (country-year).",
        "",
        "**The placebo's two terms** (two-way): post-maturity degree days above 30 °C "
        f"{_ci(pl['terms'][PLACEBO_TERMS[0]])}, p {_p(pl['terms'][PLACEBO_TERMS[0]]['p_twoway'])}"
        f", zero in {pl['edd_zero_share']:.0%} of region-years; post-maturity precipitation "
        f"(per 100 mm) {_ci(pl['terms'][PLACEBO_TERMS[1]])}, p "
        f"{_p(pl['terms'][PLACEBO_TERMS[1]]['p_twoway'])}. Leaving out "
        f"{pl['without_country']['country']} weakens the joint test most (two-way p "
        f"{_p(pl['without_country']['p_twoway'])}). After the fixed effects, post-maturity "
        f"precipitation correlates {pl['rain_vs_season_rain']:+.3f} with the season's. E1 "
        f"with both terms in the model: {pl['E1_with_placebo_terms']['estimate']:+.4f}.", "",
        "**How far heat moves within a region** (degree days above 30 °C): the median "
        f"change from one season to the next is {swing['median_change']:.1f}, and "
        f"{swing['share_changes_10']:.0%} of changes reach 10. After the fixed effects the "
        f"SD is {swing['sd_after_effects']:.1f}, {swing['share_after_effects_10']:.1%} of "
        f"region-years lie 10 or more from zero, and {swing['variance_left']:.1%} of the "
        "variance of heat is left.", "",
        "**Hours per season** in the bins above 27 °C: "
        + ", ".join(f"{_bin(b)} {h:.0f}" for b, h in ph["hours"].items()) + ".", "",
        "**E4's misses by country** (each country's part of the gap between the observed "
        "and predicted change, the two largest): "
        + "; ".join(gap(y) for y in ph["E4_gap_by_country"] if y != "2015") + ".", "",
        "**Eurostat's estimated values.** MAIZE_DATA.md counts the `e` flags of area and "
        f"production separately. In the sample, {est['region_years']} region-years carry "
        f"one, all in {', '.join(est['countries'])}, {est['years'][0]}-{est['years'][1]}.",
    ]
    return lines


def render(r: Mapping[str, Any], wheat: Mapping[str, Any], command: str) -> str:
    s, d, e1, e2, e5 = r["sample"], r["decisions"], r["E1"], r["E2"], r["E5"]
    one_rejects = (e5["p_twoway"] < ALPHA) != (e5["p_bootstrap"] < ALPHA)
    placebo = (
        "**The placebo fails: weather in the 61 days after maturity is significant under "
        "both methods. The identification is in doubt.**" if e5["significant_both"] else
        "**The placebo passes by the registered rule, which needs both methods to reject: "
        "one method rejects and the other does not.** What drives the two-way rejection is "
        "in the post-hoc section below." if one_rejects else
        "The placebo passes: neither method rejects.")
    gap, margin = heat_difference(e1, wheat["E1"])
    lines = [
        "# Grain maize: results", "",
        "Generated by `eu-wheat maize-estimate` from the maize panel "
        "([MAIZE_DATA.md](MAIZE_DATA.md)). Every model, test and decision rule was fixed in "
        "[METHODS.md](../METHODS.md), section 9, before any maize yield was looked at. "
        "Details the plan left open were fixed in the code and recorded in its change log "
        "afterwards: on 2026-10-26, after the panel was built, and on 2026-10-27, after the "
        "estimates. The coefficients are effects of weather shocks given each region's "
        "normal and each country's year.", "",
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
        f"Maize's E1 minus wheat's, as yield changes: {gap:+.1f} percentage points, with a "
        f"95% margin of about ±{margin:.1f} if the two estimates were independent (delta "
        "method; added after the review, post hoc).", "",
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
              f"- {s['years'][1] - s['years'][0] + 1} harvests: no test of a change in "
              "sensitivity.",
              *_after_review(r),
              "", "## Reproduce", "", "```", command, "```", ""]
    return "\n".join(lines)
