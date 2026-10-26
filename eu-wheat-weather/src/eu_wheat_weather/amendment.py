"""Amendment A1 (METHODS.md, section 8), reported beside the registered results.

Registered after the results and the review, before any of it was computed:

- a window that ends at each region's maturity date (8.1);
- a rule for yield series that are not regional measurements (8.2);
- E2 tested as the dry-season contrast itself, and a wild bootstrap of the two-way
  statistic, by country and by year (8.3).

Verdicts come from M1-A alone (8.4). Nothing here changes ``docs/RESULTS.md``.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from eu_wheat_weather.analysis import (
    ALPHA,
    BASE,
    OMITTED_BIN,
    POST_START,
    PRIMARY,
    Obs,
    Variant,
    _bin,
    _dry_deltas,
    _p,
    _pct,
    _round,
    _tcrit,
    coefficient_rows,
    dry_season_effect,
    estimate,
    load,
    observations,
    without_singletons,
)
from eu_wheat_weather.estimation import (
    Fit,
    holm,
    minimum_detectable,
    t_test,
    twoway_vcov,
    twoway_wild_bootstrap,
    wald_test,
)
from eu_wheat_weather.panel import read_region_map, read_wheat
from eu_wheat_weather.quality import RULES, apply_rule, review_cases
from eu_wheat_weather.regions import harmonise
from eu_wheat_weather.weather import bin_names

SEEDS = {"country": 20261101, "year": 20261102}       # section 8.3
EFFECTS = PRIMARY.effects
HARVEST = Variant("harvest", "M1-A", window="harvest", placebo="post_harvest")
EARLY = Variant("harvest_m14", "Window ends 14 days before maturity", window="harvest_m14",
                placebo="post_harvest")
PLACEBO_TERMS = ("edd_post_harvest", "p_post_harvest")
COMBINATIONS = (
    ("inference", "Registered window and sample (only the inference changes)"),
    ("window", "Harvest window alone"),
    ("data", "Data rule alone"),
    ("all", "Both: M1-A"),
)


# ----------------------------------------------------------------------------------
# Tests under the two-way method and the amended bootstrap
# ----------------------------------------------------------------------------------


def two_way(f: Fit, terms: Sequence[str]) -> dict[str, Any]:
    v, df = twoway_vcov(f)
    idx = f.index(terms)
    if len(idx) > 1:
        w, p = wald_test(f.beta, v, idx, df)
        return {"statistic": w, "p_twoway": p, "df": df}
    t, p, se = t_test(f.beta, v, idx[0], df)
    est = float(f.beta[idx[0]])
    half = _tcrit(df) * se
    return {"estimate": est, "se": se, "low": est - half, "high": est + half, "t": t,
            "p_twoway": p, "df": df, "n": f.n}


def amended_bootstrap(f: Fit, terms: Sequence[str]) -> dict[str, float]:
    """Section 8.3: weights by country and by year; the larger p-value counts."""
    out: dict[str, float] = {}
    for by, seed in SEEDS.items():
        _, out[f"p_bootstrap_{by}"] = twoway_wild_bootstrap(f, f.index(terms), by, seed=seed)
    out["p_bootstrap"] = max(out["p_bootstrap_country"], out["p_bootstrap_year"])
    return out


def both(f: Fit, terms: Sequence[str]) -> dict[str, Any]:
    return {**two_way(f, terms), **amended_bootstrap(f, terms)}


def contrast_fit(obs: Sequence[Obs]) -> Fit:
    """M1 with the precipitation terms reparametrised so that E2's dry-season contrast is
    one coefficient, ``contrast`` (section 8.3)."""
    d1, d2 = _dry_deltas(obs)
    return estimate(obs, ["gdd", "edd"], EFFECTS, extra={
        "contrast": lambda o: o.x["p"] / d1,
        "curvature": lambda o: o.x["p2"] - (d2 / d1) * o.x["p"]})


def e3_fit(obs: Sequence[Obs]) -> Fit:
    return estimate(obs, BASE, EFFECTS,
                    extra={"edd_post": lambda o: o.x["edd"] * (o.year >= POST_START)})


def sample(variant: Variant, rows: Sequence[Mapping[str, str]],
           weather: Mapping[tuple[str, int, str, str], Mapping[str, str]]) -> list[Obs]:
    obs, _ = without_singletons(observations(variant, rows, weather), EFFECTS)
    return obs


def decide(e1: Mapping[str, Any], e2: Mapping[str, Any], e3: Mapping[str, Any],
           ) -> dict[str, Any]:
    """Section 5's rules, with E2 decided by its contrast (8.3), under both methods."""
    pt = holm([e1["p_twoway"], e2["p_twoway"], e3["p_twoway"]])
    pb = holm([e1["p_bootstrap"], e2["p_bootstrap"], e3["p_bootstrap"]])
    out: dict[str, Any] = {}
    for key, a, b in zip(("E1", "E2", "E3"), pt, pb, strict=True):
        rejected = a < ALPHA and b < ALPHA
        if key == "E1":
            verdict = ("established negative" if rejected and e1["estimate"] < 0 else
                       "contrary to expectation (positive)" if rejected else "not determined")
        elif key == "E2":
            verdict = ("established negative" if rejected and e2["estimate"] < 0 else
                       "rejected but not negative" if rejected else "not determined")
        else:
            verdict = (("established: sensitivity rose" if e3["estimate"] < 0 else
                        "established: sensitivity fell") if rejected else "not determined")
        out[key] = {"p_twoway_holm": a, "p_bootstrap_holm": b, "verdict": verdict}
    return out


# ----------------------------------------------------------------------------------
# The amendment
# ----------------------------------------------------------------------------------


def leave_one_country_out(obs: Sequence[Obs], full: Mapping[str, float],
                          ) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for country in sorted({o.country for o in obs}):
        sub, _ = without_singletons([o for o in obs if o.country != country], EFFECTS)
        m1 = estimate(sub, BASE, EFFECTS)
        c = contrast_fit(sub)
        e3 = e3_fit(sub)
        rows.append({"country": country, "n": m1.n,
                     "E1": float(m1.beta[m1.index(["edd"])[0]]),
                     "E2": float(c.beta[c.index(["contrast"])[0]]),
                     "E3": float(e3.beta[e3.index(["edd_post"])[0]])})
    summary = {}
    for key in ("E1", "E2", "E3"):
        values = [r[key] for r in rows]
        most = max(rows, key=lambda r: abs(float(r[key]) - full[key]))
        summary[key] = {"low": min(values), "high": max(values),
                        "most_influential": most["country"], "without_it": most[key]}
    return {"rows": rows, "summary": summary}


def maturity_summary(path: Path, regions: set[str]) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if r["geo"] in regions]
    days = sorted(int(r["maturity_day"]) for r in rows if r["maturity_day"] != "")
    by_day = {int(r["maturity_day"]): r["maturity_date"] for r in rows if r["maturity_day"]}
    return {"regions": len(rows), "first": by_day[days[0]], "median_day": statistics.median(days),
            "last": by_day[days[-1]],
            "before_july": sum(d < 182 for d in days), "after_july": sum(d > 212 for d in days)}


def run(panel_path: Path, weather_path: Path, maturity_path: Path, crops: Path,
        region_map: Path) -> dict[str, Any]:
    panel, weather = load(panel_path, weather_path)
    matches = read_region_map(region_map)
    published = read_wheat(crops)
    harmonised, _ = harmonise(published, matches)
    clean, flags, short = apply_rule(panel, harmonised, published, matches)
    rows_for: dict[str, Sequence[Mapping[str, str]]] = {"registered": panel, "rule": clean}
    plan = {"inference": (PRIMARY, "registered"), "window": (HARVEST, "registered"),
            "data": (PRIMARY, "rule"), "all": (HARVEST, "rule")}
    decomposition: list[dict[str, Any]] = []
    for name, label in COMBINATIONS:
        variant, which = plan[name]
        obs = sample(variant, rows_for[which], weather)
        m1, c = estimate(obs, BASE, EFFECTS), contrast_fit(obs)
        decomposition.append({"combination": name, "label": label, "n": m1.n,
                              "E1": both(m1, ["edd"]), "E2": both(c, ["contrast"])})
    obs = sample(HARVEST, clean, weather)
    m1, c, e3f = estimate(obs, BASE, EFFECTS), contrast_fit(obs), e3_fit(obs)
    e1, e2 = decomposition[-1]["E1"], decomposition[-1]["E2"]
    delta = dry_season_effect(m1, obs)
    if abs(delta["effect"] - e2["estimate"]) > 1e-8:
        raise ValueError("The reparametrised contrast differs from the delta-method effect.")
    e2 = {**e2, "mean_p10_minus_p50_mm": delta["mean_p10_minus_p50"]}
    e3 = both(e3f, ["edd_post"])
    e3["minimum_detectable"] = minimum_detectable(e3["se"], e3["df"])
    e3["minimum_detectable_holm"] = minimum_detectable(e3["se"], e3["df"], ALPHA / 3)
    placebo_obs, _ = without_singletons([o for o in obs if PLACEBO_TERMS[0] in o.x], EFFECTS)
    e5 = both(estimate(placebo_obs, [*BASE, *PLACEBO_TERMS], EFFECTS), PLACEBO_TERMS)
    e5["significant_both"] = bool(e5["p_twoway"] < ALPHA and e5["p_bootstrap"] < ALPHA)
    with_bins, _ = without_singletons([o for o in obs if OMITTED_BIN in o.x], EFFECTS)
    names = [b for b in bin_names() if b != OMITTED_BIN]
    m3 = estimate(with_bins, [*names, "p", "p2"], EFFECTS)
    bins = [{**r, **amended_bootstrap(m3, [r["term"]])}
            for r in coefficient_rows(m3, "M3-A") if r["term"].startswith("t_")]
    early = sample(EARLY, clean, weather)
    early_fits = {"E1": (estimate(early, BASE, EFFECTS), "edd"),
                  "E2": (contrast_fit(early), "contrast"), "E3": (e3_fit(early), "edd_post")}
    full = {"E1": e1["estimate"], "E2": e2["estimate"], "E3": e3["estimate"]}
    sensitivity = [{"estimand": k, **two_way(f, [term]),
                    "sign_reversed": bool((two_way(f, [term])["estimate"] < 0) != (full[k] < 0))}
                   for k, (f, term) in early_fits.items()]
    regions = {o.geo for o in obs}
    return {
        "sample": {"n": m1.n, "regions": len(regions),
                   "countries": len({o.country for o in obs}),
                   "years": [min(o.year for o in obs), max(o.year for o in obs)],
                   "panel_region_years": len(panel), "after_rule": len(clean),
                   "excluded_by_rule": dict(Counter(f.rule for f in flags)),
                   "excluded_region_years": len({(f.geo, f.year) for f in flags}),
                   "excluded_by_country": dict(sorted(Counter(
                       g[:2] for g, _ in {(f.geo, f.year) for f in flags}).items())),
                   "regions_below_10_years": short,
                   "maturity": maturity_summary(maturity_path, regions)},
        "quality_flags": [asdict(f) for f in flags],
        "review_cases": review_cases(flags, panel),
        "E1": e1, "E2": e2, "E3": e3, "E5": e5, "decisions": decide(e1, e2, e3),
        "decomposition": decomposition, "M3": bins, "sensitivity": sensitivity,
        "leave_one_out": leave_one_country_out(obs, full),
    }


# ----------------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------------


def _flat(prefix: str, values: Mapping[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{k}": v for k, v in values.items()}


def write_results(results: Mapping[str, Any], registered: Mapping[str, Any],
                  output_dir: Path, report: Path, command: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    document = _round(dict(results))
    (output_dir / "results.json").write_text(json.dumps(document, indent=1) + "\n",
                                             encoding="utf-8", newline="\n")
    tables = {
        "quality_flags.csv": document["quality_flags"],
        "decomposition.csv": [{"combination": d["combination"], "n": d["n"],
                               **_flat("E1", d["E1"]), **_flat("E2", d["E2"])}
                              for d in document["decomposition"]],
        "bins.csv": document["M3"],
        "leave_one_out.csv": document["leave_one_out"]["rows"],
        "sensitivity.csv": document["sensitivity"],
    }
    for name, rows in tables.items():
        with (output_dir / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render(document, registered, command), encoding="utf-8", newline="\n")


def _date(month_day: str) -> str:
    """05-12 -> 12 May."""
    months = ("January", "February", "March", "April", "May", "June", "July", "August",
              "September", "October", "November", "December")
    return f"{int(month_day[3:])} {months[int(month_day[:2]) - 1]}"


def _ci(x: Mapping[str, Any]) -> str:
    return f"{x['estimate']:+.4f} [{x['low']:+.4f}, {x['high']:+.4f}]"


def render(r: Mapping[str, Any], registered: Mapping[str, Any], command: str) -> str:
    s, d, e1, e2, e3, e5 = r["sample"], r["decisions"], r["E1"], r["E2"], r["E3"], r["E5"]
    reg = registered["decisions"]
    placebo = ("**The amended placebo fails: weather in the 61 days after maturity is "
               "significant under both methods.**" if e5["significant_both"] else
               "The amended placebo passes: weather in the 61 days after maturity is not "
               "significant under both methods.")
    mat = s["maturity"]
    lines = [
        "# Amendment A1: results", "",
        "Generated by `eu-wheat amend`. Amendment A1 ([METHODS.md](../METHODS.md), "
        "section 8) was registered after the registered results and the critical review "
        "([REVIEW.md](REVIEW.md)), and before any of it was computed. These results stand "
        "beside the registered ones in [RESULTS.md](RESULTS.md) and do not replace them.", "",
        "## E5-A, placebo (stated first)", "",
        f"{placebo} Joint test of degree days above 30 °C and precipitation over the 61 "
        f"days after each region's maturity date: two-way p = {_p(e5['p_twoway'])}, "
        f"amended bootstrap p = {_p(e5['p_bootstrap'])} (by country "
        f"{_p(e5['p_bootstrap_country'])}, by year {_p(e5['p_bootstrap_year'])}).", "",
        "## Decisions", "",
        "Holm over E1-E3, separately under the two-way method and the amended bootstrap "
        "(the larger of its runs by country and by year); established only if rejected "
        "under both, with the required sign. E2 is tested as the dry-season contrast "
        "itself.", "",
        "| Estimand | Registered verdict | M1-A estimate | p, two-way (Holm) | "
        "p, amended bootstrap (Holm) | Amended verdict |",
        "| --- | --- | ---: | ---: | ---: | --- |",
        f"| E1, 10 more degree days above 30 °C | {reg['E1']['verdict']} | "
        f"{e1['estimate']:+.4f} ({_pct(e1['estimate'])}) | {_p(d['E1']['p_twoway_holm'])} | "
        f"{_p(d['E1']['p_bootstrap_holm'])} | {d['E1']['verdict']} |",
        f"| E2, a dry season (P10 against median) | {reg['E2']['verdict']} (rule flawed; "
        f"see the change log) | {e2['estimate']:+.4f} ({_pct(e2['estimate'])}) | "
        f"{_p(d['E2']['p_twoway_holm'])} | {_p(d['E2']['p_bootstrap_holm'])} | "
        f"{d['E2']['verdict']} |",
        f"| E3, change in E1 after 2011 | {reg['E3']['verdict']} | {e3['estimate']:+.4f} | "
        f"{_p(d['E3']['p_twoway_holm'])} | {_p(d['E3']['p_bootstrap_holm'])} | "
        f"{d['E3']['verdict']} |", "",
        "## Which change does what (descriptive)", "",
        "E1 and E2 with two-way 95% intervals and both p-values, unadjusted, under each "
        "combination of the amendment's changes. Only M1-A decides.", "",
        "| Combination | N | E1 [95% interval] | p, two-way | p, bootstrap (country, year) "
        "| E2 [95% interval] | p, two-way | p, bootstrap (country, year) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for x in r["decomposition"]:
        a, b = x["E1"], x["E2"]
        lines.append(
            f"| {x['label']} | {x['n']} | {_ci(a)} | {_p(a['p_twoway'])} | "
            f"{_p(a['p_bootstrap'])} ({_p(a['p_bootstrap_country'])}, "
            f"{_p(a['p_bootstrap_year'])}) | {_ci(b)} | {_p(b['p_twoway'])} | "
            f"{_p(b['p_bootstrap'])} ({_p(b['p_bootstrap_country'])}, "
            f"{_p(b['p_bootstrap_year'])}) |")
    excluded = ", ".join(f"{rule} {s['excluded_by_rule'].get(rule, 0)}" for rule in RULES)
    short = ", ".join(s["regions_below_10_years"]) or "none"
    by_country = ", ".join(f"{c} {n}" for c, n in s["excluded_by_country"].items())
    lines += [
        "", "## Sample", "",
        f"- **M1-A:** {s['n']} region-years in {s['regions']} regions of {s['countries']} "
        f"countries, harvest years {s['years'][0]} to {s['years'][1]}.",
        f"- **The data rule (8.2):** of the panel's {s['panel_region_years']} region-years, "
        f"{s['excluded_region_years']} are flagged (by rule, with overlaps: {excluded}). "
        f"By country: {by_country}. Regions then below 10 years and dropped: {short}. "
        "Every flag, with its reason, is in "
        "[amendment/quality_flags.csv](amendment/quality_flags.csv).",
        f"- **The harvest window (8.1):** maturity dates in the sample's "
        f"{mat['regions']} regions run from {_date(mat['first'])} to {_date(mat['last'])}; "
        f"{mat['before_july']} regions mature before July and {mat['after_july']} after "
        "July. Regions where much of the wheat is spring-sown (Finland, Sweden) use the "
        "winter-wheat calendar too.", "",
        "**The review's cases (finding D1).**", "",
        "| Case | Region-years in the panel | Excluded | Rules |", "| --- | ---: | ---: | --- |",
    ]
    lines += [f"| {c['case']} | {c['in_panel']} | {c['excluded']} | {c['rules'] or '-'} |"
              for c in r["review_cases"]]
    lines += [
        "", "## M1-A in detail", "",
        f"- **E1.** {_ci(e1)}: two-way t = {e1['t']:+.2f}, p = {_p(e1['p_twoway'])}; "
        f"amended bootstrap p = {_p(e1['p_bootstrap'])} (by country "
        f"{_p(e1['p_bootstrap_country'])}, by year {_p(e1['p_bootstrap_year'])}).",
        f"- **E2.** A dry season is on average {abs(e2['mean_p10_minus_p50_mm']):.0f} mm "
        f"below the region's median. The contrast is {_ci(e2)}: two-way "
        f"p = {_p(e2['p_twoway'])}; amended bootstrap p = {_p(e2['p_bootstrap'])} (by "
        f"country {_p(e2['p_bootstrap_country'])}, by year {_p(e2['p_bootstrap_year'])}).",
        f"- **E3.** {_ci(e3)}: two-way p = {_p(e3['p_twoway'])}; amended bootstrap "
        f"p = {_p(e3['p_bootstrap'])}. The smallest change detectable with 80% power is "
        f"{e3['minimum_detectable']:.4f} log points per 10 degree days at 5%, and "
        f"{e3['minimum_detectable_holm']:.4f} at the Holm level.", "",
        "## M3-A, the temperature bins over the harvest window (descriptive)", "",
        "Change in log yield for 24 more hours in each 3 °C bin, instead of 24 hours at "
        "15-18 °C; divide by 24 for one hour. No bin is adjusted for multiplicity.", "",
        "| Bin | Estimate | 95% interval, two-way | p, amended bootstrap |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines += [f"| {_bin(b['term'])} | {b['estimate']:+.4f} | [{b['low']:+.4f}, "
              f"{b['high']:+.4f}] | {_p(b['p_bootstrap'])} |" for b in r["M3"]]
    lines += ["", "## Sensitivity: the window ends 14 days before maturity", "",
              "Fixed with the amendment. It changes a decision only if it reverses a sign.",
              "", "| Estimand | N | Estimate [95% interval] | Sign reversed |",
              "| --- | ---: | ---: | --- |"]
    lines += [f"| {x['estimand']} | {x['n']} | {_ci(x)} | "
              f"{'**yes**' if x['sign_reversed'] else 'no'} |" for x in r["sensitivity"]]
    loo = r["leave_one_out"]["summary"]
    lines += ["", "## Leave one country out (descriptive)", "",
              "| Estimand | M1-A | Lowest | Highest | Most influential country | "
              "Without it |", "| --- | ---: | ---: | ---: | --- | ---: |"]
    for key, full in (("E1", e1), ("E2", e2), ("E3", e3)):
        x = loo[key]
        lines.append(f"| {key} | {full['estimate']:+.4f} | {x['low']:+.4f} | "
                     f"{x['high']:+.4f} | {x['most_influential']} | {x['without_it']:+.4f} |")
    lines += ["", "## Not covered", "",
              "- E4 and the sensitivity analyses of section 6 are not recomputed (8.4).",
              "- One calendar, rainfed winter wheat, is used for every region (8.1).",
              "- Errors the data rule does not catch stay in the sample; the table of the "
              "review's cases shows which.",
              "", "## Reproduce", "", "```", command, "```", ""]
    return "\n".join(lines)
