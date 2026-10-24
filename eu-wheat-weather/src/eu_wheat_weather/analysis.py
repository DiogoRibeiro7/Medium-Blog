"""The estimates of METHODS.md (sections 4-6, change log of 2026-10-24) and their report.

Everything here is fixed by the plan: the models, the variables and their units, the
two inference methods, the decision rules and the sensitivity analyses. The report
states the placebo (E5) before any other result.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eu_wheat_weather.estimation import (
    Array,
    Fit,
    dummies,
    fit,
    holm,
    minimum_detectable,
    singleton_mask,
    t_test,
    twoway_vcov,
    wald_test,
    wild_bootstrap,
)
from eu_wheat_weather.weather import bin_names

ALPHA = 0.05
POST_START = 2012
E4_YEARS = (2003, 2018, 2022)
OMITTED_BIN = "t_15_18"
BASE = ("gdd", "edd", "p", "p2")


# ----------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    """One way of building M1's sample and variables (METHODS.md, section 6)."""

    name: str
    label: str
    window: str = "mar_jul"
    weighting: str = "all"
    threshold: int = 30
    drought: str = "prec"
    effects: str = "region+country_year"     # or "region+trend+year"
    weighted: bool = False
    min_years: int = 0
    nuts2021_only: bool = False


PRIMARY = Variant("primary", "M1 as registered")
SENSITIVITIES: tuple[Variant, ...] = (
    Variant("edd27", "EDD and GDD at 27 °C", threshold=27),
    Variant("edd33", "EDD and GDD at 33 °C", threshold=33),
    Variant("apr_jul", "Season 1 April to 31 July", window="apr_jul"),
    Variant("trend_year", "Region trends and year effects", effects="region+trend+year"),
    Variant("below_800m", "Weather over cells below 800 m", weighting="below_800m"),
    Variant("area_weighted", "Weighted by wheat area", weighted=True),
    Variant("min_20_years", "Regions with at least 20 years", min_years=20),
    Variant("nuts2021_only", "NUTS-2021 codes only", nuts2021_only=True),
    Variant("water_balance", "Water balance in place of precipitation", drought="wb"),
)


@dataclass(frozen=True)
class Obs:
    geo: str
    country: str
    year: int
    log_yield: float
    area: float
    x: dict[str, float]


def load(panel_path: Path, weather_path: Path,
         ) -> tuple[list[dict[str, str]], dict[tuple[str, int, str, str], dict[str, str]]]:
    with panel_path.open(encoding="utf-8") as stream:
        panel = list(csv.DictReader(stream))
    with weather_path.open(encoding="utf-8") as stream:
        weather = {(r["geo"], int(r["year"]), r["window"], r["weighting"]): r
                   for r in csv.DictReader(stream)}
    return panel, weather


def observations(variant: Variant, panel: Sequence[Mapping[str, str]],
                 weather: Mapping[tuple[str, int, str, str], Mapping[str, str]],
                 ) -> list[Obs]:
    """The variant's sample with its variables in the registered units."""
    out = []
    for r in panel:
        if variant.nuts2021_only and r["rule"] != "nuts2021":
            continue
        geo, year = r["geo"], int(r["year"])
        w = weather.get((geo, year, variant.window, variant.weighting))
        placebo = weather.get((geo, year, "aug_sep", variant.weighting))
        if w is None or placebo is None:
            continue
        needed = (f"gdd_0_{variant.threshold}", f"edd_{variant.threshold}", variant.drought)
        if any(w[v] == "" for v in needed):
            continue
        p = float(w[variant.drought]) / 100
        x = {"gdd": float(w[f"gdd_0_{variant.threshold}"]) / 100,
             "edd": float(w[f"edd_{variant.threshold}"]) / 10, "p": p, "p2": p * p}
        if placebo["edd_30"] != "" and placebo["prec"] != "":
            x["edd_aug_sep"] = float(placebo["edd_30"]) / 10
            x["p_aug_sep"] = float(placebo["prec"]) / 100
        if all(w[b] != "" for b in bin_names()):
            x.update({b: float(w[b]) for b in bin_names()})
        out.append(Obs(geo, r["country"], year, float(r["log_yield"]), float(r["area_kha"]), x))
    if variant.min_years:
        years = Counter(o.geo for o in out)
        out = [o for o in out if years[o.geo] >= variant.min_years]
    return out


def effect_groups(obs: Sequence[Obs], effects: str) -> list[list[str]]:
    regions = [o.geo for o in obs]
    if effects == "region+country_year":
        return [regions, [f"{o.country}|{o.year}" for o in obs]]
    return [regions, [str(o.year) for o in obs]]


def controls(obs: Sequence[Obs], effects: str) -> Array:
    """The absorbed effects' dummies (and region trends)."""
    regions = dummies([o.geo for o in obs])
    if effects == "region+country_year":
        return np.column_stack([regions, dummies([f"{o.country}|{o.year}" for o in obs])])
    trend = np.array([o.year - 2012 for o in obs], dtype=float)
    parts = [regions, regions * trend[:, None]]
    if effects == "region+trend+year":
        parts.append(dummies([str(o.year) for o in obs]))
    return np.column_stack(parts)


def without_singletons(obs: Sequence[Obs], effects: str) -> tuple[list[Obs], int]:
    keep = singleton_mask(effect_groups(obs, effects))
    return [o for o, k in zip(obs, keep, strict=True) if k], int((~keep).sum())


def estimate(obs: Sequence[Obs], names: Sequence[str], effects: str,
             weighted: bool = False, extra: Mapping[str, Callable[[Obs], float]] | None = None,
             ) -> Fit:
    extra = extra or {}
    columns = [*names, *extra]
    x = np.array([[o.x[n] for n in names] + [f(o) for f in extra.values()] for o in obs])
    y = np.array([o.log_yield for o in obs])
    weights = np.array([o.area for o in obs]) if weighted else None
    return fit(columns, x, y, controls(obs, effects), [o.country for o in obs],
               [str(o.year) for o in obs], weights)


# ----------------------------------------------------------------------------------
# Estimands
# ----------------------------------------------------------------------------------


def coefficient_rows(f: Fit, model: str) -> list[dict[str, Any]]:
    v, df = twoway_vcov(f)
    rows = []
    for j, name in enumerate(f.names):
        t, p, se = t_test(f.beta, v, j, df)
        half = float(_tcrit(df) * se)
        rows.append({"model": model, "term": name, "estimate": float(f.beta[j]), "se": se,
                     "low": float(f.beta[j]) - half, "high": float(f.beta[j]) + half,
                     "t": t, "p": p, "df": df, "n": f.n})
    return rows


def _tcrit(df: int) -> float:
    from scipy import stats

    return float(stats.t.ppf(1 - ALPHA / 2, df))


def dry_season_effect(f: Fit, obs: Sequence[Obs], drought_key: str = "p",
                      ) -> dict[str, float]:
    """E2: mean over regions of g1 (P10 - P50) + g2 (P10^2 - P50^2), delta-method CI."""
    by_region: dict[str, list[float]] = {}
    for o in obs:
        by_region.setdefault(o.geo, []).append(o.x[drought_key])
    d1, d2 = [], []
    for values in by_region.values():
        p10, p50 = float(np.percentile(values, 10)), float(np.percentile(values, 50))
        d1.append(p10 - p50)
        d2.append(p10 * p10 - p50 * p50)
    grad = np.zeros(f.k)
    grad[f.index(["p"])[0]], grad[f.index(["p2"])[0]] = np.mean(d1), np.mean(d2)
    v, df = twoway_vcov(f)
    effect = float(grad @ f.beta)
    se = float(np.sqrt(grad @ v @ grad))
    half = _tcrit(df) * se
    return {"effect": effect, "se": se, "low": effect - half, "high": effect + half,
            "mean_p10_minus_p50": float(np.mean(d1)) * 100}


def tests_for(f: Fit, terms: Sequence[str], bootstrap: bool = True) -> dict[str, float]:
    """Two-way and wild-bootstrap tests that ``terms`` are zero."""
    v, df = twoway_vcov(f)
    idx = f.index(terms)
    if len(idx) == 1:
        stat, p_twoway, _ = t_test(f.beta, v, idx[0], df)
    else:
        stat, p_twoway = wald_test(f.beta, v, idx, df)
    out = {"statistic": stat, "p_twoway": p_twoway, "df": df}
    if bootstrap:
        boot_stat, p_boot = wild_bootstrap(f, idx)
        out.update({"statistic_country_cr1": boot_stat, "p_bootstrap": p_boot})
    return out


def primary_results(panel: Sequence[Mapping[str, str]],
                    weather: Mapping[tuple[str, int, str, str], Mapping[str, str]],
                    ) -> dict[str, Any]:
    raw = observations(PRIMARY, panel, weather)
    obs, singletons = without_singletons(raw, PRIMARY.effects)
    m1 = estimate(obs, BASE, PRIMARY.effects)
    e1 = tests_for(m1, ["edd"])
    e2: dict[str, Any] = dict(tests_for(m1, ["p", "p2"]))
    e2["dry_season"] = dry_season_effect(m1, obs)
    g1, g2 = m1.beta[m1.index(["p"])[0]], m1.beta[m1.index(["p2"])[0]]
    e2["optimum_mm"] = float(-g1 / (2 * g2) * 100) if g2 < 0 else None
    post = {"edd_post": lambda o: o.x["edd"] * (o.year >= POST_START)}
    m1_e3 = estimate(obs, BASE, PRIMARY.effects, extra=post)
    e3: dict[str, Any] = dict(tests_for(m1_e3, ["edd_post"]))
    v3, df3 = twoway_vcov(m1_e3)
    e3["minimum_detectable"] = minimum_detectable(float(np.sqrt(v3[-1, -1])), df3)
    placebo_obs = [o for o in obs if "edd_aug_sep" in o.x]
    placebo_obs, _ = without_singletons(placebo_obs, PRIMARY.effects)
    m1_e5 = estimate(placebo_obs, [*BASE, "edd_aug_sep", "p_aug_sep"], PRIMARY.effects)
    e5: dict[str, Any] = dict(tests_for(m1_e5, ["edd_aug_sep", "p_aug_sep"]))
    e5["significant_both"] = bool(e5["p_twoway"] < ALPHA and e5["p_bootstrap"] < ALPHA)
    decisions = decide(m1, e1, e2, m1_e3, e3)
    return {
        "sample": {"region_years": len(raw), "singletons_dropped": singletons,
                   "n": m1.n, "regions": len({o.geo for o in obs}),
                   "countries": len({o.country for o in obs}),
                   "country_years": len({(o.country, o.year) for o in obs}),
                   "years": [min(o.year for o in obs), max(o.year for o in obs)]},
        "coefficients": coefficient_rows(m1, "M1") + coefficient_rows(m1_e3, "M1 with E3")
        + coefficient_rows(m1_e5, "M1 with E5"),
        "E1": e1, "E2": e2, "E3": e3, "E5": e5, "decisions": decisions,
        "second_route": second_route(obs, m1),
        "_fits": {"m1": m1, "obs": obs},
    }


def decide(m1: Fit, e1: Mapping[str, float], e2: Mapping[str, Any], m1_e3: Fit,
           e3: Mapping[str, float]) -> dict[str, Any]:
    """Holm under each method; established only if rejected under both, with the sign."""
    keys = ("E1", "E2", "E3")
    pt = holm([e1["p_twoway"], e2["p_twoway"], e3["p_twoway"]])
    pb = holm([e1["p_bootstrap"], e2["p_bootstrap"], e3["p_bootstrap"]])
    edd = float(m1.beta[m1.index(["edd"])[0]])
    interaction = float(m1_e3.beta[m1_e3.index(["edd_post"])[0]])
    effect = float(e2["dry_season"]["effect"])
    out: dict[str, Any] = {}
    for k, a, b in zip(keys, pt, pb, strict=True):
        both = a < ALPHA and b < ALPHA
        if k == "E1":
            verdict = ("established negative" if both and edd < 0 else
                       "contrary to expectation (positive)" if both else "not determined")
        elif k == "E2":
            verdict = ("established negative" if both and effect < 0 else
                       "rejected but not negative" if both else "not determined")
        else:
            verdict = (("established: sensitivity rose" if interaction < 0 else
                        "established: sensitivity fell") if both else "not determined")
        out[k] = {"p_twoway_holm": a, "p_bootstrap_holm": b, "verdict": verdict}
    return out


SECOND_ROUTE_TOLERANCE = 1e-8


def second_route(obs: Sequence[Obs], m1: Fit) -> dict[str, Any]:
    """M1 by explicit dummy variables: the weather coefficients must agree to 1e-8.

    Only the tolerance and the outcome are recorded: the size of the floating-point
    disagreement itself differs between platforms.
    """
    x = np.array([[o.x[n] for n in BASE] for o in obs])
    design = np.column_stack([x, controls(obs, PRIMARY.effects)])
    coef, *_ = np.linalg.lstsq(design, np.array([o.log_yield for o in obs]), rcond=None)
    diff = float(np.max(np.abs(coef[:len(BASE)] - m1.beta)))
    if diff > SECOND_ROUTE_TOLERANCE:
        raise ValueError(f"The two routes to M1 differ by {diff:.2e}.")
    return {"tolerance": SECOND_ROUTE_TOLERANCE, "within_tolerance": True}


# ----------------------------------------------------------------------------------
# M3, E4 and the sensitivity analyses
# ----------------------------------------------------------------------------------


def shape(obs: Sequence[Obs]) -> list[dict[str, Any]]:
    """M3: days in 3 °C bins relative to 15-18 °C (descriptive)."""
    with_bins = [o for o in obs if OMITTED_BIN in o.x]
    with_bins, _ = without_singletons(with_bins, PRIMARY.effects)
    names = [b for b in bin_names() if b != OMITTED_BIN]
    f = estimate(with_bins, [*names, "p", "p2"], PRIMARY.effects)
    return [r for r in coefficient_rows(f, "M3") if r["term"].startswith("t_")]


def out_of_sample(obs_all: Sequence[Obs]) -> list[dict[str, Any]]:
    """E4: M2 without the year, used to predict it (change log, 2026-10-24)."""
    out = []
    for year in E4_YEARS:
        train = [o for o in obs_all if o.year != year]
        test = [o for o in obs_all if o.year == year]
        regions = {o.geo for o in train}
        test = [o for o in test if o.geo in regions]
        if not test:
            continue
        f = estimate(train, BASE, "region+trend")
        beta = f.beta
        x_train = np.array([[o.x[n] for n in BASE] for o in train])
        y_train = np.array([o.log_yield for o in train])
        c = controls(train, "region+trend")
        coef, *_ = np.linalg.lstsq(c, y_train - x_train @ beta, rcond=None)
        names = sorted(regions)
        level = dict(zip(names, coef[:len(names)], strict=True))
        slope = dict(zip(names, coef[len(names):], strict=True))
        mean_weather: dict[str, Array] = {}
        for g in names:
            rows = np.array([[o.x[n] for n in ("gdd", "edd", "p")] for o in train if o.geo == g])
            m = rows.mean(axis=0)
            mean_weather[g] = np.array([m[0], m[1], m[2], m[2] ** 2])
        predicted, observed, normal, area = [], [], [], []
        for o in test:
            x_now = np.array([o.x[n] for n in BASE])
            effect_mean = float(mean_weather[o.geo] @ beta)
            norm = level[o.geo] + slope[o.geo] * (year - 2012) + effect_mean
            predicted.append(float(x_now @ beta) - effect_mean)
            observed.append(o.log_yield - norm)
            normal.append(norm)
            area.append(o.area)
        pr, ob, nm, ar = (np.array(v) for v in (predicted, observed, normal, area))
        base = float(np.sum(ar * np.exp(nm)))
        out.append({
            "year": year, "regions": len(test),
            "correlation": float(np.corrcoef(pr, ob)[0, 1]),
            "variance_explained": float(1 - np.sum((ob - pr) ** 2)
                                        / np.sum((ob - ob.mean()) ** 2)),
            "eu_yield_observed_pct": 100 * (float(np.sum(ar * np.exp(nm + ob))) / base - 1),
            "eu_yield_predicted_pct": 100 * (float(np.sum(ar * np.exp(nm + pr))) / base - 1),
        })
    return out


def sensitivity(panel: Sequence[Mapping[str, str]],
                weather: Mapping[tuple[str, int, str, str], Mapping[str, str]],
                primary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """E1-E3 under each registered variant, with two-way 95% intervals."""
    reference = {
        "E1": next(r for r in primary["coefficients"] if r["model"] == "M1"
                   and r["term"] == "edd")["estimate"],
        "E2": primary["E2"]["dry_season"]["effect"],
        "E3": next(r for r in primary["coefficients"] if r["model"] == "M1 with E3"
                   and r["term"] == "edd_post")["estimate"],
    }
    rows = []
    for variant in SENSITIVITIES:
        effects = variant.effects
        obs, _ = without_singletons(observations(variant, panel, weather), effects)
        m = estimate(obs, BASE, effects, variant.weighted)
        m3 = estimate(obs, BASE, effects, variant.weighted,
                      extra={"edd_post": lambda o: o.x["edd"] * (o.year >= POST_START)})
        e1 = next(r for r in coefficient_rows(m, variant.name) if r["term"] == "edd")
        e3 = next(r for r in coefficient_rows(m3, variant.name) if r["term"] == "edd_post")
        e2 = dry_season_effect(m, obs)
        v, df = twoway_vcov(m)
        _, p_e2 = wald_test(m.beta, v, m.index(["p", "p2"]), df)
        for key, est, low, high, p in (
                ("E1", e1["estimate"], e1["low"], e1["high"], e1["p"]),
                ("E2", e2["effect"], e2["low"], e2["high"], p_e2),
                ("E3", e3["estimate"], e3["low"], e3["high"], e3["p"])):
            rows.append({"variant": variant.name, "label": variant.label, "estimand": key,
                         "n": m.n, "estimate": est, "low": low, "high": high, "p_twoway": p,
                         "sign_reversed": bool(np.sign(est) != np.sign(reference[key])
                                               and (low > 0 or high < 0))})
    return rows


# ----------------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------------


def run(panel_path: Path, weather_path: Path) -> dict[str, Any]:
    panel, weather = load(panel_path, weather_path)
    primary = primary_results(panel, weather)
    obs = primary.pop("_fits")["obs"]
    all_obs = observations(PRIMARY, panel, weather)
    return {**primary, "M3": shape(obs), "E4": out_of_sample(all_obs),
            "sensitivity": sensitivity(panel, weather, primary)}


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return None if math.isnan(value) else float(f"{value:.10g}")
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round(v) for v in value]
    return value


def write_results(results: Mapping[str, Any], output_dir: Path, report: Path,
                  command: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    document = _round(dict(results))
    (output_dir / "results.json").write_text(json.dumps(document, indent=1) + "\n",
                                             encoding="utf-8", newline="\n")
    for name, rows in (("coefficients.csv", document["coefficients"]),
                       ("bins.csv", document["M3"]), ("out_of_sample.csv", document["E4"]),
                       ("sensitivity.csv", document["sensitivity"])):
        with (output_dir / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render(document, command), encoding="utf-8", newline="\n")


def _pct(log_change: float) -> str:
    return f"{100 * (math.exp(log_change) - 1):+.1f}%"


def _p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _bin(term: str) -> str:
    """t_lt00 -> below 0 °C, t_ge36 -> 36 °C and above, t_03_06 -> 3-6 °C."""
    body = term.removeprefix("t_")
    if body.startswith("lt"):
        return f"below {int(body[2:])} °C"
    if body.startswith("ge"):
        return f"{int(body[2:])} °C and above"
    low, high = body.split("_")
    return f"{int(low)}-{int(high)} °C"


def _e2_caveat(dry: Mapping[str, float], verdict: str) -> list[str]:
    """When the rule establishes E2 but the dry-season effect's own interval includes 0."""
    if not verdict.startswith("established") or dry["high"] < 0:
        return []
    return [f"**E2, read with care.** The registered rule takes significance from the joint "
            f"test of the precipitation terms and the sign from the dry-season effect. The "
            f"joint test rejects under both methods, but the dry-season effect's own two-way "
            f"interval, [{dry['low']:+.4f}, {dry['high']:+.4f}], includes zero. Precipitation "
            f"clearly matters, while the size of the loss in a typical dry season is not "
            f"pinned down.", ""]


def render(r: Mapping[str, Any], command: str) -> str:
    s, e1, e2, e3, e5, d = r["sample"], r["E1"], r["E2"], r["E3"], r["E5"], r["decisions"]
    coef = {(c["model"], c["term"]): c for c in r["coefficients"]}
    edd, post = coef[("M1", "edd")], coef[("M1 with E3", "edd_post")]
    dry = e2["dry_season"]
    placebo = ("**The placebo fails: post-harvest weather is significant under both methods. "
               "The identification is in doubt, and every result below must be read with "
               "that in mind.**" if e5["significant_both"] else
               "The placebo passes: post-harvest weather is not significant under both "
               "methods.")
    lines = [
        "# Results: heat, drought and wheat yields", "",
        "Generated by `eu-wheat estimate` from the committed panel. Every model, variable, "
        "test and decision rule was fixed in [METHODS.md](../METHODS.md) before any "
        "estimate (sections 4-6 and the change log of 2026-10-24). The coefficients are "
        "effects of weather shocks given each region's normal and each country's year, "
        "not effects of a changed climate.", "",
        "## E5, placebo (stated first, as the plan requires)", "",
        f"{placebo} Joint test of post-harvest degree days above 30 °C and precipitation "
        f"(1 August to 30 September): two-way p = {_p(e5['p_twoway'])}, wild bootstrap "
        f"p = {_p(e5['p_bootstrap'])}.", "",
        "## Decisions", "",
        "Holm over E1-E3, separately under each method; established only if rejected "
        "under both, with the required sign.", "",
        "| Estimand | Estimate | p, two-way (Holm) | p, wild bootstrap (Holm) | Verdict |",
        "| --- | ---: | ---: | ---: | --- |",
        f"| E1, 10 more degree days above 30 °C | {edd['estimate']:+.4f} log points "
        f"({_pct(edd['estimate'])}) | {_p(d['E1']['p_twoway_holm'])} | "
        f"{_p(d['E1']['p_bootstrap_holm'])} | {d['E1']['verdict']} |",
        f"| E2, a dry season (P10 against median) | {dry['effect']:+.4f} log points "
        f"({_pct(dry['effect'])}) | {_p(d['E2']['p_twoway_holm'])} | "
        f"{_p(d['E2']['p_bootstrap_holm'])} | {d['E2']['verdict']} |",
        f"| E3, change in E1 after 2011 | {post['estimate']:+.4f} log points | "
        f"{_p(d['E3']['p_twoway_holm'])} | {_p(d['E3']['p_bootstrap_holm'])} | "
        f"{d['E3']['verdict']} |", "",
        *_e2_caveat(dry, d["E2"]["verdict"]),
        "## Sample", "",
        f"- {s['n']} region-years in {s['regions']} regions of {s['countries']} countries "
        f"({s['country_years']} country-years), harvest years {s['years'][0]} to "
        f"{s['years'][1]}.",
        f"- {s['region_years']} region-years have the weather M1 needs; "
        f"{s['singletons_dropped']} were alone in their region or country-year and carry "
        "no information under M1.", "",
        "## M1 in detail", "",
        "Two-way clustered (country and year) 95% intervals, with "
        f"{edd['df']} degrees of freedom. Units: GDD in hundreds of degree days, EDD in "
        "tens, precipitation in hundreds of mm, log yield in natural logs.", "",
        "| Model | Term | Estimate | 95% interval | p |", "| --- | --- | ---: | ---: | ---: |",
    ]
    lines += [f"| {c['model']} | `{c['term']}` | {c['estimate']:+.4f} | "
              f"[{c['low']:+.4f}, {c['high']:+.4f}] | {_p(c['p'])} |" for c in r["coefficients"]]
    optimum = ("" if e2["optimum_mm"] is None else
               f" Yield peaks at {e2['optimum_mm']:.0f} mm over March to July.")
    lines += [
        "",
        f"- **E1.** Two-way t = {e1['statistic']:+.2f} (p = {_p(e1['p_twoway'])}); with "
        f"country-clustered errors t = {e1['statistic_country_cr1']:+.2f}, wild bootstrap "
        f"p = {_p(e1['p_bootstrap'])}.",
        f"- **E2.** Joint test of the precipitation terms: two-way p = "
        f"{_p(e2['p_twoway'])}, wild bootstrap p = {_p(e2['p_bootstrap'])}. A dry season "
        f"is on average {abs(dry['mean_p10_minus_p50']):.0f} mm below the region's median; "
        f"it changes log yield by {dry['effect']:+.4f} [{dry['low']:+.4f}, "
        f"{dry['high']:+.4f}].{optimum}",
        f"- **E3.** The interaction is {post['estimate']:+.4f} [{post['low']:+.4f}, "
        f"{post['high']:+.4f}]: two-way p = {_p(e3['p_twoway'])}, wild bootstrap "
        f"p = {_p(e3['p_bootstrap'])}. The smallest change this design detects with 80% "
        f"power is {e3['minimum_detectable']:.4f} log points per 10 degree days.",
        f"- **Second route.** M1 by explicit dummy variables agrees within the registered "
        f"tolerance of {r['second_route']['tolerance']:.0e}.", "",
        "## M3, the shape of the temperature response (descriptive)", "",
        "Change in log yield for one more day in each 3 °C bin, instead of 15-18 °C.", "",
        "| Bin | Estimate | 95% interval |", "| --- | ---: | ---: |",
    ]
    lines += [f"| {_bin(b['term'])} | {b['estimate']:+.4f} | "
              f"[{b['low']:+.4f}, {b['high']:+.4f}] |" for b in r["M3"]]
    lines += ["", "## E4, out of sample (descriptive)", "",
              "M2 (region effects and trends) fitted without the year, then used to predict "
              "it. A negative share of variance explained means the prediction does worse "
              "than the mean anomaly. EU yields are area-weighted, relative to the normal.",
              "",
              "| Year | Regions | Correlation | Variance explained | EU yield, observed | "
              "EU yield, predicted |", "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    lines += [f"| {x['year']} | {x['regions']} | {x['correlation']:+.2f} | "
              f"{x['variance_explained']:+.2f} | {x['eu_yield_observed_pct']:+.1f}% | "
              f"{x['eu_yield_predicted_pct']:+.1f}% |" for x in r["E4"]]
    lines += ["", "## Sensitivity analyses", "",
              "E1 to E3 under each variant fixed in advance (two-way 95% intervals). A "
              "variant changes a decision only if it reverses the sign with an interval "
              "that excludes zero.", "",
              "| Variant | Estimand | N | Estimate | 95% interval | Sign reversed |",
              "| --- | --- | ---: | ---: | ---: | --- |"]
    lines += [f"| {x['label']} | {x['estimand']} | {x['n']} | {x['estimate']:+.4f} | "
              f"[{x['low']:+.4f}, {x['high']:+.4f}] | {'**yes**' if x['sign_reversed'] else 'no'} |"
              for x in r["sensitivity"]]
    lines += ["", "## Reproduce", "", "```", command, "```", ""]
    return "\n".join(lines)

