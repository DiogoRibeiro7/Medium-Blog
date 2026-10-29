"""Amendment A2 to the maize study (METHODS.md, section 10), beside the registered results.

Weather over where maize grows (10.1), a wider data rule (10.2), irrigated and rainfed
maize apart (10.3) and inference with few clusters (10.4). E1-A2 and E2-A2 are decided as
in section 9.2; everything else is descriptive. The registered maize results are read,
never recomputed.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eu_wheat_weather.amendment import _ci, both, contrast_fit, two_way
from eu_wheat_weather.analysis import (
    ALPHA,
    BASE,
    Obs,
    _dry_deltas,
    _round,
    controls,
    dry_season_effect,
    estimate,
    without_singletons,
)
from eu_wheat_weather.contract import IRRIGATION_COUNTRIES, IRRIGATION_FILTERS, DataError
from eu_wheat_weather.crops import WheatRow
from eu_wheat_weather.estimation import Fit, cluster_vcov, t_test, twoway_vcov
from eu_wheat_weather.jsonstat import decode
from eu_wheat_weather.maize import build_panel
from eu_wheat_weather.maize_analysis import EFFECTS, PLACEBO_TERMS, decide, observations
from eu_wheat_weather.nuts_changes import Change, recode_target
from eu_wheat_weather.panel import read_region_map, read_weather, read_wheat, write_csv
from eu_wheat_weather.quality import REVIEW_CASES_A2, RULES_A2, Flag, review_cases
from eu_wheat_weather.regions import CodeMatch, HarmonisedRow, harmonise

RANDOMIZATION_DRAWS = 9_999
RANDOMIZATION_SEED = 20261103
IRRIGATED_SHARE = 0.5                     # a region is irrigated at one half or more (10.3)
PRIMARY = ("maize", True)                 # M1-maize-A2: maize weights and the wider rule
COMBINATIONS: tuple[tuple[str, str, str, bool], ...] = (
    ("weights", "Maize weights alone", "maize", False),
    ("rule", "The wider data rule alone", "all", True),
    ("both", "Both: M1-maize-A2", "maize", True),
    ("below_800m", "Cells below 800 m, with the wider rule", "below_800m", True),
)
WeatherKey = tuple[str, int, str]


@dataclass(frozen=True)
class Inputs:
    published: list[WheatRow]
    matches: list[CodeMatch]
    harmonised: list[HarmonisedRow]
    equivalent: dict[str, str]
    weather: dict[tuple[str, int, str, str], dict[str, str]]


def read_changes(path: Path) -> list[Change]:
    with path.open(encoding="utf-8") as stream:
        return [Change(int(r["version_from"]), int(r["version_to"]), r["code_from"],
                       r["code_to"], r["change"]) for r in csv.DictReader(stream)]


def equivalents(published: Sequence[WheatRow], matches: Sequence[CodeMatch],
                changes: Sequence[Change]) -> dict[str, str]:
    """Q8's exception for older parent codes (change log): a NUTS-0 or NUTS-1 code whose
    NUTS-2 regions, in every version known, all lead to one NUTS-2021 region."""
    target = {m.code: m.target for m in matches}
    nuts2021 = {m.target for m in matches if m.target}
    version = {c.code_from: c.version_from for c in changes if c.code_from}
    nuts2 = ({w.geo for w in published if len(w.geo) == 4} | set(version)
             | {c.code_to for c in changes if c.code_to} | nuts2021)
    out = {}
    for code in sorted({w.geo for w in published if 2 <= len(w.geo) <= 3}):
        goals = {target.get(n) or (recode_target(n, version[n], changes)[0] if n in version
                                   else (n if n in nuts2021 else None))
                 for n in nuts2 if n.startswith(code)}
        if len(goals) == 1 and None not in goals:
            out[code] = str(goals.pop())
    return out


def load_inputs(crops: Path, region_map: Path, changes: Path, weather: Path) -> Inputs:
    published, matches = read_wheat(crops), read_region_map(region_map)
    harmonised, _ = harmonise(published, matches)
    return Inputs(published, matches, harmonised,
                  equivalents(published, matches, read_changes(changes)), read_weather(weather))


def by_window(weather: Mapping[tuple[str, int, str, str], Mapping[str, str]], weighting: str,
              ) -> dict[WeatherKey, Mapping[str, str]]:
    return {(g, y, w): r for (g, y, w, wt), r in weather.items() if wt == weighting}


def sample(inputs: Inputs, weighting: str, wider: bool,
           ) -> tuple[list[dict[str, Any]], list[Flag], list[Obs], int]:
    """A panel of section 9.1 with the weather of ``weighting`` and, if ``wider``, the rule
    of 10.2; and its observations, singletons dropped."""
    panel, _, flags = build_panel(inputs.harmonised, inputs.published, inputs.matches,
                                  inputs.weather, ("season", weighting), wider,
                                  inputs.equivalent)
    rows = [asdict(r) for r in panel]
    obs, singletons = without_singletons(
        observations(rows, by_window(inputs.weather, weighting)), EFFECTS)
    return rows, flags, obs, singletons


def e1_e2(obs: Sequence[Obs]) -> tuple[dict[str, Any], dict[str, Any], Fit]:
    m1 = estimate(obs, BASE, EFFECTS)
    return both(m1, ["edd"]), both(contrast_fit(obs), ["contrast"]), m1


def leave_one_country_out(obs: Sequence[Obs], full: Mapping[str, float]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for country in sorted({o.country for o in obs}):
        sub, _ = without_singletons([o for o in obs if o.country != country], EFFECTS)
        e1, e2 = two_way(estimate(sub, BASE, EFFECTS), ["edd"]), two_way(contrast_fit(sub),
                                                                         ["contrast"])
        rows.append({"country": country, "n": e1["n"], "E1": e1["estimate"],
                     "E1_p_twoway": e1["p_twoway"], "E2": e2["estimate"],
                     "E2_p_twoway": e2["p_twoway"]})
    summary = {}
    for key in ("E1", "E2"):
        most = max(rows, key=lambda r: abs(float(r[key]) - full[key]))
        summary[key] = {"low": min(r[key] for r in rows), "high": max(r[key] for r in rows),
                        "p_twoway_above_005": sum(1 for r in rows
                                                  if r[f"{key}_p_twoway"] >= ALPHA),
                        "most_influential": most["country"], "without_it": most[key]}
    return {"rows": rows, "summary": summary}


# ----------------------------------------------------------------------------------
# Irrigated and rainfed maize (10.3)
# ----------------------------------------------------------------------------------


def read_map(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def irrigation(obs: Sequence[Obs], shares: Mapping[str, float]) -> dict[str, Any]:
    """Heat and drought in rainfed and irrigated regions, and their differences, from M1
    with `EDD` and the dry-season terms interacted with the irrigated group."""
    irrigated = {o.geo: shares[o.geo] >= IRRIGATED_SHARE for o in obs}
    d1, d2 = _dry_deltas(obs)
    terms: dict[str, Callable[[Obs], float]] = {
        "edd": lambda o: o.x["edd"], "contrast": lambda o: o.x["p"] / d1,
        "curvature": lambda o: o.x["p2"] - (d2 / d1) * o.x["p"]}

    def in_group(term: Callable[[Obs], float], wanted: bool) -> Callable[[Obs], float]:
        return lambda o: term(o) if irrigated[o.geo] == wanted else 0.0

    split = estimate(obs, ["gdd"], EFFECTS, extra={
        f"{name}_{label}": in_group(term, wanted) for name, term in terms.items()
        for label, wanted in (("rainfed", False), ("irrigated", True))})
    diff = estimate(obs, ["gdd"], EFFECTS, extra={
        **terms, **{f"{name}_x_irrigated": in_group(term, True)
                    for name, term in terms.items()}})
    regions = {o.geo: o.country for o in obs}
    out: dict[str, Any] = {
        "regions": {"irrigated": sum(irrigated[g] for g in regions),
                    "rainfed": sum(not irrigated[g] for g in regions)},
        "region_years": {"irrigated": sum(irrigated[o.geo] for o in obs),
                         "rainfed": sum(not irrigated[o.geo] for o in obs)},
        "irrigated_by_country": dict(sorted(Counter(c for g, c in regions.items()
                                                    if irrigated[g]).items()))}
    for key, term in (("E1", "edd"), ("E2", "contrast")):
        out[key] = {"rainfed": both(split, [f"{term}_rainfed"]),
                    "irrigated": both(split, [f"{term}_irrigated"]),
                    "difference": both(diff, [f"{term}_x_irrigated"])}
    return out


def eurostat_check(directory: Path, rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    """The map's irrigated share of maize by country against Eurostat's 2023 figures."""
    values = {}
    for name, filters in IRRIGATION_FILTERS.items():
        payload = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
        values[name] = {c.codes["geo"]: c.value for c in decode(payload, filters)}
    out = []
    for country in IRRIGATION_COUNTRIES:
        mine = [r for r in rows if r["geo"].startswith(country)]
        total = sum(float(r["maize_ha"]) for r in mine)
        irr = sum(float(r["irrigated_ha"]) for r in mine)
        stat_irr, stat_all = values["ef_lus_irri"][country], values["ef_lus_allcrops"][country]
        if stat_irr is None or stat_all is None:
            raise DataError(f"Eurostat has no irrigated maize figure for {country}.")
        out.append({"country": country, "map_share": irr / total,
                    "eurostat_share": stat_irr / stat_all, "eurostat_irrigated_ha": stat_irr,
                    "eurostat_ha": stat_all})
    return out


# ----------------------------------------------------------------------------------
# Inference with few clusters (10.4)
# ----------------------------------------------------------------------------------


def country_clustered(f: Fit, term: str) -> dict[str, float]:
    """One-way country clustering with the code's small-sample factor, t with G-1 df."""
    j = f.index([term])[0]
    v, g = cluster_vcov(f.x, f.resid, f.country, f.k)
    t, p, se = t_test(f.beta, v, j, g - 1)
    return {"se": se, "t": t, "p": p, "df": g - 1}


class _TwoWay:
    """estimation.twoway_vcov for one sample, with the cluster indicators built once."""

    def __init__(self, country: np.ndarray, year: np.ndarray) -> None:
        cell = np.char.add(np.char.add(country, "|"), year)
        self.indicators = [self._indicator(c) for c in (country, year, cell)]
        self.df = min(self.indicators[0].shape[1], self.indicators[1].shape[1]) - 1

    @staticmethod
    def _indicator(codes: np.ndarray) -> np.ndarray:
        _, inverse = np.unique(codes, return_inverse=True)
        out = np.zeros((codes.size, int(inverse.max()) + 1))
        out[np.arange(codes.size), inverse] = 1.0
        return out

    def t(self, x: np.ndarray, y: np.ndarray, j: int) -> float:
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        n, k = x.shape
        bread = np.linalg.inv(x.T @ x)
        v = np.zeros((k, k))
        for sign, m in zip((1.0, 1.0, -1.0), self.indicators, strict=True):
            s = m.T @ (x * resid[:, None])
            g = m.shape[1]
            v += sign * g / (g - 1) * (n - 1) / (n - k) * bread @ (s.T @ s) @ bread
        values, vectors = np.linalg.eigh((v + v.T) / 2)
        v = vectors @ np.diag(np.maximum(values, 0.0)) @ vectors.T
        return float(beta[j] / np.sqrt(v[j, j]))


def _designs(obs: Sequence[Obs], weather: Mapping[WeatherKey, Mapping[str, str]],
             years: Mapping[int, int], d1: float, d2: float,
             ) -> tuple[np.ndarray, np.ndarray] | None:
    """M1's and the contrast model's columns with each region-year given the weather of
    ``years[year]``; None if any of it is missing."""
    rows = []
    for o in obs:
        w = weather.get((o.geo, years[o.year], "season"))
        if w is None or any(w[v] == "" for v in ("gdd_10_30", "edd_30", "prec")):
            return None
        rows.append((float(w["gdd_10_30"]) / 100, float(w["edd_30"]) / 10,
                     float(w["prec"]) / 100))
    g, e, p = (np.array(c) for c in zip(*rows, strict=True))
    return (np.column_stack([g, e, p, p * p]),
            np.column_stack([g, e, p / d1, p * p - (d2 / d1) * p]))


def randomization(obs: Sequence[Obs], weather: Mapping[WeatherKey, Mapping[str, str]],
                  draws: int = RANDOMIZATION_DRAWS, seed: int = RANDOMIZATION_SEED,
                  ) -> dict[str, Any]:
    """Section 10.4: the two-way t of E1 and of E2 under random permutations of the harvest
    years, the same permutation for every region; yields, fixed effects, d1 and d2 fixed.
    A permutation in which some region-year's weather is missing leaves those region-years
    out of that permutation."""
    years = sorted({o.year for o in obs})
    d1, d2 = _dry_deltas(obs)
    identity = {y: y for y in years}
    observed = _designs(obs, weather, identity, d1, d2)
    if observed is None:
        raise DataError("The observed sample has missing weather.")

    def t_values(sub: Sequence[Obs], designs: tuple[np.ndarray, np.ndarray]) -> list[float]:
        ctrl = controls(sub, EFFECTS)
        u, s, _ = np.linalg.svd(ctrl, full_matrices=False)
        basis = u[:, s > s[0] * 1e-10]
        y = np.array([o.log_yield for o in sub])
        yt = y - basis @ (basis.T @ y)
        tw = _TwoWay(np.array([o.country for o in sub]), np.array([str(o.year) for o in sub]))
        return [tw.t(x - basis @ (basis.T @ x), yt, j)
                for x, j in zip(designs, (1, 2), strict=True)]

    t_obs = t_values(obs, observed)
    m1, c = estimate(obs, BASE, EFFECTS), contrast_fit(obs)
    registered = [t_test(f.beta, twoway_vcov(f)[0], f.index([term])[0], 1)[0]
                  for f, term in ((m1, "edd"), (c, "contrast"))]
    if not np.allclose(t_obs, registered, rtol=1e-8, atol=1e-10):
        raise DataError(f"The fast two-way t {t_obs} differs from the registered {registered}.")
    rng = np.random.default_rng(seed)
    counts, partial = [0, 0], 0
    for _ in range(draws):
        order = rng.permutation(len(years))
        mapping = {y: years[i] for y, i in zip(years, order, strict=True)}
        designs = _designs(obs, weather, mapping, d1, d2)
        sub: Sequence[Obs] = obs
        if designs is None:
            partial += 1
            sub, _ = without_singletons(
                [o for o in obs if _designs([o], weather, mapping, d1, d2) is not None],
                EFFECTS)
            designs = _designs(sub, weather, mapping, d1, d2)
            assert designs is not None
        for k, t in enumerate(t_values(sub, designs)):
            counts[k] += abs(t) >= abs(t_obs[k]) * (1 - 1e-12)
    return {"draws": draws, "seed": seed, "permutations_with_missing_weather": partial,
            **{key: {"t": t_obs[k], "p": (1 + counts[k]) / (1 + draws)}
               for k, key in enumerate(("E1", "E2"))}}


def few_clusters(obs: Sequence[Obs], weather: Mapping[WeatherKey, Mapping[str, str]],
                 ) -> dict[str, Any]:
    m1, c = estimate(obs, BASE, EFFECTS), contrast_fit(obs)
    perm = randomization(obs, weather)
    return {"E1": {"country": country_clustered(m1, "edd"), "randomization": perm["E1"]},
            "E2": {"country": country_clustered(c, "contrast"),
                   "randomization": perm["E2"]},
            "draws": perm["draws"], "seed": perm["seed"],
            "permutations_with_missing_weather": perm["permutations_with_missing_weather"]}


# ----------------------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------------------


def _short(x: Mapping[str, Any]) -> dict[str, Any]:
    return {k: x[k] for k in ("estimate", "low", "high", "p_twoway", "p_bootstrap",
                              "p_bootstrap_country", "p_bootstrap_year", "n") if k in x}


def run(inputs: Inputs, map_path: Path, irrigation_dir: Path,
        registered_panel: Path, registered_weather: Path) -> dict[str, Any]:
    rows, flags, obs, singletons = sample(inputs, *PRIMARY)
    e1, e2, m1 = e1_e2(obs)
    e2["mean_p10_minus_p50_mm"] = dry_season_effect(m1, obs)["mean_p10_minus_p50"]
    placebo, _ = without_singletons([o for o in obs if PLACEBO_TERMS[0] in o.x], EFFECTS)
    e5 = both(estimate(placebo, [*BASE, *PLACEBO_TERMS], EFFECTS), PLACEBO_TERMS)
    e5["significant_both"] = bool(e5["p_twoway"] < ALPHA and e5["p_bootstrap"] < ALPHA)
    combinations = []
    for key, label, weighting, wider in COMBINATIONS:
        if (weighting, wider) == PRIMARY:
            ce1, ce2, n = e1, e2, m1.n
        else:
            _, _, sub, _ = sample(inputs, weighting, wider)
            ce1, ce2, _ = e1_e2(sub)
            n = ce1["n"]
        combinations.append({"key": key, "label": label, "n": n, "E1": _short(ce1),
                             "E2": _short(ce2)})
    with registered_panel.open(encoding="utf-8") as stream:
        registered_rows = list(csv.DictReader(stream))
    with registered_weather.open(encoding="utf-8") as stream:
        reg_weather = {(r["geo"], int(r["year"]), r["window"]): r
                       for r in csv.DictReader(stream)}
    reg_obs, _ = without_singletons(observations(registered_rows, reg_weather), EFFECTS)
    regions_map = read_map(map_path)
    shares = {r["geo"]: float(r["irrigated_share"]) for r in regions_map
              if r["irrigated_share"] != ""}
    new_flags = [f for f in flags if f.rule in RULES_A2]
    registered_regions = {r["geo"] for r in registered_rows}
    a2_regions = {r["geo"] for r in rows}
    return {
        "sample": {"n": m1.n, "regions": len({o.geo for o in obs}),
                   "countries": len({o.country for o in obs}), "singletons_dropped": singletons,
                   "years": [min(o.year for o in obs), max(o.year for o in obs)],
                   "registered_n": len(reg_obs)},
        "E1": e1, "E2": e2, "E5": e5, "decisions": decide(e1, e2),
        "combinations": combinations,
        "leave_one_out": leave_one_country_out(obs, {"E1": e1["estimate"],
                                                     "E2": e2["estimate"]}),
        "irrigation": irrigation(obs, shares),
        "few_clusters": {"registered": few_clusters(reg_obs, reg_weather),
                         "amended": few_clusters(obs, by_window(inputs.weather, "maize"))},
        "data_rule": {
            "flags": [asdict(f) for f in sorted(new_flags, key=lambda f: (f.rule, f.geo,
                                                                         f.year))],
            "by_rule": dict(sorted(Counter(f.rule for f in new_flags).items())),
            "by_country": dict(sorted(Counter(f.geo[:2] for f in new_flags).items())),
            "review_cases": review_cases(new_flags, registered_rows, REVIEW_CASES_A2),
            "regions_dropped": sorted(registered_regions - a2_regions),
            "equivalent_codes": inputs.equivalent},
        "map": {
            "regions": len(regions_map),
            "without_maize": sorted(r["geo"] for r in regions_map
                                    if r["irrigated_share"] == ""),
            "registered_regions_without_maize": sorted(
                g for g in registered_regions if g not in shares),
            "eurostat_check": eurostat_check(irrigation_dir, regions_map)},
        "panel": rows,
    }


# ----------------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{100 * (np.exp(x) - 1):+.1f}%"


def _p(x: float) -> str:
    return f"{x:.3f}" if x >= 0.001 else "<0.001"


def _boot(x: Mapping[str, Any]) -> str:
    return (f"{_p(x['p_bootstrap'])} ({_p(x['p_bootstrap_country'])}, "
            f"{_p(x['p_bootstrap_year'])})")


def render(r: Mapping[str, Any], registered: Mapping[str, Any], command: str) -> str:
    s, d, e1, e2, e5 = r["sample"], r["decisions"], r["E1"], r["E2"], r["E5"]
    reg = registered["decisions"]
    passes = "does not pass" if e5["significant_both"] else "passes"
    lines = [
        "# Amendment A2 to the maize study: results", "",
        "Generated by `eu-wheat maize-amend`. Amendment A2 ([METHODS.md](../METHODS.md), "
        "section 10) was registered after the maize results, the maize review "
        "([MAIZE_REVIEW.md](MAIZE_REVIEW.md)) and its corrections, and before any of it was "
        "computed. These results stand beside the registered ones in [MAIZE.md](MAIZE.md) "
        "and do not replace them.", "",
        "## E5-A2, placebo (stated first)", "",
        f"The placebo {passes}: weather in the 61 days after maturity, over where maize "
        "grows, is " + ("" if e5["significant_both"] else "not ") + "significant under both "
        f"methods. Joint test: two-way p = {_p(e5['p_twoway'])}, bootstrap p = "
        f"{_p(e5['p_bootstrap'])} (by country {_p(e5['p_bootstrap_country'])}, by year "
        f"{_p(e5['p_bootstrap_year'])}).", "",
        "## Decisions", "",
        "Holm over E1 and E2, separately under the two-way method and the bootstrap (the "
        "larger of its runs by country and by year); established only if rejected under both "
        "and negative, as in section 9.2.", "",
        "| Estimand | Registered verdict | M1-maize-A2 estimate | p, two-way (Holm) | "
        "p, bootstrap (Holm) | Amended verdict |",
        "| --- | --- | ---: | ---: | ---: | --- |"]
    for key, label, x in (("E1", "E1, 10 more degree days above 30 °C", e1),
                          ("E2", "E2, a dry season (P10 against median)", e2)):
        lines.append(f"| {label} | {reg[key]['verdict']} | {x['estimate']:+.4f} "
                     f"({_pct(x['estimate'])}) | {_p(d[key]['p_twoway_holm'])} | "
                     f"{_p(d[key]['p_bootstrap_holm'])} | {d[key]['verdict']} |")
    lines += ["", "## Which change does what (descriptive)", "",
              "E1 and E2 with two-way 95% intervals and both p-values, unadjusted. Only "
              "M1-maize-A2 decides. The first row is the registered model.", "",
              "| Combination | N | E1 [95% interval] | p, two-way | p, bootstrap (country, "
              "year) | E2 [95% interval] | p, two-way | p, bootstrap (country, year) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    re1, re2 = registered["E1"], registered["E2"]
    lines.append(f"| Registered | {registered['sample']['n']} | {_ci(re1)} | "
                 f"{_p(re1['p_twoway'])} | {_boot(re1)} | {_ci(re2)} | "
                 f"{_p(re2['p_twoway'])} | {_boot(re2)} |")
    for c in r["combinations"]:
        lines.append(f"| {c['label']} | {c['n']} | {_ci(c['E1'])} | "
                     f"{_p(c['E1']['p_twoway'])} | {_boot(c['E1'])} | {_ci(c['E2'])} | "
                     f"{_p(c['E2']['p_twoway'])} | {_boot(c['E2'])} |")
    f = r["few_clusters"]
    lines += ["", "## Inference with few clusters (descriptive)", "",
              "Two-way p as registered; one-way country clustering with G-1 degrees of "
              f"freedom; and a randomization test over {f['amended']['draws']:,} permutations "
              f"of the harvest years (seed {f['amended']['seed']}), the same permutation for "
              "every region. Unadjusted.", "",
              "| Model | Estimand | p, two-way | p, country clustering (df) | "
              "p, randomization |", "| --- | --- | ---: | ---: | ---: |"]
    for model, label, source in (("registered", "Registered", registered),
                                 ("amended", "M1-maize-A2", r)):
        for key in ("E1", "E2"):
            x = f[model][key]
            lines.append(f"| {label} | {key} | {_p(source[key]['p_twoway'])} | "
                         f"{_p(x['country']['p'])} ({x['country']['df']}) | "
                         f"{_p(x['randomization']['p'])} |")
    irr = r["irrigation"]
    lines += ["", "## Irrigated and rainfed maize (descriptive)", "",
              f"A region is irrigated if at least {IRRIGATED_SHARE:.0%} of its maize on the "
              f"map is irrigated: {irr['regions']['irrigated']} regions "
              f"({irr['region_years']['irrigated']} region-years), by country "
              + ", ".join(f"{k} {v}" for k, v in irr["irrigated_by_country"].items())
              + f"; {irr['regions']['rainfed']} rainfed regions "
              f"({irr['region_years']['rainfed']} region-years). One model of M1-maize-A2 with "
              "`EDD` and the dry-season terms interacted with the group. It decides nothing.",
              "", "| Estimand | Group | Estimate [95% interval] | p, two-way | "
              "p, bootstrap (country, year) |", "| --- | --- | ---: | ---: | ---: |"]
    for key in ("E1", "E2"):
        for group in ("rainfed", "irrigated", "difference"):
            x = irr[key][group]
            lines.append(f"| {key} | {group} | {_ci(x)} | {_p(x['p_twoway'])} | {_boot(x)} |")
    loo = r["leave_one_out"]["summary"]
    lines += ["", "## Leave one country out (descriptive)", ""]
    for key in ("E1", "E2"):
        x = loo[key]
        lines.append(f"- **{key}:** from {x['low']:+.4f} to {x['high']:+.4f}; the two-way p is "
                     f"0.05 or more in {x['p_twoway_above_005']} of the "
                     f"{len(r['leave_one_out']['rows'])} deletions. Leaving out "
                     f"{x['most_influential']} moves it most, to {x['without_it']:+.4f}.")
    dr = r["data_rule"]
    lines += ["", "## Sample and the wider data rule", "",
              f"- **M1-maize-A2:** {s['n']} region-years in {s['regions']} regions of "
              f"{s['countries']} countries, harvest years {s['years'][0]} to {s['years'][1]} "
              f"(registered: {s['registered_n']}).",
              f"- **Q6-Q8 (10.2):** {sum(dr['by_rule'].values())} region-years flagged after "
              "Q1-Q5 (by rule: " + ", ".join(f"{k} {v}" for k, v in dr["by_rule"].items())
              + "; by country: " + ", ".join(f"{k} {v}" for k, v in dr["by_country"].items())
              + "). Regions then below 10 years and dropped: "
              + (", ".join(dr["regions_dropped"]) or "none") + ". Every flag is in "
              "[maize/amendment/quality_flags.csv](maize/amendment/quality_flags.csv).",
              "", "**The review's cases (findings D3 and D5).**", "",
              "| Case | Region-years in the registered panel | Excluded | Rules |",
              "| --- | ---: | ---: | --- |"]
    lines += [f"| {c['case']} | {c['in_panel']} | {c['excluded']} | {c['rules'] or '-'} |"
              for c in dr["review_cases"]]
    m = r["map"]
    lines += ["", "## The map of maize (10.1)", "",
              f"- **Regions:** of the {m['regions']} NUTS-2021 regions, "
              f"{len(m['without_maize'])} have no maize on the map; of the registered panel's "
              f"regions, {len(m['registered_regions_without_maize'])}.",
              "- **Irrigated share, the map (2015) against Eurostat's farm statistics (2023):** "
              + "; ".join(f"{c['country']} {c['map_share']:.0%} against "
                          f"{c['eurostat_share']:.0%}" for c in m["eurostat_check"]) + ".",
              "", "## Reproduce", "", "```", command, "```", ""]
    return "\n".join(lines)


def write_results(results: Mapping[str, Any], registered: Mapping[str, Any],
                  output_dir: Path, report: Path, command: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    document = _round({k: v for k, v in results.items() if k != "panel"})
    (output_dir / "results.json").write_text(json.dumps(document, indent=1) + "\n",
                                             encoding="utf-8", newline="\n")
    panel = list(results["panel"])
    write_csv(output_dir / "panel.csv", panel, list(panel[0]))
    tables = {"quality_flags.csv": document["data_rule"]["flags"],
              "leave_one_out.csv": document["leave_one_out"]["rows"]}
    for name, rows in tables.items():
        write_csv(output_dir / name, rows, list(rows[0]))
    report.write_text(render(document, registered, command), encoding="utf-8", newline="\n")
