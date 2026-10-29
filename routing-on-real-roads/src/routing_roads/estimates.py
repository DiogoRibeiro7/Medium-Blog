"""Section 6: the estimands, from results/solves.csv and the matrices; docs/RESULTS.md.

Bootstrap: one generator (BOOTSTRAP_SEED) draws the resampled days once for the 40-day
design and then once for the 20-day sensitivity runs; every estimand and area reuses them,
and so do the descriptive analyses added after the results and after the critical review.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from routing_roads.contract import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    DAYS,
    LEVEL,
    PAIRS,
    SEEDS,
    SENSITIVITY_DAYS,
    SENSITIVITY_SEEDS,
    SIMULTANEOUS_LEVEL,
    Pair,
)
from routing_roads.experiment import AREAS, VARIANTS
from routing_roads.report import Derived, IntArray, area_indices, matrix_summary
from routing_roads.routing import Routes, cost, decode

FloatArray = NDArray[np.float64]
LEG_BIN_M = 100                     # after the review: legs against pairs of the same length
LENGTH_BANDS_M = ((0, 500), (500, 1_000), (1_000, 2_500), (2_500, None))


def load_solves(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def costs(rows: Sequence[dict[str, str]], variant: str, pair: str, area: str, plan: str,
          kind: str, days: int, seeds: Sequence[int]) -> FloatArray:
    """A days × seeds array of one plan's cost on one matrix ("road" or "straight"), for
    the first ``days`` days and the given seeds."""
    out = np.full((days, len(seeds)), np.nan)
    for r in rows:
        day, seed = int(r["day"]), int(r["seed"])
        if ((r["variant"], r["pair"], r["area"], r["plan"]) == (variant, pair, area, plan)
                and day <= days and seed in seeds):
            out[day - 1, list(seeds).index(seed)] = float(r[f"cost_{kind}"])
    if np.isnan(out).any():
        raise ValueError(f"Missing solves for {variant}, {pair}, {area}, {plan}.")
    return out


def regret_by_day(road_s: FloatArray, road_r: FloatArray) -> FloatArray:
    """E1: r(d, s) = C_R(P_S) / C_R(P_R) − 1, averaged over seeds: r̄(d)."""
    return np.asarray((road_s / road_r - 1).mean(axis=1))


def noise_by_day(road_r: FloatArray) -> FloatArray:
    """E2's placebo: the mean over ordered seed pairs s ≠ s' of |C_R(P_R(s)) / C_R(P_R(s')) − 1|."""
    k = road_r.shape[1]
    q = [np.abs(road_r[:, a] / road_r[:, b] - 1) for a in range(k) for b in range(k) if a != b]
    return np.asarray(np.mean(q, axis=0))


def resamples(days: int, rng: np.random.Generator) -> NDArray[np.int64]:
    return rng.integers(0, days, size=(BOOTSTRAP_DRAWS, days))


def interval(values: FloatArray, idx: NDArray[np.int64], level: float) -> dict[str, float]:
    """The mean over days and its percentile bootstrap interval."""
    boot = values[idx].mean(axis=1)
    tail = (1 - level) / 2
    return {"estimate": float(values.mean()), "low": float(np.quantile(boot, tail)),
            "high": float(np.quantile(boot, 1 - tail)), "level": level}


def alpha(d: Derived, pair: Pair, area: str) -> float:
    """E4's circuity factor: the median road / straight ratio over the area's pharmacies."""
    idx = area_indices(d, pair, area == "both")
    return float(matrix_summary(d.road[pair.key], d.straight[pair.key], idx)["ratio"]["p50"])


def _cells(rows: Sequence[dict[str, str]], variant: str, pair: str, area: str, days: int,
           seeds: Sequence[int]) -> dict[str, FloatArray]:
    return {f"{plan}_{kind}": costs(rows, variant, pair, area, plan, kind, days, seeds)
            for plan in ("straight", "road") for kind in ("road", "straight")}


def _shares(rows: Sequence[dict[str, str]], pair: str, area: str) -> dict[str, float]:
    out = {}
    for plan in ("straight", "road"):
        sel = [r for r in rows if (r["variant"], r["pair"], r["area"], r["plan"])
               == ("main", pair, area, plan)]
        routes = sum(int(r["routes"]) for r in sel)
        out[f"routes_{plan}"] = routes / len(sel)
        out[f"cross_bank_{plan}"] = sum(int(r["cross_bank_routes"]) for r in sel) / routes
    out["clients"] = float(np.mean([
        int(r["clients"]) for r in rows
        if (r["variant"], r["pair"], r["area"]) == ("main", pair, area)]))
    return out


def post_hoc(rows: Sequence[dict[str, str]], d: Derived) -> dict[str, Any]:
    """After the results, descriptive and not registered (METHODS.md, change log): per
    area and plan, the circuity κ(d) = C̄_R(P) / C̄_S(P) of the plan's own legs (seed
    means), averaged over days, so that E4's own-plan error is α / κ(d) − 1 day by day;
    the road distance driven a day; and the mean straight-line leg against the median
    straight-line distance between two of the area's pharmacies."""
    out: dict[str, Any] = {}
    for pair in PAIRS:
        for area in AREAS:
            idx = np.array(area_indices(d, pair, area == "both")[1:])
            s = d.straight[pair.key][np.ix_(idx, idx)]
            area_out: dict[str, Any] = {"pair_straight_m": float(
                np.median(s[~np.eye(idx.size, dtype=bool)]))}
            for plan in ("straight", "road"):
                road = costs(rows, "main", pair.key, area, plan, "road", DAYS, SEEDS)
                straight = costs(rows, "main", pair.key, area, plan, "straight", DAYS, SEEDS)
                legs = [float(r["cost_straight"]) / (int(r["clients"]) + int(r["routes"]))
                        for r in rows if (r["variant"], r["pair"], r["area"], r["plan"])
                        == ("main", pair.key, area, plan)]
                area_out[plan] = {
                    "circuity": float(np.mean(road.mean(axis=1) / straight.mean(axis=1))),
                    "road_km": float(road.mean() / 1000),
                    "leg_straight_m": float(np.mean(legs))}
            out[f"{pair.key}_{area}"] = area_out
    return out


def _depot(d: Derived, pair_key: str, role: str) -> int:
    return next(int(p["index"]) for p in d.points
                if p["pair"] == pair_key and p["role"] == role)


def oriented(routes: Routes, road: IntArray, depot: int) -> int:
    """The road cost of a plan with each route driven in its shorter direction. On the
    symmetric straight-line matrix a route and its reverse cost the same, so the solver
    leaves the direction to the seed (docs/REVIEW.md, M1)."""
    return sum(min(cost([r], road, depot), cost([r[::-1]], road, depot)) for r in routes)


def oriented_costs(rows: Sequence[dict[str, str]], d: Derived, variant: str, pair: str,
                   area: str, days: int, seeds: Sequence[int]) -> FloatArray:
    """A days × seeds array of the straight-line plan's road cost, each route oriented."""
    depot = _depot(d, pair, next(v.depot for v in VARIANTS if v.key == variant))
    out = np.full((days, len(seeds)), np.nan)
    for r in rows:
        day, seed = int(r["day"]), int(r["seed"])
        if ((r["variant"], r["pair"], r["area"], r["plan"]) == (variant, pair, area, "straight")
                and day <= days and seed in seeds):
            out[day - 1, list(seeds).index(seed)] = oriented(decode(r["plan_routes"]),
                                                             d.road[pair], depot)
    if np.isnan(out).any():
        raise ValueError(f"Missing solves for {variant}, {pair}, {area}.")
    return out


def _same_routes(rows: Sequence[dict[str, str]], pair: str, area: str) -> float:
    """The share of ordered pairs of seeds, over days, whose straight-line plans have the
    same routes up to direction."""
    plans: dict[int, list[frozenset[tuple[int, ...]]]] = {}
    for r in rows:
        if (r["variant"], r["pair"], r["area"], r["plan"]) == ("main", pair, area, "straight"):
            plans.setdefault(int(r["day"]), []).append(frozenset(
                min(tuple(x), tuple(x[::-1])) for x in decode(r["plan_routes"])))
    same = [a == b for p in plans.values() for i, a in enumerate(p) for j, b in enumerate(p)
            if i != j]
    return float(np.mean(same))


def _legs(rows: Sequence[dict[str, str]], d: Derived, pair: Pair, area: str,
          plan: str) -> dict[str, float]:
    """The circuity of the legs a plan drives (depot legs included, seeds and days pooled),
    and what pairs of the area's points with the same straight-line length would give: the
    ratio of sums within each LEG_BIN_M bin, weighted by the legs' straight-line lengths."""
    idx = np.array(area_indices(d, pair, area == "both"))
    road, straight = d.road[pair.key], d.straight[pair.key]
    r, s = road[np.ix_(idx, idx)], straight[np.ix_(idx, idx)]
    off = ~np.eye(idx.size, dtype=bool) & (s > 0)
    b = s[off] // LEG_BIN_M
    bin_ratio = np.bincount(b, weights=r[off]) / np.maximum(np.bincount(b, weights=s[off]), 1)
    driven = flat = like = 0.0
    for row in rows:
        if (row["variant"], row["pair"], row["area"], row["plan"]) == ("main", pair.key, area,
                                                                       plan):
            for route in decode(row["plan_routes"]):
                stops = [int(idx[0]), *route, int(idx[0])]
                for i, j in zip(stops, stops[1:], strict=False):
                    if straight[i, j] > 0:
                        driven += float(road[i, j])
                        flat += float(straight[i, j])
                        like += float(straight[i, j] * bin_ratio[straight[i, j] // LEG_BIN_M])
    return {"circuity": driven / flat, "like": like / flat}


def _ratios(d: Derived, pair: Pair, area: str) -> dict[str, Any]:
    """Over ordered pairs of the area's pharmacies with a positive straight-line distance:
    the mean ratio, the ratio of sums, and the median ratio by straight-line length."""
    pharm = np.array(area_indices(d, pair, area == "both")[1:])
    r = d.road[pair.key][np.ix_(pharm, pharm)]
    s = d.straight[pair.key][np.ix_(pharm, pharm)]
    keep = ~np.eye(pharm.size, dtype=bool) & (s > 0)
    r, s = r[keep], s[keep]
    q = r / s
    bands = {f"{lo}-{hi or ''}": float(np.median(q[(s >= lo) & (s < (hi or np.inf))]))
             for lo, hi in LENGTH_BANDS_M}
    return {"mean": float(q.mean()), "sums": float(r.sum()) / float(s.sum()),
            "median_by_length": bands}


def _asymmetry(d: Derived, pair: Pair) -> dict[str, float]:
    """The median asymmetry of the road distance (as in docs/DATA.md) over pairs within the
    city, within the other municipality, and across the river."""
    road = d.road[pair.key]
    city = area_indices(d, pair, False)[1:]
    other = [i for i in area_indices(d, pair, True)[1:] if i not in set(city)]

    def median(rows_: list[int], cols: list[int], within: bool) -> float:
        a = road[np.ix_(rows_, cols)]
        b = road[np.ix_(cols, rows_)].T
        keep = np.triu(np.ones_like(a, dtype=bool), 1) if within else np.ones_like(a, bool)
        a, b = a[keep], b[keep]
        mean = (a + b) / 2
        return float(np.median(np.abs(a - b)[mean > 0] / mean[mean > 0]))

    return {"city": median(city, city, True), "other": median(other, other, True),
            "across": median(city, other, False)}


def after_review(rows: Sequence[dict[str, str]], d: Derived, idx_main: NDArray[np.int64],
                 idx_sens: NDArray[np.int64]) -> dict[str, Any]:
    """After the critical review (docs/REVIEW.md), descriptive and not registered
    (METHODS.md, change log), with the registered draws: route direction, the straight-line
    plan's own seed noise, E3 without direction and in kilometres, the legs against pairs of
    the same length, and the capacity pattern paired by day."""
    out: dict[str, Any] = {"areas": {}, "water": {}, "direction_by_variant": [],
                           "capacity": {}}
    by_day: dict[tuple[str, str], dict[str, FloatArray]] = {}
    for pair in PAIRS:
        for area in AREAS:
            c = _cells(rows, "main", pair.key, area, DAYS, SEEDS)
            turned = oriented_costs(rows, d, "main", pair.key, area, DAYS, SEEDS)
            r = regret_by_day(c["straight_road"], c["road_road"])
            r_oriented = regret_by_day(turned, c["road_road"])
            km = (c["straight_road"] - c["road_road"]).mean(axis=1) / 1000
            by_day[(pair.key, area)] = {"regret": r, "oriented": r_oriented, "km": km}
            out["areas"][f"{pair.key}_{area}"] = {
                "oriented": interval(r_oriented, idx_main, LEVEL),
                "direction_share": float(1 - r_oriented.mean() / r.mean()),
                "same_routes": _same_routes(rows, pair.key, area),
                "noise_straight": interval(noise_by_day(c["straight_road"]), idx_main, LEVEL),
                "noise_straight_on_straight": float(noise_by_day(c["straight_straight"]).mean()),
                "legs": {plan: _legs(rows, d, pair, area, plan)
                         for plan in ("straight", "road")},
                "ratio": _ratios(d, pair, area)}
        city, both = by_day[(pair.key, "city")], by_day[(pair.key, "both")]
        delta = both["regret"] - city["regret"]
        delta_oriented = both["oriented"] - city["oriented"]
        out["water"][pair.key] = {
            "registered_at_95": interval(delta, idx_main, LEVEL),
            "oriented": interval(delta_oriented, idx_main, SIMULTANEOUS_LEVEL),
            "direction": interval(delta - delta_oriented, idx_main, SIMULTANEOUS_LEVEL),
            "km": interval(both["km"] - city["km"], idx_main, SIMULTANEOUS_LEVEL),
            "asymmetry": _asymmetry(d, pair)}
    for v in VARIANTS[1:]:
        for pair in PAIRS:
            for area in AREAS:
                road = costs(rows, v.key, pair.key, area, "road", "road", v.days, v.seeds)
                r = regret_by_day(costs(rows, v.key, pair.key, area, "straight", "road",
                                        v.days, v.seeds), road)
                r_oriented = regret_by_day(oriented_costs(rows, d, v.key, pair.key, area,
                                                          v.days, v.seeds), road)
                out["direction_by_variant"].append({
                    "variant": v.key, "pair": pair.key, "area": area,
                    "share": float(1 - r_oriented.mean() / r.mean())})
    main = VARIANTS[0]
    by_capacity = {v.capacity: v.key for v in VARIANTS
                   if (v.iterations, v.depot) == (main.iterations, main.depot)}
    capacities = sorted(by_capacity)
    for pair in PAIRS:
        for area in AREAS:
            by_q = {q: regret_by_day(
                costs(rows, by_capacity[q], pair.key, area, "straight", "road",
                      SENSITIVITY_DAYS, SENSITIVITY_SEEDS),
                costs(rows, by_capacity[q], pair.key, area, "road", "road", SENSITIVITY_DAYS,
                      SENSITIVITY_SEEDS)) for q in capacities}
            out["capacity"][f"{pair.key}_{area}"] = {
                f"{a}_to_{b}": interval(by_q[b] - by_q[a], idx_sens, LEVEL)
                for a, b in zip(capacities, capacities[1:], strict=False)}
    return out


def estimate(rows: Sequence[dict[str, str]], d: Derived) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx_main, idx_sens = resamples(DAYS, rng), resamples(SENSITIVITY_DAYS, rng)
    main = next(v for v in VARIANTS if v.key == "main")
    out: dict[str, Any] = {"areas": {}, "water": {}, "sensitivity": []}
    by_day: dict[tuple[str, str], FloatArray] = {}
    for pair in PAIRS:
        for area in AREAS:
            c = _cells(rows, "main", pair.key, area, DAYS, main.seeds)
            r = regret_by_day(c["straight_road"], c["road_road"])
            noise = noise_by_day(c["road_road"])
            a = alpha(d, pair, area)
            straight_mean = c["straight_straight"].mean(axis=1)
            e = a * straight_mean / c["straight_road"].mean(axis=1) - 1
            e_best = a * straight_mean / c["road_road"].mean(axis=1) - 1
            by_day[(pair.key, area)] = r
            out["areas"][f"{pair.key}_{area}"] = {
                "pair": pair.key, "area": area,
                "E1": interval(r, idx_main, LEVEL),
                "noise": interval(noise, idx_main, LEVEL),
                "E2": interval(r - noise, idx_main, SIMULTANEOUS_LEVEL),
                "alpha": a,
                "E4": {"own": {**interval(e, idx_main, LEVEL),
                               "p5": float(np.percentile(e, 5)),
                               "p95": float(np.percentile(e, 95))},
                       "best": {**interval(e_best, idx_main, LEVEL),
                                "p5": float(np.percentile(e_best, 5)),
                                "p95": float(np.percentile(e_best, 95))}},
                **_shares(rows, pair.key, area)}
            out["areas"][f"{pair.key}_{area}"]["E2"]["exceeds_noise"] = bool(
                out["areas"][f"{pair.key}_{area}"]["E2"]["low"] > 0)
        delta = by_day[(pair.key, "both")] - by_day[(pair.key, "city")]
        out["water"][pair.key] = {**interval(delta, idx_main, SIMULTANEOUS_LEVEL)}
        out["water"][pair.key]["raises_regret"] = bool(out["water"][pair.key]["low"] > 0)
    for v in VARIANTS:                      # section 7, and the main design on the same days
        source = "main" if v.key == "main" else v.key
        for pair in PAIRS:
            seeds = v.seeds if v.key != "main" else (1, 2)
            regrets = {area: regret_by_day(
                costs(rows, source, pair.key, area, "straight", "road", SENSITIVITY_DAYS, seeds),
                costs(rows, source, pair.key, area, "road", "road", SENSITIVITY_DAYS, seeds))
                for area in AREAS}
            out["sensitivity"].append({
                "variant": v.key, "pair": pair.key,
                **{f"E1_{area}": interval(regrets[area], idx_sens, LEVEL) for area in AREAS},
                "E3": interval(regrets["both"] - regrets["city"], idx_sens, LEVEL)})
    out["post_hoc"] = post_hoc(rows, d)
    out["after_review"] = after_review(rows, d, idx_main, idx_sens)
    return out


def _pct(x: float) -> str:
    return f"{100 * x:+.1f}%"


def _ci(x: dict[str, float]) -> str:
    return f"{_pct(x['estimate'])} [{_pct(x['low'])}, {_pct(x['high'])}]"


AREA_NAMES = {(p.key, "city"): p.city for p in PAIRS} | {
    (p.key, "both"): f"{p.city} with {p.other}" for p in PAIRS}
VARIANT_NAMES = {"main": "Main design (same 20 days, seeds 1-2)",
                 "depot_all": "Depot at the mean of all the pair's pharmacies",
                 "capacity_6": "Capacity 6", "capacity_24": "Capacity 24",
                 "iterations_20000": "20,000 iterations"}


def render(e: dict[str, Any], command: str) -> str:
    areas = list(e["areas"].values())
    lines = [
        "# Results: straight-line against road distances", "",
        "Generated by `routing-roads estimate` from `results/solves.csv` and the committed "
        "matrices. Every estimand, interval and decision rule is section 6 of "
        "[METHODS.md](../METHODS.md), fixed before any route was solved. Costs are total "
        "road distance; a positive regret means routes planned on straight-line distances "
        "drive further on the road than routes planned on road distances.", "",
        "## Decisions", "",
        f"Intervals that decide are simultaneous across the four areas "
        f"({SIMULTANEOUS_LEVEL:.2%} each, Bonferroni); the others are {LEVEL:.0%}. E3 makes "
        "two comparisons at the same level, which is conservative for it. The seed noise is "
        "the road plan's, as registered; the straight-line plan's own seed noise is larger "
        "(see after the critical review, below).", "",
        "| Area | E1: regret R [95%] | Seed noise, road plan [95%] | E2: R minus noise "
        f"[{SIMULTANEOUS_LEVEL:.2%}] | Exceeds solver noise |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    lines += [f"| {AREA_NAMES[(a['pair'], a['area'])]} | {_ci(a['E1'])} | {_ci(a['noise'])} | "
              f"{_ci(a['E2'])} | {'yes' if a['E2']['exceeds_noise'] else 'no'} |"
              for a in areas]
    lines += ["", f"| Pair | E3: water, Δ regret [{SIMULTANEOUS_LEVEL:.2%}] | "
              "Water raises the regret |", "| --- | ---: | --- |"]
    lines += [f"| {p.city} → {p.city} with {p.other} | {_ci(e['water'][p.key])} | "
              f"{'yes' if e['water'][p.key]['raises_regret'] else 'no'} |" for p in PAIRS]
    below = [p for p in PAIRS if e["water"][p.key]["high"] < 0]
    if below:
        lines += ["", "For " + " and ".join(f"{p.city} with {p.other}" for p in below)
                  + ", the whole interval lies below zero: adding the other bank lowers the "
                  "regret, a ratio, against the expectation stated in section 6 (positive in "
                  "both pairs, larger for Lisbon). The registered rule decides only whether "
                  "water raises the regret. After the critical review: in kilometres a day "
                  "the regret rises, and the fall comes from the direction in which routes "
                  "are driven (below)."]
    lines += ["", "## E4, the circuity estimate (descriptive)", "",
              "α is the median ratio of road to straight-line distance over the area's "
              "pharmacy pairs. The estimate α × (the straight-line plan's straight-line "
              "cost), against that plan's own road cost and against the best road cost.", "",
              "| Area | α | Error, own plan: mean [95%] | 5th / 95th percentile over days "
              "| Error, best road cost: mean [95%] | 5th / 95th |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for a in areas:
        own, best = a["E4"]["own"], a["E4"]["best"]
        lines.append(f"| {AREA_NAMES[(a['pair'], a['area'])]} | {a['alpha']:.3f} | "
                     f"{_ci(own)} | {_pct(own['p5'])} / {_pct(own['p95'])} | {_ci(best)} | "
                     f"{_pct(best['p5'])} / {_pct(best['p95'])} |")
    lines += ["", "## Plans (descriptive)", "",
              "| Area | Orders a day | Routes, straight-line plan | Routes, road plan | "
              "Routes serving both banks, straight-line plan | road plan |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for a in areas:
        both = a["area"] == "both"
        lines.append(
            f"| {AREA_NAMES[(a['pair'], a['area'])]} | {a['clients']:.1f} | "
            f"{a['routes_straight']:.2f} | {a['routes_road']:.2f} | "
            + (f"{a['cross_bank_straight']:.1%} | {a['cross_bank_road']:.1%} |" if both
               else "- | - |"))
    lines += ["", "## Sensitivity analyses (fixed in advance)", "",
              f"The first {SENSITIVITY_DAYS} days, seeds 1 and 2, {LEVEL:.0%} intervals. No "
              "variant changes a decision; the first row is the main design on the same days "
              "and seeds.", "",
              "| Variant | Pair | E1, city alone | E1, with the other bank | E3, Δ |",
              "| --- | --- | ---: | ---: | ---: |"]
    pairs = {p.key: p for p in PAIRS}
    lines += [f"| {VARIANT_NAMES[s['variant']]} | {pairs[s['pair']].city} | "
              f"{_ci(s['E1_city'])} | {_ci(s['E1_both'])} | {_ci(s['E3'])} |"
              for s in e["sensitivity"]]
    lines += ["", "## After the results (descriptive, not registered)", "",
              "Added with the article, after every registered result above had been seen; "
              "they decide nothing. A plan's circuity κ(d) is its road cost divided by its "
              "straight-line cost on day d (seed means), averaged over days; E4's error on the "
              "plan's own cost is exactly α / κ(d) − 1, day by day. A plan with n orders on k "
              "routes drives n + k legs, depot legs included; its mean leg is its "
              "straight-line cost divided by n + k.",
              "",
              "| Area | α | κ, straight-line plan | κ, road plan | Road km a day, "
              "straight-line plan | road plan | Mean leg, road plan (straight-line km) | "
              "Median distance between two pharmacies (straight-line km) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for a in areas:
        p = e["post_hoc"][f"{a['pair']}_{a['area']}"]
        lines.append(
            f"| {AREA_NAMES[(a['pair'], a['area'])]} | {a['alpha']:.3f} | "
            f"{p['straight']['circuity']:.3f} | {p['road']['circuity']:.3f} | "
            f"{p['straight']['road_km']:.1f} | {p['road']['road_km']:.1f} | "
            f"{p['road']['leg_straight_m'] / 1000:.2f} | {p['pair_straight_m'] / 1000:.2f} |")
    lines += _render_after_review(e, areas)
    lines += ["", "## Reproduce", "", "```", command, "```", ""]
    return "\n".join(lines)


def _pp(x: dict[str, float], unit: str = "") -> str:
    """A change in percentage points (or in ``unit``), with its interval."""
    scale = 1 if unit else 100
    return (f"{scale * x['estimate']:+.2f}{unit} [{scale * x['low']:+.2f}, "
            f"{scale * x['high']:+.2f}]")


def _render_after_review(e: dict[str, Any], areas: list[dict[str, Any]]) -> list[str]:
    r = e["after_review"]
    names = [AREA_NAMES[(a["pair"], a["area"])] for a in areas]
    keys = [f"{a['pair']}_{a['area']}" for a in areas]
    lines = ["", "## After the critical review (descriptive, not registered)", "",
             "Added in the corrections that [REVIEW.md](REVIEW.md) recommended, after every "
             "result above and the review had been seen; they decide nothing. Intervals use "
             "the registered bootstrap draws.", "",
             "**Route direction.** On the symmetric straight-line matrix a route and its "
             "reverse cost the same, so the solver leaves each route's direction to the seed. "
             "Here each straight-line route is driven in its shorter direction on the road, "
             "with no new solve. *Same routes* is the share of ordered pairs of seeds whose "
             "straight-line plans have the same routes up to direction. The seed noise is "
             "E2's placebo applied to the straight-line plan.", "",
             "| Area | E1: regret R | R, routes in their better direction [95%] | Share of R "
             "due to direction | Same routes | Straight-line plan's seed noise on the road "
             "[95%] | on straight lines |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, a, k in zip(names, areas, keys, strict=True):
        x = r["areas"][k]
        lines.append(f"| {name} | {_pct(a['E1']['estimate'])} | {_ci(x['oriented'])} | "
                     f"{x['direction_share']:.0%} | {x['same_routes']:.0%} | "
                     f"{_ci(x['noise_straight'])} | "
                     f"{100 * x['noise_straight_on_straight']:.2f}% |")
    variants = list(dict.fromkeys(v["variant"] for v in r["direction_by_variant"]))
    share = {(v["variant"], v["pair"], v["area"]): v["share"] for v in r["direction_by_variant"]}
    lines += ["", "The share of the regret due to direction in each variant fixed in advance "
              "(its own days and seeds):", "",
              "| Variant | " + " | ".join(names) + " |",
              "| --- |" + " ---: |" * len(names)]
    lines += [f"| {VARIANT_NAMES[v]} | " + " | ".join(
        f"{share[(v, a['pair'], a['area'])]:.0%}" for a in areas) + " |" for v in variants]
    lines += ["", f"**The water.** Δ as registered, at {LEVEL:.0%}; Δ with every straight-line "
              "route in its better direction, and the part of Δ due to direction; and Δ in "
              f"road kilometres a day of regret. The last three at {SIMULTANEOUS_LEVEL:.2%}. "
              "The asymmetry is docs/DATA.md's, over pairs of pharmacies within each "
              "municipality and across the river.", "",
              f"| Pair | Δ [{LEVEL:.0%}] | Δ, routes in their better direction | Δ due to "
              "direction | Δ, km a day | Asymmetry within the city | within the other "
              "municipality | across |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for p in PAIRS:
        w = r["water"][p.key]
        a = w["asymmetry"]
        lines.append(f"| {p.city} → {p.city} with {p.other} | {_pp(w['registered_at_95'])} | "
                     f"{_pp(w['oriented'])} | {_pp(w['direction'])} | {_pp(w['km'], ' km')} | "
                     f"{a['city']:.1%} | {a['other']:.1%} | {a['across']:.1%} |")
    lines += ["", "**The legs.** The circuity of the legs each plan drives (road over "
              "straight-line distance, depot legs included, pooled over days and seeds), "
              "against pairs of the area's points with the same straight-line length "
              f"({LEG_BIN_M} m bins). Over the area's pairs of pharmacies: the mean ratio and "
              "the ratio of sums, against α, the median; and the median ratio by "
              "straight-line distance.", "",
              "| Area | α | Mean ratio | Ratio of sums | Median, under 0.5 km | 0.5-1 km | "
              "1-2.5 km | over 2.5 km | Straight-line plan's legs | pairs of the same length | "
              "Road plan's legs | pairs of the same length |",
              "| --- |" + " ---: |" * 11]
    for name, a, k in zip(names, areas, keys, strict=True):
        x = r["areas"][k]
        q = x["ratio"]
        lines.append(f"| {name} | {a['alpha']:.3f} | {q['mean']:.3f} | {q['sums']:.3f} | "
                     + " | ".join(f"{m:.2f}" for m in q["median_by_length"].values()) + " | "
                     + " | ".join(f"{x['legs'][plan]['circuity']:.3f} | "
                                  f"{x['legs'][plan]['like']:.3f}"
                                  for plan in ("straight", "road")) + " |")
    steps = list(next(iter(r["capacity"].values())))
    lines += ["", f"**Capacity, paired by day.** The change in the regret between vans of "
              f"different capacity, on the first {SENSITIVITY_DAYS} days with seeds 1 and 2, in "
              f"percentage points [{LEVEL:.0%}].", "",
              "| Area | " + " | ".join(s.replace("_to_", " → ") for s in steps) + " |",
              "| --- |" + " ---: |" * len(steps)]
    lines += [f"| {name} | " + " | ".join(_pp(r["capacity"][k][s]) for s in steps) + " |"
              for name, k in zip(names, keys, strict=True)]
    return lines


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return float(f"{value:.10g}")
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round(v) for v in value]
    return value


def write(e: dict[str, Any], results: Path, report: Path, command: str) -> None:
    document = _round(e)
    (results / "estimates.json").write_text(json.dumps(document, indent=1) + "\n",
                                            encoding="utf-8", newline="\n")
    report.write_text(render(document, command), encoding="utf-8", newline="\n")
