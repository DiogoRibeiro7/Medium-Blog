"""Section 6: the estimands, from results/solves.csv and the matrices; docs/RESULTS.md.

Bootstrap: one generator (BOOTSTRAP_SEED) draws the resampled days once for the 40-day
design and then once for the 20-day sensitivity runs; every estimand and area reuses them.
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
    SENSITIVITY_DAYS,
    SIMULTANEOUS_LEVEL,
    Pair,
)
from routing_roads.experiment import AREAS, VARIANTS
from routing_roads.report import Derived, area_indices, matrix_summary

FloatArray = NDArray[np.float64]


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
        f"({SIMULTANEOUS_LEVEL:.2%} each, Bonferroni); the others are {LEVEL:.0%}.", "",
        "| Area | E1: regret R [95%] | Seed noise [95%] | E2: R minus noise "
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
                  "regret, against the expectation stated in section 6 (positive in both "
                  "pairs, larger for Lisbon). The registered rule decides only whether water "
                  "raises the regret."]
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
    lines += ["", "## Reproduce", "", "```", command, "```", ""]
    return "\n".join(lines)


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
