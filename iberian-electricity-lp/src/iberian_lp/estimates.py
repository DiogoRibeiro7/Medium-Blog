"""Sections 5-7: the estimands, from results/programmes.csv and the derived tables;
results/estimates.json and docs/RESULTS.md.

Prices are compared in whole cents, as OMIE publishes them, so every comparison with π is
exact. One generator (BOOTSTRAP_SEED) draws the resampled days once; every estimand
reuses them. A mean over periods is, in each resample, the sum over the drawn days divided
by their number of periods.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from iberian_lp.contract import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    E1_THRESHOLD,
    E1_TOLERANCE,
    E1_TOLERANCE_SENSITIVITY,
    E3_MW,
    E3_SENSITIVITY_MW,
    LEVEL,
    MIDDAY_HOURS,
    MOVED,
    NON_CONVEX,
    SIMULTANEOUS_LEVEL,
    SOURCE,
    VOLUME_TOLERANCE_MW,
)
from iberian_lp.experiment import SPANISH
from iberian_lp.report import TYPE_NAMES, Derived

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
ZONES = {"MI": "joint", "ES": "Spain", "PT": "Portugal"}


def load_programmes(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class Table:
    """The programmes table as arrays, prices in cents (±inf for an unbounded end)."""

    def __init__(self, rows: list[dict[str, str]], days: list[str]) -> None:
        self.rows = rows
        self.days = days
        self.day = np.array([days.index(r["date"]) for r in rows])
        self.zone = np.array([r["zone"] for r in rows])
        self.hour = np.array([int(r["hour"]) for r in rows])
        self.price = self.cents("price")

    def cents(self, name: str) -> FloatArray:
        return np.array([np.nan if r[name] == "" else float(r[name]) * 100 for r in self.rows]
                        ).round()

    def mw(self, name: str) -> FloatArray:
        return np.array([float(r[name]) for r in self.rows])

    def interval(self, programme: str) -> tuple[FloatArray, FloatArray]:
        return self.cents(f"{programme}_lo"), self.cents(f"{programme}_hi")

    def nearest(self, programme: str) -> FloatArray:
        """Section 5's point rule, minus π: the point of Λ nearest π, minus π (EUR/MWh)."""
        lo, hi = self.interval(programme)
        return (np.clip(self.price, lo, hi) - self.price) / 100

    def midpoint(self, programme: str) -> FloatArray:
        """Section 7: the midpoint of Λ, minus π; NaN when Λ is unbounded."""
        lo, hi = self.interval(programme)
        with np.errstate(invalid="ignore"):
            mid = np.where(np.isfinite(lo) & np.isfinite(hi), (lo + hi) / 2, np.nan)
        return (mid - self.price) / 100


class Bootstrap:
    def __init__(self, n_days: int) -> None:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        self.idx = rng.integers(0, n_days, size=(BOOTSTRAP_DRAWS, n_days))
        self.n_days = n_days

    def mean(self, values: FloatArray, day: NDArray[np.int64], mask: BoolArray
             ) -> tuple[float, FloatArray]:
        """The mean of ``values`` over the masked periods, and its resampled values."""
        keep = mask & ~np.isnan(values)
        sums = np.bincount(day[keep], weights=values[keep], minlength=self.n_days)
        counts = np.bincount(day[keep], minlength=self.n_days).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = sums[self.idx].sum(axis=1) / counts[self.idx].sum(axis=1)
        return float(sums.sum() / counts.sum()), boot


def interval(estimate: float, boot: FloatArray, level: float) -> dict[str, float]:
    """A percentile interval; resamples without a period in the subset are dropped."""
    tail = (1 - level) / 2
    boot = boot[~np.isnan(boot)]
    return {"estimate": estimate, "low": float(np.quantile(boot, tail)),
            "high": float(np.quantile(boot, 1 - tail)), "level": level}


def _mean(b: Bootstrap, t: Table, values: FloatArray, mask: BoolArray,
          level: float) -> dict[str, float]:
    return interval(*b.mean(values, t.day, mask), level)


def _gap(b: Bootstrap, t: Table, m: FloatArray, mask: BoolArray,
         level: float = SIMULTANEOUS_LEVEL) -> dict[str, Any]:
    """E3: G = mean |m| outside midday − mean |m| at midday, with both means."""
    midday = np.isin(t.hour, list(MIDDAY_HOURS))
    out_est, out_boot = b.mean(np.abs(m), t.day, mask & ~midday)
    mid_est, mid_boot = b.mean(np.abs(m), t.day, mask & midday)
    return {"mean": _mean(b, t, m, mask, level),
            "outside_abs": interval(out_est, out_boot, level),
            "midday_abs": interval(mid_est, mid_boot, level),
            "G": interval(out_est - mid_est, out_boot - mid_boot, level),
            "unbounded": int(np.isnan(m[mask]).sum())}


def e1(b: Bootstrap, t: Table) -> dict[str, Any]:
    lo, hi = t.interval("R")
    every = np.ones(len(t.rows), dtype=bool)

    def inside(tolerance: float) -> FloatArray:
        widen = round(tolerance * 100)
        return ((lo - widen <= t.price) & (t.price <= hi + widen)).astype(float)

    hit = inside(E1_TOLERANCE)
    distance = np.maximum(np.maximum(lo - t.price, t.price - hi), 0) / 100
    misses = hit == 0
    matched = t.mw("matched_buy")
    volume = ((t.mw("R_buy_min") - VOLUME_TOLERANCE_MW - 1e-9 <= matched)
              & (matched <= t.mw("R_buy_max") + VOLUME_TOLERANCE_MW + 1e-9)).astype(float)
    share = _mean(b, t, hit, every, LEVEL)
    return {
        "share": share, "reproduces": share["estimate"] >= E1_THRESHOLD,
        "by_zone": {z: {"inside": int(hit[t.zone == z].sum()),
                        "zone_periods": int((t.zone == z).sum())} for z in ZONES},
        "tolerance_sensitivity": _mean(b, t, inside(E1_TOLERANCE_SENSITIVITY), every, LEVEL),
        "volume": {**_mean(b, t, volume, every, LEVEL), "mismatched": int((volume == 0).sum())},
        "misses": {
            "count": int(misses.sum()),
            "by_zone": {z: int((misses & (t.zone == z)).sum()) for z in ZONES},
            "days": int(len(set(t.day[misses]))),
            "distance_median": float(np.median(distance[misses])) if misses.any() else 0.0,
            "distance_max": float(np.max(distance)),
            "by_hour": {int(h): int(n) for h, n in sorted(Counter(t.hour[misses]).items())}}}


def e2(b: Bootstrap, t: Table, spanish: BoolArray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, programme in (("all", "X"), ("blocks", "X_blocks"), ("sco", "X_sco")):
        delta = t.nearest(programme)
        out[name] = {
            "mean": _mean(b, t, delta, spanish, SIMULTANEOUS_LEVEL),
            "moved": _mean(b, t, (np.abs(delta) > MOVED + 1e-9).astype(float), spanish,
                           SIMULTANEOUS_LEVEL),
            "mean_abs": _mean(b, t, np.abs(delta), spanish, SIMULTANEOUS_LEVEL),
            "up": int(((delta > MOVED + 1e-9) & spanish).sum()),
            "down": int(((delta < -MOVED - 1e-9) & spanish).sum())}
    mean = out["all"]["mean"]
    out["excludes_zero"] = mean["low"] > 0 or mean["high"] < 0
    return out


def e3(b: Bootstrap, t: Table, spanish: BoolArray) -> dict[str, Any]:
    m = t.nearest(f"M{E3_MW}")
    out = _gap(b, t, m, spanish)
    out["by_hour"] = {int(h): _mean(b, t, m, spanish & (t.hour == h), SIMULTANEOUS_LEVEL)
                      for h in sorted(set(t.hour[spanish]))}
    out["midday_smaller"] = out["G"]["low"] > 0
    out["sensitivity"] = {
        **{f"{q} MW": _gap(b, t, t.nearest(f"M{q}"), spanish) for q in E3_SENSITIVITY_MW},
        "midpoint": _gap(b, t, t.midpoint(f"M{E3_MW}"), spanish)}
    return out


def e4(b: Bootstrap, t: Table, spanish: BoolArray) -> dict[str, Any]:
    f = t.nearest("F0")
    out = {}
    for name in ("imports", "exports"):
        mask = spanish & (t.mw(f"{name}_fr") > 0)
        out[name] = {**_mean(b, t, f, mask, LEVEL), "periods": int(mask.sum())}
    return out


def descriptive(t: Table, d: Derived, spanish: BoolArray) -> dict[str, Any]:
    lo, hi = t.interval("R")
    wide = hi > lo
    finite = wide & np.isfinite(lo) & np.isfinite(hi)
    rows = [r for r in d.periods if r["zone"] in SPANISH]
    return {
        "periods": int(spanish.sum()), "zone_periods": len(t.rows),
        "split": int((t.zone == "ES").sum()),
        "steps_median": float(np.median([float(r["steps_offered"]) for r in rows])),
        "steps_summed_median": float(np.median([float(r["steps_aggregated"]) for r in rows])),
        "accepted_share": {k: sum(float(r[f"matched_sell_{k}"]) for r in d.periods)
                           / sum(float(r[f"offered_sell_{k}"]) for r in d.periods)
                           for k in NON_CONVEX},
        "wide": {"count": int(wide.sum()), "share": float(wide.mean()),
                 "unbounded": int((wide & ~finite).sum()),
                 "median_width": float(np.median((hi - lo)[finite]) / 100)
                 if finite.any() else 0.0}}


BANDS = (0, 10, 20, 40, 60, 80, 120, np.inf)     # EUR/MWh, for the zero floor (review)


def line_full(t: Table, d: Derived) -> FloatArray:
    """Per row: 1 if the France line had no free capacity in the direction of the flow,
    0 if it had room, NaN outside the Spanish or joint zone or with no flow (review)."""
    rows = {(r["date"], r["period"], r["zone"]): r for r in d.periods}
    out = np.full(len(t.rows), np.nan)
    for i, x in enumerate(t.rows):
        r = rows[(x["date"], x["period"], x["zone"])]
        if x["zone"] in SPANISH:
            side = ("import" if float(r["imports_fr"]) > 0
                    else "export" if float(r["exports_fr"]) > 0 else None)
            if side is not None and r[f"fr_{side}_free"] != "":
                out[i] = float(float(r[f"fr_{side}_free"]) == 0)
    return out


def post_hoc(b: Bootstrap, t: Table, d: Derived, spanish: BoolArray) -> dict[str, Any]:
    """Added after the results were seen, and after the critical review (METHODS.md,
    change log; docs/REVIEW.md); descriptive, 95%."""
    m = t.nearest(f"M{E3_MW}")
    midday = np.isin(t.hour, list(MIDDAY_HOURS))
    nonpositive = spanish & (t.price <= 0)
    positive = spanish & (t.price > 0)
    lo, hi = t.interval(f"M{E3_MW}")
    new_price = np.round(t.price + 100 * m)               # cents, exact
    band = np.digitize(t.price / 100, BANDS[1:-1], right=True)
    full = line_full(t, d)
    delta = t.nearest("X")
    return {
        # after the results
        "nonpositive": {
            name: {"periods": int((spanish & mask).sum()),
                   "count": int((nonpositive & mask).sum())}
            for name, mask in (("midday", midday), ("outside", ~midday))},
        "nonpositive_max_abs_m": float(np.max(np.abs(m[nonpositive]), initial=0.0)),
        "positive": _gap(b, t, m, positive, LEVEL),
        "midpoint_wide": int(((hi > lo) & spanish).sum()),
        # after the review
        "floor": {
            "above_price": int((positive & (np.abs(m) * 100 > t.price + 1e-6)).sum()),
            "below_zero": int((positive & (new_price < 0)).sum()),
            **{name: {"positive": int((positive & mask).sum()),
                      "binding": int((positive & mask & (new_price == 0)).sum()),
                      "median_price": float(np.median(t.price[positive & mask]) / 100)}
               for name, mask in (("midday", midday), ("outside", ~midday))},
            "bands": [{"low": BANDS[k], "high": BANDS[k + 1],
                       **{name: {"periods": int((positive & mask & (band == k)).sum()),
                                 "mean_abs": float(np.mean(np.abs(m[positive & mask
                                                                    & (band == k)])))}
                          for name, mask in (("midday", midday), ("outside", ~midday))}}
                      for k in range(len(BANDS) - 1)]},
        "line": {
            "with_room": int((full == 0).sum()), "periods": int((~np.isnan(full)).sum()),
            **{name: {"delta": _mean(b, t, delta, spanish & (full == state), LEVEL),
                      "m": _mean(b, t, m, spanish & (full == state), LEVEL)}
               for name, state in (("full", 1), ("room", 0))}},
        "blocks": {name: {**_mean(b, t, t.nearest(p), spanish, LEVEL),
                          "down": int(((t.nearest(p) < -MOVED - 1e-9) & spanish).sum())}
                   for name, p in (("c01_only", "X_c01"), ("dedup", "X_blocks_dedup"),
                                   ("all_dedup", "X_dedup"))},
        "zones": {name: _mean(b, t, delta, t.zone == z, LEVEL)
                  for name, z in (("joint", "MI"), ("split", "ES"))}}


def estimate(rows: list[dict[str, str]], d: Derived, solver: dict[str, Any]) -> dict[str, Any]:
    t = Table(rows, d.days)
    b = Bootstrap(len(d.days))
    spanish = np.isin(t.zone, SPANISH)
    return {"source": SOURCE, "days": len(d.days), "bootstrap": {
                "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "level": LEVEL,
                "simultaneous_level": SIMULTANEOUS_LEVEL},
            "solver": solver, "E1": e1(b, t), "E2": e2(b, t, spanish),
            "E3": e3(b, t, spanish), "E4": e4(b, t, spanish),
            "descriptive": descriptive(t, d, spanish),
            "post_hoc": post_hoc(b, t, d, spanish)}


def write(results: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")


def _ci(x: dict[str, float], fmt: str = "{:+.2f}") -> str:
    return f"{fmt.format(x['estimate'])} [{fmt.format(x['low'])}, {fmt.format(x['high'])}]"


def _pct(x: dict[str, float]) -> str:
    return _ci(x, "{:.1%}")


def render(r: dict[str, Any], command: str) -> str:
    e1_, e2_, e3_, e4_, d = r["E1"], r["E2"], r["E3"], r["E4"], r["descriptive"]
    decided = e1_["reproduces"]
    e1_outcome = ("the realised programme reproduces the clearing" if decided
                  else "below the threshold: E2-E4 are descriptive")
    s = r["solver"]
    lines = [
        "# Results: the Iberian price as a linear programme", "",
        f"Generated by `iberian-lp estimates` from `results/programmes.csv`, against the plan "
        "in [METHODS.md](../METHODS.md) (registered 2026-10-28) and its change log. Data: "
        f"{r['source']}. Prices are in EUR/MWh.", "",
        f"- **Programmes:** {s['solves']:,} solved with HiGHS (SciPy "
        f"{s['versions']['scipy']}), each checked against its exact dual interval and "
        f"optimal value (section 4); every check passed.",
        f"- **Periods:** {d['periods']:,} quarter-hours of {r['days']} days; "
        f"{d['zone_periods']:,} zone-periods, since {d['split']:,} periods split into Spain "
        "and Portugal.",
        f"- **Intervals:** percentile bootstrap over days, {r['bootstrap']['draws']:,} "
        f"resamples, seed {r['bootstrap']['seed']}. Every interval of E2 and E3, including "
        f"their breakdowns and hourly means, is at {r['bootstrap']['simultaneous_level']:.1%}"
        " (Bonferroni over the two questions, not over the hours); the others are "
        f"{r['bootstrap']['level']:.0%}. E1's decision uses its estimate.", "",
        "## The decisions", "",
        "| Estimand | Registered rule | Estimate [interval] | Outcome |",
        "| --- | --- | --- | --- |",
        f"| E1, the price as a dual | share ≥ {E1_THRESHOLD:.0%} | {_pct(e1_['share'])} | "
        f"{e1_outcome} |"]
    if decided:
        lines += [
            f"| E2, the cost of non-convexity | the interval of mean δ excludes 0 | "
            f"{_ci(e2_['all']['mean'])} | "
            f"{'excludes zero' if e2_['excludes_zero'] else 'includes zero'} |",
            f"| E3, the merit-order effect | G's interval lies above 0 | {_ci(e3_['G'])} | "
            f"{'smaller at midday' if e3_['midday_smaller'] else 'not established'} |"]
    else:
        lines += ["| E2, E3 | (not decided: E1 below the threshold) | see below | descriptive |"]
    moved = e2_["all"]["moved"]["estimate"]
    e2_held = not e2_["excludes_zero"] and moved <= 0.5
    lines += [
        "", "Expectations stated in the plan (section 6):", "",
        f"- **E2, no direction and δ = 0 in most periods:** "
        f"{'borne out' if e2_held else 'not borne out'}; the mean of δ is "
        f"{_ci(e2_['all']['mean'])} and |δ| > 0.01 in {moved:.1%} of periods.",
        f"- **E3, G > 0:** {'borne out' if e3_['midday_smaller'] else 'not borne out'} for "
        "its sign; the section after the results examines the reason the plan gave."]
    misses = e1_["misses"]
    lines += [
        "", "## E1: the price as a dual (Q1)", "",
        f"The published price π lies in the dual interval Λ_R of the realised programme, "
        f"widened by {E1_TOLERANCE} EUR/MWh at each end, in {_pct(e1_['share'])} of "
        "zone-periods."
        + (" With every zone-period inside, the bootstrap interval is degenerate and "
           "carries no information." if e1_["misses"]["count"] == 0 else ""), "",
        "EUPHEMIA, the algorithm that clears the market, must accept every in-the-money "
        "period order in full and reject every out-of-the-money one, and it validates its "
        "solutions by complementary slackness for its selection of block and complex orders "
        "(EUPHEMIA Public Description, 18 December 2025, sections 5.1 and 8.2). `LP_R` fixes "
        "that selection, so E1 checks that OMIE's public files and this reconstruction agree; "
        "it can fail only through an error in the data or the code ([REVIEW.md](REVIEW.md), "
        "finding M1).", "",
        "| Zone | Zone-periods | π in Λ_R |", "| --- | ---: | ---: |",
        *(f"| {name} | {e1_['by_zone'][z]['zone_periods']:,} | "
          f"{e1_['by_zone'][z]['inside']:,} |" for z, name in ZONES.items()), "",
        (f"- **Where it fails:** {misses['count']:,} zone-periods on {misses['days']} days; "
         f"π lies a median {misses['distance_median']:.2f} and at most "
         f"{misses['distance_max']:.2f} EUR/MWh outside Λ_R." if misses["count"]
         else "- **Where it fails:** nowhere."),
        *(["- **Hours of the misses** (OMIE's numbering, count): " + ", ".join(
            f"{h} ({n})" for h, n in misses["by_hour"].items()) + "."]
          if misses["count"] else []),
        f"- **Tolerance of {E1_TOLERANCE_SENSITIVITY} EUR/MWh (section 7):** "
        f"{_pct(e1_['tolerance_sensitivity'])}.",
        f"- **Accepted volume:** the published accepted buy volume lies within "
        f"{VOLUME_TOLERANCE_MW} MW of the realised programme's optimal accepted volume in "
        f"{_pct(e1_['volume'])} of zone-periods ({e1_['volume']['mismatched']:,} do not).",
        f"- **Λ_R wider than a point:** {d['wide']['count']:,} zone-periods "
        f"({d['wide']['share']:.1%}), median width {d['wide']['median_width']:.2f} EUR/MWh"
        + (f"; {d['wide']['unbounded']} unbounded." if d['wide']['unbounded'] else "."), "",
        "## E2: the cost of non-convexity (Q2)", "",
        "δ = the point of Λ_X nearest π, minus π, in the Spanish or joint zone. Relaxing "
        "the order types of section 7 separately:", "",
        "| Relaxed | Mean δ | Share with \\|δ\\| > 0.01 | Mean \\|δ\\| | Periods up | "
        "Periods down |",
        "| --- | --- | --- | --- | ---: | ---: |"]
    for key, name in (("all", "every non-convex order (`LP_X`)"),
                      ("blocks", "blocks only (C01, C04)"), ("sco", "scalable complex only (C02)")):
        x = e2_[key]
        lines.append(f"| {name} | {_ci(x['mean'])} | {_pct(x['moved'])} | "
                     f"{_ci(x['mean_abs'], '{:.2f}')} | {x['up']:,} | {x['down']:,} |")
    lines += [
        "", "The rows are not additive. The blocks-only row treats the alternatives of an "
        "exclusive group as additive supply, and every relaxation drops the scalable complex "
        "orders' fixed terms and keeps the France flows fixed, so each |δ| is an upper bound "
        "for its kind of relaxation; see the section after the results."]
    lines += [
        "", "## E3: the merit-order effect (Q3)", "",
        f"m = the price of `LP_R` with one more sell step of {E3_MW:,} MW at 0 EUR/MWh, minus "
        f"π. Midday is OMIE's hours {MIDDAY_HOURS[0]}-{MIDDAY_HOURS[-1]}; G = mean |m| "
        "outside midday − mean |m| at midday.", "",
        "| Variant | Mean m | Mean \\|m\\| outside midday | Mean \\|m\\| at midday | G |",
        "| --- | --- | --- | --- | --- |"]
    for name, x in ((f"{E3_MW:,} MW (registered)", e3_),
                    *((f"{q:,} MW", e3_["sensitivity"][f"{q} MW"]) for q in E3_SENSITIVITY_MW),
                    (f"{E3_MW:,} MW, midpoint rule", e3_["sensitivity"]["midpoint"])):
        lines.append(f"| {name} | {_ci(x['mean'])} | {_ci(x['outside_abs'], '{:.2f}')} | "
                     f"{_ci(x['midday_abs'], '{:.2f}')} | {_ci(x['G'])} |")
    unbounded = e3_["sensitivity"]["midpoint"]["unbounded"]
    if unbounded:
        lines += ["", f"The midpoint rule leaves out {unbounded:,} periods whose interval is "
                  "unbounded."]
    lines += ["", f"Mean m by hour ({E3_MW:,} MW):", "", "| Hour | Mean m |", "| ---: | --- |",
              *(f"| {h} | {_ci(x)} |" for h, x in e3_["by_hour"].items()),
              "", "## E4: the French border (Q4, descriptive)", "",
              "f = the price of `LP_R` with the France flows at zero, minus π.", "",
              "| Periods | Count | Mean f |", "| --- | ---: | --- |",
              f"| importing from France | {e4_['imports']['periods']:,} | "
              f"{_ci(e4_['imports'])} |",
              f"| exporting to France | {e4_['exports']['periods']:,} | "
              f"{_ci(e4_['exports'])} |",
              "", "## Descriptive", "",
              f"- **Split periods:** {d['split']:,} of {d['periods']:,} "
              f"({d['split'] / d['periods']:.1%}).",
              f"- **Steps per Spanish or joint zone-period:** median {d['steps_median']:,.0f} "
              f"offered, {d['steps_summed_median']:,.0f} after summing by price.",
              "- **Share of offered volume accepted:** " + ", ".join(
                  f"{TYPE_NAMES[k]} ({k}) {v:.1%}" for k, v in d["accepted_share"].items())
              + "."]
    ph = r["post_hoc"]
    mid, out, pos = ph["nonpositive"]["midday"], ph["nonpositive"]["outside"], ph["positive"]
    fl, line, blocks, zones = ph["floor"], ph["line"], ph["blocks"], ph["zones"]
    zero = ("and m = 0 in every one of them" if ph["nonpositive_max_abs_m"] == 0
            else f"yet |m| reaches {ph['nonpositive_max_abs_m']:.2f} there")
    both = [x for x in fl["bands"] if x["midday"]["periods"] and x["outside"]["periods"]]
    larger = sum(1 for x in both if x["midday"]["mean_abs"] > x["outside"]["mean_abs"])

    def band(x: dict[str, Any]) -> str:
        name = (f"above {x['low']:g}" if x["high"] == np.inf
                else f"({x['low']:g}, {x['high']:g}]")
        return (f"| {name} | {x['midday']['mean_abs']:.2f} ({x['midday']['periods']:,}) | "
                f"{x['outside']['mean_abs']:.2f} ({x['outside']['periods']:,}) |")

    lines += [
        "", "## After the results (descriptive, not registered)", "",
        "Added after the results above had been seen (*results*), or after the critical "
        "review (*review*, [REVIEW.md](REVIEW.md)); see the change log of "
        "[METHODS.md](../METHODS.md). They change no decision. Intervals are 95%.", "",
        f"- **Where G comes from** (*results*). π ≤ 0 in {mid['count']:,} of "
        f"{mid['periods']:,} midday periods ({mid['count'] / mid['periods']:.1%}) and in "
        f"{out['count']:,} of {out['periods']:,} others "
        f"({out['count'] / out['periods']:.1%}). A sell step at 0 EUR/MWh changes no step's "
        "position at prices below zero, so a price at or below zero that clears `LP_R` "
        f"still clears it with the extra step, {zero}.",
        f"- **The zero floor binds at every price** (*review*). With the extra step, the "
        f"price never falls below zero ({fl['below_zero']} exceptions), so |m| ≤ π in every "
        f"period priced above zero ({fl['above_price']} exceptions). The floor binds "
        f"exactly, with the new price at 0, in {fl['midday']['binding']:,} of "
        f"{fl['midday']['positive']:,} positive midday periods and "
        f"{fl['outside']['binding']:,} of {fl['outside']['positive']:,} others, and positive "
        f"midday prices are lower: median {fl['midday']['median_price']:.2f} against "
        f"{fl['outside']['median_price']:.2f}. So the comparison of the periods priced above "
        f"zero, mean |m| {_ci(pos['outside_abs'], '{:.2f}')} outside midday against "
        f"{_ci(pos['midday_abs'], '{:.2f}')} at midday (outside minus midday "
        f"{_ci(pos['G'])}), mixes the floor with the slope of the supply curve and does not "
        "test the slope.",
        f"- **The reason the plan gave** (*review*). Within price bands, mean |m| is larger "
        f"at midday in {larger} of {len(both)}. The plan's reason for G > 0, a flatter supply "
        "curve at midday, gets no support; G comes from the periods at or below zero.", "",
        "| π (EUR/MWh) | Mean \\|m\\| at midday (periods) | Mean \\|m\\| at other "
        "hours (periods) |", "| --- | ---: | ---: |",
        *(band(x) for x in fl["bands"]), "",
        f"- **The France line** (*review*). The line had free capacity in the direction of "
        f"the flow in {line['with_room']:,} of {line['periods']:,} periods "
        f"({line['with_room'] / line['periods']:.1%}), where holding the flow fixed is a strong "
        f"assumption. Mean δ: {_ci(line['full']['delta'])} with the line full, "
        f"{_ci(line['room']['delta'])} with room. Mean m: {_ci(line['full']['m'])} and "
        f"{_ci(line['room']['m'])}.",
        "- **Exclusive blocks** (*review*). The curves do not say which exclusive blocks are "
        "alternatives within one group. Two variants show how much of the blocks-only row "
        "comes from counting them as additive supply; neither models a group exactly. With "
        "the exclusive blocks deduplicated (of the steps with the same quantity in a period, "
        f"only the cheapest), mean δ is {_ci(blocks['dedup'])}; with the exclusive blocks "
        f"not relaxed, {_ci(blocks['c01_only'])}. With every order relaxed and the exclusive "
        f"blocks deduplicated, {_ci(blocks['all_dedup'])}.",
        f"- **Joint and split periods** (*review*). Mean δ is {_ci(zones['joint'])} in joint "
        f"periods, where the relaxation acts on the whole Iberian market, and "
        f"{_ci(zones['split'])} in Spain in split periods, where Portugal's orders and the "
        "flow between the two stay fixed.",
        f"- **The midpoint rule** (*results*). E3's counterfactual interval is wider than a "
        f"point in {ph['midpoint_wide']:,} of {d['periods']:,} periods; in the others the "
        "two rules give the same price.",
        "", "## Reproduce", "", "```", "iberian-lp programmes", command, "```", ""]
    return "\n".join(lines)
