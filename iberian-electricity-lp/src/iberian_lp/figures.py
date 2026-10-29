"""Figures for the article, drawn from the committed tables, programmes and estimates only
(no solving and no bootstrap here)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from iberian_lp.contract import E3_MW, MIDDAY_HOURS, NON_CONVEX, SIMPLE
from iberian_lp.estimates import Table
from iberian_lp.experiment import fixed, read_steps
from iberian_lp.publication import RELAXATIONS, daily, day_text
from iberian_lp.report import Derived

INK = "#1f2933"
MUTED = "#7b8794"
BLUE = "#2563a8"
ORANGE = "#d97706"
GRID = "#e4e7eb"
SOURCE = "Data: OMIE (OMI-Polo Español, S.A.), omie.es. "
RELAXATION_NAMES = {"all": "Every indivisible\norder", "sco": "Scalable complex\norders only",
                    "blocks": "Block orders only"}


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.hashsalt": "iberian-lp"})
    return plt


def _save(plt: Any, fig: Any, path: Path, note: str) -> Path:
    # Just below everything drawn so far; the tight bounding box grows to include it.
    fig.canvas.draw()
    bottom = fig.get_tightbbox().y0 / fig.get_figheight()
    fig.text(0.01, bottom - 0.01, SOURCE + note, fontsize=7, color=MUTED, ha="left",
             va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return path


def _curve(prices: list[int], tenths: list[int], start: float, descending: bool
           ) -> tuple[np.ndarray, np.ndarray]:
    """A stepwise curve: prices in EUR/MWh, in merit order, and the quantity edges in MW."""
    order = np.argsort(prices, kind="stable")[::-1 if descending else 1]
    p = np.asarray(prices, dtype=float)[order] / 100
    q = np.asarray(tenths, dtype=float)[order] / 10
    return p, start + np.concatenate(([0.0], np.cumsum(q)))


def example(d: Derived, t: Table, e: dict[str, Any]) -> int:
    """Figure 1's period: of the first drawn day's joint quarter-hours, the one whose δ is
    nearest the mean δ over all periods (the earliest, on a tie)."""
    delta = t.nearest("X")
    candidates = [i for i, r in enumerate(t.rows) if r["date"] == d.days[0]
                  and r["zone"] == "MI"]
    return min(candidates, key=lambda i: abs(delta[i] - e["E2"]["all"]["mean"]["estimate"]))


def curves(derived: Path, d: Derived, t: Table, e: dict[str, Any], path: Path) -> Path:
    """One period's bids: the realised and relaxed supply curves against demand."""
    i = example(d, t, e)
    day, hour, quarter = t.rows[i]["date"], int(t.rows[i]["hour"]), int(t.rows[i]["quarter"])
    row = next(r for r in d.periods if r["date"] == day and r["period"] == t.rows[i]["period"]
               and r["zone"] == "MI")
    bids = read_steps(derived / "steps" / f"{day.replace('-', '')}.csv.gz")[
        (int(row["period"]), row["zone"])]
    mw = {k: fixed(row[k], 1) / 10 for k in row if k.startswith(("matched", "imports",
                                                                  "exports"))}
    imports = mw["imports_fr"] + mw["imports_ib"]
    exports = mw["exports_fr"] + mw["exports_ib"]
    accepted_nc = sum(mw[f"matched_sell_{k}"] for k in NON_CONVEX)
    realised = _curve(*bids[("V", SIMPLE)], imports + accepted_nc, False)
    relaxed_prices = [p for k in (SIMPLE, *NON_CONVEX) for p in bids.get(("V", k), ([], []))[0]]
    relaxed_qty = [q for k in (SIMPLE, *NON_CONVEX) for q in bids.get(("V", k), ([], []))[1]]
    relaxed = _curve(relaxed_prices, relaxed_qty, imports, False)
    demand = _curve(*bids[("C", SIMPLE)], exports, True)
    price = t.price[i] / 100
    relaxed_price = price + float(t.nearest("X")[i])
    volume = mw["matched_buy"] + exports

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.stairs(*realised, baseline=None, color=BLUE, lw=2, label="Supply, realised programme")
    ax.stairs(*relaxed, baseline=None, color=ORANGE, lw=2, ls=(0, (4, 2)),
              label="Supply, indivisible orders made divisible")
    ax.stairs(*demand, baseline=None, color=INK, lw=2, label="Demand")
    for y, text in ((price, f"published price {price:.2f}"),
                    (relaxed_price, f"relaxed price {relaxed_price:.2f}")):
        ax.axhline(y, color=MUTED, lw=0.8, ls=":")
        ax.annotate(text, (volume - 5900, y), xytext=(0, 3), textcoords="offset points",
                    fontsize=8, color=INK, va="bottom")
    ax.set_xlim(volume - 6000, volume + 6000)
    ax.set_ylim(min(price, relaxed_price) - 60, max(price, relaxed_price) + 60)
    ax.set_xlabel("Cumulative quantity (MW)")
    ax.set_ylabel("Price (EUR/MWh)")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=8, frameon=False)
    return _save(plt, fig, path, f"{day_text(date.fromisoformat(day))}, "
                 f"{hour - 1}:{15 * (quarter - 1):02d} to {hour - 1}:{15 * quarter:02d}, the "
                 "joint Iberian market.\nThe realised curve starts at the fixed supply: "
                 "imports and the accepted indivisible orders.")


def relaxation(t: Table, e: dict[str, Any], path: Path) -> Path:
    """E2: the change in price from making the indivisible orders divisible, day by day."""
    days = daily(t)
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    rng = np.random.default_rng(0)
    names = [name for name, _ in RELAXATIONS]
    for y, name in enumerate(names):
        x = days[name]
        ax.scatter(x, y + rng.uniform(-0.12, 0.12, x.size), s=14, color=BLUE, alpha=0.6, lw=0,
                   label="One day's mean" if y == 0 else None)
        m = e["E2"][name]["mean"]
        ax.plot([m["low"], m["high"]], [y - 0.3, y - 0.3], color=INK, lw=2)
        ax.scatter([m["estimate"]], [y - 0.3], s=30, color=INK, zorder=3,
                   label=f"Mean over days, {m['level']:.1%} interval" if y == 0 else None)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(names)), [RELAXATION_NAMES[n] for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("Price with the orders made divisible, minus the published price (EUR/MWh)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=8, frameon=False)
    return _save(plt, fig, path, "Spanish or joint zone, mean over each day's quarter-hours.")


def hours(t: Table, e: dict[str, Any], path: Path) -> Path:
    """E3 by hour, and the share of quarter-hours priced at or below zero."""
    by_hour = e["E3"]["by_hour"]
    hrs = np.array([int(h) for h in by_hour])
    est = np.array([x["estimate"] for x in by_hour.values()])
    low = np.array([x["low"] for x in by_hour.values()])
    high = np.array([x["high"] for x in by_hour.values()])
    share = daily(t)["nonpositive_by_hour"]
    plt = _pyplot()
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.5, 5.0), sharex=True,
                                      height_ratios=(3, 2))
    for ax in (top, bottom):
        ax.axvspan(MIDDAY_HOURS[0] - 1, MIDDAY_HOURS[-1], color=GRID, alpha=0.7, lw=0)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    centre = hrs - 0.5
    top.vlines(centre, low, high, color=BLUE, lw=1.5)
    top.scatter(centre, est, s=20, color=BLUE, zorder=3)
    top.axhline(0, color=MUTED, lw=0.8)
    top.set_ylabel(f"Change in price with\n{E3_MW:,} MW more at\nzero price (EUR/MWh)")
    top.annotate("midday", (MIDDAY_HOURS[0] - 0.8, float(np.min(low))), fontsize=8, color=INK,
                 va="bottom")
    bottom.bar(centre, [100 * share[int(h)] for h in hrs], width=0.8, color=ORANGE, lw=0)
    bottom.set_ylabel("Quarter-hours priced\nat or below zero (%)")
    bottom.set_xlabel("Hour of the day (Spanish time)")
    bottom.set_xticks(range(0, 25, 3))
    bottom.set_xlim(0, 24)
    level = next(iter(by_hour.values()))["level"]
    return _save(plt, fig, path, f"Spanish or joint zone. Top: mean over quarter-hours with "
                 f"{level:.1%} intervals. Shaded: the registered midday.")


def draw_all(derived: Path, d: Derived, t: Table, e: dict[str, Any], out: Path) -> list[Path]:
    return [curves(derived, d, t, e, out / "fig1_curves.png"),
            relaxation(t, e, out / "fig2_relaxation.png"),
            hours(t, e, out / "fig3_hours.png")]
