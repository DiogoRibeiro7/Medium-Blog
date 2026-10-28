"""Figures for the article, drawn from the committed tables, solves and estimates only (no
solving and no bootstrap here)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from routing_roads.contract import DAYS, EXTRACT, PAIRS, SEEDS
from routing_roads.estimates import AREA_NAMES, costs, noise_by_day, regret_by_day
from routing_roads.experiment import AREAS
from routing_roads.report import Derived, area_indices, pair_ratios

INK = "#1f2933"
MUTED = "#7b8794"
BLUE = "#2563a8"
ORANGE = "#d97706"
GRID = "#e4e7eb"
SOURCE = (f"Map data © OpenStreetMap contributors (ODbL), Geofabrik extract {EXTRACT}. "
          "Simulated daily orders from OSM pharmacies. Routes: PyVRP.")
AREA_KEYS = [(p.key, a) for p in PAIRS for a in AREAS]
SHORT = {"main": "Main design", "depot_all": "Depot at the\npair's mean"}


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.hashsalt": "routing-roads"})
    return plt


def _save(plt: Any, fig: Any, path: Path, note: str = "") -> Path:
    # Just below everything drawn so far; the tight bounding box grows to include it.
    fig.canvas.draw()
    bottom = fig.get_tightbbox().y0 / fig.get_figheight()
    fig.text(0.01, bottom - 0.01, SOURCE + (f"\n{note}" if note else ""), fontsize=7,
             color=MUTED, ha="left", va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return path


def _pct(x: Any) -> Any:
    return 100 * np.asarray(x)


def _cells(rows: Sequence[dict[str, str]], pair: str, area: str) -> dict[str, Any]:
    return {f"{plan}_{kind}": costs(rows, "main", pair, area, plan, kind, DAYS, SEEDS)
            for plan in ("straight", "road") for kind in ("road", "straight")}


def ratios(d: Derived, path: Path) -> Path:
    """The ratio of road to straight-line distance over pairs of pharmacies, by area."""
    plt = _pyplot()
    fig, axes = plt.subplots(len(AREA_KEYS), 1, figsize=(7.5, 4.6), sharex=True)
    top = 2.6
    bins = np.linspace(1.0, top, 65)
    for ax, (pair_key, area) in zip(axes, AREA_KEYS, strict=True):
        pair = next(p for p in PAIRS if p.key == pair_key)
        r = pair_ratios(d.road[pair_key], d.straight[pair_key],
                        area_indices(d, pair, area == "both"))
        counts, _ = np.histogram(np.clip(r, None, top), bins=bins)
        share = counts / counts.sum()
        ax.stairs(share, bins, fill=True, color=BLUE, alpha=0.22, lw=0)
        ax.stairs(share, bins, color=BLUE, lw=1.2)
        median = float(np.median(r))
        ax.axvline(median, color=INK, lw=1.0)
        ax.annotate(f"median {median:.2f}", (median, float(np.max(share))), xytext=(4, -2),
                    textcoords="offset points", fontsize=8, color=INK, va="top")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylabel(AREA_NAMES[(pair_key, area)].replace(" with ", "\nwith "), rotation=0,
                      ha="right", va="center", fontsize=8)
    axes[-1].set_xlabel(f"Road distance over straight-line distance (pairs above {top} "
                        "drawn at the right edge)")
    axes[0].set_title("How much further the road is than the straight line, over every pair "
                      "of pharmacies", fontsize=10, loc="left", color=INK)
    return _save(plt, fig, path, "One-way streets respected: each ordered pair counts once "
                 "in each direction.")


def regret(rows: Sequence[dict[str, str]], e: dict[str, Any], path: Path) -> Path:
    """Per day: the regret of the straight-line plan, and the solver's seed noise."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    rng = np.random.default_rng(0)                 # vertical jitter only
    for i, (pair_key, area) in enumerate(AREA_KEYS):
        c = _cells(rows, pair_key, area)
        r = regret_by_day(c["straight_road"], c["road_road"])
        noise = noise_by_day(c["road_road"])
        y = len(AREA_KEYS) - 1 - i
        ax.scatter(_pct(r), y + 0.12 + rng.uniform(-0.08, 0.08, r.size), s=14, color=BLUE,
                   alpha=0.55, lw=0, label="Straight-line plan against road plan"
                   if i == 0 else None)
        ax.scatter(_pct(noise), y - 0.22 + rng.uniform(-0.05, 0.05, noise.size), s=14,
                   color=MUTED, alpha=0.6, lw=0,
                   label="Road plan, one seed against another" if i == 0 else None)
        e1 = e["areas"][f"{pair_key}_{area}"]["E1"]
        ax.plot(_pct([e1["low"], e1["high"]]), [y + 0.32] * 2, color=INK, lw=1.6)
        ax.plot(_pct(e1["estimate"]), y + 0.32, marker="o", color=INK, ms=5)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(AREA_KEYS)),
                  [AREA_NAMES[k].replace(" with ", "\nwith ") for k in AREA_KEYS[::-1]],
                  fontsize=8)
    ax.set_xlabel("Extra road distance (%), one dot a day")
    ax.set_title("Extra road distance of routes planned on straight lines, and the solver's "
                 "own noise", fontsize=10, loc="left", color=INK, pad=24)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=8,
              frameon=False, borderaxespad=0.2)
    return _save(plt, fig, path, "Black: mean over days with its 95% interval. Seed noise "
                 "averages the 12 ordered pairs of seeds.")


def water(rows: Sequence[dict[str, str]], e: dict[str, Any], path: Path) -> Path:
    """Per day: the regret with the other bank minus the regret in the city alone."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    rng = np.random.default_rng(0)
    for i, pair in enumerate(PAIRS):
        by_area = {}
        for area in AREAS:
            c = _cells(rows, pair.key, area)
            by_area[area] = regret_by_day(c["straight_road"], c["road_road"])
        delta = by_area["both"] - by_area["city"]
        y = len(PAIRS) - 1 - i
        ax.scatter(_pct(delta), y - 0.1 + rng.uniform(-0.08, 0.08, delta.size), s=14,
                   color=BLUE, alpha=0.55, lw=0)
        w = e["water"][pair.key]
        ax.plot(_pct([w["low"], w["high"]]), [y + 0.2] * 2, color=INK, lw=1.6)
        ax.plot(_pct(w["estimate"]), y + 0.2, marker="o", color=INK, ms=5)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(PAIRS)), [f"{p.city}\n+ {p.other}" for p in PAIRS[::-1]],
                  fontsize=8)
    ax.set_ylim(-0.6, len(PAIRS) - 0.4)
    ax.set_xlabel("Change in regret when the other bank is added (percentage points), "
                  "one dot a day")
    ax.set_title("What adding the far bank does to the regret", fontsize=10, loc="left",
                 color=INK)
    ax.grid(axis="x", color=GRID, lw=0.6)
    level = e["water"][PAIRS[0].key]["level"]
    return _save(plt, fig, path, f"Black: mean over days with its {level:.2%} interval "
                 "(simultaneous across the four areas).")


def estimate(rows: Sequence[dict[str, str]], e: dict[str, Any], path: Path) -> Path:
    """Per day: α × the straight-line plan's straight-line cost, against the road distance
    that plan drives and against the road plan's."""
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.6))
    for ax, (pair_key, area) in zip(axes.flat, AREA_KEYS, strict=True):
        c = _cells(rows, pair_key, area)
        alpha = e["areas"][f"{pair_key}_{area}"]["alpha"]
        guess = alpha * c["straight_straight"].mean(axis=1) / 1000
        own = c["straight_road"].mean(axis=1) / 1000
        best = c["road_road"].mean(axis=1) / 1000
        low = min(guess.min(), best.min()) * 0.95
        high = max(own.max(), guess.max()) * 1.03
        ax.plot([low, high], [low, high], color=MUTED, lw=0.8)
        ax.scatter(own, guess, s=16, color=BLUE, alpha=0.7, lw=0,
                   label="the road distance the straight-line plan drives")
        ax.scatter(best, guess, s=16, facecolor="none", edgecolor=MUTED, lw=0.8,
                   label="the road plan's road distance")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal")
        ax.set_title(AREA_NAMES[(pair_key, area)], fontsize=9, loc="left", color=INK)
        ax.grid(color=GRID, lw=0.6)
        ax.annotate("estimate = road distance", (high, high), xytext=(-4, -12),
                    textcoords="offset points", fontsize=7, color=MUTED, ha="right")
    for ax in axes[1]:
        ax.set_xlabel("Road distance a day (km)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Circuity estimate (km)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=8,
               frameon=False, title="Estimate α × straight-line cost, against",
               title_fontsize=8, alignment="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(plt, fig, path, "One dot a day (seed means). Below the line, the estimate "
                 "falls short.")


def sensitivity(e: dict[str, Any], path: Path) -> Path:
    plt = _pyplot()
    from routing_roads.estimates import VARIANT_NAMES

    variants = list(dict.fromkeys(s["variant"] for s in e["sensitivity"]))
    by = {(s["variant"], s["pair"]): s for s in e["sensitivity"]}
    panels = (("E1_city", "Regret, city alone (%)"),
              ("E1_both", "Regret, with the other bank (%)"),
              ("E3", "Change from the other bank (points)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(7.5, 3.4), sharey=True)
    colours = (BLUE, ORANGE)
    for ax, (key, label) in zip(axes, panels, strict=True):
        for j, (pair, colour) in enumerate(zip(PAIRS, colours, strict=True)):
            for i, v in enumerate(variants):
                x = by[(v, pair.key)][key]
                y = len(variants) - 1 - i + (0.14 if j == 0 else -0.14)
                ax.plot(_pct([x["low"], x["high"]]), [y, y], color=colour, lw=1.6)
                ax.plot(_pct(x["estimate"]), y, marker="o", color=colour, ms=4.5,
                        label=pair.city if i == 0 and key == "E1_city" else None)
        if key == "E3":
            ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_xlabel(label, fontsize=8)
        ax.grid(axis="x", color=GRID, lw=0.6)
    names = [SHORT.get(v, VARIANT_NAMES[v]) for v in variants]
    axes[0].set_yticks(range(len(variants)), names[::-1], fontsize=8)
    fig.legend(loc="upper right", bbox_to_anchor=(0.98, 0.92), fontsize=8, frameon=False,
               ncol=2)
    fig.suptitle("Each variation fixed in advance, beside the main design on the same days "
                 "and seeds", fontsize=10, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(plt, fig, path, "Lines: 95% intervals over days.")


def draw_all(d: Derived, rows: Sequence[dict[str, str]], e: dict[str, Any],
             out: Path) -> list[Path]:
    return [ratios(d, out / "fig1_ratios.png"),
            regret(rows, e, out / "fig2_regret.png"),
            water(rows, e, out / "fig3_water.png"),
            estimate(rows, e, out / "fig4_estimate.png"),
            sensitivity(e, out / "fig5_sensitivity.png")]
