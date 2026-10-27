"""Figures for the article, drawn from the committed results only (no estimation here)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

INK = "#1f2933"
MUTED = "#7b8794"
BLUE = "#2563a8"
ORANGE = "#d97706"
GRID = "#e4e7eb"


def _pct(log_points: float) -> float:
    return 100 * (math.exp(log_points) - 1)


def _per_hour(per_24_hours: float) -> float:
    """M3's bins count time in units of 24 hours; one hour is a 24th of the coefficient."""
    return _pct(per_24_hours / 24)


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.hashsalt": "eu-wheat"})
    return plt


def _save(plt: Any, fig: Any, path: Path, source: str) -> None:
    # Below the axes; the tight bounding box grows to include it.
    fig.text(0.01, -0.03, source, fontsize=7, color=MUTED, ha="left", va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


SOURCE = ("Data: Eurostat (apro_cpshr); E-OBS v33.0e (ECA&D, Copernicus). "
          "EU-27 NUTS-2 regions, 2000-2024.")
MAIZE_SOURCE = ("Data: Eurostat (apro_cpshr, grain maize); E-OBS v33.0e (ECA&D, Copernicus); "
                "GGCMI Phase 3 crop calendar. EU-27 NUTS-2 regions, 2010-2024.")


def temperature_response(bins: Sequence[Mapping[str, Any]], path: Path,
                         season: str = "1 March to 31 July", source: str = SOURCE) -> None:
    plt = _pyplot()
    labels: list[str] = []
    mids: list[float] = []
    for b in bins:
        body = b["term"].removeprefix("t_")
        if body.startswith("lt"):
            labels.append("<0")
            mids.append(-1.5)
        elif body.startswith("ge"):
            labels.append("≥36")
            mids.append(37.5)
        else:
            lo, hi = (int(x) for x in body.split("_"))
            labels.append(f"{lo}-{hi}")
            mids.append((lo + hi) / 2)
    est = [_per_hour(b["estimate"]) for b in bins]
    low = [_per_hour(b["low"]) for b in bins]
    high = [_per_hour(b["high"]) for b in bins]
    rows = sorted(zip(mids, labels, est, low, high, strict=True), key=lambda r: r[0])
    rows.insert(next(i for i, r in enumerate(rows) if r[0] > 16.5), (16.5, "15-18", 0.0, 0.0, 0.0))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    xs = range(len(rows))
    ax.axhline(0, color=MUTED, lw=0.8)
    for x, (_, label, e, bar_low, bar_high) in zip(xs, rows, strict=True):
        colour = ORANGE if bar_high < 0 else BLUE
        if label == "15-18":
            ax.plot(x, 0, marker="o", mfc="white", mec=INK, ms=5)
            continue
        ax.plot([x, x], [bar_low, bar_high], color=colour, lw=1.6)
        ax.plot(x, e, marker="o", color=colour, ms=5)
    ax.set_xticks(list(xs), [r[1] for r in rows], rotation=0, fontsize=8)
    ax.set_xlabel(f"Temperature (°C), {season}")
    ax.set_ylabel("Yield change per hour (%)")
    ax.set_title("One more hour in each temperature range, instead of an hour at 15-18 °C",
                 fontsize=10, loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    _save(plt, fig, path, source + " Bars: 95% intervals (two-way clustered).")


def precipitation_response(curve: Mapping[str, Any], optimum: Mapping[str, float] | None,
                           path: Path) -> None:
    plt = _pyplot()
    pts = curve["points"]
    x = [p["mm"] for p in pts]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.fill_between(x, [_pct(p["low"]) for p in pts], [_pct(p["high"]) for p in pts],
                    color=BLUE, alpha=0.18, lw=0)
    ax.plot(x, [_pct(p["effect"]) for p in pts], color=BLUE, lw=2)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(curve["median_mm"], color=MUTED, lw=0.8, ls="--")
    ax.annotate("median", (curve["median_mm"], ax.get_ylim()[0]), xytext=(4, 4),
                textcoords="offset points", fontsize=8, color=MUTED)
    if optimum is not None:
        ax.axvspan(optimum["low"], optimum["high"], color=ORANGE, alpha=0.10, lw=0)
        ax.axvline(optimum["estimate"], color=ORANGE, lw=0.8, ls=":")
        ax.annotate("peak (shaded: 95% interval)", (optimum["estimate"], ax.get_ylim()[0]),
                    xytext=(4, 18), textcoords="offset points", fontsize=8, color=ORANGE)
    ax.set_xlabel("Precipitation, 1 March to 31 July (mm)")
    ax.set_ylabel("Yield change vs. median (%)")
    ax.set_title("Rainfall over the season and wheat yields", fontsize=10, loc="left",
                 color=INK)
    ax.grid(color=GRID, lw=0.6)
    _save(plt, fig, path, SOURCE + " Band: 95% interval (two-way clustered).")


def out_of_sample(regions: Sequence[Mapping[str, Any]], years: Sequence[Mapping[str, Any]],
                  path: Path, source: str = SOURCE, area: str = "EU") -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(years), figsize=(10.5, 3.9), sharex=True, sharey=True)
    # Shared limits that hold every region, rounded out to 10 points: no dot is clipped.
    every = [_pct(r[k]) for r in regions for k in ("predicted", "observed")]
    low, high = 10 * math.floor(min(every) / 10), 10 * math.ceil(max(every) / 10)
    for ax, summary in zip(axes, years, strict=True):
        rows = [r for r in regions if r["year"] == summary["year"]]
        pred = [_pct(r["predicted"]) for r in rows]
        obs = [_pct(r["observed"]) for r in rows]
        ax.scatter(pred, obs, s=10, color=BLUE, alpha=0.6, lw=0)
        ax.plot([low, high], [low, high], color=MUTED, lw=0.8)
        ax.axhline(0, color=GRID, lw=0.6)
        ax.axvline(0, color=GRID, lw=0.6)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal")
        label = f"{area}: " if area else ""
        ax.set_title(f"{summary['year']}\n{label}{summary['eu_yield_observed_pct']:+.1f}% "
                     f"observed, {summary['eu_yield_predicted_pct']:+.1f}% predicted",
                     fontsize=8.5, loc="left", color=INK)
        ax.set_xlabel("Predicted from weather (%)")
    axes[0].set_ylabel("Observed yield anomaly (%)")
    _save(plt, fig, path, source + " Each dot is a region; the model never saw that year.")


def sensitivity(rows: Sequence[Mapping[str, Any]], primary: Mapping[str, Mapping[str, float]],
                path: Path, source: str = SOURCE) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    variants = [("Registered model", "")] + list(dict.fromkeys(
        (r["label"], r["variant"]) for r in rows))
    for ax, key, title in zip(axes, ("E1", "E2"),
                              ("Heat: 10 more degree days above the threshold\n"
                               "(30 °C unless stated)",
                               "Drought: a dry season\n(10th percentile against the median)"),
                              strict=True):
        ys = list(range(len(variants)))[::-1]
        for y, (_label, name) in zip(ys, variants, strict=True):
            r = primary[key] if not name else next(
                x for x in rows if x["variant"] == name and x["estimand"] == key)
            colour = INK if not name else BLUE
            ax.plot([_pct(r["low"]), _pct(r["high"])], [y, y], color=colour, lw=1.6)
            ax.plot(_pct(r["estimate"]), y, "o", color=colour, ms=5)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(ys, [v[0] for v in variants], fontsize=8)
        ax.set_title(title, fontsize=9.5, loc="left", color=INK)
        ax.set_xlabel("Yield change (%)")
        ax.grid(axis="x", color=GRID, lw=0.6)
    _save(plt, fig, path, source + " Lines: 95% intervals (two-way clustered).")


AMENDMENT_ROWS = (("inference", "Registered analysis"),
                  ("window", "Season ends at harvest"),
                  ("data", "Without suspect yield series"),
                  ("all", "Both: the second analysis"))


def amendment(decomposition: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Heat and drought under the registered analysis and Amendment A1's changes."""
    plt = _pyplot()
    rows = {d["combination"]: d for d in decomposition}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    ys = list(range(len(AMENDMENT_ROWS)))[::-1]
    for ax, key, title in zip(axes, ("E1", "E2"),
                              ("Heat: 10 more degree days\nabove 30 °C",
                               "Drought: a dry season\n(10th percentile against the median)"),
                              strict=True):
        for y, (name, _) in zip(ys, AMENDMENT_ROWS, strict=True):
            r = rows[name][key]
            colour = INK if name == "inference" else ORANGE if name == "all" else BLUE
            ax.plot([_pct(r["low"]), _pct(r["high"])], [y, y], color=colour, lw=1.6)
            ax.plot(_pct(r["estimate"]), y, "o", color=colour, ms=5)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(ys, [label for _, label in AMENDMENT_ROWS], fontsize=8)
        ax.set_title(title, fontsize=9.5, loc="left", color=INK)
        ax.set_xlabel("Yield change (%)")
        ax.grid(axis="x", color=GRID, lw=0.6)
    _save(plt, fig, path, SOURCE + " Crop calendar: GGCMI Phase 3. "
          "Lines: 95% intervals (two-way clustered).")


def draw_all(results: Mapping[str, Any], directory: Path,
             amended: Mapping[str, Any] | None = None) -> list[Path]:
    """Every figure of the article, from ``results.json`` (and the amendment's)."""
    coef = {(c["model"], c["term"]): c for c in results["coefficients"]}
    dry = results["E2"]["dry_season"]
    primary = {"E1": coef[("M1", "edd")],
               "E2": {"estimate": dry["effect"], "low": dry["low"], "high": dry["high"]}}
    paths = [directory / name for name in ("fig1_temperature.png", "fig2_precipitation.png",
                                           "fig3_out_of_sample.png", "fig4_sensitivity.png")]
    temperature_response(results["M3"], paths[0])
    # The peak's interval is a post-hoc check (results["post_hoc"]); no peak, no interval.
    optimum = (None if results["E2"]["optimum_mm"] is None
               else results["post_hoc"]["optimum_mm"])
    precipitation_response(results["precipitation_curve"], optimum, paths[1])
    out_of_sample(results["E4_regions"], results["E4"], paths[2])
    sensitivity(results["sensitivity"], primary, paths[3])
    if amended is not None:
        paths.append(directory / "fig5_amendment.png")
        amendment(amended["decomposition"], paths[4])
    return paths


def crops(rows: Sequence[tuple[str, Mapping[str, Mapping[str, float]]]], path: Path) -> None:
    """Heat and drought for maize beside wheat: (label, {"E1": ..., "E2": ...}) per row."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    ys = list(range(len(rows)))[::-1]
    for ax, key, title in zip(axes, ("E1", "E2"),
                              ("Heat: 10 more degree days\nabove 30 °C",
                               "Drought: a dry season\n(10th percentile against the median)"),
                              strict=True):
        for y, (label, estimates) in zip(ys, rows, strict=True):
            r = estimates[key]
            colour = ORANGE if label.startswith("Maize") else BLUE
            ax.plot([_pct(r["low"]), _pct(r["high"])], [y, y], color=colour, lw=1.6)
            ax.plot(_pct(r["estimate"]), y, "o", color=colour, ms=5)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(ys, [label for label, _ in rows], fontsize=8)
        ax.set_title(title, fontsize=9.5, loc="left", color=INK)
        ax.set_xlabel("Yield change (%)")
        ax.grid(axis="x", color=GRID, lw=0.6)
    _save(plt, fig, path, MAIZE_SOURCE + " Wheat: 2000-2024. "
          "Lines: 95% intervals (two-way clustered).")


def draw_maize(results: Mapping[str, Any], wheat_amended: Mapping[str, Any],
               wheat: Mapping[str, Any], directory: Path) -> list[Path]:
    """Every figure of the maize article, from the maize and wheat results."""
    paths = [directory / name for name in ("fig1_temperature.png", "fig2_crops.png",
                                           "fig3_out_of_sample.png", "fig4_sensitivity.png")]
    temperature_response(results["M3"], paths[0], "planting to maturity", MAIZE_SOURCE)
    edd = next(c for c in wheat["coefficients"] if c["model"] == "M1" and c["term"] == "edd")
    dry = wheat["E2"]["dry_season"]
    registered = {"E1": edd, "E2": {"estimate": dry["effect"], "low": dry["low"],
                                    "high": dry["high"]}}
    crops([("Maize", results), ("Wheat, second analysis", wheat_amended),
           ("Wheat, registered analysis", registered)], paths[1])
    out_of_sample(results["E4_regions"], results["E4"], paths[2], MAIZE_SOURCE, "")
    sensitivity(results["sensitivity"], {"E1": results["E1"], "E2": results["E2"]}, paths[3],
                MAIZE_SOURCE)
    return paths
