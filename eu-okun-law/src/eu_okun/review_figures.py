"""Diagnostic figures for the classical-results review (static PNG, matplotlib).

Colour use follows one rule: at most three categorical hues per chart (the first three
slots of a colour-vision-deficiency validated palette) plus a neutral grey, never
colour alone - every chart has a legend or axis labels, and every figure has a table
counterpart in the report. Countries are always ordered alphabetically, never by
estimate, because no ranking has been established.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from eu_okun.contract import EU27, quarter_sequence
from eu_okun.review import (
    HEADLINE_LAGS,
    ReviewResult,
    grid_lookup,
    in_pandemic,
)

SURFACE, INK, INK2, MUTED, GRID, AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#c3c2b7"
FOCUS = ("PT", "ES", "NL", "LU")


def _pyplot() -> Any:
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "Figures need matplotlib: install the 'report' extra (pip install -e '.[report]')."
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5, "text.color": INK,
        "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "xtick.color": MUTED,
        "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 100,
        "savefig.dpi": 160, "svg.hashsalt": "eu-okun-review",
    })
    return plt


def _finish(plt: Any, fig: Any, path: Path, source: str, credit_y: float = 0.005) -> None:
    fig.text(0.995, credit_y, source, ha="right", va="bottom", fontsize=7, color=MUTED)
    fig.savefig(path, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def _zero_line(ax: Any, vertical: bool = True) -> None:
    if vertical:
        ax.axvline(0.0, color=AXIS, lw=1.0, zorder=1)
    else:
        ax.axhline(0.0, color=AXIS, lw=1.0, zorder=1)


def figure_data_quality(review: ReviewResult, path: Path, source: str) -> None:
    """Country-by-quarter grid of missing cells and Eurostat flags, two panels."""
    plt = _pyplot()
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    start = str(review.manifest["period"]["start"])
    quarters = quarter_sequence(start, review.end)
    lfs = np.zeros((len(EU27), len(quarters)))
    gdp = np.zeros_like(lfs)
    lfs_code = {"b": 2, "bu": 2, "u": 3, "d": 4}
    for i, geo in enumerate(EU27):
        for j, quarter in enumerate(quarters):
            row = review.panel[(geo, quarter)]
            flags = review.flags[(geo, quarter)]
            if row["unemployment"] is None:
                lfs[i, j] = 1
            elif flags["lfs"]:
                lfs[i, j] = lfs_code.get(flags["lfs"], 2)
            if row["gdp"] is None:
                gdp[i, j] = 1
            elif flags["gdp"] == "b":
                gdp[i, j] = 2
            elif flags["gdp"] == "p":
                gdp[i, j] = 3
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 9.2), sharex=True)
    lfs_map = ListedColormap([SURFACE, NEUTRAL, ORANGE, BLUE, AQUA])
    gdp_map = ListedColormap([SURFACE, NEUTRAL, ORANGE, BLUE])
    for ax, grid, cmap, top, title in (
        (axes[0], lfs, lfs_map, 4, "Labour-force survey: unemployment and participation"),
        (axes[1], gdp, gdp_map, 3, "Quarterly real GDP growth"),
    ):
        ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=top, interpolation="nearest")
        ax.set_yticks(range(len(EU27)))
        ax.set_yticklabels(EU27, fontsize=7.5)
        ax.grid(False)
        ax.set_title(title)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        pandemic = [j for j, q in enumerate(quarters) if in_pandemic(q)]
        ax.axvspan(pandemic[0] - 0.5, pandemic[-1] + 0.5, color=MUTED, alpha=0.08, lw=0)
    ticks = [j for j, q in enumerate(quarters) if q.endswith("Q1") and int(q[:4]) % 2 == 0]
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels([quarters[j][:4] for j in ticks])
    axes[0].legend(handles=[
        Patch(color=NEUTRAL, label="Missing"),
        Patch(color=ORANGE, label="Break in series (b, bu)"),
        Patch(color=BLUE, label="Low reliability (u)"),
        Patch(color=AQUA, label="Definition differs (d)"),
    ], loc="upper left", bbox_to_anchor=(0.0, -0.01), ncol=4, fontsize=8)
    axes[1].legend(handles=[
        Patch(color=ORANGE, label="Break in series (b)"),
        Patch(color=BLUE, label="Provisional (p)"),
        Patch(facecolor=MUTED, alpha=0.2, label="2020-2021 (shaded)"),
    ], loc="upper left", bbox_to_anchor=(0.0, -0.06), ncol=3, fontsize=8)
    fig.subplots_adjust(hspace=0.22)
    _finish(plt, fig, path, source)


def figure_pandemic_influence(review: ReviewResult, path: Path, source: str) -> None:
    """Published full-sample sums against the same model without 2020-2021."""
    plt = _pyplot()
    from matplotlib.lines import Line2D

    lookup = grid_lookup(review.grid)
    fig, ax = plt.subplots(figsize=(7.6, 8.6))
    ys = np.arange(len(EU27))[::-1]
    for y, geo in zip(ys, EU27, strict=True):
        full = lookup[("full", geo, HEADLINE_LAGS, False)]
        excluded = lookup[("pandemic_excluded", geo, HEADLINE_LAGS, False)]
        ax.plot([excluded.ci_low, excluded.ci_high], [y - 0.16] * 2, color=BLUE, lw=2,
                solid_capstyle="round", alpha=0.35, zorder=2)
        ax.plot([full.gdp_sum, excluded.gdp_sum], [y, y - 0.16], color=NEUTRAL, lw=1, zorder=2)
        ax.plot([full.gdp_sum], [y], marker="o", ms=7, color=ORANGE, mec=SURFACE, mew=1.5,
                lw=0, zorder=4)
        ax.plot([excluded.gdp_sum], [y - 0.16], marker="o", ms=7, color=BLUE, mec=SURFACE,
                mew=1.5, lw=0, zorder=4)
    _zero_line(ax)
    ax.set_yticks(ys)
    ax.set_yticklabels(EU27)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Sum of GDP coefficients, lags 0-2 (pp of unemployment-rate change "
                  "per pp of quarterly growth)")
    ax.set_title("Including 2020-2021 pulls many coefficient sums towards zero")
    ax.legend(handles=[
        Line2D([], [], marker="o", color=ORANGE, lw=0, ms=7, label="2009-2026, as published"),
        Line2D([], [], marker="o", color=BLUE, lw=0, ms=7,
               label="2020-2021 excluded (bar: 95% HAC interval)"),
    ], loc="upper left", bbox_to_anchor=(0.0, -0.075), ncol=2, fontsize=8.5)
    _finish(plt, fig, path, source, credit_y=-0.045)


def figure_focus_scatter(review: ReviewResult, path: Path, source: str) -> None:
    """Unemployment-rate change against same-quarter GDP growth, by period."""
    plt = _pyplot()
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6))
    periods = (
        ("2009-2019", BLUE, lambda q: q <= "2019-Q4"),
        ("2020-2021", ORANGE, in_pandemic),
        ("2022 onwards", AQUA, lambda q: q >= "2022-Q1"),
    )
    quarters = quarter_sequence("2009-Q1", review.end)
    for ax, geo in zip(axes.ravel(), FOCUS, strict=True):
        for _, colour, member in periods:
            points = []
            for quarter in quarters:
                now = review.panel[(geo, quarter)]
                before = review.panel.get((geo, _previous(quarter)))
                if (member(quarter) and before is not None and now["gdp"] is not None
                        and now["unemployment"] is not None
                        and before["unemployment"] is not None):
                    points.append((now["gdp"], now["unemployment"] - before["unemployment"]))
            if points:
                xs, ys = zip(*points, strict=True)
                ax.plot(xs, ys, marker="o", ms=5.5, lw=0, color=colour, mec=SURFACE, mew=1,
                        alpha=0.9)
        _zero_line(ax)
        _zero_line(ax, vertical=False)
        ax.set_title(geo)
        ax.set_xlabel("Real GDP growth, % q/q")
        ax.set_ylabel("Change in unemployment rate, pp")
    fig.legend(handles=[
        Line2D([], [], marker="o", color=colour, lw=0, ms=6, label=label)
        for label, colour, _ in periods
    ], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(hspace=0.38, wspace=0.25, bottom=0.11)
    _finish(plt, fig, path, source, credit_y=-0.06)


def _previous(quarter: str) -> str:
    year, number = int(quarter[:4]), int(quarter[-1])
    return f"{year - 1}-Q4" if number == 1 else f"{year}-Q{number - 1}"


def figure_focus_sensitivity(review: ReviewResult, path: Path, source: str) -> None:
    """Coefficient sums for the four focus countries across samples and lag orders."""
    plt = _pyplot()
    from matplotlib.lines import Line2D

    lookup = grid_lookup(review.grid)
    samples = ("full", "pandemic_excluded", "pre_2020", "quality_screened")
    labels = {"full": "2009-26 incl. pandemic", "pandemic_excluded": "pandemic excluded",
              "pre_2020": "2009-2019", "quality_screened": "flags screened"}
    rows = [(s, k) for s in samples for k in (1, 2, 4)]
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 5.6), sharey=True)
    ys = np.arange(len(rows))[::-1]
    for ax, geo in zip(axes, FOCUS, strict=True):
        for offset, participation, colour in ((0.15, False, BLUE), (-0.15, True, ORANGE)):
            for y, (sample, lags) in zip(ys, rows, strict=True):
                row = lookup[(sample, geo, lags, participation)]
                ax.plot([row.ci_low, row.ci_high], [y + offset] * 2, color=colour, lw=1.6,
                        solid_capstyle="round", zorder=2)
                ax.plot([row.gdp_sum], [y + offset], marker="o", ms=5.5, color=colour,
                        mec=SURFACE, mew=1, lw=0, zorder=3)
        _zero_line(ax)
        ax.set_title(geo)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        for boundary in (2.5, 5.5, 8.5):
            ax.axhline(boundary, color=GRID, lw=0.8)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([f"{labels[s]}, K={k}" for s, k in rows], fontsize=8)
    fig.supxlabel("Sum of GDP coefficients with 95% HAC interval (pp per pp)", fontsize=9,
                  color=INK2)
    fig.legend(handles=[
        Line2D([], [], marker="o", color=BLUE, lw=1.6, ms=5.5, label="No participation control"),
        Line2D([], [], marker="o", color=ORANGE, lw=1.6, ms=5.5,
               label="Controlling for participation change"),
    ], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.subplots_adjust(wspace=0.08, bottom=0.14)
    _finish(plt, fig, path, source, credit_y=-0.11)


def figure_stability(review: ReviewResult, path: Path, source: str) -> None:
    """Pre-2020 and post-2021 sums (K=1) with the post leave-one-quarter-out range."""
    plt = _pyplot()
    from matplotlib.lines import Line2D

    rows = {r.geo: r for r in review.stability if not r.participation}
    fig, ax = plt.subplots(figsize=(7.6, 8.6))
    ys = np.arange(len(EU27))[::-1]
    for y, geo in zip(ys, EU27, strict=True):
        r = rows[geo]
        ax.plot([r.post_loo_min, r.post_loo_max], [y - 0.16] * 2, color=ORANGE, lw=2,
                alpha=0.4, solid_capstyle="round", zorder=2)
        ax.plot([r.pre_sum], [y + 0.16], marker="o", ms=7, color=BLUE, mec=SURFACE, mew=1.5,
                lw=0, zorder=3)
        ax.plot([r.post_sum], [y - 0.16], marker="o", ms=7, color=ORANGE, mec=SURFACE,
                mew=1.5, lw=0, zorder=3)
    _zero_line(ax)
    ax.set_yticks(ys)
    ax.set_yticklabels(EU27)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Sum of GDP coefficients, lags 0-1, no participation control (pp per pp)")
    ax.set_title("Post-2021 sums rest on 17 quarters; some flip sign when one is dropped")
    ax.legend(handles=[
        Line2D([], [], marker="o", color=BLUE, lw=0, ms=7, label="2009-2019"),
        Line2D([], [], marker="o", color=ORANGE, lw=0, ms=7,
               label="2022 onwards (bar: range when any one quarter is dropped)"),
    ], loc="upper left", bbox_to_anchor=(0.0, -0.075), ncol=1, fontsize=8.5)
    _finish(plt, fig, path, source, credit_y=-0.07)


def figure_forecast_skill(review: ReviewResult, path: Path, source: str,
                          model: str = "gdp_unemployment_ar1") -> None:
    """MAE skill of the AR(1)-plus-GDP model over AR(1), three evaluation sets."""
    plt = _pyplot()
    from matplotlib.lines import Line2D

    sets = (("published", ORANGE, "All targets, as published"),
            ("non_pandemic_targets", BLUE, "Targets outside 2020-Q1..2022-Q2"),
            ("pandemic_free", AQUA, "2020-2021 also removed from training"))
    fig, ax = plt.subplots(figsize=(7.6, 8.6))
    ys = np.arange(len(EU27))[::-1]
    for offset, (name, colour, _) in zip((0.22, 0.0, -0.22), sets, strict=True):
        values = review.forecast_skill[name][model]
        ax.plot([100 * values[g] for g in EU27], ys + offset, marker="o", ms=6, lw=0,
                color=colour, mec=SURFACE, mew=1.2, zorder=3)
    _zero_line(ax)
    ax.set_yticks(ys)
    ax.set_yticklabels(EU27)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Reduction in mean absolute error versus AR(1), % (right of zero = better)")
    ax.set_title("Lagged GDP adds little once pandemic quarters stop dominating the score")
    ax.legend(handles=[
        Line2D([], [], marker="o", color=colour, lw=0, ms=6, label=label)
        for _, colour, label in sets
    ], loc="upper left", bbox_to_anchor=(0.0, -0.075), ncol=1, fontsize=8.5)
    _finish(plt, fig, path, source, credit_y=-0.09)


FigureFunction = Callable[[ReviewResult, Path, str], None]
FIGURES: dict[str, FigureFunction] = {
    "data_quality.png": figure_data_quality,
    "pandemic_influence.png": figure_pandemic_influence,
    "focus_scatter.png": figure_focus_scatter,
    "focus_sensitivity.png": figure_focus_sensitivity,
    "post2021_stability.png": figure_stability,
    "forecast_skill.png": figure_forecast_skill,
}


def write_figures(review: ReviewResult, directory: Path, source: str) -> list[str]:
    """Render every review figure into ``directory`` and return the file names."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, function in FIGURES.items():
        function(review, directory / name, source)
    return list(FIGURES)
