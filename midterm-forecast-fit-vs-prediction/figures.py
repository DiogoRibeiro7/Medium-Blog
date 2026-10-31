"""Render nine original, source-linked visualizations of midterm model evidence.

All plotted coordinates come from ``analysis.compute_evidence``; no sourced
graphic is copied or edited. The exercise is reproducible and seed-fixed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from analysis import Evidence, ROOT, compute_evidence, influence, prediction_interval

Array = npt.NDArray[np.float64]
INK = "#183047"
BLUE = "#1e6988"
RED = "#ad4542"
GOLD = "#c58e37"
GREEN = "#387a68"
GRAY = "#647481"
LIGHT = "#e9eff2"
PAPER = "#fffefa"

FIGURE_FILES: tuple[str, ...] = (
    "fig1_historical_fit.png",
    "fig2_residuals_over_time.png",
    "fig3_leave_one_out.png",
    "fig4_forward_validation.png",
    "fig5_error_comparison.png",
    "fig6_2026_prediction_intervals.png",
    "fig7_polling_sensitivity.png",
    "fig8_influence_diagnostics.png",
    "fig9_scenario_heatmap.png",
)


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 16, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": GRAY, "axes.labelcolor": INK,
        "xtick.color": GRAY, "ytick.color": GRAY,
        "text.color": INK, "axes.titleweight": "bold",
        "figure.facecolor": PAPER, "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER, "grid.alpha": 0.2,
    })


def _footer(fig: Figure, text: str) -> None:
    fig.text(0.04, 0.015, text, color=GRAY, size=8, ha="left", va="bottom")


def _save(fig: Figure, path: Path) -> None:
    fig.savefig(path, dpi=185, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)


def _axis(ax: Axes, ylabel: str, xlabel: str) -> None:
    ax.set_xlabel(xlabel, labelpad=9)
    ax.set_ylabel(ylabel, labelpad=9)
    ax.grid(True, axis="y", linewidth=0.6, zorder=0)


def historical_fit(e: Evidence, dest: Path) -> None:
    """Figure 1: model-estimated versus observed seat changes, 18 elections."""

    fig, ax = plt.subplots(figsize=(10.0, 7.4))
    x = e.model.fitted
    y = e.model.outcomes
    limits = (-72.0, 18.0)
    ax.plot(limits, limits, ls="--", lw=1.6, color=GRAY, label="Perfect fitted value")
    for i, row in enumerate(e.elections):
        ax.scatter(x[i], y[i], s=100, color=BLUE if row.party == "D" else RED,
                   edgecolors=PAPER, lw=1.5, zorder=3)
        ax.annotate(str(row.year), (x[i], y[i]), xytext=(5, 5),
                    textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    _axis(ax, "Observed presidential-party House seat change", "In-sample fitted change")
    ax.set_title(f"A striking fit is only the beginning  |  R² = {e.model.r_squared:.3f}")
    ax.scatter([], [], color=BLUE, s=75, label="Democratic president")
    ax.scatter([], [], color=RED, s=75, label="Republican president")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    _footer(fig, "18 elections, 1950–2022; 1986 missing. Sources: Gallup, RCP, American Presidency Project.")
    _save(fig, dest)


def residual_timeline(e: Evidence, dest: Path) -> None:
    """Figure 2: sign and magnitude of historically fitted residuals."""

    fig, ax = plt.subplots(figsize=(11.0, 6.1))
    years = np.array([r.year for r in e.elections])
    residuals = e.model.residuals
    colors = [GREEN if r >= 0 else RED for r in residuals]
    ax.axhline(0, color=INK, lw=1)
    ax.bar(years, residuals, width=2.7, color=colors, alpha=0.87, zorder=3)
    for year, r in zip(years, residuals):
        ax.annotate(str(year), (year, r), xytext=(0, 4 if r >= 0 else -13),
                    textcoords="offset points", size=8, ha="center")
    ax.set_ylim(-20.5, 25.0)
    ax.set_xlim(1947, 2026)
    ax.set_xticks([1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020])
    _axis(ax, "Actual − fitted House seats", "Midterm election year")
    ax.set_title("The unexplained seats are substantive, not rounding errors")
    _footer(fig, "In-sample residuals; positive bars mean more seats than the model expected. No 1986 polling input.")
    _save(fig, dest)


def held_out_scatter(e: Evidence, dest: Path) -> None:
    """Figure 3: each election predicted without fitting its own outcome."""

    fig, ax = plt.subplots(figsize=(9.8, 7.2))
    limits = (-78.0, 20.0)
    ax.plot(limits, limits, ls="--", color=GRAY, lw=1.7, label="Perfect prediction")
    for row in e.leave_one_out:
        ax.scatter(row.predicted, row.actual, s=98, color=BLUE,
                   edgecolor=PAPER, lw=1.4, zorder=3)
        ax.annotate(str(row.year), (row.predicted, row.actual),
                    xytext=(5, 5), textcoords="offset points", size=8)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    _axis(ax, "Actual presidential-party seat change", "Leave-one-election-out prediction")
    ax.set_title(f"Held-out predictions  |  RMSE {e.metrics['loocv_rmse']:.1f} seats")
    ax.legend(loc="upper left", frameon=False)
    _footer(fig, "Each fitted point excludes that year's outcome, but the training set may contain later years.")
    _save(fig, dest)


def forward_rolling(e: Evidence, dest: Path) -> None:
    """Figure 4: expanding-origin temporal validation, no future leakage."""

    fig, ax = plt.subplots(figsize=(11.2, 6.7))
    years: Array = np.array([row.year for row in e.forward], dtype=float)
    y: Array = np.array([row.actual for row in e.forward], dtype=float)
    predicted: Array = np.array([row.predicted for row in e.forward], dtype=float)
    baseline: Array = np.array([row.baseline for row in e.forward], dtype=float)
    ax.plot(years, y, "o-", color=INK, lw=2.2, ms=7, label="Observed result", zorder=4)
    ax.plot(years, predicted, "s-", color=BLUE, lw=2.0, ms=6,
            label="Model, fitted only on earlier elections", zorder=3)
    ax.plot(years, baseline, "--", color=GOLD, lw=1.9,
            label="Earlier-election mean baseline")
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xticks(years)
    ax.tick_params(axis="x", rotation=45)
    _axis(ax, "Presidential-party House seat change", "Held-out election year")
    ax.set_title("Forward-in-time testing: can the relationship travel?")
    ax.legend(frameon=False, loc="lower left", fontsize=10)
    _footer(fig, f"Initial 8-election training window; {len(e.forward)} later forecasts. 1986 omitted, never imputed.")
    _save(fig, dest)


def error_comparison(e: Evidence, dest: Path) -> None:
    """Figure 5: distinguish in-sample and genuine predictive error scales."""

    labels = ["In-sample OLS", "Leave-one-out", "Forward-in-time", "Forward mean\nbaseline"]
    values = [e.metrics["in_sample_rmse"], e.metrics["loocv_rmse"],
              e.metrics["forward_rmse"], e.metrics["forward_mean_baseline_rmse"]]
    fig, ax = plt.subplots(figsize=(10.3, 6.1))
    bars = ax.barh(np.arange(4), values, color=[GRAY, BLUE, GREEN, GOLD],
                   edgecolor=PAPER, height=0.65, zorder=3)
    ax.set_yticks(np.arange(4), labels=labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.18)
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + 0.45, bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}", va="center", color=INK, weight="bold")
    ax.set_xlabel("Root mean squared error in House seats (lower is better)")
    ax.set_title("Fit, resampling and future-facing tests answer different questions")
    ax.grid(True, axis="x", alpha=0.2)
    _footer(fig, "In-sample and LOOCV use 18 elections; forward model and baseline use the same 10 later elections.")
    _save(fig, dest)


def forecast_intervals(e: Evidence, dest: Path) -> None:
    """Figure 6: 2026 point prediction and two assumption-dependent intervals."""

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    point: float = e.metrics["forecast"]
    methods = [
        ("Student-t regression, 95%", e.metrics["parametric_95_interval"], BLUE, 2.0),
        ("Pairs + residual bootstrap, 95%", e.metrics["bootstrap_95_interval"], GREEN, 1.0),
    ]
    for label, (lower, upper), color, y in methods:
        ax.plot([lower, upper], [y, y], lw=8, solid_capstyle="round", color=color, alpha=0.86)
        ax.plot([lower, lower], [y - 0.15, y + 0.15], color=color, lw=2)
        ax.plot([upper, upper], [y - 0.15, y + 0.15], color=color, lw=2)
        ax.scatter([point], [y], marker="D", s=85, color=INK, zorder=5)
        ax.text(upper + 1, y, f"{lower:.1f} to {upper:.1f}", va="center", size=9)
    ax.axvline(-33, ls="--", color=RED, lw=1.8, label="Rattner: −33")
    ax.set_yticks([2.0, 1.0], [item[0] for item in methods])
    ax.set_xlim(-65, 0)
    ax.set_ylim(0.45, 2.52)
    ax.set_title(f"2026: point estimate {point:.1f} seats is not a probability distribution")
    ax.set_xlabel("Predicted change in Republican House seats")
    ax.grid(True, axis="x", alpha=0.22)
    ax.legend(frameon=False, loc="upper right")
    _footer(fig, "Intervals are conditional on model/sampling assumptions; the bootstrap is not regime-shift robust.")
    _save(fig, dest)


def polling_sensitivity(e: Evidence, dest: Path) -> None:
    """Figure 7: what changing the generic-ballot assumption does to seats."""

    fig, ax = plt.subplots(figsize=(10.3, 6.3))
    advantages: Array = np.linspace(2.0, 14.0, 49)
    for seats, color, style in ((210, GOLD, "--"), (220, BLUE, "-"), (230, GREEN, "--")):
        predictions = [e.model.predict(-float(margin), float(seats)) for margin in advantages]
        ax.plot(advantages, predictions, style, lw=2.4, color=color,
                label=f"{seats} Republican seats defended")
        if seats == 220:
            intervals: Array = np.array([
                prediction_interval(e.model, -float(margin), float(seats), 0.95)
                for margin in advantages
            ])
            ax.fill_between(advantages, intervals[:, 0], intervals[:, 1],
                            color=BLUE, alpha=0.12, label="95% model-based PI, 220 seats")
    ax.scatter([8], [e.metrics["forecast"]], marker="*", s=240,
               color=RED, edgecolor=PAPER, zorder=5, label="Rattner D+8 scenario")
    _axis(ax, "Predicted Republican House seat change", "Democratic generic-ballot lead (points)")
    ax.set_title("A forecast is conditional on its polling input")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    _footer(fig, "Other inputs held fixed; scenarios are alternatives, not forecasts of polling movement.")
    _save(fig, dest)


def influence_diagnostics(e: Evidence, dest: Path) -> None:
    """Figure 8: leverage, residual and Cook's-distance of each election."""

    fig, ax = plt.subplots(figsize=(10.3, 6.4))
    leverages, standardized, cooks = influence(e.model)
    for i, election in enumerate(e.elections):
        ax.scatter(leverages[i], standardized[i], s=95 + 1050 * cooks[i],
                   color=BLUE if election.party == "D" else RED,
                   alpha=0.78, edgecolor=PAPER, lw=1.1, zorder=3)
        ax.annotate(str(election.year), (leverages[i], standardized[i]),
                    xytext=(5, 5), textcoords="offset points", size=8)
    ax.axhline(0, color=INK, lw=0.85)
    ax.axhline(2, color=GRAY, lw=0.9, ls="--")
    ax.axhline(-2, color=GRAY, lw=0.9, ls="--")
    ax.set_xlim(0, max(leverages) * 1.23)
    _axis(ax, "Internally studentized residual", "OLS leverage (hat diagonal)")
    ax.set_title("An 18-election regression has no expendable observations")
    _footer(fig, "Bubble area scales with Cook's distance, a diagnostic rather than a causal test.")
    _save(fig, dest)


def scenario_heatmap(e: Evidence, dest: Path) -> None:
    """Figure 9: two-dimensional conditional 2026 point-forecast surface."""

    fig, ax = plt.subplots(figsize=(10.0, 6.9))
    advantages: Array = np.linspace(4.0, 12.0, 101)
    defended: Array = np.linspace(210.0, 230.0, 101)
    xx, yy = np.meshgrid(advantages, defended)
    zz: Array = np.asarray(
        e.model.coefficients[0] - e.model.coefficients[1] * xx
        + e.model.coefficients[2] * yy,
        dtype=float,
    )
    heat = ax.contourf(xx, yy, zz, levels=16, cmap="RdBu_r", alpha=0.87)
    ax.contour(xx, yy, zz, levels=[-40, -35, -30, -25], colors=INK,
               alpha=0.7, linewidths=0.9)
    ax.scatter([8], [220], marker="*", s=250, facecolor=GOLD,
               edgecolor=INK, lw=1.2, zorder=6)
    ax.annotate(f"Oct 30 input\n{e.metrics['forecast']:.1f} seats", (8, 220),
                xytext=(18, 16), textcoords="offset points",
                bbox={"facecolor": PAPER, "edgecolor": LIGHT, "alpha": 0.95}, size=10)
    ax.set_title("Two assumptions, one conditional prediction surface")
    _axis(ax, "Republican House seats defended", "Democratic generic-ballot lead (points)")
    cb = fig.colorbar(heat, ax=ax, shrink=0.9, pad=0.03)
    cb.set_label("Predicted change in Republican seats")
    _footer(fig, "Grid is an input sensitivity analysis; it is not a probability distribution of possible elections.")
    _save(fig, dest)


def generate_figures(evidence: Evidence, output: Path) -> tuple[Path, ...]:
    """Create every figure deterministically and return the verified paths."""

    if not isinstance(output, Path):
        raise TypeError("output must be a pathlib.Path")
    output.mkdir(parents=True, exist_ok=True)
    _style()
    functions = (
        historical_fit, residual_timeline, held_out_scatter, forward_rolling,
        error_comparison, forecast_intervals, polling_sensitivity,
        influence_diagnostics, scenario_heatmap,
    )
    for function, filename in zip(functions, FIGURE_FILES, strict=True):
        function(evidence, output / filename)
    paths: tuple[Path, ...] = tuple(output / name for name in FIGURE_FILES)
    if not all(path.exists() and path.stat().st_size > 10_000 for path in paths):
        raise OSError("A figure was not correctly generated")
    return paths


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT, help="Output root with figures/")
    parser.add_argument("--draws", type=int, default=5000)
    args: argparse.Namespace = parser.parse_args()
    files = generate_figures(compute_evidence(draws=args.draws), args.output / "figures")
    print(f"Generated {len(files)} original figures in {args.output / 'figures'}")


if __name__ == "__main__":
    main()
