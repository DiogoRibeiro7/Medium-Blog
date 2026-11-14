"""Build four original, schematic illustrations for the causal inference article.

All numeric values were constructed to demonstrate identification problems.
The figures are teaching aids, not empirical estimates. Both PNG and SVG
formats are written for each graphic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch

OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "figures"
INK = "#16283C"
MUTED = "#617184"
TEAL = "#147D7B"
CORAL = "#D95E4C"
GOLD = "#E5AA42"
BLUE = "#4779AD"
PALE = "#E8EFF3"
BACKGROUND = "#F8FAFB"
WHITE = "#FFFFFF"


def configure_style() -> None:
    """Apply a high-contrast, publication-friendly design."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "axes.edgecolor": PALE,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "savefig.facecolor": BACKGROUND,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: Figure, stem: str) -> None:
    """Write bitmap and vector versions of a figure."""
    if not isinstance(stem, str) or not stem:
        raise ValueError("Figure stem must be a nonempty string.")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIRECTORY / f"{stem}.png", dpi=190, pad_inches=0.24)
    fig.savefig(OUTPUT_DIRECTORY / f"{stem}.svg", pad_inches=0.24)
    plt.close(fig)


def remove_spines(ax: Axes) -> None:
    """Hide decorative boundaries and tick marks."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)


def make_identification_figure() -> None:
    """Show that selection alone can produce a large retention gap."""
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.7))
    fig.suptitle(
        "The retention gap is real. The causal story is not identified.",
        x=0.055, ha="left", fontsize=19, fontweight="bold", color=INK, y=0.98,
    )

    observed_ax = axes[0]
    groups = ["Adopters", "Non-adopters"]
    values = np.array([65.0, 50.0], dtype=float)
    observed_ax.bar(groups, values, color=[TEAL, CORAL], width=0.58)
    observed_ax.set_ylim(0, 82)
    observed_ax.set_ylabel("Observed retention (%)")
    observed_ax.set_title("What the dashboard shows", loc="left", pad=22, fontsize=14)
    observed_ax.set_yticks([0, 20, 40, 60, 80])
    observed_ax.yaxis.grid(True, color=PALE, zorder=0)
    observed_ax.set_axisbelow(True)
    for index, percentage in enumerate(values):
        observed_ax.text(
            index, percentage + 2.2, f"{percentage:.0f}%",
            ha="center", va="bottom", fontsize=19, fontweight="bold",
        )
    observed_ax.text(
        0.5, 76, "30% higher relative retention", color=TEAL,
        fontweight="bold", ha="center", fontsize=11,
    )
    remove_spines(observed_ax)

    composition_ax = axes[1]
    positions = np.arange(2, dtype=float)
    high_share = np.array([80.0, 20.0])
    low_share = 100.0 - high_share
    composition_ax.bar(
        positions, high_share, color=BLUE, width=0.58,
        label="Highly engaged users",
    )
    composition_ax.bar(
        positions, low_share, bottom=high_share, color=GOLD,
        width=0.58, label="Casually engaged users",
    )
    composition_ax.set_xticks(positions, groups)
    composition_ax.set_ylim(0, 113)
    composition_ax.set_yticks([0, 25, 50, 75, 100])
    composition_ax.set_ylabel("Composition of each group (%)")
    composition_ax.set_title(
        "One explanation: different user composition",
        loc="left", pad=22, fontsize=14,
    )
    for index in range(2):
        composition_ax.text(
            index, high_share[index] / 2, f"{high_share[index]:.0f}%",
            ha="center", va="center", color=WHITE,
            fontweight="bold", fontsize=14,
        )
        composition_ax.text(
            index, high_share[index] + low_share[index] / 2,
            f"{low_share[index]:.0f}%", ha="center", va="center",
            color=INK, fontweight="bold", fontsize=14,
        )
    composition_ax.legend(
        frameon=False, loc="upper center",
        bbox_to_anchor=(0.50, -0.12), fontsize=10,
    )
    remove_spines(composition_ax)
    fig.text(
        0.06, 0.025,
        "Constructed example: neither segment benefits. Selection alone "
        "generates the 65% vs 50% gap.",
        fontsize=11, color=MUTED,
    )
    fig.subplots_adjust(
        left=0.09, right=0.97, bottom=0.22, top=0.80, wspace=0.35,
    )
    save_figure(fig, "01_same_numbers_different_worlds")


def make_universal_rollout_figure() -> None:
    """Show rival counterfactuals after a universal launch."""
    fig, ax = plt.subplots(figsize=(13.0, 6.5))
    weeks = np.arange(-5, 6, dtype=int)
    observed = np.array(
        [49.6, 49.9, 50.1, 50.0, 50.2, 50.1, 51.5, 52.5, 54.0, 55.0, 55.5],
        dtype=float,
    )
    after_launch = np.arange(0, 6, dtype=int)
    flat_path = np.array([50.1, 50.0, 50.0, 50.0, 50.0, 50.0])
    seasonal_path = np.array([50.1, 51.2, 52.4, 53.6, 54.7, 55.4])
    ax.axvspan(0.5, 5.35, facecolor=PALE, alpha=0.8, zorder=0)
    ax.axvline(0.5, color=MUTED, linestyle=":", lw=1.7)
    ax.plot(
        weeks, observed, color=TEAL, marker="o", linewidth=3,
        markersize=7, label="Observed retention",
    )
    ax.plot(
        after_launch, flat_path, color=CORAL, linestyle="--",
        linewidth=2.6, label="Possible untreated path A",
    )
    ax.plot(
        after_launch, seasonal_path, color=GOLD, linestyle="--",
        linewidth=2.6, label="Possible untreated path B",
    )
    ax.set_xlim(-5.35, 5.35)
    ax.set_ylim(47.8, 58)
    ax.set_xticks(weeks)
    ax.set_ylabel("Retention (%)")
    ax.set_xlabel("Weeks relative to launch")
    ax.set_title(
        "One universal rollout. Several counterfactual stories.",
        loc="left", fontsize=19, fontweight="bold", pad=24,
    )
    ax.text(
        0.65, 57.1, "FEATURE LAUNCH", fontsize=11,
        color=MUTED, fontweight="bold",
    )
    ax.text(-5.1, 56.2, "Pre-launch data", fontsize=12, color=MUTED)
    ax.annotate(
        "The dashed paths are hypotheses.\nObserved data cannot choose.",
        xy=(3.5, 50.1), xytext=(-4.7, 53.5),
        fontsize=11, color=INK,
        arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.3},
        bbox={"boxstyle": "round,pad=0.5", "fc": WHITE, "ec": PALE},
    )
    ax.yaxis.grid(True, color=PALE)
    ax.set_axisbelow(True)
    ax.legend(
        ncol=3, frameon=False, loc="upper left",
        bbox_to_anchor=(0.0, -0.16), fontsize=11,
    )
    remove_spines(ax)
    fig.subplots_adjust(
        left=0.07, right=0.98, top=0.86, bottom=0.24,
    )
    save_figure(fig, "02_universal_rollout")


def draw_node(ax: Axes, x: float, y: float, name: str, color: str) -> None:
    """Draw a text node with a rounded background."""
    ax.text(
        x, y, name, ha="center", va="center", fontsize=10.5,
        color=WHITE if color != PALE else INK, fontweight="bold",
        bbox={"boxstyle": "round,pad=0.62", "fc": color, "ec": "none"},
        zorder=5,
    )


def draw_arrow(
    ax: Axes,
    start: tuple[float, float],
    finish: tuple[float, float],
    color: str = MUTED,
) -> None:
    """Draw an arrow in fractional axes coordinates."""
    ax.add_patch(
        FancyArrowPatch(
            start, finish, arrowstyle="-|>", mutation_scale=17,
            linewidth=2.0, color=color, zorder=2,
            transform=ax.transAxes,
        )
    )


def make_control_selection_figure() -> None:
    """Contrast confounder, mediator, and collider DAGs."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
    fig.suptitle(
        "A variable's causal role matters more than its predictive score.",
        x=0.04, ha="left", y=0.98, fontsize=19, fontweight="bold",
    )
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    conf = axes[0]
    conf.set_title("1  Confounder", fontsize=15, loc="left", pad=20)
    draw_arrow(conf, (0.50, 0.70), (0.22, 0.37))
    draw_arrow(conf, (0.53, 0.70), (0.80, 0.37))
    draw_arrow(conf, (0.29, 0.32), (0.73, 0.32), color=TEAL)
    draw_node(conf, 0.50, 0.76, "Prior engagement", BLUE)
    draw_node(conf, 0.20, 0.32, "Adoption", TEAL)
    draw_node(conf, 0.79, 0.32, "Retention", CORAL)
    conf.text(
        0.5, 0.08, "Pre-treatment adjustment may\nblock a backdoor path.",
        ha="center", color=MUTED, fontsize=11,
    )

    med = axes[1]
    med.set_title("2  Mediator", fontsize=15, loc="left", pad=20)
    draw_arrow(med, (0.25, 0.51), (0.42, 0.51), color=TEAL)
    draw_arrow(med, (0.56, 0.51), (0.74, 0.51), color=TEAL)
    draw_node(med, 0.18, 0.51, "Feature", TEAL)
    draw_node(med, 0.49, 0.51, "Daily habit", BLUE)
    draw_node(med, 0.83, 0.51, "Retention", CORAL)
    med.text(
        0.5, 0.08, "Controlling the mechanism can\nremove part of the total effect.",
        ha="center", color=MUTED, fontsize=11,
    )

    col = axes[2]
    col.set_title("3  Collider", fontsize=15, loc="left", pad=20)
    draw_arrow(col, (0.26, 0.70), (0.44, 0.48), color=TEAL)
    draw_arrow(col, (0.74, 0.70), (0.56, 0.48))
    draw_arrow(col, (0.78, 0.67), (0.78, 0.30), color=CORAL)
    draw_node(col, 0.20, 0.74, "Feature", TEAL)
    draw_node(col, 0.79, 0.74, "Motivation", BLUE)
    draw_node(col, 0.50, 0.45, "Activity", GOLD)
    draw_node(col, 0.79, 0.25, "Retention", CORAL)
    col.text(
        0.5, 0.08, "Conditioning on activity may\nopen a noncausal path.",
        ha="center", color=MUTED, fontsize=11,
    )
    fig.subplots_adjust(
        left=0.045, right=0.98, top=0.83, bottom=0.06, wspace=0.19,
    )
    save_figure(fig, "03_which_controls")


def make_did_pretrends_figure() -> None:
    """Explain non-parallel trends with a constructed two-group example."""
    fig, ax = plt.subplots(figsize=(13.0, 6.7))
    times = np.array([-2, -1, 0, 1], dtype=int)
    treated = np.array([30.0, 40.0, 50.0, 65.0])
    control = np.array([30.0, 35.0, 40.0, 45.0])
    ax.axvspan(0.5, 1.32, facecolor=PALE, alpha=0.95)
    ax.axvline(0.5, color=MUTED, lw=1.5, linestyle=":")
    ax.plot(
        times, treated, color=TEAL, marker="o", linewidth=3.0,
        markersize=9, label="Treated segment (observed)",
    )
    ax.plot(
        times, control, color=BLUE, marker="s", linewidth=3.0,
        markersize=7, label="Control segment (observed)",
    )
    ax.plot(
        [0, 1], [50, 60], color=CORAL, linestyle="--",
        linewidth=2.7, marker="x", markersize=10,
        label="Treated without launch (hypothesis)",
    )
    ax.annotate(
        "Illustrative causal effect:\n65% − 60% = 5 pp",
        xy=(1, 61.8), xytext=(-0.9, 68.3), fontsize=11,
        color=CORAL, fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": CORAL, "lw": 1.5},
    )
    ax.text(
        -1.9, 58, "Treated pre-trend: +10 pp / period",
        color=TEAL, fontsize=11, fontweight="bold",
    )
    ax.text(
        -1.9, 54, "Control pre-trend: +5 pp / period",
        color=BLUE, fontsize=11, fontweight="bold",
    )
    ax.text(0.56, 27.5, "POST", color=MUTED, fontsize=10.5, fontweight="bold")
    ax.set_xticks(times, ["−2", "−1", "0", "+1"])
    ax.set_xlim(-2.25, 1.34)
    ax.set_ylim(25, 74)
    ax.set_ylabel("Retention (%)")
    ax.set_xlabel("Periods relative to launch")
    ax.set_title(
        "Difference-in-differences needs a counterfactual trend.",
        loc="left", fontsize=19, fontweight="bold", pad=24,
    )
    ax.yaxis.grid(True, color=PALE)
    ax.set_axisbelow(True)
    ax.legend(
        ncol=2, frameon=False, loc="upper left",
        bbox_to_anchor=(0.0, -0.15), fontsize=10.5,
    )
    remove_spines(ax)
    fig.subplots_adjust(
        left=0.07, right=0.98, top=0.85, bottom=0.26,
    )
    save_figure(fig, "04_nonparallel_trends")


def main() -> None:
    """Create all article graphics with deterministic inputs."""
    builders: tuple[Callable[[], None], ...] = (
        make_identification_figure,
        make_universal_rollout_figure,
        make_control_selection_figure,
        make_did_pretrends_figure,
    )
    configure_style()
    for builder in builders:
        builder()
    print(f"Created {len(builders)} figure pairs.")


if __name__ == "__main__":
    main()
