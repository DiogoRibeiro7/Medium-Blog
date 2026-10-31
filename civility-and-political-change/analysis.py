"""Reproducible illustrations for the essay on civility and political change.

This module produces *hypothetical* bargaining and delay illustrations, plus
conceptual diagrams. It neither estimates historical effects nor fits data.

Run ``python analysis.py`` from this directory. Python 3.10+ is required.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


INK = "#192331"
BLUE = "#147b93"
RED = "#ba493b"
GOLD = "#cc923d"
MUTED = "#667789"
PALE = "#e7eff2"
CREAM = "#fcfaf6"
FIGURE_NAMES: tuple[str, ...] = (
    "fig1_midpoint_is_not_justice.png",
    "fig2_politeness_and_disruption.png",
    "fig3_hypothesized_mechanisms.png",
    "fig4_outside_options.png",
    "fig5_bargaining_sensitivity.png",
    "fig6_cost_of_procedural_restraint.png",
    "fig7_cost_of_waiting.png",
    "fig8_what_evidence_can_establish.png",
)


@dataclass(frozen=True)
class BargainingScenario:
    """A hypothetical normalized two-party transferable-surplus game."""

    label: str
    reformer_outside: float
    incumbent_outside: float
    weight: float = 0.5


def _unit(value: float, name: str) -> None:
    """Require a finite real number in the closed unit interval."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")


def bargaining_solution(
    reformer_outside: float,
    incumbent_outside: float,
    reformer_weight: float = 0.5,
) -> tuple[float, float]:
    """Solve a weighted Nash split of a *transferable* normalized unit surplus.

    The feasible pairs satisfy ``r+i=1``, ``r>=d_r`` and ``i>=d_i``.
    The weighted Nash product maximized is

        (r-d_r)**weight * (i-d_i)**(1-weight).

    For positive residual surplus the solution is
    ``r=d_r + weight*(1-d_r-d_i)``. The closed-form limiting solution
    also handles endpoint weights and a zero residual surplus.

    This is a stylized allocation model, not an estimate of political rights,
    welfare, or historical events. Transferable utility is a strong assumption.
    """
    _unit(reformer_outside, "reformer_outside")
    _unit(incumbent_outside, "incumbent_outside")
    _unit(reformer_weight, "reformer_weight")
    residual = 1.0 - reformer_outside - incumbent_outside
    if residual < -1e-12:
        raise ValueError("Disagreement utilities exceed feasible surplus")
    # Avoid a negative residual induced by round-off at the feasible boundary.
    r = reformer_outside + reformer_weight * max(0.0, residual)
    return r, 1.0 - r


def effective_outside_option(
    baseline: float, procedural_cost: float, exposure: float = 0.3
) -> float:
    """Hypothetical reduction in a reformer's outside option from a restraint cost."""
    _unit(baseline, "baseline")
    _unit(procedural_cost, "procedural_cost")
    _unit(exposure, "exposure")
    return max(0.0, baseline - procedural_cost * exposure)


def discounted_delay_loss(years: float, annual_harm: float, rate: float) -> float:
    """Compute integral of constant flow harm over a delay, in normalized units."""
    for name, val in (("years", years), ("annual_harm", annual_harm), ("rate", rate)):
        if isinstance(val, bool) or not isinstance(val, (float, int)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(val) or val < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if rate == 0:
        return years * annual_harm
    # expm1 provides a stable value near zero.
    return annual_harm * (-math.expm1(-rate * years)) / rate


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": CREAM,
            "axes.facecolor": CREAM,
            "savefig.facecolor": CREAM,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )


def _save(fig: Figure, destination: Path) -> None:
    fig.savefig(destination, dpi=190, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def _box(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    color: str,
) -> None:
    rect = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.015,rounding_size=0.035",
        facecolor=color,
        edgecolor="none",
        transform=ax.transAxes,
    )
    ax.add_patch(rect)
    ax.text(x + width / 2, y + height * 0.65, title, ha="center", va="center",
            fontsize=13, fontweight="bold", color="white", transform=ax.transAxes)
    ax.text(x + width / 2, y + height * 0.29, body, ha="center", va="center",
            fontsize=9, color="white", transform=ax.transAxes, linespacing=1.3)


def figure1(path: Path) -> None:
    """Show why the arithmetic political midpoint is not a rights assessment."""
    fig, ax = plt.subplots(figsize=(11.2, 4.1))
    ax.set_xlim(-0.07, 1.07)
    ax.set_ylim(-0.55, 1.1)
    ax.hlines(0.35, 0, 1, lw=8, color=PALE, zorder=1)
    for x, label, col, vertical in (
        (0.0, "Existing rule", RED, 0.75),
        (0.5, "Arithmetic midpoint", GOLD, -0.23),
        (0.8, "Illustrative rights floor", BLUE, 0.75),
        (1.0, "Full reform proposal", INK, -0.23),
    ):
        ax.scatter([x], [0.35], color=col, s=165, zorder=3)
        ax.plot([x, x], [0.35, vertical - (0.07 if vertical > 0 else -0.07)],
                lw=1.4, color=col)
        ax.text(x, vertical, label, ha="center", va="center", fontsize=10,
                fontweight="bold" if x in (0.5, 0.8) else "normal", color=col)
    ax.text(0.5, -0.48, "Hypothetical policy axis; the 0.8 rights floor is a normative assumption, not data.",
            ha="center", fontsize=9, color=MUTED)
    ax.set_title("A midpoint answers an arithmetic question, not a justice question", fontsize=16, pad=15)
    ax.axis("off")
    _save(fig, path)


def figure2(path: Path) -> None:
    """Separately map interpersonal manners and public political disruption."""
    fig, ax = plt.subplots(figsize=(9.4, 6.2))
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, color="#fae4df"))
    ax.add_patch(patches.Rectangle((1, 0), 1, 1, color="#e4edf1"))
    ax.add_patch(patches.Rectangle((0, 1), 1, 1, color="#e8f0eb"))
    ax.add_patch(patches.Rectangle((1, 1), 1, 1, color="#d1e6ea"))
    ax.axhline(1, color="white", lw=4)
    ax.axvline(1, color="white", lw=4)
    cells = (
        (0.5, 1.5, "Polite + low disruption", "Conventional meetings"),
        (1.5, 1.5, "Polite + high disruption", "Nonviolent sit-ins / strikes"),
        (0.5, 0.5, "Impolite + low disruption", "Insult without organisation"),
        (1.5, 0.5, "Impolite + high disruption", "Confrontation + mobilisation"),
    )
    for x, y, title, example in cells:
        ax.text(x, y + 0.09, title, ha="center", fontsize=11, fontweight="bold")
        ax.text(x, y - 0.1, example, ha="center", fontsize=9, color=MUTED)
    ax.set_xticks((0.5, 1.5), ("Low disruption", "High disruption"))
    ax.set_yticks((0.5, 1.5), ("Impolite", "Polite"))
    ax.set_xlabel("Institutional / tactical disruption", labelpad=15)
    ax.set_ylabel("Interpersonal manner", labelpad=15)
    ax.set_title("Politeness and political disruption are different dimensions", fontsize=15, pad=15)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    _save(fig, path)


def figure3(path: Path) -> None:
    """Draw two *hypothesized*, unestimated pathways from protest to policy."""
    fig, ax = plt.subplots(figsize=(12.0, 6.3))
    ax.set_axis_off()
    _box(ax, 0.035, 0.62, 0.18, 0.28, "Grievance", "Public claim of\ninjustice", INK)
    _box(ax, 0.29, 0.62, 0.18, 0.28, "Mobilisation", "Strikes, sit-ins,\nboycotts", BLUE)
    _box(ax, 0.545, 0.62, 0.18, 0.28, "Leverage", "Cost of refusing\nnegotiation rises", BLUE)
    _box(ax, 0.80, 0.62, 0.17, 0.28, "Reform", "Enforceable\ncommitment", BLUE)
    _box(ax, 0.29, 0.12, 0.18, 0.28, "Civility appeal", "Conduct replaces\nsubstance", RED)
    _box(ax, 0.545, 0.12, 0.18, 0.28, "Pressure falls", "Disruption is\ncurtailed", RED)
    _box(ax, 0.80, 0.12, 0.17, 0.28, "Delay", "Status quo\ncontinues", RED)
    def arrow(x1: float, y1: float, x2: float, y2: float, color: str) -> None:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), xycoords="axes fraction",
                    arrowprops={"arrowstyle": "-|>", "lw": 2.5, "color": color})
    for x1, x2 in ((0.215, 0.29), (0.47, 0.545), (0.725, 0.80)):
        arrow(x1, 0.76, x2, 0.76, BLUE)
    arrow(0.375, 0.62, 0.375, 0.40, RED)
    arrow(0.47, 0.26, 0.545, 0.26, RED)
    arrow(0.725, 0.26, 0.80, 0.26, RED)
    ax.text(0.5, 0.975, "Hypothesized mechanisms — not a causal estimate", ha="center",
            va="top", fontsize=16, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.02, "Other influences (courts, elections, media, repression, coalitions) are omitted.",
            ha="center", fontsize=9, color=MUTED, transform=ax.transAxes)
    _save(fig, path)


def _scenarios() -> list[BargainingScenario]:
    return [
        BargainingScenario("Incumbent advantage", 0.05, 0.55),
        BargainingScenario("Balanced outside options", 0.30, 0.30),
        BargainingScenario("Reformer advantage", 0.55, 0.05),
    ]


def figure4(path: Path, scenarios: list[BargainingScenario]) -> None:
    """Show how equal bargaining weights can hide unequal outside options."""
    fig, ax = plt.subplots(figsize=(11.1, 5.0))
    names = [s.label for s in scenarios]
    r = [bargaining_solution(s.reformer_outside, s.incumbent_outside, s.weight)[0]
         for s in scenarios]
    y = np.arange(len(names))
    ax.barh(y, r, color=BLUE, height=0.57, label="Reformer share")
    ax.barh(y, [1-v for v in r], left=r, color=GOLD, height=0.57, label="Incumbent share")
    for yi, val in zip(y, r):
        ax.text(val / 2, yi, f"{val:.0%}", ha="center", va="center",
                color="white", fontweight="bold", fontsize=12)
        ax.text((1 + val) / 2, yi, f"{1-val:.0%}", ha="center", va="center",
                color="white", fontweight="bold", fontsize=12)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of a hypothetical transferable unit surplus")
    ax.set_title("Equal formal weights do not imply equal outcomes", fontsize=16, pad=12)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.33), ncol=2, frameon=False)
    fig.text(0.53, -0.015, "Assumed Nash weights: 0.5 / 0.5. Outside options vary; no historical data.",
             ha="center", color=MUTED, fontsize=9)
    _save(fig, path)


def figure5(path: Path, grid: list[dict[str, float]]) -> None:
    """Plot algebraic bargaining sensitivity to outside option and weight."""
    values = np.array([v["reformer_share"] for v in grid], dtype=float).reshape(51, 51)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(values, origin="lower", aspect="auto", extent=[0.1, 0.9, 0, 0.4],
                   cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xlabel("Reformer bargaining weight")
    ax.set_ylabel("Reformer outside option")
    ax.set_title("The outcome depends on leverage and the disagreement point", fontsize=15, pad=14)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Reformer share")
    ax.text(0.11, 0.375, "Incumbent outside option fixed at 0.4", color=INK, fontsize=9,
            bbox={"boxstyle": "round", "facecolor": CREAM, "alpha": 0.84, "edgecolor": "none"})
    _save(fig, path)


def figure6(path: Path, costs: list[dict[str, float]]) -> None:
    """Illustrate hypothetical pressure/loss of bargaining leverage."""
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    c = [v["restraint_cost"] for v in costs]
    ax.plot(c, [v["effective_outside"] for v in costs], color=RED, lw=3,
            label="Effective reformer outside option")
    ax.plot(c, [v["reformer_share"] for v in costs], color=BLUE, lw=3,
            label="Negotiated reformer share")
    ax.set_xlabel("Hypothetical restraint intensity (0–1)")
    ax.set_ylabel("Normalized utility")
    ax.set_ylim(0, 0.56)
    ax.grid(alpha=0.15)
    ax.set_title("When procedural restraint lowers leverage: an assumed mechanism", fontsize=15)
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.02, 0.045, "Not measured; assumed dR(c) = max(0, 0.35 − 0.30c), dI = 0.45, weight = 0.5.",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    _save(fig, path)


def figure7(path: Path, costs: list[dict[str, float]]) -> None:
    """Display cumulative discounted harm of delayed action under specified rates."""
    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    years = np.linspace(0, 20, 201)
    for rate, col in ((0.0, INK), (0.03, BLUE), (0.08, GOLD)):
        ax.plot(years, [discounted_delay_loss(float(t), 1.0, rate) for t in years],
                color=col, lw=2.8, label=f"Annual discount rate {rate:.0%}")
    ax.set_xlabel("Years until reform")
    ax.set_ylabel("Present-value harm (normalized annual unit)")
    ax.set_title("A delay has a cost even when the language of delay is gentle", fontsize=15)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.15)
    ax.text(0.5, -0.19, "Illustration assumes one unit of avoidable harm per year, not an empirical welfare estimate.",
            ha="center", transform=ax.transAxes, color=MUTED, fontsize=9)
    _save(fig, path)


def figure8(path: Path) -> None:
    """Distinguish evidence types from what they can actually identify."""
    fig, ax = plt.subplots(figsize=(11.8, 6.1))
    ax.axis("off")
    data = [
        ("Historical records", "Texts, actions, sequences", "Document speech and chronology", "Prove a single motive or counterfactual"),
        ("Empirical research", "Designs + measured outcomes", "Estimate defined associations/effects", "Validate every historical analogy"),
        ("Formal model", "Explicit assumptions + equations", "Show possible mechanisms", "Demonstrate that the mechanism occurred"),
        ("Normative judgment", "Rights + ethical commitments", "Evaluate whether a compromise is just", "Follow from data alone"),
    ]
    x0, ys = 0.045, [0.74, 0.55, 0.36, 0.17]
    headers = ["Evidence", "Can help establish", "Cannot by itself establish"]
    for x, width, h in ((x0, 0.23, headers[0]), (0.30, 0.29, headers[1]), (0.61, 0.35, headers[2])):
        ax.text(x, 0.91, h, transform=ax.transAxes, fontsize=11, color=INK, fontweight="bold")
    for (title, typ, supports, not_supports), y in zip(data, ys):
        box = patches.FancyBboxPatch((x0-0.01,y-0.09),0.92,0.165,
                   transform=ax.transAxes,boxstyle="round,pad=0.01,rounding_size=0.02",
                   facecolor=PALE if y in (0.74,0.36) else "#f0f3f0", edgecolor="none")
        ax.add_patch(box)
        ax.text(x0,y+0.025,title,transform=ax.transAxes,fontweight="bold",fontsize=11)
        ax.text(x0,y-0.04,typ,transform=ax.transAxes,color=MUTED,fontsize=8)
        ax.text(0.30,y-0.012,supports,transform=ax.transAxes,fontsize=10)
        ax.text(0.61,y-0.012,not_supports,transform=ax.transAxes,fontsize=10)
    ax.set_title("Four different kinds of argument — four different inferential limits",fontsize=16,pad=15)
    _save(fig, path)


def _csv(path: Path, columns: tuple[str, ...], rows: Iterable[dict[str, float | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def generate(output: Path) -> tuple[list[Path], list[Path]]:
    """Create deterministic images and tables in a caller-selected directory."""
    if not isinstance(output, Path):
        raise TypeError("output must be a pathlib.Path")
    figs = output / "figures"
    results = output / "results"
    figs.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    _style()

    cases = _scenarios()
    scenario_rows: list[dict[str, float | str]] = []
    for case in cases:
        r, i = bargaining_solution(case.reformer_outside, case.incumbent_outside, case.weight)
        scenario_rows.append({"scenario": case.label, "reformer_outside": case.reformer_outside,
                              "incumbent_outside": case.incumbent_outside,
                              "reformer_weight": case.weight, "reformer_share": r,
                              "incumbent_share": i})
    grid = [
        {"reformer_weight": float(weight), "reformer_outside": float(d_r),
         "incumbent_outside": 0.4,
         "reformer_share": bargaining_solution(float(d_r), 0.4, float(weight))[0]}
        for d_r in np.linspace(0, 0.4, 51)
        for weight in np.linspace(0.1, 0.9, 51)
    ]
    costs = [
        {"restraint_cost": float(c),
         "effective_outside": effective_outside_option(0.35, float(c), 0.3),
         "reformer_share": bargaining_solution(
             effective_outside_option(0.35, float(c), 0.3), 0.45, 0.5
         )[0]}
        for c in np.linspace(0, 1, 101)
    ]
    delay = [
        {"years": float(y), "discount_rate": rate,
         "normalized_loss": discounted_delay_loss(float(y), 1.0, rate)}
        for rate in (0.0, 0.03, 0.08) for y in np.linspace(0, 20, 201)
    ]

    figures: tuple[Callable[[Path], None], ...] = (
        figure1, figure2, figure3,
        lambda path: figure4(path, cases),
        lambda path: figure5(path, grid),
        lambda path: figure6(path, costs),
        lambda path: figure7(path, delay),
        figure8,
    )
    paths: list[Path] = []
    for filename, draw in zip(FIGURE_NAMES, figures):
        target = figs / filename
        draw(target)
        paths.append(target)

    tables: list[Path] = []
    specifications = (
        ("bargaining_scenarios.csv", ("scenario", "reformer_outside", "incumbent_outside",
                                      "reformer_weight", "reformer_share", "incumbent_share"), scenario_rows),
        ("bargaining_sensitivity.csv", ("reformer_weight", "reformer_outside",
                                        "incumbent_outside", "reformer_share"), grid),
        ("restraint_cost.csv", ("restraint_cost", "effective_outside", "reformer_share"), costs),
        ("delay_cost.csv", ("years", "discount_rate", "normalized_loss"), delay),
    )
    for filename, columns, data in specifications:
        target = results / filename
        _csv(target, columns, data)
        tables.append(target)
    return paths, tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent,
                        help="Directory to hold figures/ and results/")
    args = parser.parse_args()
    figures, tables = generate(args.output)
    print(f"Generated {len(figures)} deterministic PNG figures and {len(tables)} CSV tables")


if __name__ == "__main__":
    main()
