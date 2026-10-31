"""Reproducible figures for a sourced Progressive Era essay.

Figures 1–4 are historical transcriptions or a sourced event chronology.
Figures 5–8 are explicitly hypothetical mathematical demonstrations, never
estimates of the policies' actual effects. All output is deterministic.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURE_NAMES = (
    "figure_01_urban_shift.png",
    "figure_02_income_tax_rates.png",
    "figure_03_union_membership.png",
    "figure_04_reform_timeline.png",
    "figure_05_implementation_bottleneck.png",
    "figure_06_regulatory_capacity.png",
    "figure_07_formal_vs_effective_access.png",
    "figure_08_distributional_incidence.png",
)

INK = "#192B36"
TEAL = "#087E8B"
GOLD = "#D38B35"
RED = "#B65052"
BLUE = "#426E9A"
GREY = "#7C8C98"
PAPER = "#FCFCFA"


def read_numeric_csv(filename: str, columns: Iterable[str]) -> dict[str, np.ndarray]:
    """Read a source transcription with exact requested numeric columns.

    Reject nonfinite values and unsorted or duplicate year observations.
    This function intentionally does not attempt network access.
    """
    path = DATA_DIR / filename
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    result = {key: np.asarray([float(row[key]) for row in rows]) for key in columns}
    if not all(np.all(np.isfinite(series)) for series in result.values()):
        raise ValueError(f"Non-finite source values: {path}")
    years = result.get("year")
    if years is not None and not np.all(np.diff(years) > 0):
        raise ValueError(f"Years not strictly increasing: {path}")
    return result


def urban_data() -> dict[str, np.ndarray]:
    """Continental U.S., 1924 Statistical Abstract, Table 34."""
    return read_numeric_csv(
        "urban_population.csv",
        ("year", "urban_population", "rural_population", "urban_percent"),
    )


def union_data() -> dict[str, np.ndarray]:
    """Leo Wolman (1929), Table 38, selected trade-union membership years."""
    return read_numeric_csv("union_membership.csv", ("year", "members"))


def tax_data() -> dict[str, np.ndarray]:
    """IRS Taxpayer Advocate (2011), Table 2, statutory brackets."""
    return read_numeric_csv(
        "income_tax_rates.csv",
        ("year", "lowest_bracket_percent", "highest_bracket_percent",
         "highest_bracket_threshold_usd"),
    )


@dataclass(frozen=True)
class Implementation:
    """Hypothetical serial conditional probabilities, all on [0, 1]."""

    legal_reach: float
    enforcement: float
    compliance: float
    practical_access: float

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")

    @property
    def realized_fraction(self) -> float:
        """Probability chain under the *assumed* conditional transitions."""
        return math.prod(vars(self).values())


def effective_access(
    legal: float, registration: float, free_participation: float
) -> float:
    """Hypothetical accessible franchise; no demographic data is inferred."""
    return Implementation(legal, registration, free_participation, 1.0).realized_fraction


def detection_probability(
    monitoring_budget: float, coverage: float, total_targets: float = 100.0
) -> float:
    """Poisson monitoring example, P(detection) = 1-exp(-B/(c*N)).

    Zero legal coverage implies no detection. This is not an empirical fit.
    """
    if not all(math.isfinite(float(x)) for x in (
        monitoring_budget, coverage, total_targets
    )):
        raise ValueError("Arguments must be finite")
    if monitoring_budget < 0 or not 0 <= coverage <= 1 or total_targets <= 0:
        raise ValueError("Invalid budget, coverage, or target count")
    if coverage == 0:
        return 0.0
    return -math.expm1(-monitoring_budget / (coverage * total_targets))


def regulatory_reach(
    monitoring_budget: float, coverage: float, total_targets: float = 100.0
) -> float:
    """Fraction of all targets covered AND detected in a hypothetical model."""
    return coverage * detection_probability(monitoring_budget, coverage, total_targets)


def group_benefits(
    baseline_harm: np.ndarray,
    access: np.ndarray,
    enforcement: float,
    legal_reach: float = 1.0,
) -> np.ndarray:
    """Illustrative avoided harm; no actual groups or years are represented."""
    harms = np.asarray(baseline_harm, dtype=float)
    access_values = np.asarray(access, dtype=float)
    if harms.shape != access_values.shape or harms.ndim != 1:
        raise ValueError("Inputs must be same-length one-dimensional arrays")
    if not np.all(np.isfinite(harms)) or np.any(harms < 0):
        raise ValueError("Baseline harms must be finite and nonnegative")
    if not np.all(np.isfinite(access_values)) or np.any(
        (access_values < 0) | (access_values > 1)
    ):
        raise ValueError("Access probabilities must lie in [0, 1]")
    Implementation(legal_reach, enforcement, 1.0, 1.0)
    return harms * access_values * enforcement * legal_reach


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "axes.edgecolor": "#BBC5C8",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "font.size": 11,
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "savefig.facecolor": PAPER,
        "savefig.dpi": 175,
    })


def _finish(fig: plt.Figure, target: Path, note: str) -> None:
    fig.subplots_adjust(bottom=.20)
    fig.text(0.075, 0.025, note, color=GREY, fontsize=8.5)
    fig.savefig(target, bbox_inches="tight", pad_inches=0.21)
    plt.close(fig)


def _historic_urban(target: Path) -> None:
    d = urban_data()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bars = ax.bar(d["year"].astype(int).astype(str), d["urban_percent"], color=TEAL, width=0.52)
    ax.axhline(50.0, color=GOLD, linestyle="--", linewidth=1.7)
    ax.text(2.38, 50.7, "Half of population", color=GOLD, ha="right", fontsize=10)
    for bar, pct in zip(bars, d["urban_percent"]):
        ax.text(bar.get_x() + bar.get_width()/2, pct + 1, f"{pct:.1f}%", ha="center", weight="bold")
    ax.set(ylim=(0, 63), ylabel="Share living in urban places (%)",
           title="The urban majority emerged by 1920")
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="y", alpha=0.17)
    _finish(fig, target, "SOURCE • U.S. Census, Statistical Abstract 1924, Table 34 (continental U.S.; contemporary classification).")


def _historic_unions(target: Path) -> None:
    d = union_data()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(d["year"], d["members"] / 1e6, color=TEAL, marker="o", markersize=7, lw=3)
    ax.axvspan(1917, 1919, alpha=.12, color=GOLD)
    ax.text(1917.3, 1.12, "WWI and immediate aftermath", color=GOLD, fontsize=10)
    ax.annotate("5.11 million", (1920, 5.1108), xytext=(1911.5, 4.8),
                arrowprops={"arrowstyle": "->", "color": INK}, fontsize=12, weight="bold")
    ax.set(xlim=(1899, 1921), ylim=(0, 6), ylabel="Reported membership (millions)",
           xlabel="Year", title="Organized labor grew, but the trajectory was not smooth")
    ax.set_xticks([1900, 1905, 1910, 1915, 1920])
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="y", alpha=.17)
    _finish(fig, target, "SOURCE • Leo Wolman (1929), Recent Economic Changes, Table 38; selected reported observations.")


def _historic_tax(target: Path) -> None:
    d = tax_data()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.step(d["year"], d["highest_bracket_percent"], where="mid", color=RED, lw=3, label="Highest marginal statutory rate")
    ax.plot(d["year"], d["highest_bracket_percent"], "o", color=RED, ms=7)
    ax.plot(d["year"], d["lowest_bracket_percent"], color=BLUE, lw=2, marker="s", label="Lowest statutory bracket rate")
    ax.fill_between(d["year"], d["lowest_bracket_percent"], d["highest_bracket_percent"],
                    step="mid", alpha=.06, color=RED)
    ax.set(xlim=(1912.6, 1920.4), ylim=(0, 88), ylabel="Federal individual income-tax rate (%)",
           xlabel="Tax year", title="The top tax schedule changed with war finance")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="y", alpha=.17)
    _finish(fig, target, "SOURCE • National Taxpayer Advocate (2011), Table 2; rates are marginal, not effective tax paid.")


def _historic_timeline(target: Path) -> None:
    with (DATA_DIR / "reform_milestones.csv").open(newline="", encoding="utf-8") as handle:
        events = list(csv.DictReader(handle))
    years = np.array([int(row["year"]) + (int(row["month"]) - 1) / 12.0 for row in events])
    colors = {"Consumers": TEAL, "Competition": GOLD, "Taxation": RED,
              "Elections": BLUE, "Banking": "#70579E", "Child labor": "#637F42",
              "Voting": BLUE}
    fig, ax = plt.subplots(figsize=(12.5, 7.4))
    for i, (event, year) in enumerate(zip(events, years)):
        color = colors[event["area"]]
        ax.hlines(i, 1905.5, year, color=color, lw=1.3, alpha=.34)
        ax.scatter(year, i, color=color, s=72, zorder=3)
        ax.annotate(event["label"], (year, i), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=10, color=color)
    ax.set(xlim=(1905, 1925.5), ylim=(-.7, len(events)-.3), xlabel="Calendar year",
           title="Legislation, institutions, and judicial reversal")
    ax.set_xticks([1906, 1909, 1912, 1915, 1918, 1921, 1924])
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.spines[["right", "top", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=.13)
    _finish(fig, target, "SOURCE • NARA, FDA, FTC, Federal Reserve, DOJ, Library of Congress; selected events, not an exhaustive list.")


def _bottleneck(target: Path) -> None:
    rates = [.9, .7, .6, .8]
    cumulative = np.cumprod([1.0] + rates)
    labels = ["Nominally addressed", "Legally covered", "Enforced",
              "Complied with", "Accessible in practice"]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bars = ax.barh(labels[::-1], (100*cumulative)[::-1], color=[TEAL, BLUE, GOLD, RED, INK][::-1])
    for bar in bars:
        ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                f"{bar.get_width():.1f}%", va="center", weight="bold")
    ax.set(xlim=(0, 113), xlabel="Retained share of the initial population (%)",
           title="An ambitious law can narrow at each implementation gate")
    ax.spines[["right", "top", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=.15)
    _finish(fig, target, "ILLUSTRATION • Arbitrary conditional factors: .90 × .70 × .60 × .80 = .3024; not historical data.")


def _capacity(target: Path) -> None:
    coverage = np.linspace(.05, 1, 96)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.8))
    for budget, color in [(10., RED), (25., GOLD), (50., TEAL)]:
        individual = [detection_probability(budget, c) for c in coverage]
        combined = [regulatory_reach(budget, c) for c in coverage]
        for ax, ys in zip(axes, (individual, combined)):
            ax.plot(coverage, ys, color=color, lw=2.8, label=f"Budget = {budget:.0f}")
    axes[0].set_title("Detection per covered target", fontsize=13, pad=14)
    axes[1].set_title("Detection across all targets", fontsize=13, pad=14)
    for ax in axes:
        ax.set(xlabel="Fraction of targets legally covered", ylim=(0, 1.02), xlim=(0, 1))
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(alpha=.13)
    axes[0].set_ylabel("Probability / population fraction")
    axes[1].legend(frameon=False, loc="upper left")
    fig.subplots_adjust(top=.79, wspace=.26)
    fig.suptitle("Wider jurisdiction and fixed monitoring resources", fontsize=17,
                 weight="bold", y=.99)
    _finish(fig, target, "ILLUSTRATION • 100 imaginary targets; independent Poisson monitoring, arbitrary budget units.")


def _access(target: Path) -> None:
    groups = ["Example A", "Example B"]
    reg = [.95, .55]
    freedom = [.95, .45]
    accessible = [effective_access(1, r, f) for r, f in zip(reg, freedom)]
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    width = .25
    for offset, values, label, color in [
        (-width, [1., 1.], "Formal legal protection", BLUE),
        (0, reg, "Register / qualify", GOLD),
        (width, accessible, "Able to participate", TEAL),
    ]:
        ax.bar(x+offset, values, width=width, color=color, label=label)
    ax.set_xticks(x, groups)
    ax.set(ylim=(0, 1.13), ylabel="Illustrative fraction",
           title="Identical formal rights need not mean identical access")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="y", alpha=.17)
    _finish(fig, target, "ILLUSTRATION • Two invented groups; factors are assumed probabilities, not racial voting-rate estimates.")


def _incidence(target: Path) -> None:
    names = ["High initial harm", "Medium initial harm", "Low initial harm"]
    harm = np.array([100., 65., 30.])
    scenarios = {
        "Minimal implementation": group_benefits(harm, np.array([.35, .65, .9]), .25),
        "Uniform stronger enforcement": group_benefits(harm, np.array([.35, .65, .9]), .75),
        "Access-oriented implementation": group_benefits(harm, np.array([.85, .85, .9]), .75),
    }
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    width = .24
    for shift, (label, values), color in zip(
        [-width, 0, width], scenarios.items(), [GREY, BLUE, TEAL]
    ):
        ax.bar(x+shift, values, width=width, color=color, label=label)
    ax.set_xticks(x, names)
    ax.set(ylabel="Avoided harm (arbitrary units)",
           title="The distribution of benefit depends on implementation")
    ax.legend(frameon=False, loc="upper right", fontsize=9.2)
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="y", alpha=.15)
    _finish(fig, target, "ILLUSTRATION • Invented baseline harms [100,65,30], access and enforcement; no policy effect estimated.")


def _write_csv(target: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_all(output: Path = ROOT) -> None:
    """Render eight PNG figures and three audit-friendly CSV result tables."""
    figures = output / "figures"
    results = output / "results"
    figures.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    _style()
    functions = (
        _historic_urban, _historic_tax, _historic_unions, _historic_timeline,
        _bottleneck, _capacity, _access, _incidence,
    )
    for func, name in zip(functions, FIGURE_NAMES):
        func(figures / name)

    summary: list[dict[str, object]] = []
    series_metadata = [
        ("Urban population", urban_data(), "urban_percent",
         "Percent, continental U.S.", "Census 1924 Statistical Abstract, Table 34"),
        ("Union members", union_data(), "members",
         "Persons, selected years", "Wolman 1929, Table 38"),
        ("Top statutory tax rate", tax_data(), "highest_bracket_percent",
         "Percent marginal", "IRS National Taxpayer Advocate 2011, Table 2"),
    ]
    for name, values, column, units, source in series_metadata:
        for year, value in zip(values["year"], values[column]):
            summary.append({"series": name, "year": int(year),
                            "value": float(value), "unit": units, "source": source,
                            "evidence_class": "historical_transcription"})
    _write_csv(results / "source_based_summary.csv",
               ["series", "year", "value", "unit", "source", "evidence_class"], summary)

    curves = []
    for coverage in np.linspace(.05, 1, 96):
        for budget in (10., 25., 50.):
            curves.append({
                "coverage": round(float(coverage), 4),
                "budget": budget,
                "detect_probability": detection_probability(budget, float(coverage)),
                "total_reach": regulatory_reach(budget, float(coverage)),
                "evidence_class": "hypothetical",
            })
    _write_csv(results / "hypothetical_capacity.csv",
               ["coverage", "budget", "detect_probability", "total_reach",
                "evidence_class"], curves)
    harm = np.array([100., 65., 30.])
    incidence = []
    for scenario, access, enforcement in (
        ("minimal", np.array([.35, .65, .9]), .25),
        ("uniform_enforcement", np.array([.35, .65, .9]), .75),
        ("access_oriented", np.array([.85, .85, .9]), .75),
    ):
        for group, amount in zip(("high_harm", "medium_harm", "low_harm"),
                                 group_benefits(harm, access, enforcement)):
            incidence.append({"scenario": scenario, "invented_group": group,
                              "avoided_harm_arbitrary_units": amount,
                              "evidence_class": "hypothetical"})
    _write_csv(results / "hypothetical_incidence.csv",
               ["scenario", "invented_group", "avoided_harm_arbitrary_units",
                "evidence_class"], incidence)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT,
                        help="Output folder containing figures/ and results/")
    args = parser.parse_args()
    generate_all(args.output.resolve())
    print(f"Generated {len(FIGURE_NAMES)} figures and 3 CSV results in {args.output.resolve()}")


if __name__ == "__main__":
    main()
