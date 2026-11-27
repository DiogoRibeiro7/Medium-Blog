"""Exact, synthetic counterexamples for interpreting a loneliness survey.

No Gallup respondent data are used. These scenarios demonstrate identification
limits, not estimates of reporting bias or the effect of income.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Cell:
    """One country-by-financial-status cell in a constructed population."""

    country: str
    status: str
    respondents: int
    yes: int

    def __post_init__(self) -> None:
        if self.respondents <= 0 or not 0 <= self.yes <= self.respondents:
            raise ValueError("Invalid cell counts")

    @property
    def prevalence(self) -> float:
        return self.yes / self.respondents


def cells() -> tuple[Cell, ...]:
    """Construct equal within-country rates and a 32% vs 15% pooled gap."""
    return (
        Cell("A", "Struggling", 880, 308),
        Cell("A", "Comfortable", 200, 70),
        Cell("B", "Struggling", 120, 12),
        Cell("B", "Comfortable", 800, 80),
    )


def pooled(rows: Iterable[Cell], status: str) -> float:
    """Compute the person-weighted response proportion for one status."""
    selected = [row for row in rows if row.status == status]
    if not selected:
        raise ValueError("No rows for status")
    return sum(row.yes for row in selected) / sum(row.respondents for row in selected)


def reported_prevalence(latent: float, sensitivity: float, false_positive: float) -> float:
    """Map a hypothetical binary latent state to a binary survey response."""
    if any(not 0 <= value <= 1 for value in (latent, sensitivity, false_positive)):
        raise ValueError("Probabilities must lie in [0, 1]")
    return sensitivity * latent + false_positive * (1 - latent)


def daily_membership(rotating: bool, days: int = 7) -> tuple[frozenset[int], ...]:
    """Return daily groups of exactly 23 out of 100 synthetic individuals."""
    if days <= 0:
        raise ValueError("days must be positive")
    return tuple(
        frozenset(((23 * day if rotating else 0) + j) % 100 for j in range(23))
        for day in range(days)
    )


def generate(output: Path) -> None:
    """Write deterministic tables and illustrative publication figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    rows = cells()
    with (output / "composition.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("country", "status", "respondents", "yes", "prevalence"))
        writer.writerows((r.country, r.status, r.respondents, r.yes, r.prevalence) for r in rows)

    # Equal within-country differences; unequal pooled composition.
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    x = range(3)
    struggling = [100 * rows[0].prevalence, 100 * rows[2].prevalence, 100 * pooled(rows, "Struggling")]
    comfortable = [100 * rows[1].prevalence, 100 * rows[3].prevalence, 100 * pooled(rows, "Comfortable")]
    left = ax.bar([v - 0.18 for v in x], struggling, width=0.36, label="Struggling", color="#a45a36")
    right = ax.bar([v + 0.18 for v in x], comfortable, width=0.36, label="Comfortable", color="#35688e")
    for bars in (left, right):
        ax.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=10)
    ax.set(xticks=list(x), xticklabels=["Country A", "Country B", "Pooled"], ylabel="Yes responses (%)", ylim=(0, 42), title="An apparent income gap without a within-country gap")
    ax.legend(frameon=False)
    ax.set_xlabel("Synthetic counterexample — not Gallup estimates", fontsize=10, labelpad=12)
    fig.savefig(output / "composition.png", dpi=180)
    plt.close(fig)

    # Identical latent prevalence with different response mechanisms.
    scenarios = [(0.20, 0.90, 0.125), (0.20, 0.70, 0.025)]
    with (output / "measurement.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("scenario", "latent", "sensitivity", "false_positive", "reported"))
        writer.writerows((name, *s, reported_prevalence(*s)) for name, s in zip(("A", "B"), scenarios))
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    left = ax.bar([-0.18, 0.82], [100 * s[0] for s in scenarios], width=0.36, label="Latent prevalence (assumed)", color="#8a929a")
    right = ax.bar([0.18, 1.18], [100 * reported_prevalence(*s) for s in scenarios], width=0.36, label="Reported prevalence", color="#35688e")
    for bars in (left, right):
        ax.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=10)
    ax.set(xticks=[0, 1], xticklabels=["Scenario A", "Scenario B"], ylabel="Prevalence (%)", ylim=(0, 35), title="Different answers need not imply different latent prevalence")
    ax.legend(frameon=False)
    ax.set_xlabel("Illustrative assumptions only; no reporting-error rates estimated", fontsize=10, labelpad=12)
    fig.savefig(output / "measurement.png", dpi=180)
    plt.close(fig)

    # The same daily margin does not identify who experiences loneliness repeatedly.
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), layout="constrained")
    with (output / "daily.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("scenario", "day", "person_id", "yes"))
        for rotating, ax in zip((False, True), axes):
            groups = daily_membership(rotating)
            name = "rotating" if rotating else "persistent"
            matrix = [[int(person in group) for person in range(100)] for group in groups]
            for day, row in enumerate(matrix, start=1):
                writer.writerows((name, day, person, yes) for person, yes in enumerate(row))
            ax.imshow(matrix, aspect="auto", interpolation="none", cmap="Blues", vmin=0, vmax=1)
            ax.set(title=f"{name.title()}: {len(set().union(*groups))} people affected across seven days", ylabel="Day", yticks=range(7), yticklabels=range(1, 8))
    axes[-1].set_xlabel("Synthetic individual (dark blue = yes response)\nSame 23% daily prevalence; no observed survey respondents", labelpad=10)
    fig.savefig(output / "daily.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    generate(Path(__file__).resolve().parent / "outputs")
