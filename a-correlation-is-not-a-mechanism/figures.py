"""Publication-quality diagnostics for the nine synthetic causal studies.

Figures summarize generated Monte Carlo evidence and display each estimator's
error against its structural target. The vertical whisker is 1.96 times the
Monte Carlo standard error of the estimator's *mean*, not a confidence interval
for one observational dataset or a guarantee of causal identification.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis import STUDY_IDS


TITLES: dict[str, str] = {
    "confounding": "A common cause changes the observational slope",
    "simpson": "Stratification reverses the aggregate association",
    "collider": "Controlling a common effect opens a noncausal path",
    "mediator": "Adjusting for a mediator changes the estimand",
    "instrument": "Exclusion violations shift the IV estimand",
    "backdoor": "Not every additional control closes a backdoor path",
    "overlap": "Poor overlap inflates IPW uncertainty",
    "nonidentification": "The same observed law can imply different causal effects",
    "prediction": "A predictive proxy need not be an intervention lever",
}


def read_summary(path: Path) -> list[dict[str, str]]:
    """Load and validate an archived per-study summary table."""

    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"study", "scenario", "method", "target", "mean_estimate", "bias", "monte_carlo_se"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"invalid summary table: {path}")
    return rows


def plot_study(study: str, summary_path: Path, output_path: Path) -> None:
    """Plot estimator means with Monte Carlo intervals and causal targets."""

    if study not in TITLES:
        raise ValueError(f"unknown study: {study}")
    rows = read_summary(summary_path)
    if any(row["study"] != study for row in rows):
        raise ValueError("summary table contains a different study")
    labels = [f"{row['scenario'].replace('_', ' ')} · {row['method'].replace('_', ' ')}" for row in rows]
    estimates = [float(row["mean_estimate"]) for row in rows]
    targets = [float(row["target"]) for row in rows]
    errors = [1.96 * float(row["monte_carlo_se"]) for row in rows]
    positions = list(range(len(rows)))
    fig_height = max(3.6, 0.65 * len(rows) + 1.8)
    fig, axis = plt.subplots(figsize=(10.6, fig_height), constrained_layout=True)
    axis.errorbar(estimates, positions, xerr=errors, fmt="o", color="#0F6474",
                  ecolor="#0F6474", capsize=4, markersize=7, label="Monte Carlo mean ± 1.96 MCSE")
    axis.scatter(targets, positions, marker="x", s=85, color="#BF5734",
                 linewidths=2, label="Structural target", zorder=4)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Coefficient, probability difference, or effect (study units)")
    axis.set_title(TITLES[study], fontsize=13, pad=15)
    axis.grid(axis="x", alpha=0.18)
    axis.legend(loc="best", frameon=False)
    # The legend makes the whisker definition explicit without colliding with
    # the x-axis label in shorter figures.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    """Generate all nine figures from an explicit archive directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for study in STUDY_IDS:
        plot_study(study, args.data_dir / f"{study}_summary.csv",
                   args.output_dir / f"{study}.png")


if __name__ == "__main__":
    main()
