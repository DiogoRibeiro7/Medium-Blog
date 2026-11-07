"""Render a diagnostic figure from the no-cure simulation's committed cells.

The figure is independent of the editorial manuscript and makes the baseline
model-selection error visible next to the stricter false-report decision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from no_cure_audit import LATENCIES, RESULTS, SIZES

FIGURES: Final[Path] = Path(__file__).resolve().parent / "figures"
REQUIRED: Final[frozenset[str]] = frozenset(
    {
        "latency", "n", "ratio", "aic_prefers_cure", "false_cure_report",
        "false_report_ci_low", "false_report_ci_high",
    }
)


def create_figure(cells: pd.DataFrame, destination: Path) -> None:
    """Save a three-panel comparison of model selection and false reports."""
    if not isinstance(cells, pd.DataFrame):
        raise TypeError("cells must be a DataFrame")
    if not REQUIRED.issubset(cells.columns) or cells.empty:
        raise ValueError("Missing or empty no-cure cell summary")
    if not isinstance(destination, Path):
        raise TypeError("destination must be a pathlib.Path")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), sharey=True)
    palettes: dict[int, str] = {250: "#286090", 1000: "#ba4a00"}
    for ax, latency in zip(axes, LATENCIES):
        for n in SIZES:
            group: pd.DataFrame = cells[
                (cells["latency"] == latency) & (cells["n"] == n)
            ].sort_values("ratio")
            if group.empty:
                raise ValueError(f"Missing cell summary for {latency}, n={n}")
            ax.plot(
                group["ratio"],
                100 * group["aic_prefers_cure"],
                color=palettes[n],
                linestyle="--",
                marker="o",
                label=f"n={n}: AIC cure" if latency == LATENCIES[0] else None,
            )
            ax.plot(
                group["ratio"],
                100 * group["false_cure_report"],
                color=palettes[n],
                linestyle="-",
                marker="s",
                label=f"n={n}: false report" if latency == LATENCIES[0] else None,
            )
            ax.fill_between(
                group["ratio"].to_numpy(dtype=float),
                100 * group["false_report_ci_low"].to_numpy(dtype=float),
                100 * group["false_report_ci_high"].to_numpy(dtype=float),
                color=palettes[n],
                alpha=0.12,
            )
        ax.set_title(latency.replace("-", " (") + ")")
        ax.set_xlabel("Longest follow-up / susceptible 95th percentile")
        ax.set_ylim(0, 40)
        ax.set_xticks([0.25, 0.5, 1.0, 2.0])
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Percentage of no-cure datasets")
    fig.suptitle(
        "False cure signals when the true cure fraction is zero",
        fontsize=14,
        y=0.99,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.045))
    fig.text(
        0.5,
        0.015,
        "False report = AIC cure preference, family agreement < 0.05, selected cure fraction > 0.10; shaded = 95% Wilson intervals",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 0.95))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Load numerical evidence and regenerate its figure deterministically."""
    cells: pd.DataFrame = pd.read_csv(RESULTS / "no_cure_cells.csv")
    output: Path = FIGURES / "fig7_no_cure_false_discovery.png"
    create_figure(cells, output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
