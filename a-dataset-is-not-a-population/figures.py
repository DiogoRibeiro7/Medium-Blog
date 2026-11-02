"""Render nine synthetic study figures from archived replication-level CSVs.

Each plot reports the finite-population estimation error, or for predictive
models the excess squared-error risk over Bayes risk. All intervals display
Monte Carlo uncertainty for the across-replication mean, not survey intervals.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import METHODS, STUDIES

FIGURE_NAMES = {
    "srs": "fig1_simple_random_sampling",
    "strata": "fig2_disproportionate_stratification",
    "nonresponse": "fig3_outcome_dependent_nonresponse",
    "coverage": "fig4_population_undercoverage",
    "volunteer": "fig5_opt_in_selection_bias",
    "calibration": "fig6_joint_vs_marginal_calibration",
    "weighting": "fig7_weight_clipping_tradeoff",
    "predictive": "fig8_covariate_shift_risk",
    "drift": "fig9_composition_vs_concept_shift",
}

TITLES = {
    "srs": "Simple random sampling from a fixed population",
    "strata": "Disproportionate stratification",
    "nonresponse": "Response depends on an unobserved outcome driver",
    "coverage": "A sampling frame omits 20% of the population",
    "volunteer": "A large voluntary sample versus a small probability sample",
    "calibration": "Correct margins, incorrect joint distribution",
    "weighting": "Inverse weights and deliberate clipping",
    "predictive": "Model deployment under covariate shift",
    "drift": "Changing population shares and conditional mechanisms",
}


def read_replications(path: Path, study: str) -> dict[str, np.ndarray]:
    """Read exact-schema numeric replications and enforce complete methods."""
    values: dict[str, list[float]] = {method: [] for method in METHODS[study]}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["study", "scenario", "replicate", "method", "estimate", "target", "error"]:
            raise ValueError(f"Unexpected archive schema: {path}")
        seen: set[tuple[int, str]] = set()
        for row in reader:
            method = row["method"]
            if row["study"] != study or method not in values:
                raise ValueError(f"Unexpected study or method: {row}")
            index = int(row["replicate"])
            key = (index, method)
            if key in seen or index < 0:
                raise ValueError(f"Duplicate or negative replicate: {key}")
            seen.add(key)
            error = float(row["error"])
            if not math.isfinite(error) or not math.isclose(
                float(row["estimate"]) - float(row["target"]), error, abs_tol=1e-10
            ):
                raise ValueError("Nonfinite or inconsistent error")
            values[method].append(error)
    counts = {len(group) for group in values.values()}
    if len(counts) != 1 or not counts or min(counts) < 2:
        raise ValueError("Incomplete experiment or insufficient repetitions")
    return {method: np.asarray(group, dtype=float) for method, group in values.items()}


def draw_study(study: str, values: dict[str, np.ndarray], destination: Path) -> None:
    """Render mean signed error and 95% Monte Carlo confidence limits."""
    names = list(METHODS[study])
    means = np.asarray([values[name].mean() for name in names])
    errors = np.asarray([1.96 * values[name].std(ddof=1) / math.sqrt(len(values[name])) for name in names])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 180})
    fig, ax = plt.subplots(figsize=(9.2, max(4.0, 0.65 * len(names) + 2.6)), layout="constrained")
    positions = np.arange(len(names))
    colors = ["#237a8a" if abs(mean) < 0.20 else "#bd6044" for mean in means]
    ax.barh(positions, means, color=colors, alpha=0.87, height=0.52)
    ax.errorbar(means, positions, xerr=errors, fmt="none", ecolor="#1d2a32", capsize=4, linewidth=1.5)
    ax.axvline(0.0, color="#303b41", linewidth=1.2, linestyle="--")
    ax.set_yticks(positions, [name.replace("_", " ") for name in names])
    ax.invert_yaxis()
    ax.set_xlabel("Mean excess risk above Bayes risk" if study == "predictive" else "Mean estimate minus target")
    ax.set_title(TITLES[study], fontsize=13, loc="left", fontweight="semibold", pad=24)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.text(0.0, 1.015, "Bars: Monte Carlo means  •  Whiskers: ±1.96 MC SE  •  Synthetic data only",
            transform=ax.transAxes, fontsize=9, color="#52606b", va="bottom")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def render_all(data: Path, destination: Path, suffix: str = "") -> list[Path]:
    """Render exactly one labelled PNG per study, using supplied CSV evidence."""
    paths: list[Path] = []
    for study in STUDIES:
        values = read_replications(data / f"{study}_replicates{suffix}.csv", study)
        output = destination / f"{FIGURE_NAMES[study]}{suffix}.png"
        draw_study(study, values, output)
        paths.append(output)
    return paths


def main() -> None:
    """CLI for publishing or checking generated figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    paths = render_all(args.data, args.output, "_quick" if args.quick else "")
    print(f"Rendered {len(paths)} figures to {args.output}")


if __name__ == "__main__":
    main()
