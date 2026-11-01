"""Stress-test the fixed Porto annual forecasts against simple alternatives.

All outputs remain descriptive predictive checks of annual municipality
housing medians. The 2021 undirected work/study network is not residential
movement data, and no causal or displacement inference is attempted.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import Observation, ROOT, SUBSET_SHA256, load_panel
from commuting_network import load_weights
from extract_commuting import PROVENANCE as COMMUTING_PROVENANCE
from sequential_pressure import (
    METRICS, TEST_YEARS, annual_log_growth, lagged_design,
    rolling_predictions,
)


FIGURES = (
    "fig16_rolling_housing_naive_benchmarks.png",
    "fig17_network_prediction_municipal_influence.png",
)
RESULTS = (
    "forecast_robustness_benchmark_mae.csv",
    "forecast_robustness_municipality_deletions.csv",
    "forecast_robustness_year_deletions.csv",
    "forecast_robustness_provenance.json",
)
PREDICTORS = ("no_change", "persistence", "amp_prior_mean", "baseline_ridge", "network_ridge")


@dataclass(frozen=True, slots=True)
class BenchmarkCell:
    """A genuinely held-out target with three unfitted and two fitted forecasts."""

    outcome: str
    year: int
    municipality: str
    actual: float
    no_change: float
    persistence: float
    amp_prior_mean: float
    baseline_ridge: float
    network_ridge: float


def benchmark_predictions(
    panel: dict[tuple[str, int], Observation], names: Sequence[str], weights: np.ndarray,
) -> list[BenchmarkCell]:
    """Use prior-year known outcomes for fair zero/persistence/AMP comparisons.

    The two ridge predictions come from the established, leakage-audited
    rolling-origin implementation. No naive comparator learns from a test year.
    """
    if len(set(names)) != len(names) or not names:
        raise ValueError("Municipality ordering must be unique and nonempty")
    existing = rolling_predictions(panel, names, weights)
    prior_growth = {
        outcome: annual_log_growth(panel, names, field=field, base_year=base)
        for outcome, (field, base, _) in METRICS.items()
    }
    positions = {name: idx for idx, name in enumerate(names)}
    result: list[BenchmarkCell] = []
    for forecast in existing:
        prior = prior_growth[forecast.outcome][forecast.year - 1]
        result.append(BenchmarkCell(
            forecast.outcome, forecast.year, forecast.municipality,
            forecast.actual_log_growth_pct, 0.0,
            float(prior[positions[forecast.municipality]]), float(prior.mean()),
            forecast.baseline_prediction_pct, forecast.network_prediction_pct,
        ))
    return result


def mean_absolute_error(rows: Sequence[BenchmarkCell], predictor: str) -> float:
    """Compute MAE in annual log-growth percentage points; reject bad inputs."""
    if not rows or predictor not in PREDICTORS:
        raise ValueError("Nonempty forecasts and a registered predictor required")
    values = np.array([getattr(row, predictor) for row in rows], dtype=float)
    observed = np.array([row.actual for row in rows], dtype=float)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(observed)):
        raise ValueError("Non-finite held-out forecast values")
    return float(np.mean(np.abs(observed - values)))


def forecast_gain(rows: Sequence[BenchmarkCell]) -> float:
    """Positive values favor network ridge over the original baseline ridge."""
    return mean_absolute_error(rows, "baseline_ridge") - mean_absolute_error(rows, "network_ridge")


def benchmark_table(rows: Sequence[BenchmarkCell]) -> list[dict[str, str | int]]:
    """Three calendar folds plus pooled summaries for both distinct outcomes."""
    municipalities = {r.municipality for r in rows}
    if not municipalities:
        raise ValueError("Incomplete annual forecast folds")
    identifiers = [(r.outcome, r.year, r.municipality) for r in rows]
    if len(set(identifiers)) != len(rows):
        raise ValueError("Duplicate held-out forecast cells")
    records: list[dict[str, str | int]] = []
    for outcome in METRICS:
        for target in (*TEST_YEARS, "pooled"):
            subset = [r for r in rows if r.outcome == outcome and (target == "pooled" or r.year == target)]
            expected = len(municipalities) * (len(TEST_YEARS) if target == "pooled" else 1)
            if len(subset) != expected:
                raise ValueError("Incomplete annual forecast folds")
            record: dict[str, str | int] = {
                "outcome": outcome, "target_year": str(target),
                "n_municipality_years": len(subset),
            }
            for predictor in PREDICTORS:
                record[f"{predictor}_mae_log_points"] = f"{mean_absolute_error(subset, predictor):.6f}"
            record["baseline_minus_network_mae_log_points"] = f"{forecast_gain(subset):.6f}"
            records.append(record)
    return records


def municipality_deletions(rows: Sequence[BenchmarkCell]) -> list[dict[str, str | int]]:
    """Descriptive deletion of *evaluation cells*, never a model refit or CI."""
    output: list[dict[str, str | int]] = []
    for outcome in METRICS:
        values = [r for r in rows if r.outcome == outcome]
        full = forecast_gain(values)
        for municipality in sorted({r.municipality for r in values}):
            omitted = [r for r in values if r.municipality == municipality]
            retained = [r for r in values if r.municipality != municipality]
            if len(omitted) != len(TEST_YEARS) or len(retained) != len(values) - len(TEST_YEARS):
                raise ValueError("Municipality has missing held-out calendar years")
            output.append({
                "outcome": outcome, "excluded_municipality": municipality,
                "n_remaining": len(retained),
                "full_pooled_gain": f"{full:.6f}",
                "gain_after_evaluation_deletion": f"{forecast_gain(retained):.6f}",
                "excluded_municipality_own_gain": f"{forecast_gain(omitted):.6f}",
            })
    return output


def year_deletions(rows: Sequence[BenchmarkCell]) -> list[dict[str, str | int]]:
    """Re-score the existing held-out forecasts after deleting one test year."""
    output: list[dict[str, str | int]] = []
    for outcome in METRICS:
        values = [r for r in rows if r.outcome == outcome]
        for excluded_year in TEST_YEARS:
            retained = [r for r in values if r.year != excluded_year]
            if len(retained) != len(values) - len({r.municipality for r in values}):
                raise ValueError("Incomplete year-deletion fold")
            output.append({
                "outcome": outcome, "excluded_test_year": excluded_year,
                "n_remaining": len(retained),
                "gain_after_evaluation_deletion": f"{forecast_gain(retained):.6f}",
            })
    return output


def porto_training_ablation(
    panel: dict[tuple[str, int], Observation], names: Sequence[str], weights: np.ndarray,
    full: Sequence[BenchmarkCell],
) -> dict[str, dict[str, float | int]]:
    """Refit and evaluate on 16 municipalities, also deleting Porto network edges.

    Contrasts 16-unit refit with the *evaluation-only* deletion from the
    original 17-unit forecasts. Not a counterfactual Porto treatment effect.
    """
    if "Porto" not in names:
        raise ValueError("Core-municipality ablation requires Porto")
    indices = [i for i, name in enumerate(names) if name != "Porto"]
    remaining_names = [names[i] for i in indices]
    remaining_weights = weights[np.ix_(indices, indices)]
    new_rows = benchmark_predictions(panel, remaining_names, remaining_weights)
    report: dict[str, dict[str, float | int]] = {}
    for outcome in METRICS:
        original = [r for r in full if r.outcome == outcome]
        original_16 = [r for r in original if r.municipality != "Porto"]
        revised = [r for r in new_rows if r.outcome == outcome]
        report[outcome] = {
            "all_17_gain": round(forecast_gain(original), 6),
            "original_models_evaluation_without_porto_gain": round(forecast_gain(original_16), 6),
            "refitted_16_and_network_without_porto_gain": round(forecast_gain(revised), 6),
            "held_out_municipality_years_after_refit": len(revised),
        }
    return report


def first_fold_geometry(
    panel: dict[tuple[str, int], Observation], names: Sequence[str], weights: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Expose the 2023 fold's unidentifiable year-constant metro predictor."""
    output: dict[str, dict[str, float | int]] = {}
    for outcome, (field, base, _) in METRICS.items():
        growth = annual_log_growth(panel, names, field=field, base_year=base)
        design = lagged_design(growth, 2022, weights)
        # All rows are from target year 2022, so its metropolitan mean
        # is a constant and indistinguishable from an intercept.
        output[outcome] = {
            "train_targets": len(design),
            "amp_prior_lag_train_sd": round(float(design[:, 1].std()), 12),
            "baseline_intercept_augmented_rank": int(np.linalg.matrix_rank(
                np.column_stack([np.ones(len(design)), design[:, :2]]))),
            "network_intercept_augmented_rank": int(np.linalg.matrix_rank(
                np.column_stack([np.ones(len(design)), design]))),
        }
    return output


def _save_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty forecast diagnostic")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_benchmarks(rows: Sequence[dict[str, str | int]], path: Path) -> None:
    """Pooled error against unfitted no-change, persistence and metro lags."""
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), sharey=True)
    labels = ("No change", "Own persistence", "AMP prior mean", "Baseline ridge", "+ Network ridge")
    colors = ("#b8c0c9", "#b8c0c9", "#b8c0c9", "#5485a7", "#c37b51")
    for axis, outcome, title in zip(axes, METRICS, ("Sold-price medians", "New-contract rent medians")):
        pooled = next(row for row in rows if row["outcome"] == outcome and row["target_year"] == "pooled")
        values = [float(pooled[f"{model}_mae_log_points"]) for model in PREDICTORS]
        x = np.arange(len(PREDICTORS))
        axis.bar(x, values, color=colors, width=0.72)
        axis.set_title(title, fontsize=12)
        axis.set_xticks(x, labels=labels, rotation=26, ha="right", fontsize=8.5)
        axis.grid(axis="y", alpha=0.18)
        axis.set_axisbelow(True)
        for pos, value in enumerate(values):
            axis.text(pos, value + 0.12, f"{value:.2f}", ha="center", fontsize=9)
    axes[0].set_ylabel("Pooled held-out MAE, annual log-growth percentage points")
    fig.suptitle("Three one-year-ahead folds: does added model complexity pay?", fontsize=13)
    fig.text(0.025, 0.02, "17 AMP municipalities × 2023–2025; all comparator inputs predate targets. "
             "Errors are predictive, not causal evidence.", fontsize=8.6, color="#596979")
    fig.subplots_adjust(left=0.095, right=0.98, bottom=0.27, top=0.85, wspace=0.13)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_influence(rows: Sequence[dict[str, str | int]], path: Path) -> None:
    """Show how heterogeneous municipality-level forecast gains really are."""
    fig, axes = plt.subplots(1, 2, figsize=(12.7, 8.1), sharey=True)
    for axis, outcome, title in zip(axes, METRICS, ("Sold-price medians", "New-contract rent medians")):
        subset = [r for r in rows if r["outcome"] == outcome]
        names = [str(r["excluded_municipality"]) for r in subset]
        gains = np.array([float(r["excluded_municipality_own_gain"]) for r in subset])
        colors = ["#5485a7" if gain >= 0 else "#c37b51" for gain in gains]
        axis.barh(np.arange(len(names)), gains, color=colors, height=0.7)
        axis.axvline(0, color="#344e65", lw=0.9)
        axis.set_yticks(range(len(names)), names, fontsize=8.8)
        axis.invert_yaxis()
        axis.set_xlabel("Baseline − network absolute error (mean, pp)")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.14)
        axis.set_axisbelow(True)
    fig.suptitle("Network forecast contributions vary across municipalities", fontsize=13)
    fig.text(0.02, 0.022,
             "Each bar averages the 3 held-out years of one municipality. No refitting or independent errors assumed; "
             "not a migration flow.", fontsize=8.2, color="#596979")
    fig.subplots_adjust(left=0.23, right=0.98, top=0.90, bottom=0.105, wspace=0.24)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(output: Path = ROOT) -> dict[str, object]:
    """Generate a fixed, offline forecast stress test with explicit limitations."""
    names, weights = load_weights()
    panel = load_panel()
    rows = benchmark_predictions(panel, names, weights)
    table = benchmark_table(rows)
    municipality_table = municipality_deletions(rows)
    year_table = year_deletions(rows)
    core_ablation = porto_training_ablation(panel, names, weights, rows)
    geometry = first_fold_geometry(panel, names, weights)
    fig_dir, result_dir = output / "figures", output / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_benchmarks(table, fig_dir / FIGURES[0])
    plot_influence(municipality_table, fig_dir / FIGURES[1])
    for filename, data in zip(RESULTS[:3], (table, municipality_table, year_table)):
        _save_csv(result_dir / filename, data)
    report: dict[str, object] = {
        "evidence_class": "observed_median_outcomes_exploratory_forecast_stress_test",
        "source_panel_sha256": SUBSET_SHA256,
        "network_source": "fixed_2021_undirected_work_study_interaction_matrix_not_residential_od",
        "network_source_provenance": json.loads(COMMUTING_PROVENANCE.read_text(encoding="utf-8")),
        "predictor_set": list(PREDICTORS),
        "test_years": list(TEST_YEARS),
        "held_out_years": len(TEST_YEARS),
        "municipalities": len(names),
        "benchmark_definitions": {
            "no_change": "predict_annual_log_growth_zero",
            "persistence": "predict_own_previous_year_annual_log_growth",
            "amp_prior_mean": "predict_17_municipality_mean_previous_year_annual_log_growth",
            "baseline_ridge": "own_prior_and_amp_prior_growth_fixed_ridge_lambda_1",
            "network_ridge": "baseline_plus_2021_commuting_weighted_neighbor_prior_deviation_from_amp",
        },
        "pooled_benchmark": [row for row in table if row["target_year"] == "pooled"],
        "municipality_evaluation_deletion": {
            outcome: {
                "min_remaining_gain": round(min(float(r["gain_after_evaluation_deletion"])
                                                for r in municipality_table if r["outcome"] == outcome), 6),
                "max_remaining_gain": round(max(float(r["gain_after_evaluation_deletion"])
                                                for r in municipality_table if r["outcome"] == outcome), 6),
            } for outcome in METRICS
        },
        "year_evaluation_deletion": year_table,
        "porto_network_and_training_ablation": core_ablation,
        "first_training_fold_feature_identifiability": geometry,
        "limitations": [
            "Only three held-out annual folds; municipality-years are spatially and temporally correlated",
            "Historical prior-year observed medians may not have been published before real-time forecast decisions",
            "Evaluation-only deletions are not refits, cluster bootstrap confidence intervals or formal hypothesis tests",
            "The 2023 fold trains on a single calendar year, so its AMP prior mean is collinear with intercept",
            "Porto removal changes training composition and row-normalized undirected weights, not an exogenous treatment",
            "Nominal municipality medians are not fixed-quality housing indices or household affordability burdens",
            "The commuting network does not identify directed residential moves or causal transmission",
        ],
    }
    (result_dir / RESULTS[3]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    report = run(args.output)
    print(f"Evaluated {len(report['predictor_set'])} forecasts across {report['held_out_years']} held-out years")


if __name__ == "__main__":
    main()
