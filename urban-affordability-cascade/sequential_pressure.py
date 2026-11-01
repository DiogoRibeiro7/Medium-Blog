"""Observed timing and rolling one-year predictive diagnostics for AMP housing.

The 2019–2025 municipal panel contains annual nominal *medians* for sold
homes and new rental contracts, not household moves or a price index with
constant dwelling quality. The 2021 undirected commuting-interaction weights
are used only as fixed lagged predictors for 2022–2025 targets. This module
never interprets predictive performance as contagion or displacement.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import Observation, ROOT, load_panel, require_float
from commuting_network import load_weights, weights_for_mode
from extract_commuting import PROVENANCE as COMMUTING_PROVENANCE
from extract_residential_mobility import DEFAULT_PROVENANCE as RESIDENTIAL_PROVENANCE


SEED = 20261031
RIDGE_LAMBDA = 1.0
TEST_YEARS = (2023, 2024, 2025)
FIGURES = (
    "fig14_nominal_housing_first_passage.png",
    "fig15_rolling_housing_prediction_mae.png",
)
RESULTS = (
    "sequential_housing_first_passage.csv",
    "sequential_housing_rolling_predictions.csv",
    "sequential_housing_rolling_summary.csv",
    "sequential_housing_provenance.json",
)
METRICS = {
    "sold_price": ("price_m2", 2019, (25, 50, 75, 100)),
    "new_contract_rent": ("rent_m2", 2020, (25, 50, 75)),
}


@dataclass(frozen=True, slots=True)
class ForecastCell:
    """One held-out municipality-year outcome and two pre-target predictions."""

    outcome: str
    year: int
    municipality: str
    actual_log_growth_pct: float
    baseline_prediction_pct: float
    network_prediction_pct: float


def first_passage(
    panel: dict[tuple[str, int], Observation], *, municipality: str,
    field: str, base_year: int, threshold_pct: float,
) -> int | None:
    """First available year reaching a nominal growth threshold, else censored.

    Never interpolate an unobserved passage date or equate first crossing
    with a persistent or causal metropolitan housing-market event.
    """
    if not math.isfinite(threshold_pct) or threshold_pct <= 0:
        raise ValueError("First-passage threshold must be positive and finite")
    if base_year not in (2019, 2020):
        raise ValueError("Only the two published complete baseline periods are supported")
    reference = require_float(getattr(panel[(municipality, base_year)], field), field)
    if reference <= 0:
        raise ValueError("Non-positive housing baseline")
    for year in range(base_year + 1, 2026):
        current = require_float(getattr(panel[(municipality, year)], field), field)
        if current / reference >= 1 + threshold_pct / 100:
            return year
    return None


def threshold_table(
    panel: dict[tuple[str, int], Observation], names: Sequence[str],
) -> list[dict[str, str | int]]:
    """Observed first-passage years for fixed round-number thresholds."""
    output: list[dict[str, str | int]] = []
    for municipality in names:
        row: dict[str, str | int] = {"municipality": municipality}
        for outcome, (field, base, thresholds) in METRICS.items():
            for threshold in thresholds:
                year = first_passage(panel, municipality=municipality, field=field,
                                     base_year=base, threshold_pct=float(threshold))
                row[f"{outcome}_plus_{threshold}_first_year"] = "" if year is None else year
        output.append(row)
    return output


def annual_log_growth(
    panel: dict[tuple[str, int], Observation], names: Sequence[str],
    *, field: str, base_year: int,
) -> dict[int, np.ndarray]:
    """Annual 100×log-ratios of reported municipality medians, no imputation."""
    if field not in {"price_m2", "rent_m2"} or base_year not in (2019, 2020):
        raise ValueError("Unsupported observed housing outcome or base year")
    history: dict[int, np.ndarray] = {}
    for year in range(base_year + 1, 2026):
        current = np.array([require_float(getattr(panel[(m, year)], field), field) for m in names])
        previous = np.array([require_float(getattr(panel[(m, year - 1)], field), field) for m in names])
        if np.any(current <= 0) or np.any(previous <= 0):
            raise ValueError("Observed medians must be strictly positive")
        history[year] = 100.0 * np.log(current / previous)
    return history


def lagged_design(growth: dict[int, np.ndarray], target_year: int, weights: np.ndarray) -> np.ndarray:
    """Own and metropolitan prior changes; network only uses prior changes."""
    if target_year <= 2021 or target_year - 1 not in growth:
        raise ValueError("No complete prior annual growth observations")
    prior = growth[target_year - 1]
    if weights.shape != (len(prior), len(prior)) or np.any(weights < 0):
        raise ValueError("Invalid commuting network geometry or signed weights")
    if not np.allclose(weights, weights.T) or not np.allclose(np.diag(weights), 0):
        raise ValueError("Network must be undirected and have zero diagonal")
    degrees = weights.sum(axis=1)
    if np.any(degrees <= 0):
        raise ValueError("Disconnected or empty network rows")
    # The incremental network predictor is the neighbors' deviation from the
    # metropolitan mean. Centering removes the shared time shock that a dense
    # 17-node commuter graph would otherwise duplicate and extrapolate.
    metropolitan = prior.mean()
    network_deviation = weights @ prior / degrees - metropolitan
    result = np.column_stack((prior, np.full(len(prior), metropolitan), network_deviation))
    if not np.all(np.isfinite(result)):
        raise ValueError("Non-finite lagged predictor")
    return result


def ridge_forecast(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
    *, penalty: float = RIDGE_LAMBDA,
) -> np.ndarray:
    """Fit a fixed-penalty ridge model with training-only standardization.

    The intercept is not penalized. Constant train features are zero after
    centering and given unit scale, avoiding near-singular first folds.
    """
    if not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("Ridge penalty must be positive and finite")
    if train_x.ndim != 2 or test_x.ndim != 2 or train_x.shape[1] != test_x.shape[1]:
        raise ValueError("Incompatible ridge feature matrices")
    if train_y.shape != (len(train_x),) or len(train_x) < train_x.shape[1] + 1:
        raise ValueError("Not enough training rows for ridge regression")
    if not all(np.all(np.isfinite(array)) for array in (train_x, train_y, test_x)):
        raise ValueError("Missing or non-finite forecast data")
    center = train_x.mean(axis=0)
    deviation = train_x.std(axis=0)
    deviation = np.where(deviation > 1e-12, deviation, 1.0)
    normalized_train = (train_x - center) / deviation
    normalized_test = (test_x - center) / deviation
    intercept = train_y.mean()
    normal = normalized_train.T @ normalized_train + penalty * np.eye(train_x.shape[1])
    coefficients = np.linalg.solve(normal, normalized_train.T @ (train_y - intercept))
    return intercept + normalized_test @ coefficients


def rolling_predictions(
    panel: dict[tuple[str, int], Observation], names: Sequence[str], weights: np.ndarray,
) -> list[ForecastCell]:
    """One-year-ahead rolling-origin validation on 2023, 2024, and 2025.

    Each fold fits only 2022..(test year − 1) targets and uses the previous
    calendar year's observed changes. The fixed 2021 network precedes every
    evaluated target year. The held-out outcome never informs scaling,
    penalty choice, regression coefficients or prior-year predictors.
    """
    outputs: list[ForecastCell] = []
    for outcome, (field, base_year, _) in METRICS.items():
        growth = annual_log_growth(panel, names, field=field, base_year=base_year)
        for test_year in TEST_YEARS:
            train_years = range(2022, test_year)
            train = np.vstack([lagged_design(growth, year, weights) for year in train_years])
            train_target = np.concatenate([growth[year] for year in train_years])
            predictors = lagged_design(growth, test_year, weights)
            baseline = ridge_forecast(train[:, :2], train_target, predictors[:, :2])
            network = ridge_forecast(train, train_target, predictors)
            for index, municipality in enumerate(names):
                outputs.append(ForecastCell(
                    outcome, test_year, municipality, float(growth[test_year][index]),
                    float(baseline[index]), float(network[index]),
                ))
    return outputs


def summarize_forecasts(rows: Sequence[ForecastCell]) -> list[dict[str, str | int]]:
    """Fold and pooled MAE, showing improvement as baseline minus network."""
    result: list[dict[str, str | int]] = []
    for outcome in METRICS:
        subset = [r for r in rows if r.outcome == outcome]
        for label, group in [(str(y), [r for r in subset if r.year == y]) for y in TEST_YEARS] + [("pooled", subset)]:
            if not group:
                raise ValueError("Empty rolling forecast fold")
            baseline_mae = sum(abs(r.actual_log_growth_pct - r.baseline_prediction_pct) for r in group) / len(group)
            network_mae = sum(abs(r.actual_log_growth_pct - r.network_prediction_pct) for r in group) / len(group)
            result.append({
                "outcome": outcome, "test_year": label, "n_municipality_years": len(group),
                "baseline_mae_log_points": f"{baseline_mae:.6f}",
                "network_mae_log_points": f"{network_mae:.6f}",
                "baseline_minus_network_mae": f"{baseline_mae - network_mae:.6f}",
            })
    return result


def network_shuffle_benchmark(
    panel: dict[tuple[str, int], Observation], names: Sequence[str], weights: np.ndarray,
    *, permutations: int = 999, seed: int = SEED,
) -> dict[str, dict[str, float | int]]:
    """Compare fixed network predictive gains with shuffled municipal labels.

    This is a *descriptive assignment benchmark*, not a randomized experiment
    and not a formal null p-value. Municipal exchangeability is implausible.
    """
    if permutations < 99 or not isinstance(permutations, int) or isinstance(permutations, bool):
        raise ValueError("At least 99 integer network shuffles required")
    actual = {r["outcome"]: float(r["baseline_minus_network_mae"])
              for r in summarize_forecasts(rolling_predictions(panel, names, weights))
              if r["test_year"] == "pooled"}
    random_gains: dict[str, list[float]] = {key: [] for key in METRICS}
    rng = np.random.default_rng(seed)
    for _ in range(permutations):
        order = rng.permutation(len(names))
        relabeled = weights[np.ix_(order, order)]
        for stat in summarize_forecasts(rolling_predictions(panel, names, relabeled)):
            if stat["test_year"] == "pooled":
                random_gains[str(stat["outcome"])].append(float(stat["baseline_minus_network_mae"]))
    return {
        key: {
            "observed_network_mae_improvement": actual[key],
            "shuffle_median_improvement": round(float(np.median(values)), 6),
            "shuffle_90th_percentile": round(float(np.quantile(values, 0.90)), 6),
            "fraction_shuffled_at_least_observed": round((1 + sum(v >= actual[key] for v in values)) / (permutations + 1), 6),
            "shuffles": permutations,
        }
        for key, values in random_gains.items()
    }


def _write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        raise ValueError("Cannot write empty observed/forecast output")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_thresholds(rows: Sequence[dict[str, str | int]], output: Path) -> None:
    """A single first-passage heatmap with explicit unpassed cells."""
    items = (("sold_price_plus_50_first_year", "Sale +50%"),
             ("sold_price_plus_100_first_year", "Sale +100%"),
             ("new_contract_rent_plus_25_first_year", "New rent +25%"),
             ("new_contract_rent_plus_50_first_year", "New rent +50%"))
    data = np.array([[float(r[field]) if r[field] != "" else np.nan for field, _ in items] for r in rows])
    fig, ax = plt.subplots(figsize=(10.6, 8.2))
    palette = plt.get_cmap("YlGnBu").copy()
    palette.set_bad("#e0e4e7")
    im = ax.imshow(np.ma.masked_invalid(data), cmap=palette, vmin=2021, vmax=2025, aspect="auto")
    ax.set_xticks(range(len(items)), labels=[label for _, label in items], fontsize=10)
    ax.set_yticks(range(len(rows)), labels=[str(r["municipality"]) for r in rows], fontsize=9)
    for i, row in enumerate(rows):
        for j, (field, _) in enumerate(items):
            value = row[field]
            ax.text(j, i, str(value) if value != "" else "not yet", ha="center", va="center",
                    fontsize=8.7, color="white" if value != "" and int(value) >= 2024 else "#203548")
        if row["municipality"] == "Porto":
            ax.get_yticklabels()[i].set_fontweight("bold")
    ax.set_title("First observed year above fixed nominal housing thresholds", fontsize=13, pad=15)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_ticks([2021, 2022, 2023, 2024, 2025])
    cb.set_label("First crossing year")
    fig.text(0.025, 0.023,
             "Source: fixed INE-derived annual medians. Sale baseline 2019; new-contract rent baseline 2020. "
             "Not a moving-house or inflation-adjusted series.", fontsize=8.4, color="#596979")
    fig.subplots_adjust(left=0.23, right=0.93, bottom=0.105, top=0.90)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_mae(rows: Sequence[dict[str, str | int]], output: Path) -> None:
    """Show within-year held-out absolute-error comparison, not causal effects."""
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 5.2), sharey=True)
    for ax, outcome, title in zip(axes, METRICS, ("Sold-price median", "New-contract rent median")):
        subset = [r for r in rows if r["outcome"] == outcome and r["test_year"] != "pooled"]
        pos = np.arange(len(TEST_YEARS))
        ax.bar(pos - 0.18, [float(r["baseline_mae_log_points"]) for r in subset],
               width=0.36, color="#739db8", label="Own lag + AMP lag")
        ax.bar(pos + 0.18, [float(r["network_mae_log_points"]) for r in subset],
               width=0.36, color="#d49b68", label="Add 2021 network lag")
        ax.set_xticks(pos, [str(y) for y in TEST_YEARS])
        ax.set_title(title, fontsize=11.5)
        ax.set_xlabel("Held-out target year")
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        pooled = next(r for r in rows if r["outcome"] == outcome and r["test_year"] == "pooled")
        ax.text(0.03, 0.96,
                f"Pooled baseline − network MAE: {float(pooled['baseline_minus_network_mae']):+.2f}",
                transform=ax.transAxes, va="top", fontsize=9.0, color="#344b5f")
    axes[0].set_ylabel("MAE in annual log-growth percentage points")
    axes[0].legend(frameon=False, loc="upper right", fontsize=8.4)
    fig.suptitle("Out-of-year forecasts, 17 AMP municipalities: does the network help?", fontsize=13.4)
    fig.text(0.025, 0.023,
             "2022 is the first training target. A fixed 2021 undirected commuting network predicts 2023–2025 only. "
             "Three folds; no causal inference.", fontsize=8.4, color="#596979")
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.19, top=0.84, wspace=0.22)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run(output: Path = ROOT, *, shuffles: int = 999) -> dict[str, object]:
    """Reproduce first-passage and rolling forecasts with all evidence labels."""
    names, network = load_weights()
    panel = load_panel()
    thresholds = threshold_table(panel, names)
    predictions = rolling_predictions(panel, names, network)
    summaries = summarize_forecasts(predictions)
    shuffle = network_shuffle_benchmark(panel, names, network, permutations=shuffles)

    figure_dir, result_dir = output / "figures", output / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_thresholds(thresholds, figure_dir / FIGURES[0])
    plot_mae(summaries, figure_dir / FIGURES[1])
    _write_csv(result_dir / RESULTS[0], thresholds)
    _write_csv(result_dir / RESULTS[1], [
        {"outcome": row.outcome, "target_year": row.year, "municipality": row.municipality,
         "observed_log_growth_pct": f"{row.actual_log_growth_pct:.6f}",
         "baseline_prediction_pct": f"{row.baseline_prediction_pct:.6f}",
         "network_prediction_pct": f"{row.network_prediction_pct:.6f}"}
        for row in predictions
    ])
    _write_csv(result_dir / RESULTS[2], summaries)

    sensitive = {}
    for mode in ("sqrt", "top3_union"):
        alternative = summarize_forecasts(rolling_predictions(panel, names, weights_for_mode(network, mode)))
        sensitive[mode] = {str(r["outcome"]): float(r["baseline_minus_network_mae"])
                           for r in alternative if r["test_year"] == "pooled"}
    provenance = {
        "evidence_classes": ["observed_nominal_median_threshold", "exploratory_prediction"],
        "municipality_count": len(names),
        "source_panel_sha256": "88502e15b55a4d46536eed42ddc1924d30f555498d2c4a7e7a1b41fbeed71c66",
        "commuting_source": json.loads(COMMUTING_PROVENANCE.read_text(encoding="utf-8")),
        "residential_source_identity_not_used_for_predictions": json.loads(RESIDENTIAL_PROVENANCE.read_text(encoding="utf-8"))["derived_csv_sha256"],
        "outcome_definitions": {
            "sold_price": "annual 100*log median sold EUR/m2 ratio, available 2019–2025",
            "new_contract_rent": "annual 100*log new-lease median EUR/m2 ratio, available 2020–2025",
        },
        "thresholds_nominal_pct": {key: list(val[2]) for key, val in METRICS.items()},
        "first_training_target": 2022,
        "rolling_test_years": list(TEST_YEARS),
        "baseline_features": ["municipality_prior_annual_log_growth", "amp_prior_annual_mean_log_growth"],
        "network_additional_feature": "2021_undirected_interaction_weighted_prior_annual_neighbor_log_growth_minus_amp_prior_mean",
        "ridge_lambda_on_train_standardized_features": RIDGE_LAMBDA,
        "network_weights": "raw_symmetric_interaction_counts_row_normalized_for_feature",
        "network_shuffle_benchmark_seed": SEED,
        "network_shuffles": shuffles,
        "pooled_results": [r for r in summaries if r["test_year"] == "pooled"],
        "network_label_shuffle": shuffle,
        "alternative_network_weight_improvement_mae_points": sensitive,
        "limits": [
            "seven sale years, six new-rent years, and only three forecast evaluation years",
            "nominal transaction and new-contract medians, not fixed-quality prices or incumbent rent",
            "one 2021 commuting network, not a residential migration OD or a pretreatment instrument",
            "forecast error difference is not an identified effect of network transmission",
            "network-label permutations are an informal benchmark, not calibrated causal p-values",
            "no measurement of household moves, eviction, affordability shocks, or real rent burdens",
        ],
    }
    (result_dir / RESULTS[3]).write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return provenance


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--shuffles", type=int, default=999,
                        help="Fixed-seed informal network relabeling benchmark (min 99)")
    args = parser.parse_args(argv)
    report = run(args.output, shuffles=args.shuffles)
    print(f"Reproduced {len(report['rolling_test_years'])} annual forecast folds for both housing outcomes")


if __name__ == "__main__":
    main()
