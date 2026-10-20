"""Training-only model selection under uncertain future intervention costs.

This is a predictive decision experiment, not causal policy evaluation.
Costs are treated as known when an action is taken, but their future values
are uncertain when the regression model is selected. A discrete uniform
cost grid represents that selection-time uncertainty.

Model tuning occurs within each repeated train/validation split; the
independent test set is reserved for final performance and regret evaluation.
Repeated overlapping splits are diagnostic, not independent replicates.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import ShuffleSplit

from analysis import (
    FloatArray,
    Scenario,
    expected_profit,
    policy_regret,
    simulate_data,
)
from stress_analysis import (
    SIGNALS,
    STRESSES,
    Stress,
    StressedData,
    apply_stress,
    stress_estimators,
)

Criterion = Literal["RMSE", "Nominal utility", "Cost-grid utility"]
Protocol = Literal["Single split", "Repeated splits"]
CostGrid = tuple[float, ...]
DEFAULT_COSTS: CostGrid = (0.2, 0.6, 1.0, 1.4)
CRITERIA: tuple[Criterion, ...] = (
    "RMSE",
    "Nominal utility",
    "Cost-grid utility",
)
PROTOCOLS: tuple[Protocol, ...] = ("Single split", "Repeated splits")
OBJECTIVE_COLUMNS: dict[Criterion, tuple[str, bool]] = {
    "RMSE": ("validation_mse", False),
    "Nominal utility": ("validation_nominal_utility", True),
    "Cost-grid utility": ("validation_grid_utility", True),
}


def validate_design(
    costs: CostGrid,
    nominal_cost: float,
    split_count: int,
    validation_fraction: float = 0.30,
) -> CostGrid:
    """Validate an ascending grid of finite, distinct, non-negative costs."""
    if not isinstance(costs, tuple) or len(costs) < 2:
        raise ValueError("costs must be a tuple of at least two values")
    for cost in costs:
        if isinstance(cost, bool) or not isinstance(cost, (int, float, np.number)):
            raise TypeError("Every cost must be a real numeric value")
    grid = tuple(float(c) for c in costs)
    if not np.all(np.isfinite(np.asarray(grid))) or any(c < 0 for c in grid):
        raise ValueError("Costs must be non-negative and finite")
    if any(next_c <= c for c, next_c in zip(grid, grid[1:])):
        raise ValueError("Costs must be strictly increasing without duplicates")
    if isinstance(nominal_cost, bool) or not isinstance(
        nominal_cost, (int, float, np.number)
    ):
        raise TypeError("nominal_cost must be numeric")
    if not np.isfinite(nominal_cost) or nominal_cost < 0:
        raise ValueError("nominal_cost must be non-negative and finite")
    if isinstance(split_count, bool) or not isinstance(split_count, int):
        raise TypeError("split_count must be an integer")
    if split_count < 2:
        raise ValueError("split_count must be at least two")
    if not (np.isfinite(validation_fraction) and 0 < validation_fraction < 1):
        raise ValueError("validation_fraction must be between zero and one")
    return grid


def score_validation(
    observed_benefit: FloatArray,
    predicted_benefit: FloatArray,
    costs: CostGrid,
    nominal_cost: float,
) -> tuple[float, float, float]:
    """Compute prediction MSE, nominal utility and grid-averaged utility.

    An action is taken when predicted benefit exceeds its corresponding cost.
    At evaluation time, the cost is known and the action threshold can adapt.
    This is *not* a pre-committed action chosen before cost realisation.
    """
    y = np.asarray(observed_benefit, dtype=np.float64)
    pred = np.asarray(predicted_benefit, dtype=np.float64)
    if y.ndim != 1 or pred.shape != y.shape or y.size == 0:
        raise ValueError("Observed and predicted benefits must match as 1-D arrays")
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Scores must contain finite values")
    # The grid is validated at the caller boundary, and here for public use.
    checked_grid = validate_design(costs, nominal_cost, split_count=2)
    squared_error = float(np.mean((y - pred) ** 2))
    nominal_utility = float(
        np.mean((y - nominal_cost) * (pred > nominal_cost))
    )
    grid_utility = float(
        np.mean(
            [
                np.mean((y - cost) * (pred > cost))
                for cost in checked_grid
            ]
        )
    )
    return squared_error, nominal_utility, grid_utility


def validation_audit(
    x_train: FloatArray,
    y_train: FloatArray,
    costs: CostGrid = DEFAULT_COSTS,
    nominal_cost: float = 0.6,
    split_count: int = 3,
    validation_fraction: float = 0.30,
    seed: int = 2026,
) -> pd.DataFrame:
    """Score fresh candidates using repeated training-only holdout splits.

    Any median imputation, scaling, dimension reduction or estimator tuning
    occurs on each inner training subset, never on its validation subset.
    Test data and oracle conditional means are absent from this interface.
    """
    grid = validate_design(costs, nominal_cost, split_count, validation_fraction)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    x = np.asarray(x_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or len(y) != len(x) or len(x) < 64:
        raise ValueError("At least 64 rows of aligned X and Y are required")
    if x.shape[1] < 2 or not np.isfinite(y).all() or np.isinf(x).any():
        raise ValueError("X may contain NaN but not infinity; Y must be finite")
    splitter = ShuffleSplit(
        n_splits=split_count, test_size=validation_fraction, random_state=seed
    )
    records: list[dict[str, str | int | float]] = []
    for split_index, (fit_indices, valid_indices) in enumerate(splitter.split(x)):
        observed = y[valid_indices]
        for name, model in stress_estimators().items():
            # These estimators own their preprocessing and hyperparameter tuning.
            model.fit(x[fit_indices], y[fit_indices])
            predicted = np.asarray(
                model.predict(x[valid_indices]), dtype=np.float64
            ).reshape(-1)
            mse, nominal_utility, grid_utility = score_validation(
                observed, predicted, grid, float(nominal_cost)
            )
            records.append(
                {
                    "split": split_index,
                    "model": name,
                    "validation_mse": mse,
                    "validation_nominal_utility": nominal_utility,
                    "validation_grid_utility": grid_utility,
                }
            )
    return pd.DataFrame.from_records(records)


def select_policies(audit: pd.DataFrame, split_count: int) -> pd.DataFrame:
    """Select six policies and quantify agreement across repeated holdouts.

    A policy combines its selection objective (RMSE, utility at the nominal
    cost, or utility across costs) with one or many validation splits.

    Agreement is the fraction of *overlapping* split winners matching the
    repeated-split aggregate winner; it is descriptive, not a probability.
    """
    required = {
        "split",
        "model",
        "validation_mse",
        "validation_nominal_utility",
        "validation_grid_utility",
    }
    if not required.issubset(audit.columns):
        raise ValueError(f"Missing audit columns: {required - set(audit.columns)}")
    if isinstance(split_count, bool) or not isinstance(split_count, int):
        raise TypeError("split_count must be an integer")
    if split_count < 2 or audit.empty:
        raise ValueError("Need two or more audited validation splits")
    if audit.duplicated(["split", "model"]).any():
        raise ValueError("Audit contains duplicate split/model pairs")
    if set(audit["split"]) != set(range(split_count)):
        raise ValueError("Audit must contain the complete split index range")
    models = set(audit["model"])
    if len(models) < 2 or any(
        set(group["model"]) != models for _, group in audit.groupby("split")
    ):
        raise ValueError("All candidate models must be evaluated in every split")
    for col, _ in OBJECTIVE_COLUMNS.values():
        if not np.isfinite(audit[col].to_numpy(dtype=np.float64)).all():
            raise ValueError("Non-finite validation scores are not allowed")

    output: list[dict[str, str | int | float]] = []
    for protocol in PROTOCOLS:
        included = audit.loc[
            audit["split"] == 0
        ] if protocol == "Single split" else audit
        for criterion in CRITERIA:
            score_column, maximize = OBJECTIVE_COLUMNS[criterion]
            by_model = included.groupby("model", sort=True)[score_column].mean()
            winner = str(
                by_model.idxmax() if maximize else by_model.idxmin()
            )
            # The same dataset is repeatedly re-split; the split winners are
            # dependent and should never be treated as iid observations.
            if protocol == "Repeated splits":
                winners = [
                    str(
                        split_scores.set_index("model")[score_column].idxmax()
                        if maximize
                        else split_scores.set_index("model")[score_column].idxmin()
                    )
                    for _, split_scores in audit.groupby("split", sort=True)
                ]
                agreement = float(np.mean(np.asarray(winners) == winner))
            else:
                agreement = float("nan")
            output.append(
                {
                    "protocol": protocol,
                    "criterion": criterion,
                    "policy": f"{protocol} / {criterion}",
                    "selected_model": winner,
                    "number_of_splits": 1 if protocol == "Single split" else split_count,
                    "split_agreement": agreement,
                }
            )
    return pd.DataFrame.from_records(output)


def run_cost_selection_experiment(
    repeats: int = 6,
    n_train_values: tuple[int, ...] = (180,),
    n_test: int = 500,
    costs: CostGrid = DEFAULT_COSTS,
    nominal_cost: float = 0.6,
    split_count: int = 3,
    seed: int = 2026,
    scenarios: tuple[Scenario, ...] = SIGNALS,
    stresses: tuple[Stress, ...] = ("clean", "mar_outliers"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare policy selection and resulting decision regret on paired tests.

    In one (scenario, sample size, replicate) all stress conditions share
    the underlying data and full-information test ground truth. Repeated
    validation splits and internal hyperparameter tuning use training only.

    Returns (test_performance, validation_audit). Each simulated test sample
    is independent of its training sample; cost outcomes are paired because
    they are evaluated on the *same* test sample, not independent trials.
    """
    grid = validate_design(costs, nominal_cost, split_count)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(n_test, bool) or not isinstance(n_test, int) or n_test < 10:
        raise ValueError("n_test must be at least 10")
    if not n_train_values or any(
        isinstance(n, bool) or not isinstance(n, int) or n < 64
        for n in n_train_values
    ):
        raise ValueError("n_train_values must be a nonempty tuple of integers >= 64")
    if len(set(n_train_values)) != len(n_train_values):
        raise ValueError("Training sample sizes must not repeat")
    if not scenarios or any(s not in SIGNALS for s in scenarios):
        raise ValueError("Unknown or absent signal scenarios")
    if not stresses or any(s not in STRESSES for s in stresses):
        raise ValueError("Unknown or absent stress scenarios")
    if len(set(stresses)) != len(stresses) or len(set(scenarios)) != len(scenarios):
        raise ValueError("Scenario labels must be distinct")

    evaluated: list[dict[str, str | int | float]] = []
    audits: list[dict[str, str | int | float]] = []
    for scenario_idx, scenario in enumerate(scenarios):
        for n_train in n_train_values:
            for replicate in range(repeats):
                local_seed = (
                    seed + scenario_idx * 10_000_000 + n_train * 1000 + replicate
                )
                original = simulate_data(
                    seed=local_seed,
                    scenario=scenario,
                    n_train=n_train,
                    n_test=n_test,
                )
                for stress_idx, stress in enumerate(stresses):
                    observed = apply_stress(
                        original,
                        stress=stress,
                        seed=local_seed + (stress_idx + 1) * 100_000,
                    )
                    audit = validation_audit(
                        observed.x_train, observed.y_train, costs=grid,
                        nominal_cost=float(nominal_cost),
                        split_count=split_count, seed=local_seed,
                    )
                    selected = select_policies(audit, split_count)
                    metadata: dict[str, str | int] = {
                        "scenario": scenario,
                        "stress": stress,
                        "n_train": n_train,
                        "replicate": replicate,
                    }
                    audits.extend({**metadata, **record} for record in audit.to_dict("records"))

                    # Refit once per distinct winning estimator; no test-based
                    # selection and no oracle information is used to fit models.
                    predictions: dict[str, FloatArray] = {}
                    for name in selected["selected_model"].unique():
                        model = stress_estimators()[str(name)]
                        model.fit(observed.x_train, observed.y_train)
                        predicted = np.asarray(
                            model.predict(observed.x_test), dtype=np.float64
                        ).reshape(-1)
                        if predicted.shape != observed.y_test.shape or not np.isfinite(
                            predicted
                        ).all():
                            raise ValueError("Invalid test predictions")
                        predictions[str(name)] = predicted

                    for policy_record in selected.to_dict("records"):
                        predicted = predictions[str(policy_record["selected_model"])]
                        test_rmse = float(
                            np.sqrt(np.mean((observed.y_test - predicted) ** 2))
                        )
                        for cost in grid:
                            evaluated.append(
                                {
                                    **metadata,
                                    **policy_record,
                                    "cost": cost,
                                    "test_rmse": test_rmse,
                                    "expected_profit": expected_profit(
                                        observed.mean_test, predicted, cost
                                    ),
                                    "regret": policy_regret(
                                        observed.mean_test, predicted, cost
                                    ),
                                    "oracle_profit": float(
                                        np.mean(
                                            np.maximum(observed.mean_test - cost, 0)
                                        )
                                    ),
                                }
                            )
    return pd.DataFrame.from_records(evaluated), pd.DataFrame.from_records(audits)


def paired_policy_regret(
    evaluation: pd.DataFrame,
    reference_policy: str,
    comparison_policy: str,
    bootstrap_draws: int = 500,
    seed: int = 17,
) -> pd.DataFrame:
    """Estimate paired regret change: comparison minus reference.

    Negative changes favour the comparison policy. Bootstrap sampling units
    are independent simulated *replicates*, not validation splits, costs or
    test rows. The cost-specific estimates remain correlated across costs.
    """
    if not reference_policy or not comparison_policy:
        raise ValueError("Both policy labels must be provided")
    if reference_policy == comparison_policy:
        raise ValueError("Comparison and reference policies must differ")
    if isinstance(bootstrap_draws, bool) or not isinstance(bootstrap_draws, int):
        raise TypeError("bootstrap_draws must be an integer")
    if bootstrap_draws < 10:
        raise ValueError("At least ten bootstrap draws are required")
    keys = ["scenario", "stress", "n_train", "replicate", "cost"]
    required = set(keys) | {"policy", "regret"}
    if not required.issubset(evaluation.columns):
        raise ValueError(f"Evaluation is missing: {required - set(evaluation.columns)}")
    if evaluation.duplicated(keys + ["policy"]).any():
        raise ValueError("Duplicate evaluation policy records")

    reference = evaluation.loc[
        evaluation["policy"] == reference_policy, keys + ["regret"]
    ].rename(columns={"regret": "reference_regret"})
    comparison = evaluation.loc[
        evaluation["policy"] == comparison_policy, keys + ["regret"]
    ].rename(columns={"regret": "comparison_regret"})
    if reference.empty or comparison.empty:
        raise ValueError("Both named policies must be present in evaluation")
    joined = reference.merge(comparison, how="outer", on=keys, validate="one_to_one")
    if joined[["reference_regret", "comparison_regret"]].isna().any().any():
        raise ValueError("Every replicate/cost must match across both policies")
    joined["delta"] = joined["comparison_regret"] - joined["reference_regret"]
    rng = np.random.default_rng(seed)
    results: list[dict[str, str | int | float]] = []
    for (scenario, stress, n_train, cost), group in joined.groupby(
        ["scenario", "stress", "n_train", "cost"], sort=True
    ):
        changes: NDArray[np.float64] = group["delta"].to_numpy(dtype=np.float64)
        draws = rng.integers(
            0, len(changes), size=(bootstrap_draws, len(changes))
        )
        means: NDArray[np.float64] = changes[draws].mean(axis=1)
        results.append(
            {
                "scenario": str(scenario),
                "stress": str(stress),
                "n_train": int(n_train),
                "cost": float(cost),
                "reference_policy": reference_policy,
                "comparison_policy": comparison_policy,
                "replicates": int(len(changes)),
                "mean_regret_change": float(changes.mean()),
                "bootstrap_lower_95": float(np.quantile(means, 0.025)),
                "bootstrap_upper_95": float(np.quantile(means, 0.975)),
            }
        )
    return pd.DataFrame.from_records(results)
