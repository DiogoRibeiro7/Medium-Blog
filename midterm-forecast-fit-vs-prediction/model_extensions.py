"""Small-sample alternatives to the frozen two-predictor House midterm OLS.

These are *comparisons*, not a competition in which the best historical model
is retrospectively declared the 2026 forecasting winner. Parameters are fixed
before scoring: Student-t degrees of freedom=4, ridge alpha=2, and recency
half-life=24 years. Every fold uses only earlier elections; centering and
scaling use its training rows alone. No outcomes from 2026 are present.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from analysis import (
    Election, ROOT, Scenario, _errors, alternative_gallup, expanding_backtest,
    fit_model, predictor, read_elections, read_scenario,
)

Array = npt.NDArray[np.float64]
STUDENT_DF = 4.0
RIDGE_ALPHA = 2.0
RECENCY_HALF_LIFE = 24.0
MODEL_NAMES = ("ols", "student_t", "ridge", "recency_wls")


@dataclass(frozen=True)
class LocationFit:
    """A fitted conditional mean in original percentage-point/seat units."""

    coefficients: Array
    residual_scale: float | None = None
    effective_sample_size: float | None = None

    def predict(self, margin: float, seats: float) -> float:
        return float(predictor(margin, seats) @ self.coefficients)


@dataclass(frozen=True)
class ModelFold:
    """Same election and same training cutoff for every competitor."""

    year: int
    actual: int
    train_count: int
    ols: float
    student_t: float
    ridge: float
    recency_wls: float


@dataclass(frozen=True)
class ModelPath:
    """2026 scenario prediction as the historical fitting window expands."""

    train_end_year: int
    train_count: int
    ols: float
    student_t: float
    ridge: float
    recency_wls: float


@dataclass(frozen=True)
class ModelEvidence:
    """Single calculation shared by the manuscript, tables and figures."""

    folds: tuple[ModelFold, ...]
    alternative_folds: tuple[ModelFold, ...]
    coefficient_path: tuple[ModelPath, ...]
    summary: dict[str, object]


def _rows(elections: Iterable[Election]) -> tuple[Election, ...]:
    records = tuple(elections)
    if len(records) < 5 or [r.year for r in records] != sorted(r.year for r in records):
        raise ValueError("At least five chronologically ordered elections are required")
    if len({r.year for r in records}) != len(records):
        raise ValueError("Election years must be unique")
    for row in records:
        predictor(row.party_margin, row.seats)
        if abs(row.change) > 435:
            raise ValueError("Invalid House seat change")
    return records


def _standardized_design(rows: tuple[Election, ...]) -> tuple[Array, Array, Array, Array]:
    raw: Array = np.array([[r.party_margin, r.seats] for r in rows], dtype=float)
    means: Array = np.mean(raw, axis=0)
    scales: Array = np.std(raw, axis=0)
    if np.any(scales < 1e-10):
        raise ValueError("Predictor is constant within a training window")
    z: Array = np.column_stack((np.ones(len(rows)), (raw - means) / scales))
    if np.linalg.matrix_rank(z) != 3:
        raise ValueError("Training design is rank deficient")
    y: Array = np.array([r.change for r in rows], dtype=float)
    return z, y, means, scales


def _to_original(beta: Array, means: Array, scales: Array) -> Array:
    slope: Array = beta[1:] / scales
    return np.array([beta[0] - float(means @ slope), *slope], dtype=float)


def fit_ridge(
    elections: Iterable[Election], alpha: float = RIDGE_ALPHA,
) -> LocationFit:
    """Unpenalized intercept; ridge penalty on fold-standardized slopes only."""

    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("Ridge alpha must be finite and nonnegative")
    rows = _rows(elections)
    z, y, means, scales = _standardized_design(rows)
    penalty: Array = np.diag(np.array([0.0, alpha, alpha], dtype=float))
    coefficients: Array = np.linalg.solve(z.T @ z + penalty, z.T @ y)
    return LocationFit(coefficients=_to_original(coefficients, means, scales))


def _student_t_loss_gradient(parameters: Array, design: Array, y: Array, df: float) -> tuple[float, Array]:
    """Negative t log-likelihood (up to a constant) and analytic gradient."""

    beta: Array = parameters[:3]
    log_scale: float = float(parameters[3])
    scale: float = float(np.exp(log_scale))
    residual: Array = (y - design @ beta) / scale
    weights: Array = (df + 1.0) * residual / (df + residual * residual)
    loss: float = float(len(y) * log_scale + 0.5 * (df + 1.0) * np.sum(np.log1p(residual**2 / df)))
    grad_beta: Array = -(design.T @ weights) / scale
    grad_log_scale: float = float(len(y) - weights @ residual)
    gradient: Array = np.r_[grad_beta, grad_log_scale]
    return loss, gradient


def fit_student_t(
    elections: Iterable[Election], df: float = STUDENT_DF,
) -> LocationFit:
    """Deterministic, heavy-tailed Student-t likelihood with fixed df=4.

    We deliberately do not estimate df from 8–18 elections. The fitted scale
    is a t-distribution scale, not a calibrated future prediction interval.
    """

    if not math.isfinite(df) or df <= 2.0:
        raise ValueError("Student-t df must exceed two to have finite variance")
    rows = _rows(elections)
    z, y, means, scales = _standardized_design(rows)
    ols_beta: Array = np.linalg.lstsq(z, y, rcond=None)[0]
    initial_scale: float = max(float(np.sqrt(np.mean((y - z @ ols_beta) ** 2))), 0.5)
    initial: Array = np.r_[ols_beta, math.log(initial_scale)]
    result = minimize(
        _student_t_loss_gradient, initial, args=(z, y, df), jac=True,
        method="L-BFGS-B",
        bounds=[(None, None)] * 3 + [(math.log(0.1), math.log(200.0))],
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 3000},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ArithmeticError(f"Student-t fit did not converge: {result.message}")
    return LocationFit(
        coefficients=_to_original(np.asarray(result.x[:3], dtype=float), means, scales),
        residual_scale=float(np.exp(result.x[3])),
    )


def fit_recency_wls(
    elections: Iterable[Election], prediction_year: int,
    half_life_years: float = RECENCY_HALF_LIFE,
) -> LocationFit:
    """Weighted least squares where the newest past elections count more.

    Time weights are fixed from years only; no future outcomes or test errors
    enter their calculation. The half-life is a design assumption, not fitted.
    """

    if not math.isfinite(half_life_years) or half_life_years <= 0.0:
        raise ValueError("Recency half-life must be positive and finite")
    rows = _rows(elections)
    if isinstance(prediction_year, bool) or not isinstance(prediction_year, int):
        raise TypeError("Prediction year must be an integer")
    if prediction_year <= rows[-1].year:
        raise ValueError("Recency fit must target a later election")
    z, y, means, scales = _standardized_design(rows)
    ages: Array = np.asarray([prediction_year - row.year for row in rows], dtype=float)
    weights: Array = np.exp2(-ages / half_life_years)
    root_weight: Array = np.sqrt(weights)
    beta: Array = np.linalg.lstsq(z * root_weight[:, None], y * root_weight, rcond=None)[0]
    effective_n: float = float(np.sum(weights) ** 2 / np.sum(weights**2))
    return LocationFit(
        coefficients=_to_original(beta, means, scales),
        effective_sample_size=effective_n,
    )


def _fits(rows: tuple[Election, ...], prediction_year: int) -> dict[str, LocationFit]:
    reference = fit_model(rows)
    return {
        "ols": LocationFit(coefficients=reference.coefficients),
        "student_t": fit_student_t(rows),
        "ridge": fit_ridge(rows),
        "recency_wls": fit_recency_wls(rows, prediction_year),
    }


def forward_model_folds(
    elections: tuple[Election, ...], minimum_train: int = 8,
) -> tuple[ModelFold, ...]:
    """Compare all candidate rules using exactly identical chronological folds."""

    rows = _rows(elections)
    # Reuse the independently verified full-model backtest to validate origins.
    reference = expanding_backtest(rows, minimum_train=minimum_train)
    results: list[ModelFold] = []
    for i, full_fold in enumerate(reference, start=minimum_train):
        target = rows[i]
        if full_fold.year != target.year or full_fold.train_count != i:
            raise ArithmeticError("Reference and alternative folds are misaligned")
        fitted = _fits(rows[:i], target.year)
        assert math.isclose(fitted["ols"].predict(target.party_margin, target.seats),
                            full_fold.predicted, abs_tol=1e-9)
        predictions = {name: fit.predict(target.party_margin, target.seats)
                       for name, fit in fitted.items()}
        results.append(ModelFold(target.year, target.change, i, **predictions))
    return tuple(results)


def _scores(folds: tuple[ModelFold, ...], scenario: Scenario,
            rows: tuple[Election, ...]) -> dict[str, dict[str, float]]:
    actual: Array = np.array([row.actual for row in folds], dtype=float)
    final_fits = _fits(rows, scenario.year)
    return {
        name: {
            **_errors(actual, np.array([getattr(row, name) for row in folds], dtype=float)),
            "forecast_2026": final_fits[name].predict(scenario.party_margin, scenario.seats),
        }
        for name in MODEL_NAMES
    }


def compute_model_evidence() -> ModelEvidence:
    """Recalculate all comparisons without selecting parameters from errors."""

    rows = read_elections()
    scenario = read_scenario()
    primary = forward_model_folds(rows)
    alternative_rows = alternative_gallup(rows)
    alternative = forward_model_folds(alternative_rows)
    if [(f.year, f.actual, f.train_count) for f in primary] != [
        (f.year, f.actual, f.train_count) for f in alternative
    ]:
        raise ArithmeticError("Gallup source change altered the validation targets")
    trajectories: list[ModelPath] = []
    for i in range(8, len(rows) + 1):
        fitted = _fits(rows[:i], scenario.year)
        predictions = {name: fit.predict(scenario.party_margin, scenario.seats)
                       for name, fit in fitted.items()}
        trajectories.append(ModelPath(rows[i - 1].year, i, **predictions))
    metrics = {
        "as_of": "2026-10-31",
        "observations": len(rows),
        "forward_elections": [fold.year for fold in primary],
        "forward_count": len(primary),
        "student_t_df_fixed": STUDENT_DF,
        "ridge_alpha_fixed": RIDGE_ALPHA,
        "recency_half_life_years_fixed": RECENCY_HALF_LIFE,
        "primary": _scores(primary, scenario, rows),
        "alternative_gallup": _scores(alternative, scenario, alternative_rows),
        "recency_effective_n_at_2026": fit_recency_wls(rows, scenario.year).effective_sample_size,
    }
    return ModelEvidence(primary, alternative, tuple(trajectories), metrics)


def _csv(path: Path, fieldnames: tuple[str, ...], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer: csv.DictWriter[str] = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_model_evidence(evidence: ModelEvidence, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fields = ("year", "actual", "train_count", *MODEL_NAMES)
    _csv(output / "alternative_model_forward.csv", fields,
         (vars(row) for row in evidence.folds))
    _csv(output / "alternative_model_forward_gallup.csv", fields,
         (vars(row) for row in evidence.alternative_folds))
    trajectory_fields = ("train_end_year", "train_count", *MODEL_NAMES)
    _csv(output / "alternative_model_path.csv", trajectory_fields,
         (vars(row) for row in evidence.coefficient_path))
    (output / "alternative_models_summary.json").write_text(
        json.dumps(evidence.summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT,
                        help="Output root containing results/")
    args = parser.parse_args()
    evidence = compute_model_evidence()
    save_model_evidence(evidence, args.output / "results")
    for name in MODEL_NAMES:
        primary = evidence.summary["primary"][name]
        alternate = evidence.summary["alternative_gallup"][name]
        print(f"{name}: forward RMSE={primary['rmse']:.3f}, "
              f"2026 scenario={primary['forecast_2026']:.2f}, "
              f"alternate Gallup RMSE={alternate['rmse']:.3f}")


if __name__ == "__main__":
    main()
