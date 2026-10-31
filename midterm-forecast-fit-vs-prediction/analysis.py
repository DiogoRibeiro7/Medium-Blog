"""Reconstruct and stress-test an October 2026 US House midterm forecast.

Data are frozen, source-transcribed *pre-election polls* and historical seat
changes, not election outcomes substituted for polls. All inferences are
predictive and associational. In particular, the 2026 election is not in the
training sample. Run ``python analysis.py`` to regenerate machine-readable
evidence; charts are generated separately by ``figures.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import numpy.typing as npt
from scipy.stats import t as student_t

ROOT: Path = Path(__file__).resolve().parent
DATA: Path = ROOT / "data"
Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Election:
    """One historical midterm with a pre-election polling measurement."""

    year: int
    party: str
    seats: int
    change: int
    dem_margin: float
    source: str

    @property
    def party_margin(self) -> float:
        """Positive values favor the president's party in percentage points."""

        return self.dem_margin if self.party == "D" else -self.dem_margin


@dataclass(frozen=True)
class Scenario:
    """Pre-election inputs for a future midterm, never a labelled outcome."""

    year: int
    party: str
    seats: int
    dem_margin: float

    @property
    def party_margin(self) -> float:
        return self.dem_margin if self.party == "D" else -self.dem_margin


@dataclass(frozen=True)
class OLSFit:
    """Full-rank OLS model including intercept, party margin and seat count."""

    coefficients: Array
    design: Array
    outcomes: Array
    fitted: Array
    residuals: Array
    xtx_inverse: Array
    residual_variance: float
    r_squared: float

    @property
    def degrees_of_freedom(self) -> int:
        return int(self.design.shape[0] - self.design.shape[1])

    @property
    def residual_sd(self) -> float:
        return math.sqrt(self.residual_variance)

    def predict(self, margin: float, seats: float) -> float:
        """Make a conditional point prediction after finite-value validation."""

        x: Array = predictor(margin, seats)
        return float(x @ self.coefficients)


@dataclass(frozen=True)
class Holdout:
    """One historical election predicted without its outcome in the fit."""

    year: int
    actual: int
    predicted: float
    baseline: float
    train_count: int
    lower_90: float
    upper_90: float
    lower_95: float
    upper_95: float

    @property
    def error(self) -> float:
        return self.predicted - self.actual


@dataclass(frozen=True)
class ReducedHoldout:
    """Chronological holdout forecasts from single-feature OLS benchmarks."""

    year: int
    actual: int
    train_count: int
    generic_only: float
    seats_only: float


@dataclass(frozen=True)
class Evidence:
    """Computed values shared by the tables, figures and publication draft."""

    elections: tuple[Election, ...]
    scenario: Scenario
    model: OLSFit
    leave_one_out: tuple[Holdout, ...]
    forward: tuple[Holdout, ...]
    reduced_forward: tuple[ReducedHoldout, ...]
    bootstrapped_future: Array
    alternative_model: OLSFit
    metrics: dict[str, Any]


def _valid_real(value: float, name: str) -> float:
    """Reject nonnumeric, boolean, NaN or infinite numerical inputs."""

    if isinstance(value, bool) or not isinstance(value, (float, int, np.number)):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def predictor(party_margin: float, seats: float) -> Array:
    """Return an intercept and the two covariates, in fixed order."""

    margin: float = _valid_real(party_margin, "party_margin")
    defended: float = _valid_real(seats, "seats")
    if abs(margin) > 100.0 or not 0 <= defended <= 435:
        raise ValueError("Generic margin or defended seats outside valid bounds")
    return np.array([1.0, margin, defended], dtype=float)


def _validate_election(item: Election) -> None:
    if isinstance(item.year, bool) or not isinstance(item.year, int) or item.year < 1900:
        raise ValueError("Invalid election year")
    if item.party not in {"D", "R"}:
        raise ValueError("President's party must be D or R")
    if isinstance(item.seats, bool) or not isinstance(item.seats, int):
        raise TypeError("House seats must be an integer")
    if isinstance(item.change, bool) or not isinstance(item.change, int):
        raise TypeError("House seat change must be an integer")
    predictor(item.party_margin, item.seats)
    if abs(item.change) > 435 or not item.source:
        raise ValueError("Invalid seat change or missing source")


def read_elections(path: Path = DATA / "historical_midterms.csv") -> tuple[Election, ...]:
    """Load the exact audited CSV contract, without filling missing elections."""

    expected: tuple[str, ...] = (
        "year", "president_party", "house_seats_defended",
        "president_party_seat_change", "generic_dem_minus_rep", "generic_source",
    )
    with path.open(encoding="utf-8", newline="") as handle:
        reader: csv.DictReader[str] = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError("Historical input CSV headers do not match the source contract")
        elections: tuple[Election, ...] = tuple(
            Election(
                year=int(row["year"]), party=row["president_party"],
                seats=int(row["house_seats_defended"]),
                change=int(row["president_party_seat_change"]),
                dem_margin=float(row["generic_dem_minus_rep"]),
                source=row["generic_source"],
            )
            for row in reader
        )
    if len(elections) < 5:
        raise ValueError("Too few elections to fit and assess the proposed model")
    for election in elections:
        _validate_election(election)
    years: list[int] = [item.year for item in elections]
    if len(years) != len(set(years)) or years != sorted(years):
        raise ValueError("Elections must have unique, chronologically ordered years")
    return elections


def read_scenario(path: Path = DATA / "2026_scenario.csv") -> Scenario:
    """Read a single unlabeled 2026 input scenario with no outcome column."""

    expected: tuple[str, ...] = (
        "year", "president_party", "house_seats_defended",
        "generic_dem_minus_rep", "scenario_source",
    )
    with path.open(encoding="utf-8", newline="") as handle:
        reader: csv.DictReader[str] = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError("Scenario must not contain a 2026 outcome")
        records: list[dict[str, str]] = list(reader)
    if len(records) != 1 or not records[0]["scenario_source"]:
        raise ValueError("Expected one sourced and unlabeled forecast scenario")
    record: dict[str, str] = records[0]
    scenario = Scenario(
        year=int(record["year"]), party=record["president_party"],
        seats=int(record["house_seats_defended"]),
        dem_margin=float(record["generic_dem_minus_rep"]),
    )
    if scenario.party not in {"D", "R"} or scenario.year != 2026:
        raise ValueError("Unexpected forecast year or political party")
    predictor(scenario.party_margin, scenario.seats)
    return scenario


def fit_model(elections: Iterable[Election]) -> OLSFit:
    """Fit an OLS line from first principles using least squares and SVD."""

    rows: tuple[Election, ...] = tuple(elections)
    if len(rows) < 5:
        raise ValueError("At least five source observations required")
    for item in rows:
        _validate_election(item)
    x: Array = np.stack([predictor(row.party_margin, row.seats) for row in rows])
    y: Array = np.array([row.change for row in rows], dtype=float)
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("Design matrix is rank deficient")
    coefficients: Array = np.linalg.lstsq(x, y, rcond=None)[0]
    fitted: Array = x @ coefficients
    residuals: Array = y - fitted
    rss: float = float(residuals @ residuals)
    tss: float = float(np.sum((y - np.mean(y)) ** 2))
    if tss <= 0:
        raise ValueError("Outcome is constant, so R-squared is undefined")
    df: int = len(rows) - x.shape[1]
    if df <= 0:
        raise ValueError("No degrees of freedom for residual uncertainty")
    return OLSFit(
        coefficients=coefficients, design=x, outcomes=y,
        fitted=fitted, residuals=residuals,
        xtx_inverse=np.linalg.inv(x.T @ x),
        residual_variance=rss / df, r_squared=1.0 - rss / tss,
    )


def prediction_interval(
    model: OLSFit, margin: float, seats: float,
    level: float = 0.95, *, future_observation: bool = True,
) -> tuple[float, float]:
    """Student-t OLS interval conditional on stable, homoskedastic errors.

    ``future_observation`` adds the irreducible future residual variance.
    These are *model-based* intervals, not distribution-free guarantees.
    """

    if not 0.0 < level < 1.0:
        raise ValueError("Interval coverage must lie strictly between zero and one")
    x: Array = predictor(margin, seats)
    leverage: float = float(x @ model.xtx_inverse @ x)
    variance: float = model.residual_variance * (leverage + float(future_observation))
    if variance < 0:
        raise ArithmeticError("Computed negative prediction variance")
    half_width: float = float(student_t.ppf((1.0 + level) / 2.0, model.degrees_of_freedom)) * math.sqrt(variance)
    center: float = float(x @ model.coefficients)
    return center - half_width, center + half_width


def _holdout(train: tuple[Election, ...], test: Election) -> Holdout:
    model: OLSFit = fit_model(train)
    predicted: float = model.predict(test.party_margin, test.seats)
    low90, high90 = prediction_interval(model, test.party_margin, test.seats, 0.90)
    low95, high95 = prediction_interval(model, test.party_margin, test.seats, 0.95)
    return Holdout(
        year=test.year, actual=test.change, predicted=predicted,
        baseline=float(np.mean([item.change for item in train])),
        train_count=len(train), lower_90=low90, upper_90=high90,
        lower_95=low95, upper_95=high95,
    )


def leave_one_out(elections: tuple[Election, ...]) -> tuple[Holdout, ...]:
    """LOOCV: each year held out; past and future years can train its fold."""

    return tuple(
        _holdout(elections[:i] + elections[i + 1 :], test)
        for i, test in enumerate(elections)
    )


def expanding_backtest(
    elections: tuple[Election, ...], minimum_train: int = 8,
) -> tuple[Holdout, ...]:
    """One-election-ahead backtest: absolutely no later data in training."""

    if isinstance(minimum_train, bool) or not isinstance(minimum_train, int):
        raise TypeError("minimum_train must be an integer")
    if not 5 <= minimum_train < len(elections):
        raise ValueError("Training window must leave at least one future election")
    return tuple(
        _holdout(elections[:i], elections[i])
        for i in range(minimum_train, len(elections))
    )


def reduced_predict(
    elections: Iterable[Election], feature: Literal["generic", "seats"],
    *, margin: float, seats: float,
) -> float:
    """Fit intercept plus one feature; never use the test outcome in estimation.

    Both benchmarks use the same OLS solver and source observations as the
    headline model. The test prediction uses its contemporaneous feature,
    but not its observed seat change or any later election.
    """

    if feature not in {"generic", "seats"}:
        raise ValueError("Feature must be 'generic' or 'seats'")
    rows: tuple[Election, ...] = tuple(elections)
    if len(rows) < 3:
        raise ValueError("At least three training elections are required")
    for election in rows:
        _validate_election(election)
    predictor(margin, seats)
    training_feature: Array = np.asarray(
        [row.party_margin if feature == "generic" else row.seats for row in rows],
        dtype=float,
    )
    x: Array = np.column_stack((np.ones(len(rows), dtype=float), training_feature))
    if np.linalg.matrix_rank(x) != 2:
        raise ValueError("Reduced benchmark has a rank-deficient design")
    y: Array = np.asarray([row.change for row in rows], dtype=float)
    coefficients: Array = np.linalg.lstsq(x, y, rcond=None)[0]
    test_feature: float = margin if feature == "generic" else seats
    return float(np.array([1.0, test_feature]) @ coefficients)


def expanding_reduced_backtest(
    elections: tuple[Election, ...], minimum_train: int = 8,
) -> tuple[ReducedHoldout, ...]:
    """Use identical historical origins for two simpler predictive rules."""

    if isinstance(minimum_train, bool) or not isinstance(minimum_train, int):
        raise TypeError("minimum_train must be an integer")
    if not 5 <= minimum_train < len(elections):
        raise ValueError("Training window must leave at least one future election")
    return tuple(
        ReducedHoldout(
            year=elections[i].year,
            actual=elections[i].change,
            train_count=i,
            generic_only=reduced_predict(
                elections[:i], "generic", margin=elections[i].party_margin,
                seats=elections[i].seats,
            ),
            seats_only=reduced_predict(
                elections[:i], "seats", margin=elections[i].party_margin,
                seats=elections[i].seats,
            ),
        )
        for i in range(minimum_train, len(elections))
    )


def influence(model: OLSFit) -> tuple[Array, Array, Array]:
    """Return leverage, internally studentized residuals, and Cook's distance."""

    xh: Array = model.design @ model.xtx_inverse
    leverage: Array = np.sum(xh * model.design, axis=1)
    if np.any(leverage >= 1):
        raise ArithmeticError("Leverage cannot reach one for leave-one-out analysis")
    standardized: Array = model.residuals / np.sqrt(model.residual_variance * (1.0 - leverage))
    cooks: Array = (
        (model.residuals ** 2) / (model.design.shape[1] * model.residual_variance)
        * leverage / ((1.0 - leverage) ** 2)
    )
    return leverage, standardized, cooks


def bootstrap_future(
    elections: tuple[Election, ...], scenario: Scenario,
    draws: int = 5000, seed: int = 20261031,
) -> Array:
    """Pairs bootstrap plus centered residual draw for a future seat change.

    This is an illustrative iid-bootstrap sensitivity analysis; no bootstrap
    is automatically robust to political regime change or polling drift.
    """

    if not isinstance(draws, int) or isinstance(draws, bool) or draws < 100:
        raise ValueError("At least 100 positive integer bootstrap draws required")
    rng: np.random.Generator = np.random.default_rng(seed)
    output: Array = np.empty(draws, dtype=float)
    completed: int = 0
    attempts: int = 0
    while completed < draws:
        attempts += 1
        if attempts > draws * 20:
            raise ArithmeticError("Repeated bootstrap design rank failures")
        indices: Array = rng.integers(0, len(elections), size=len(elections))
        rows: tuple[Election, ...] = tuple(elections[int(i)] for i in indices)
        try:
            model: OLSFit = fit_model(rows)
        except ValueError:
            continue
        noise: Array = model.residuals - np.mean(model.residuals)
        shock: float = float(rng.choice(noise))
        output[completed] = model.predict(scenario.party_margin, scenario.seats) + shock
        completed += 1
    return output


def alternative_gallup(
    elections: tuple[Election, ...],
    path: Path = DATA / "gallup_alternative.csv",
) -> tuple[Election, ...]:
    """Replace only explicitly sourced conflicting retrospective Gallup cells."""

    with path.open(encoding="utf-8", newline="") as handle:
        reader: csv.DictReader[str] = csv.DictReader(handle)
        expected: tuple[str, ...] = (
            "year", "alternative_generic_dem_minus_rep", "source", "reason",
        )
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError("Alternative Gallup file contract changed")
        revisions: dict[int, tuple[float, str]] = {}
        for row in reader:
            year: int = int(row["year"])
            if year in revisions or not row["reason"]:
                raise ValueError("Duplicate or unexplained Gallup discrepancy")
            revisions[year] = (float(row["alternative_generic_dem_minus_rep"]), row["source"])
    years: set[int] = {election.year for election in elections}
    if not set(revisions).issubset(years):
        raise ValueError("An alternative source has an unobserved election")
    return tuple(
        Election(
            year=e.year, party=e.party, seats=e.seats, change=e.change,
            dem_margin=revisions[e.year][0] if e.year in revisions else e.dem_margin,
            source=revisions[e.year][1] if e.year in revisions else e.source,
        )
        for e in elections
    )


def _errors(actual: Array, predicted: Array) -> dict[str, float]:
    """Report two loss scales; R² may be negative on held-out observations."""

    residual: Array = predicted - actual
    tss: float = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - np.sum(residual ** 2) / tss) if tss > 0 else float("nan"),
    }


def compute_evidence(draws: int = 5000, seed: int = 20261031) -> Evidence:
    """Compute every quantitative claim from the frozen source transcripts."""

    elections: tuple[Election, ...] = read_elections()
    scenario: Scenario = read_scenario()
    if scenario.year <= elections[-1].year or scenario.year in {r.year for r in elections}:
        raise ValueError("Forecast cannot use an election already in training data")
    model: OLSFit = fit_model(elections)
    loocv: tuple[Holdout, ...] = leave_one_out(elections)
    forward: tuple[Holdout, ...] = expanding_backtest(elections)
    reduced_forward: tuple[ReducedHoldout, ...] = expanding_reduced_backtest(elections)
    if [(row.year, row.actual, row.train_count) for row in forward] != [
        (row.year, row.actual, row.train_count) for row in reduced_forward
    ]:
        raise ArithmeticError("Forward benchmark test sets do not align")
    alternative_elections: tuple[Election, ...] = alternative_gallup(elections)
    alternative: OLSFit = fit_model(alternative_elections)
    alternative_forward: tuple[Holdout, ...] = expanding_backtest(alternative_elections)
    alternative_reduced: tuple[ReducedHoldout, ...] = expanding_reduced_backtest(alternative_elections)
    bootstrap: Array = bootstrap_future(elections, scenario, draws=draws, seed=seed)
    y: Array = model.outcomes
    heldout_actual: Array = np.array([row.actual for row in loocv], dtype=float)
    heldout_pred: Array = np.array([row.predicted for row in loocv], dtype=float)
    forward_actual: Array = np.array([row.actual for row in forward], dtype=float)
    forward_pred: Array = np.array([row.predicted for row in forward], dtype=float)
    baseline_pred: Array = np.array([row.baseline for row in forward], dtype=float)
    generic_pred: Array = np.array([row.generic_only for row in reduced_forward], dtype=float)
    seats_pred: Array = np.array([row.seats_only for row in reduced_forward], dtype=float)
    loo_metrics: dict[str, float] = _errors(heldout_actual, heldout_pred)
    forward_metrics: dict[str, float] = _errors(forward_actual, forward_pred)
    baseline_metrics: dict[str, float] = _errors(forward_actual, baseline_pred)
    generic_metrics: dict[str, float] = _errors(forward_actual, generic_pred)
    seats_metrics: dict[str, float] = _errors(forward_actual, seats_pred)
    alternative_full_pred: Array = np.array([row.predicted for row in alternative_forward], dtype=float)
    alternative_generic_pred: Array = np.array([row.generic_only for row in alternative_reduced], dtype=float)
    alternative_actual: Array = np.array([row.actual for row in alternative_forward], dtype=float)
    if [row.year for row in alternative_forward] != [row.year for row in forward]:
        raise ArithmeticError("Alternative Gallup and primary fold years do not match")
    alternative_forward_metrics: dict[str, float] = _errors(alternative_actual, alternative_full_pred)
    alternative_generic_metrics: dict[str, float] = _errors(alternative_actual, alternative_generic_pred)
    q90: tuple[float, float] = prediction_interval(model, scenario.party_margin, scenario.seats, 0.9)
    q95: tuple[float, float] = prediction_interval(model, scenario.party_margin, scenario.seats, 0.95)
    forecast_loo: list[float] = [
        fit_model(elections[:i] + elections[i + 1:]).predict(scenario.party_margin, scenario.seats)
        for i in range(len(elections))
    ]
    leverage, standardized, cooks = influence(model)
    metrics: dict[str, Any] = {
        "as_of": "2026-10-31", "data_period": "1950-2022", "observations": len(elections),
        "missing_1986": True, "predictors_excluding_intercept": 2,
        "model_coefficients": [float(beta) for beta in model.coefficients],
        "degrees_of_freedom": model.degrees_of_freedom,
        "in_sample_r2": model.r_squared,
        "in_sample_rmse": _errors(y, model.fitted)["rmse"],
        "in_sample_mae": _errors(y, model.fitted)["mae"],
        "loocv_rmse": loo_metrics["rmse"], "loocv_mae": loo_metrics["mae"],
        "loocv_r2": loo_metrics["r2"],
        "forward_test_count": len(forward),
        "forward_rmse": forward_metrics["rmse"],
        "forward_mae": forward_metrics["mae"],
        "forward_r2": forward_metrics["r2"],
        "forward_mean_baseline_rmse": baseline_metrics["rmse"],
        "forward_mean_baseline_mae": baseline_metrics["mae"],
        "forward_mean_baseline_r2": baseline_metrics["r2"],
        "forward_generic_only_rmse": generic_metrics["rmse"],
        "forward_generic_only_mae": generic_metrics["mae"],
        "forward_generic_only_r2": generic_metrics["r2"],
        "forward_seats_only_rmse": seats_metrics["rmse"],
        "forward_seats_only_mae": seats_metrics["mae"],
        "forward_seats_only_r2": seats_metrics["r2"],
        "forward_full_beats_generic_absolute_count": int(np.sum(
            np.abs(forward_pred - forward_actual) < np.abs(generic_pred - forward_actual)
        )),
        "loocv_90_coverage": float(np.mean([row.lower_90 <= row.actual <= row.upper_90 for row in loocv])),
        "loocv_95_coverage": float(np.mean([row.lower_95 <= row.actual <= row.upper_95 for row in loocv])),
        "forward_90_coverage": float(np.mean([row.lower_90 <= row.actual <= row.upper_90 for row in forward])),
        "forward_95_coverage": float(np.mean([row.lower_95 <= row.actual <= row.upper_95 for row in forward])),
        "scenario_party_margin": scenario.party_margin,
        "scenario_seats": scenario.seats,
        "forecast": model.predict(scenario.party_margin, scenario.seats),
        "forecast_generic_only": reduced_predict(
            elections, "generic", margin=scenario.party_margin, seats=scenario.seats,
        ),
        "forecast_seats_only": reduced_predict(
            elections, "seats", margin=scenario.party_margin, seats=scenario.seats,
        ),
        "original_rattner_forecast": -33,
        "parametric_90_interval": list(q90),
        "parametric_95_interval": list(q95),
        "bootstrap_draws": draws, "bootstrap_seed": seed,
        "bootstrap_90_interval": [float(v) for v in np.quantile(bootstrap, [0.05, 0.95])],
        "bootstrap_95_interval": [float(v) for v in np.quantile(bootstrap, [0.025, 0.975])],
        "alternative_gallup_r2": alternative.r_squared,
        "alternative_gallup_forecast": alternative.predict(scenario.party_margin, scenario.seats),
        "alternative_forward_rmse": alternative_forward_metrics["rmse"],
        "alternative_forward_generic_only_rmse": alternative_generic_metrics["rmse"],
        "alternative_forward_full_beats_generic_absolute_count": int(np.sum(
            np.abs(alternative_full_pred - alternative_actual)
            < np.abs(alternative_generic_pred - alternative_actual)
        )),
        "forecast_omit_one_year_min": min(forecast_loo),
        "forecast_omit_one_year_max": max(forecast_loo),
        "largest_cook_year": elections[int(np.argmax(cooks))].year,
        "max_cooks_distance": float(np.max(cooks)),
        "max_leverage": float(np.max(leverage)),
        "max_studentized_absolute_residual": float(np.max(np.abs(standardized))),
    }
    return Evidence(elections, scenario, model, loocv, forward, reduced_forward,
                    bootstrap, alternative, metrics)


def _write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> None:
    """Write stable, machine-readable numeric tables with explicit column order."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer: csv.DictWriter[str] = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_tables(evidence: Evidence, directory: Path) -> None:
    """Materialize all published figures' underlying numbers and metrics."""

    directory.mkdir(parents=True, exist_ok=True)
    for row_name, results in (("loocv_predictions.csv", evidence.leave_one_out),
                              ("forward_predictions.csv", evidence.forward)):
        fields: tuple[str, ...] = (
            "year", "actual", "predicted", "error", "baseline", "train_count",
            "lower_90", "upper_90", "lower_95", "upper_95",
        )
        _write_csv(directory / row_name, fields, (
            {name: (row.error if name == "error" else getattr(row, name)) for name in fields}
            for row in results
        ))
    _write_csv(
        directory / "forward_model_comparison.csv",
        ("year", "actual", "train_count", "two_predictor", "generic_only",
         "seats_only", "historical_mean"),
        (
            {"year": full.year, "actual": full.actual,
             "train_count": full.train_count, "two_predictor": full.predicted,
             "generic_only": reduced.generic_only, "seats_only": reduced.seats_only,
             "historical_mean": full.baseline}
            for full, reduced in zip(evidence.forward, evidence.reduced_forward, strict=True)
        ),
    )
    _write_csv(
        directory / "historical_fits.csv",
        ("year", "president_party", "house_seats_defended", "party_margin",
         "actual_change", "fitted_change", "residual", "poll_source"),
        (
            {"year": e.year, "president_party": e.party, "house_seats_defended": e.seats,
             "party_margin": e.party_margin, "actual_change": e.change,
             "fitted_change": evidence.model.fitted[i],
             "residual": evidence.model.residuals[i], "poll_source": e.source}
            for i, e in enumerate(evidence.elections)
        ),
    )
    leverage, standardized, cooks = influence(evidence.model)
    _write_csv(
        directory / "influence_diagnostics.csv",
        ("year", "leverage", "studentized_residual", "cooks_distance"),
        (
            {"year": e.year, "leverage": leverage[i],
             "studentized_residual": standardized[i], "cooks_distance": cooks[i]}
            for i, e in enumerate(evidence.elections)
        ),
    )
    interval_values: list[dict[str, Any]] = []
    for seat_count in (210, 215, 220, 225, 230):
        for dem_margin in np.arange(4.0, 12.01, 1.0):
            party_margin: float = -dem_margin  # 2026 president's party is Republican.
            lo, hi = prediction_interval(evidence.model, party_margin, seat_count)
            interval_values.append({
                "democratic_generic_advantage": dem_margin,
                "republican_seats_defended": seat_count,
                "predicted_republican_seat_change": evidence.model.predict(party_margin, seat_count),
                "model_based_95_lower": lo, "model_based_95_upper": hi,
            })
    _write_csv(
        directory / "scenario_grid.csv",
        ("democratic_generic_advantage", "republican_seats_defended",
         "predicted_republican_seat_change", "model_based_95_lower", "model_based_95_upper"),
        interval_values,
    )
    coefficient_names: tuple[str, ...] = ("intercept", "president_party_ballot_margin", "seats_defended")
    errors: Array = np.sqrt(np.diag(evidence.model.xtx_inverse) * evidence.model.residual_variance)
    threshold: float = float(student_t.ppf(0.975, evidence.model.degrees_of_freedom))
    _write_csv(
        directory / "coefficient_intervals.csv",
        ("term", "estimate", "standard_error", "model_based_95_lower", "model_based_95_upper"),
        (
            {"term": term, "estimate": evidence.model.coefficients[i],
             "standard_error": errors[i],
             "model_based_95_lower": evidence.model.coefficients[i] - threshold * errors[i],
             "model_based_95_upper": evidence.model.coefficients[i] + threshold * errors[i]}
            for i, term in enumerate(coefficient_names)
        ),
    )
    _write_csv(
        directory / "source_sensitivity.csv",
        ("historical_gallup_source", "observations", "r_squared", "forecast_2026",
         "forward_rmse", "forward_generic_only_rmse", "full_beats_generic_absolute_count"),
        (
            {"historical_gallup_source": source, "observations": len(evidence.elections),
             "r_squared": model.r_squared,
             "forecast_2026": model.predict(evidence.scenario.party_margin, evidence.scenario.seats),
             "forward_rmse": evidence.metrics[full_key],
             "forward_generic_only_rmse": evidence.metrics[generic_key],
             "full_beats_generic_absolute_count": evidence.metrics[wins_key]}
            for source, model, full_key, generic_key, wins_key in (
                ("Gallup_2002_primary", evidence.model,
                 "forward_rmse", "forward_generic_only_rmse",
                 "forward_full_beats_generic_absolute_count"),
                ("Gallup_2004_alternative", evidence.alternative_model,
                 "alternative_forward_rmse", "alternative_forward_generic_only_rmse",
                 "alternative_forward_full_beats_generic_absolute_count"),
            )
        ),
    )
    (directory / "summary.json").write_text(
        json.dumps(evidence.metrics, indent=2, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Generate all numerical evidence without introducing network access."""

    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT, help="Output root with results/")
    parser.add_argument("--draws", type=int, default=5000, help="Seeded bootstrap replications")
    args: argparse.Namespace = parser.parse_args()
    evidence: Evidence = compute_evidence(draws=args.draws)
    save_tables(evidence, args.output / "results")
    for key in (
        "observations", "in_sample_r2", "in_sample_rmse", "loocv_rmse",
        "loocv_r2", "forward_rmse", "forward_mean_baseline_rmse", "forecast",
        "forward_generic_only_rmse", "forward_seats_only_rmse",
        "parametric_95_interval", "bootstrap_95_interval", "alternative_gallup_forecast",
    ):
        print(f"{key}: {evidence.metrics[key]}")


if __name__ == "__main__":
    main()
