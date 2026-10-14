"""Classical, descriptive Okun regressions using explicit calendar-quarter lags."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from eu_okun.audit import audit_snapshot
from eu_okun.contract import EU27, EurostatDataError, quarter_sequence


@dataclass(frozen=True)
class RegressionResult:
    """OLS coefficients with Newey-West standard errors (not causal effects)."""

    geo: str
    lags: int
    participation: bool
    n: int
    start: str
    end: str
    coefficients: dict[str, float]
    hac_se: dict[str, float]
    r_squared: float
    condition_number: float


def _quarter_before(quarter: str, steps: int) -> str:
    """Shift a period identifier by exactly the requested calendar quarters."""
    idx = int(quarter[:4]) * 4 + int(quarter[-1]) - 1 - steps
    return f"{idx // 4:04d}-Q{idx % 4 + 1}"


def _read_panel(path: Path) -> dict[tuple[str, str], dict[str, float | None]]:
    """Parse the vetted quarterly CSV without filling missing values."""
    data: dict[tuple[str, str], dict[str, float | None]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["geo"], row["quarter"])
            if key in data:
                raise EurostatDataError(f"Duplicate panel key {key}.")
            data[key] = {
                name: float(row[column]) if row[column] else None
                for name, column in (
                    ("gdp", "gdp_growth_qoq_pct"),
                    ("unemployment", "unemployment_rate_pct"),
                    ("participation", "participation_rate_pct"),
                )
            }
    return data


def design_matrix(
    panel: dict[tuple[str, str], dict[str, float | None]],
    country: str,
    quarters: list[str],
    lags: int,
    participation: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Construct complete cases with *calendar*, not compressed, GDP lags."""
    if lags < 0 or country not in EU27:
        raise ValueError("Invalid country or lag count.")
    predictors: list[list[float]] = []
    response: list[float] = []
    dates: list[str] = []
    for quarter in quarters:
        now = panel.get((country, quarter))
        previous = panel.get((country, _quarter_before(quarter, 1)))
        if now is None or previous is None:
            continue
        unemployment = now["unemployment"]
        prev_unemployment = previous["unemployment"]
        if unemployment is None or prev_unemployment is None:
            continue
        values: list[float] = [1.0]
        complete = True
        for lag in range(lags + 1):
            lagged = panel.get((country, _quarter_before(quarter, lag)))
            value = None if lagged is None else lagged["gdp"]
            if value is None:
                complete = False
                break
            values.append(value)
        if not complete:
            continue
        if participation:
            current_p, previous_p = now["participation"], previous["participation"]
            if current_p is None or previous_p is None:
                continue
            values.append(current_p - previous_p)
        predictors.append(values)
        response.append(unemployment - prev_unemployment)
        dates.append(quarter)
    return np.asarray(predictors, dtype=float).reshape(-1, lags + 2 + int(participation)), np.asarray(
        response, dtype=float
    ), dates


def fit_hac(
    x: np.ndarray, y: np.ndarray, maxlags: int = 4,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit full-rank OLS and Bartlett-kernel HAC covariance (no small-sample scaling).

    Regressors are fixed observed covariates. The covariance uses the usual
    sandwich (X'X)^(-1) S (X'X)^(-1), with Bartlett weights and uncorrected
    OLS residuals. This is descriptive, and requires a sufficiently long series.
    """
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.size:
        raise ValueError("Incompatible design and response dimensions.")
    n, k = x.shape
    if n < max(12, 3 * k) or maxlags < 0 or maxlags >= n:
        raise ValueError("Insufficient observations or invalid HAC bandwidth.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite inputs are not allowed.")
    if np.linalg.matrix_rank(x) != k:
        raise ValueError("The regression design is rank deficient.")
    condition_number = float(np.linalg.cond(x))
    if condition_number > 1e10:
        raise ValueError("Ill-conditioned regression design.")
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ beta
    scores = x * residuals[:, None]
    meat = scores.T @ scores
    for lag in range(1, maxlags + 1):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1.0 - lag / (maxlags + 1.0)) * (cross + cross.T)
    bread = np.linalg.inv(x.T @ x)
    covariance = bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float("nan") if total <= 0 else 1.0 - float(np.sum(residuals**2)) / total
    return beta, se, r_squared, condition_number


def run_baselines(
    directory: Path, lag_orders: tuple[int, ...] = (0, 1, 2, 4),
    hac_lags: int = 4,
) -> list[RegressionResult]:
    """Require a validated frozen snapshot and fit eligible country models."""
    report = audit_snapshot(directory)
    start = report["analysis_period"]["start"]
    end = report["analysis_period"]["end"]
    if not isinstance(start, str) or not isinstance(end, str):
        raise EurostatDataError("Invalid analysis period.")
    panel = _read_panel(directory / "processed" / "eu27_quarterly.csv")
    quarters = quarter_sequence(start, end)
    fits: list[RegressionResult] = []
    for country in EU27:
        for lags in lag_orders:
            for participation in (False, True):
                x, y, periods = design_matrix(panel, country, quarters, lags, participation)
                if not periods:
                    continue
                try:
                    beta, se, r_squared, condition = fit_hac(x, y, maxlags=hac_lags)
                except ValueError:
                    continue
                names = ["intercept"] + [f"gdp_lag_{j}" for j in range(lags + 1)]
                if participation:
                    names.append("participation_change")
                fits.append(
                    RegressionResult(
                        country, lags, participation, len(y), periods[0], periods[-1],
                        dict(zip(names, map(float, beta), strict=True)),
                        dict(zip(names, map(float, se), strict=True)),
                        r_squared, condition,
                    )
                )
    return fits


def write_baselines(directory: Path, output: Path) -> int:
    """Save descriptive estimates with snapshot provenance; fail if none are estimable."""
    results = run_baselines(directory)
    if not results:
        raise EurostatDataError("No estimable country models; inspect data audit first.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "description": "Descriptive OLS with Bartlett HAC uncertainty; not causal",
                "models": [asdict(result) for result in results],
            },
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    return len(results)
