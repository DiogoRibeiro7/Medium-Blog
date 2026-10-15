"""Quarter-ahead pseudo-out-of-sample forecasts of unemployment-rate changes.

No model is fitted to the held-out target or uses contemporaneous GDP,
unemployment, or participation as a predictor. All features refer to t-1 or
earlier; macroeconomic revisions and release lags are NOT reconstructed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eu_okun.audit import audit_snapshot
from eu_okun.baseline import _quarter_before, _read_panel
from eu_okun.contract import EU27, EurostatDataError, quarter_sequence

TRAIN_START = "2009-Q1"
TEST_START = "2018-Q1"
MIN_TRAIN = 32
MODEL_FEATURES: dict[str, tuple[str, ...]] = {
    "zero_change": (),
    "historical_mean": (),
    "unemployment_ar1": ("du_lag1",),
    "gdp_lags": ("gdp_lag1", "gdp_lag2"),
    "gdp_unemployment_ar1": ("du_lag1", "gdp_lag1", "gdp_lag2"),
    "gdp_ar1_participation": ("du_lag1", "gdp_lag1", "gdp_lag2", "dp_lag1"),
}
BASELINE_MODEL = "unemployment_ar1"
GDP_MODEL = "gdp_unemployment_ar1"

Panel = dict[tuple[str, str], dict[str, float | None]]


@dataclass(frozen=True)
class ForecastExample:
    """A complete outcome and explicitly lagged feature row."""

    quarter: str
    actual_du: float
    last_unemployment: float
    du_lag1: float
    dp_lag1: float
    gdp_lag1: float
    gdp_lag2: float


@dataclass(frozen=True)
class ForecastRecord:
    """One forecast with its training cutoff and common evaluation target."""

    geo: str
    target_quarter: str
    information_cutoff: str
    train_last_quarter: str
    train_n: int
    model: str
    actual_du_pp: float
    predicted_du_pp: float
    last_unemployment_pct: float
    predicted_unemployment_pct: float


@dataclass(frozen=True)
class Score:
    """Predictive errors in percentage points, without post-hoc model selection."""

    geo: str
    model: str
    n: int
    mae_pp: float
    rmse_pp: float
    bias_pp: float
    skill_vs_zero: float
    skill_vs_ar1: float


def _lagged_example(panel: Panel, geo: str, quarter: str) -> ForecastExample | None:
    """Compute target and features, with every feature dated no later than t-1."""
    p1 = _quarter_before(quarter, 1)
    p2 = _quarter_before(quarter, 2)
    now, previous, earlier = (
        panel.get((geo, quarter)),
        panel.get((geo, p1)),
        panel.get((geo, p2)),
    )
    if now is None or previous is None or earlier is None:
        return None
    current_u = now["unemployment"]
    prev_u, older_u = previous["unemployment"], earlier["unemployment"]
    prev_p, older_p = previous["participation"], earlier["participation"]
    g1, g2 = previous["gdp"], earlier["gdp"]
    if any(v is None for v in (current_u, prev_u, older_u, prev_p, older_p, g1, g2)):
        return None
    # The guards above establish all fields are observed. Convert explicitly
    # so missing data never silently becomes a zero-valued feature.
    assert current_u is not None and prev_u is not None and older_u is not None
    assert prev_p is not None and older_p is not None and g1 is not None and g2 is not None
    return ForecastExample(
        quarter=quarter,
        actual_du=current_u - prev_u,
        last_unemployment=prev_u,
        du_lag1=prev_u - older_u,
        dp_lag1=prev_p - older_p,
        gdp_lag1=g1,
        gdp_lag2=g2,
    )


def complete_panel_examples(
    panel: Panel,
    quarters: Sequence[str],
    countries: Sequence[str] = EU27,
) -> tuple[dict[str, dict[str, ForecastExample]], list[str]]:
    """Intersect target quarters across all countries and predictor definitions."""
    if not countries or len(set(countries)) != len(countries):
        raise ValueError("Countries must be unique and nonempty.")
    if list(quarters) != sorted(set(quarters)):
        raise ValueError("Target quarters must be sorted and unique.")
    examples: dict[str, dict[str, ForecastExample]] = {}
    valid = set(quarters)
    for country in countries:
        country_rows: dict[str, ForecastExample] = {}
        for quarter in quarters:
            record = _lagged_example(panel, country, quarter)
            if record is not None:
                country_rows[quarter] = record
        examples[country] = country_rows
        valid.intersection_update(country_rows)
    common = [q for q in quarters if q in valid]
    return examples, common


def _features(record: ForecastExample, names: tuple[str, ...]) -> list[float]:
    """Extract an explicit approved predictor set (no current-quarter inputs)."""
    allowed = {
        "du_lag1": record.du_lag1,
        "dp_lag1": record.dp_lag1,
        "gdp_lag1": record.gdp_lag1,
        "gdp_lag2": record.gdp_lag2,
    }
    return [allowed[key] for key in names]


def _forecast_model(
    training: Sequence[ForecastExample], target: ForecastExample, model: str,
) -> float:
    """Estimate with training outcomes only and predict the withheld quarter."""
    if model not in MODEL_FEATURES:
        raise ValueError(f"Unknown forecast model: {model}.")
    if model == "zero_change":
        return 0.0
    responses = np.asarray([row.actual_du for row in training], dtype=float)
    if model == "historical_mean":
        return float(responses.mean())
    predictors = MODEL_FEATURES[model]
    x = np.asarray([[1.0, *_features(row, predictors)] for row in training], dtype=float)
    if x.shape[0] < max(MIN_TRAIN, 3 * x.shape[1]):
        raise ValueError("Insufficient training observations.")
    if np.linalg.matrix_rank(x) != x.shape[1] or np.linalg.cond(x) > 1e10:
        raise ValueError("Rank-deficient or ill-conditioned training regression.")
    coefficients, *_ = np.linalg.lstsq(x, responses, rcond=None)
    forecast = float(np.dot(coefficients, [1.0, *_features(target, predictors)]))
    if not math.isfinite(forecast):
        raise ValueError("Nonfinite model prediction.")
    return forecast


def expanding_forecasts(
    panel: Panel,
    end_quarter: str,
    countries: Sequence[str] = EU27,
    test_start: str = TEST_START,
    min_train: int = MIN_TRAIN,
) -> tuple[list[ForecastRecord], list[str]]:
    """Evaluate every country/model on identical held-out calendar quarters.

    Train each regression on samples strictly preceding the forecast quarter.
    GDP and participation predictors use t-1 or t-2 only. The same historical
    complete-case intersection is used for every candidate model/country.
    """
    if min_train < MIN_TRAIN or test_start < TRAIN_START:
        raise ValueError("Invalid minimum training size or test start.")
    quarters = quarter_sequence(TRAIN_START, end_quarter)
    examples, common = complete_panel_examples(panel, quarters, countries)
    forecast_quarters = [
        q for i, q in enumerate(common) if q >= test_start and i >= min_train
    ]
    if len(forecast_quarters) < 12:
        raise EurostatDataError("Fewer than 12 common held-out quarterly origins.")
    rows: list[ForecastRecord] = []
    for forecast_quarter in forecast_quarters:
        past = [q for q in common if q < forecast_quarter]
        if len(past) < min_train or past[-1] >= forecast_quarter:
            raise EurostatDataError("Non-chronological expanding-window split.")
        issue_cutoff = _quarter_before(forecast_quarter, 1)
        for geo in countries:
            fitted_rows = [examples[geo][q] for q in past]
            observed = examples[geo][forecast_quarter]
            for model in MODEL_FEATURES:
                try:
                    prediction = _forecast_model(fitted_rows, observed, model)
                except ValueError as exc:
                    raise EurostatDataError(
                        f"Forecast failure: {geo}, {forecast_quarter}, {model}: {exc}"
                    ) from exc
                rows.append(
                    ForecastRecord(
                        geo=geo,
                        target_quarter=forecast_quarter,
                        information_cutoff=issue_cutoff,
                        train_last_quarter=past[-1],
                        train_n=len(past),
                        model=model,
                        actual_du_pp=observed.actual_du,
                        predicted_du_pp=prediction,
                        last_unemployment_pct=observed.last_unemployment,
                        predicted_unemployment_pct=observed.last_unemployment + prediction,
                    )
                )
    return rows, forecast_quarters


def _errors(rows: Sequence[ForecastRecord]) -> np.ndarray:
    """Return signed unemployment-change forecast errors in percentage points."""
    return np.asarray([r.predicted_du_pp - r.actual_du_pp for r in rows], dtype=float)


def summarize_forecasts(rows: Sequence[ForecastRecord]) -> list[Score]:
    """Summarize every model against the same zero-change and AR1 targets."""
    groups: dict[tuple[str, str], list[ForecastRecord]] = defaultdict(list)
    for row in rows:
        groups[(row.geo, row.model)].append(row)
    countries = sorted({row.geo for row in rows})
    results: list[Score] = []
    for geo in countries:
        zero = groups[(geo, "zero_change")]
        ar1 = groups[(geo, BASELINE_MODEL)]
        if not zero or not ar1:
            raise EurostatDataError(f"Missing benchmark forecasts for {geo}.")
        dates = [r.target_quarter for r in zero]
        denominator_zero = float(np.abs(_errors(zero)).mean())
        denominator_ar1 = float(np.abs(_errors(ar1)).mean())
        if denominator_zero <= 0 or denominator_ar1 <= 0:
            raise EurostatDataError(f"Benchmark MAE is zero for {geo}.")
        for model in MODEL_FEATURES:
            sample = groups[(geo, model)]
            if [r.target_quarter for r in sample] != dates:
                raise EurostatDataError(f"Forecast origins differ for {geo}/{model}.")
            residuals = _errors(sample)
            mae = float(np.abs(residuals).mean())
            results.append(
                Score(
                    geo=geo,
                    model=model,
                    n=len(sample),
                    mae_pp=mae,
                    rmse_pp=float(np.sqrt(np.mean(residuals**2))),
                    bias_pp=float(np.mean(residuals)),
                    skill_vs_zero=1.0 - mae / denominator_zero,
                    skill_vs_ar1=1.0 - mae / denominator_ar1,
                )
            )
    return results


def write_validation(data_dir: Path, output_dir: Path) -> int:
    """Audit official sources, evaluate actual out-of-sample forecasts and persist."""
    audit = audit_snapshot(data_dir)
    end = audit["analysis_period"]["end"]
    if not isinstance(end, str):
        raise EurostatDataError("Invalid study endpoint.")
    panel = _read_panel(data_dir / "processed" / "eu27_quarterly.csv")
    forecasts, quarters = expanding_forecasts(panel, end)
    scores = summarize_forecasts(forecasts)
    if len(scores) != len(EU27) * len(MODEL_FEATURES):
        raise EurostatDataError("Incomplete EU-27 forecast model evaluation.")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "origin_forecasts.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(forecasts[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in forecasts)
    with (output_dir / "country_scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(scores[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in scores)
    payload: dict[str, Any] = {
        "design": "Expanding-window pseudo-out-of-sample, revised historical data",
        "claim": "Retrospective predictive association, not real-time vintage forecasting",
        "target": "Next-quarter unemployment-rate change in percentage points",
        "features": "Only t-1 and earlier; contemporaneous target-quarter GDP excluded",
        "training_start": TRAIN_START,
        "first_evaluation": quarters[0],
        "last_evaluation": quarters[-1],
        "evaluation_quarters": quarters,
        "min_train": MIN_TRAIN,
        "models": {key: list(value) for key, value in MODEL_FEATURES.items()},
        "source_manifest_sha256": hashlib.sha256(
            (data_dir / "snapshot_manifest.json").read_bytes()
        ).hexdigest(),
        "processed_csv_sha256": audit["processed_csv_sha256"],
        "countries": len(EU27),
        "forecast_count": len(forecasts),
        "scores": [asdict(score) for score in scores],
    }
    (output_dir / "scores.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return len(forecasts)
