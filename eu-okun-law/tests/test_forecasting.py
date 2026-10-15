"""Leakage and benchmark contracts for expanding-window validation."""

from __future__ import annotations

import numpy as np
import pytest

from eu_okun.forecasting import (
    _forecast_model,
    _lagged_example,
    complete_panel_examples,
    expanding_forecasts,
    summarize_forecasts,
)


def synthetic_panel() -> dict[tuple[str, str], dict[str, float | None]]:
    """Generate deterministic data with a predictable lagged GDP relationship."""
    rng = np.random.default_rng(42)
    panel = {}
    quarters = [f"{year}-Q{q}" for year in range(2008, 2027) for q in range(1, 5)]
    for geo in ("PT", "ES"):
        u = 9.0
        previous_gdp = 0.0
        for quarter in quarters:
            gdp = float(rng.normal())
            u += -0.2 * previous_gdp + float(rng.normal(scale=0.05))
            panel[(geo, quarter)] = {
                "gdp": gdp, "unemployment": u, "participation": 70.0 + 0.01 * len(panel),
            }
            previous_gdp = gdp
    return panel


def test_only_lagged_predictors_used() -> None:
    panel = synthetic_panel()
    a = _lagged_example(panel, "PT", "2025-Q1")
    assert a is not None
    panel[("PT", "2025-Q1")]["gdp"] = 99999.0
    panel[("PT", "2025-Q1")]["participation"] = 99999.0
    b = _lagged_example(panel, "PT", "2025-Q1")
    assert b is not None
    assert a == b


def test_future_target_does_not_change_earlier_forecast() -> None:
    panel = synthetic_panel()
    first, dates = expanding_forecasts(
        panel, "2026-Q2", countries=("PT", "ES"), test_start="2022-Q1"
    )
    assert len(dates) >= 5
    earliest = dates[0]
    earlier_predictions = [
        (r.geo, r.model, r.predicted_du_pp) for r in first
        if r.target_quarter == earliest
    ]
    panel[("PT", "2026-Q2")]["unemployment"] = 90.0
    second, _ = expanding_forecasts(
        panel, "2026-Q2", countries=("PT", "ES"), test_start="2022-Q1"
    )
    assert earlier_predictions == [
        (r.geo, r.model, r.predicted_du_pp) for r in second
        if r.target_quarter == earliest
    ]
    assert all(r.train_last_quarter < r.target_quarter for r in second)
    assert all(r.information_cutoff < r.target_quarter for r in second)


def test_country_intersection_removes_missing_quarter() -> None:
    panel = synthetic_panel()
    quarters = [f"2025-Q{q}" for q in range(1, 5)]
    panel[("ES", "2025-Q2")]["gdp"] = None
    _, common = complete_panel_examples(panel, quarters, ("PT", "ES"))
    assert "2025-Q3" not in common


def test_zero_change_baseline_and_scores() -> None:
    panel = synthetic_panel()
    forecasts, _ = expanding_forecasts(
        panel, "2026-Q2", countries=("PT", "ES"), test_start="2022-Q1"
    )
    assert all(row.predicted_du_pp == 0.0 for row in forecasts
               if row.model == "zero_change")
    scores = summarize_forecasts(forecasts)
    assert len(scores) == 12
    assert all(np.isfinite(score.mae_pp) for score in scores)
    assert all(score.n == 18 for score in scores)


def test_unknown_model_fails() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        _forecast_model([], object(), "future_gdp_model")  # type: ignore[arg-type]
