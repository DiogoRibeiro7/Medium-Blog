"""Contract tests for the independent zero-cure simulation audit."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from no_cure_audit import (  # noqa: E402
    annotate,
    format_summary,
    run_experiment,
    summarize,
    wilson_interval,
)
from simulate import simulate_dataset  # noqa: E402


def fake_raw() -> pd.DataFrame:
    """Two deterministic rows exercising both diagnostic decisions."""
    return pd.DataFrame(
        {
            "pi": [0.0, 0.0],
            "latency": ["lognormal-1.0", "lognormal-1.0"],
            "n": [250, 250],
            "ratio": [0.5, 0.5],
            "rep": [0, 1],
            "events": [80, 100],
            "pi_best": [0.30, 0.04],
            "family_gap": [0.02, 0.12],
            "cure_preferred": [1, 0],
            "converged": [1, 1],
        }
    )


def test_zero_cure_generator_is_reproducible() -> None:
    first: tuple[np.ndarray, np.ndarray] = simulate_dataset(
        0.0, "lognormal-1.0", 300, 1.0, np.random.default_rng(404)
    )
    second: tuple[np.ndarray, np.ndarray] = simulate_dataset(
        0.0, "lognormal-1.0", 300, 1.0, np.random.default_rng(404)
    )
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[1].sum() > 0


def test_combined_rule_counts_only_false_cure_reports() -> None:
    rows: pd.DataFrame = annotate(fake_raw())
    assert rows["false_positive_fit"].tolist() == [1, 0]
    assert rows["screen_pass"].tolist() == [1, 0]
    assert rows["false_cure_report"].tolist() == [1, 0]
    cells: pd.DataFrame = summarize(fake_raw())
    assert cells.iloc[0]["false_cure_report"] == 0.5
    assert cells.iloc[0]["false_report_ci_low"] < 0.5
    assert cells.iloc[0]["false_report_ci_high"] > 0.5
    assert cells.iloc[0]["mean_event_fraction"] == pytest.approx(0.36)
    assert "false report 0.500" in format_summary(fake_raw(), cells)


def test_invalid_or_nonzero_cure_audit_is_rejected() -> None:
    with pytest.raises(ValueError, match="Missing audit columns"):
        annotate(pd.DataFrame({"pi": [0.0]}))
    with pytest.raises(ValueError, match="requires pi=0"):
        annotate(fake_raw().assign(pi=0.2))
    with pytest.raises(ValueError, match="nonempty"):
        annotate(fake_raw().iloc[:0])
    with pytest.raises(ValueError, match="positive integer"):
        run_experiment(0, 1)
    with pytest.raises(ValueError, match="positive integer"):
        run_experiment(1, 0)


def test_small_real_grid_uses_only_zero_cure_cases() -> None:
    # One replicate per grid cell is sufficient to catch API regressions.
    # Full evidence and numerical convergence are evaluated by CI.
    raw: pd.DataFrame = run_experiment(reps=1, workers=1)
    assert len(raw) == 24
    assert (raw["pi"] == 0).all()
    assert len(summarize(raw)) == 24


def test_wilson_interval_boundaries() -> None:
    lower, upper = wilson_interval(24, 100)
    assert lower == pytest.approx(0.167, abs=0.002)
    assert upper == pytest.approx(0.332, abs=0.002)
    assert wilson_interval(0, 100)[0] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        wilson_interval(2, 1)
