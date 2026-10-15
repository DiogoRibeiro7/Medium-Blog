"""Economic-regime exclusions and minimum-sample diagnostics."""

from __future__ import annotations

from eu_okun.regimes import regime_comparisons


def test_short_incomplete_regimes_record_non_estimability() -> None:
    """Unavailable data must yield diagnostics, not invented estimates."""
    rows, windows = regime_comparisons({}, "2026-Q2")
    assert rows == []
    assert all(item["status"] == "not_estimable" for item in windows.values())


def test_pandemic_is_not_invented_when_lags_unavailable() -> None:
    """A sparse single-country panel cannot support EU-27 estimates."""
    panel = {
        ("PT", "2019-Q4"): {
            "gdp": 1.0, "unemployment": 6.0, "participation": 70.0
        },
        ("PT", "2022-Q1"): {
            "gdp": 2.0, "unemployment": 5.0, "participation": 71.0
        },
    }
    rows, info = regime_comparisons(panel, "2026-Q2")
    assert not rows
    assert info["pandemic_excluded"]["status"] == "not_estimable"
