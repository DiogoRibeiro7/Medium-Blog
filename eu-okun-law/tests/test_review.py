"""Tests of the independent verification and sensitivity procedures in ``review``."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from eu_okun.baseline import fit_hac_covariance
from eu_okun.comparison import cumulative_gdp
from eu_okun.contract import EU27, SERIES, EurostatDataError, quarter_sequence
from eu_okun.review import (
    SAMPLES,
    GridRow,
    ReviewResult,
    _numeric_difference,
    _sample_panel,
    flagged_change,
    independent_ols_hac,
    quarter_runs,
    reparameterized_sum,
    sample_designs,
    score_forecast_set,
    stability_designs,
    verdict_for,
)
from eu_okun.review_report import REPORT_NAME, names, render_report, write_review
from eu_okun.stability import fit_stability

Panel = dict[tuple[str, str], dict[str, float | None]]
QUARTERS = quarter_sequence("2008-Q1", "2026-Q2")
PANDEMIC_GDP = {"2020-Q1": -3.0, "2020-Q2": -12.0, "2020-Q3": 10.0, "2020-Q4": 1.0}


def synthetic_panel(countries: tuple[str, ...] = EU27, seed: int = 5) -> Panel:
    """Okun-like data: du = -0.3 g + noise, except muted responses in 2020."""
    rng = np.random.default_rng(seed)
    panel: Panel = {}
    for geo in countries:
        u, p, previous = 8.0, 70.0, 0.4
        for quarter in QUARTERS:
            g = PANDEMIC_GDP.get(quarter, float(rng.normal(0.4, 0.7)))
            if quarter.startswith("2020"):        # job retention: unemployment barely moves
                du = -0.02 * g + float(rng.normal(0, 0.05))
            else:
                du = -0.3 * g - 0.1 * previous + float(rng.normal(0, 0.15))
            u = min(max(u + du, 2.0), 25.0)
            p = p + float(rng.normal(0.02, 0.2))
            panel[(geo, quarter)] = {"gdp": g, "unemployment": u, "participation": p}
            previous = g
    return panel


def random_design(n: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.column_stack([np.ones(n), rng.normal(size=(n, k - 1))])
    y = x @ rng.normal(size=k) + rng.normal(size=n)
    return x, y


# ----------------------------------------------------------------------------------
# Independent numerical route
# ----------------------------------------------------------------------------------


def test_kernel_matrix_hac_matches_lag_loop_with_calendar_gaps() -> None:
    x, y = random_design(50, 4, seed=1)
    periods = [q for q in quarter_sequence("2010-Q1", "2023-Q2")][:54]
    for gap in ("2012-Q3", "2015-Q1", "2015-Q2", "2019-Q4"):
        periods.remove(gap)
    beta, covariance, _, _ = fit_hac_covariance(x, y, maxlags=4, periods=periods)
    own_beta, own_covariance = independent_ols_hac(x, y, 4, periods)
    np.testing.assert_allclose(own_beta, beta, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(own_covariance, covariance, rtol=1e-11, atol=1e-15)


def test_reparameterised_sum_equals_full_covariance_contrast() -> None:
    x, y = random_design(60, 5, seed=2)
    periods = quarter_sequence("2009-Q1", "2023-Q4")
    beta, covariance, _, _ = fit_hac_covariance(x, y, maxlags=4, periods=periods)
    total, se, _, _ = cumulative_gdp(beta, covariance, 3)
    own_total, own_se = reparameterized_sum(x, y, 3, 4, periods)
    assert own_total == pytest.approx(total, abs=1e-12)
    assert own_se == pytest.approx(se, abs=1e-12)


def test_reparameterised_sum_differs_from_summed_marginal_errors() -> None:
    """Guard against the classic mistake of adding marginal standard errors."""
    x, y = random_design(60, 4, seed=3)
    x[:, 2] = x[:, 1] + 0.1 * x[:, 2]           # strongly correlated GDP lags
    periods = quarter_sequence("2009-Q1", "2023-Q4")
    _, covariance, _, _ = fit_hac_covariance(x, y, maxlags=4, periods=periods)
    _, se = reparameterized_sum(x, y, 2, 4, periods)
    marginal = math.sqrt(covariance[1, 1]) + math.sqrt(covariance[2, 2])
    assert se < 0.5 * marginal


def test_pooled_stability_equals_separate_period_fits() -> None:
    panel = synthetic_panel(("PT", "ES"))
    rows, _ = fit_stability(panel, "2026-Q2", countries=("PT", "ES"))
    pre, pre_dates, post, post_dates = stability_designs(panel, "2026-Q2", ("PT", "ES"))
    for row in rows:
        key = (row.geo, 1, row.controls_participation)
        b_pre, v_pre = independent_ols_hac(*pre[key], 4, pre_dates)
        b_post, v_post = independent_ols_hac(*post[key], 4, post_dates)
        contrast = np.zeros(b_pre.size)
        contrast[1:3] = 1.0
        assert row.change_post_minus_pre == pytest.approx(contrast @ (b_post - b_pre))
        assert row.change_hac_se == pytest.approx(
            math.sqrt(contrast @ (v_pre + v_post) @ contrast)
        )


# ----------------------------------------------------------------------------------
# Samples, flags and verdicts
# ----------------------------------------------------------------------------------


def test_pandemic_exclusion_removes_outcomes_and_lag_history() -> None:
    panel = synthetic_panel(("PT", "ES"))
    spec = next(s for s in SAMPLES if s.name == "pandemic_excluded")
    designs = sample_designs(panel, {}, spec, "2026-Q2", ("PT", "ES"))
    _, _, dates = designs[("PT", 4, False)]
    assert not [q for q in dates if "2020-Q1" <= q <= "2021-Q4"]
    # Four GDP lags may not reach back into 2021: the first later outcome is 2023-Q1.
    assert min(q for q in dates if q > "2021-Q4") == "2023-Q1"
    restricted, keep = _sample_panel(panel, "pandemic_excluded")
    assert not keep("2021-Q3") and keep("2022-Q1")
    assert ("PT", "2020-Q2") not in restricted


def test_full_sample_matches_published_common_sample() -> None:
    panel = synthetic_panel(("PT", "ES"))
    designs = sample_designs(panel, {}, SAMPLES[0], "2026-Q2", ("PT", "ES"))
    dates = {tuple(d) for _, _, d in designs.values()}
    assert len(dates) == 1
    assert next(iter(dates))[0] == "2009-Q1"


@pytest.mark.parametrize(
    ("flags", "quarter", "expected"),
    [
        ({("ES", "2021-Q1"): {"lfs": "b", "gdp": ""}}, "2021-Q1", True),
        ({("ES", "2021-Q1"): {"lfs": "b", "gdp": ""}}, "2021-Q2", False),
        ({("ES", "2021-Q1"): {"lfs": "u", "gdp": ""}}, "2021-Q2", True),
        ({("ES", "2021-Q1"): {"lfs": "d", "gdp": ""}}, "2021-Q1", True),
        ({("ES", "2021-Q1"): {"lfs": "", "gdp": "b"}}, "2021-Q3", True),
        ({("ES", "2021-Q1"): {"lfs": "", "gdp": "p"}}, "2021-Q3", False),
    ],
)
def test_flagged_change_rules(
    flags: dict[tuple[str, str], dict[str, str]], quarter: str, expected: bool
) -> None:
    assert flagged_change(flags, "ES", quarter, max_lag=4) is expected


def test_gdp_break_only_matters_within_the_lag_window() -> None:
    flags = {("ES", "2021-Q1"): {"lfs": "", "gdp": "b"}}
    assert flagged_change(flags, "ES", "2021-Q3", max_lag=2)
    assert not flagged_change(flags, "ES", "2021-Q3", max_lag=1)


def _row(total: float, low: float, high: float) -> GridRow:
    return GridRow("full", "PT", 2, False, 40, "2009-Q2", "2019-Q4", total, 0.1, low, high, 0.3)


def test_verdict_rules() -> None:
    assert verdict_for([_row(-0.3, -0.5, -0.1), _row(-0.2, -0.4, -0.05)]) == "robust_negative"
    assert verdict_for([_row(-0.3, -0.5, -0.1), _row(-0.1, -0.3, 0.1)]) == (
        "negative_precision_dependent"
    )
    assert verdict_for([_row(-0.3, -0.5, -0.1), _row(0.0, -0.2, 0.2)]) == "sign_unstable"


# ----------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------


def test_quarter_runs_and_names() -> None:
    quarters = ["2021-Q1", "2021-Q2", "2021-Q3", "2023-Q1", "2020-Q4"]
    assert quarter_runs(quarters) == "2020-Q4..2021-Q3, 2023-Q1"
    assert names([]) == "none"
    assert names(["ES"]) == "ES"
    assert names(["ES", "FR", "PT"]) == "ES, FR and PT"


def test_archived_comparison_reports_largest_numeric_difference() -> None:
    left = {"a": [1.0, {"b": 2.0}], "label": "x"}
    right = {"a": [1.0, {"b": 2.5}], "label": "x"}
    assert _numeric_difference(left, right) == pytest.approx(0.5)
    with pytest.raises(EurostatDataError, match="keys differ"):
        _numeric_difference({"a": 1}, {"b": 1})
    with pytest.raises(EurostatDataError, match="differs"):
        _numeric_difference({"a": "x"}, {"a": "y"})


def test_loss_differential_ratio_matches_hac_mean() -> None:
    from eu_okun.forecasting import ForecastRecord

    rng = np.random.default_rng(9)
    quarters = quarter_sequence("2018-Q1", "2024-Q4")
    records = []
    for geo in ("PT", "ES"):
        for quarter in quarters:
            actual = float(rng.normal())
            for model, noise in (("zero_change", 0.5), ("historical_mean", 0.4),
                                 ("unemployment_ar1", 0.3), ("gdp_lags", 0.3),
                                 ("gdp_unemployment_ar1", 0.2),
                                 ("gdp_ar1_participation", 0.25)):
                predicted = actual + float(rng.normal(scale=noise))
                records.append(ForecastRecord(
                    geo, quarter, quarter, quarter, 40, model, actual, predicted, 8.0,
                    8.0 + predicted,
                ))
    rows, skill = score_forecast_set("test", records)
    assert {r.model for r in rows} == {
        "zero_change", "historical_mean", "gdp_lags", "gdp_unemployment_ar1",
        "gdp_ar1_participation",
    }
    assert all(r.origins == len(quarters) for r in rows)
    assert set(skill["gdp_lags"]) == {"PT", "ES"}


# ----------------------------------------------------------------------------------
# End to end on a synthetic, offline EU-27 snapshot
# ----------------------------------------------------------------------------------


def write_snapshot(directory: Path, panel: Panel) -> None:
    """A complete snapshot: JSON-stat-like raw files, manifest and processed CSV."""
    (directory / "raw").mkdir(parents=True)
    (directory / "processed").mkdir()
    sources = {}
    for spec in SERIES:
        payload = {
            "label": f"Synthetic {spec.name}",
            "updated": "2026-10-15T00:00:00+0200",
            "dimension": {
                dim: {"category": {"label": {code: f"{dim} {code}"}}}
                for dim, code in spec.filters.items()
            },
            "extension": {"status": {"label": {"d": "definition differs (see metadata)"}},
                          "annotation": []},
        }
        content = json.dumps(payload).encode()
        (directory / "raw" / f"{spec.dataset}.json").write_bytes(content)
        sources[spec.name] = {"dataset": spec.dataset,
                              "sha256": hashlib.sha256(content).hexdigest()}
    (directory / "snapshot_manifest.json").write_text(json.dumps({
        "countries": list(EU27), "downloaded_at_utc": "2026-10-15T00:00:00+00:00",
        "period": {"start": QUARTERS[0], "end": QUARTERS[-1]}, "series": sources,
    }))
    columns = ["geo", "quarter"]
    for spec in SERIES:
        columns += [spec.column, f"{spec.name}_flag"]
    with (directory / "processed" / "eu27_quarterly.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for geo in EU27:
            for quarter in QUARTERS:
                row = panel[(geo, quarter)]
                lfs_flag = "d" if geo == "ES" and quarter >= "2021-Q1" else ""
                writer.writerow({
                    "geo": geo, "quarter": quarter,
                    "gdp_growth_qoq_pct": row["gdp"], "gdp_flag": "",
                    "unemployment_rate_pct": row["unemployment"],
                    "unemployment_flag": lfs_flag,
                    "participation_rate_pct": row["participation"],
                    "participation_flag": lfs_flag,
                })


@pytest.fixture(scope="module")
def review_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, ReviewResult]:
    root = tmp_path_factory.mktemp("review")
    snapshot = root / "snapshot"
    write_snapshot(snapshot, synthetic_panel())
    output = root / "out"
    result = write_review(snapshot, output, figures=False, pair_replicates=199)
    return output, result


def test_review_verifies_every_published_quantity(
    review_run: tuple[Path, ReviewResult],
) -> None:
    _, result = review_run
    checks = result.checks
    assert len(checks) == 17
    assert all(check.passed for check in checks), [c for c in checks if not c.passed]


def test_review_detects_the_synthetic_pandemic_attenuation(
    review_run: tuple[Path, ReviewResult],
) -> None:
    _, result = review_run
    robustness = result.robustness
    # Muted 2020 responses with large GDP swings pull the full-sample sum towards zero.
    assert all(abs(x.headline_full) < abs(x.headline_excluded) for x in robustness)
    assert all(x.pandemic_leverage_share > 0.5 for x in robustness)


def test_review_writes_report_and_tables(review_run: tuple[Path, ReviewResult]) -> None:
    output, result = review_run
    report = (output / REPORT_NAME).read_text(encoding="utf-8")
    for heading in ("## Summary", "## 3. Independent verification",
                    "## 9. What can be reported", "## 11. Unresolved research questions"):
        assert heading in report
    assert "**FAIL**" not in report
    assert "{assets}" not in report
    for name in ("grid.csv", "robustness.csv", "verification.csv", "review.json"):
        assert (output / name).exists()
    document = json.loads((output / "review.json").read_text(encoding="utf-8"))
    assert len(document["robustness"]) == 27
    first = render_report(result, ".", output / REPORT_NAME, output, Path("snapshot"), "")
    assert first == render_report(result, ".", output / REPORT_NAME, output, Path("snapshot"), "")


def test_baseline_hac_uses_calendar_lags_across_gaps(tmp_path: Path) -> None:
    """A missing quarter must not make the HAC pair observations a year apart as neighbours."""
    from eu_okun.baseline import design_matrix, run_baselines

    panel = synthetic_panel()
    panel[("PT", "2015-Q2")]["unemployment"] = None
    write_snapshot(tmp_path / "snapshot", panel)
    fits = run_baselines(tmp_path / "snapshot", lag_orders=(1,))
    fit = next(f for f in fits if f.geo == "PT" and not f.participation)
    x, y, dates = design_matrix(panel, "PT", quarter_sequence("2009-Q1", "2026-Q2"), 1, False)
    assert "2015-Q2" not in dates and "2015-Q3" not in dates
    _, calendar, _, _ = fit_hac_covariance(x, y, 4, dates)
    _, rows, _, _ = fit_hac_covariance(x, y, 4)
    assert fit.hac_se["gdp_lag_0"] == pytest.approx(math.sqrt(calendar[1, 1]), rel=1e-12)
    assert not math.isclose(math.sqrt(rows[1, 1]), fit.hac_se["gdp_lag_0"], rel_tol=1e-9)
