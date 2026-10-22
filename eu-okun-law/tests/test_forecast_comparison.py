"""Tests of the Bayesian forecasting comparison: the framework's sample and point
forecasts, no look-ahead, the hierarchical model's likelihood, the block bootstrap,
refits and the report. Sampling is limited to one tiny smoke run."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import stats

from eu_okun.contract import EU27, quarter_sequence
from eu_okun.forecast_comparison import (
    BAYES_MODELS,
    CLASSICAL_MODELS,
    FAMILY,
    bayes_components,
    bootstrap_comparison,
    build_forecast_panel,
    circular_block_indices,
    classical_predictions,
    holm_verdicts,
)
from eu_okun.forecasting import expanding_forecasts
from eu_okun.predictive import write_predictions

Panel = dict[tuple[str, str], dict[str, float | None]]


def synthetic(countries: tuple[str, ...] = tuple(EU27), seed: int = 3) -> Panel:
    rng = np.random.default_rng(seed)
    panel: Panel = {}
    for geo in countries:
        u, p, g_prev, du = 8.0, 70.0, 0.4, 0.0
        for quarter in quarter_sequence("2008-Q1", "2026-Q2"):
            g = float(rng.normal(0.4, 0.8))
            du = 0.3 * du - 0.2 * g_prev + float(rng.normal(0, 0.2))
            u, p = u + du, p + float(rng.normal(0, 0.2))
            panel[(geo, quarter)] = {"gdp": g, "unemployment": u, "participation": p}
            g_prev = g
    return panel


def test_panel_and_point_forecasts_are_the_frameworks() -> None:
    panel = synthetic()
    fpanel = build_forecast_panel(panel, "2026-Q2")
    records, quarters = expanding_forecasts(panel, "2026-Q2")
    assert fpanel.target_quarters == quarters and len(quarters) == 34
    framework = {(r.model, r.geo, r.target_quarter): r for r in records}
    for model in CLASSICAL_MODELS:
        for p in classical_predictions(fpanel, model):
            r = framework[(model, p.geo, p.quarter)]
            assert p.mean == pytest.approx(r.predicted_du_pp, abs=1e-10)
            assert p.actual == pytest.approx(r.actual_du_pp)
            assert p.q05 < p.median < p.q95


def test_forecasts_do_not_use_values_dated_in_the_target_quarter() -> None:
    panel = synthetic()
    fpanel = build_forecast_panel(panel, "2026-Q2")
    column = fpanel.origins[5]
    target = fpanel.quarters[column]
    changed = {k: dict(v) for k, v in panel.items()}
    for geo in EU27:
        row = changed[(geo, target)]
        row["unemployment"] = float(row["unemployment"] or 0) + 3.0
        row["gdp"], row["participation"] = 9.0, 50.0
    other = build_forecast_panel(changed, "2026-Q2")
    one = replace(fpanel, origins=(column,))
    two = replace(other, origins=(column,))
    for model in CLASSICAL_MODELS:
        before, after = classical_predictions(one, model), classical_predictions(two, model)
        for a, b in zip(before, after, strict=True):
            assert (a.mean, a.q05, a.q95) == pytest.approx((b.mean, b.q05, b.q95))
            assert b.actual == pytest.approx(a.actual + 3.0)
    rng = np.random.default_rng(0)
    draws = {"alpha": rng.normal(size=(10, 27)), "beta": rng.normal(size=(10, 27, 3)),
             "sigma": rng.uniform(0.1, 0.3, (10, 27)), "nu": rng.uniform(4, 9, 10)}
    names = ("du_lag1", "gdp_lag1", "gdp_lag2")
    c1 = bayes_components(draws, fpanel.design(names)[:, column], False)
    c2 = bayes_components(draws, other.design(names)[:, column], False)
    np.testing.assert_allclose(c1.centre, c2.centre)


def test_hierarchical_likelihood_matches_an_independent_reference() -> None:
    numpyro = pytest.importorskip("numpyro")
    numpyro.enable_x64()
    import jax.numpy as jnp
    from numpyro import handlers

    from eu_okun.forecast_bayes import forecast_model

    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(4, 20, 2)), rng.normal(size=(4, 20))
    for factor in (False, True):
        trace = handlers.trace(handlers.seed(forecast_model, 2)).get_trace(
            jnp.asarray(x), jnp.asarray(y), factor)
        v = {k: np.asarray(site["value"]) for k, site in trace.items()}
        mean = v["alpha"][:, None] + np.einsum("ntk,nk->nt", x, v["beta"])
        if factor:
            mean = mean + v["lam"][:, None] * v["f"][None, :]
        expected = stats.t.logpdf(y, v["nu"], mean, v["sigma"][:, None]).sum()
        ours = float(np.asarray(trace["y"]["fn"].log_prob(jnp.asarray(y))))
        assert ours == pytest.approx(expected, rel=1e-10)


def test_circular_blocks_bootstrap_and_holm() -> None:
    idx = circular_block_indices(34, 4, 200, np.random.default_rng(0))
    assert idx.shape == (200, 34) and idx.min() >= 0 and idx.max() < 34
    steps = (idx[:, 1:4] - idx[:, 0:3]) % 34
    assert (steps == 1).all()                       # each block is consecutive (circular)
    with pytest.raises(ValueError):
        circular_block_indices(6, 4, 10, np.random.default_rng(0))
    rng = np.random.default_rng(1)
    loss = rng.uniform(0.1, 0.4, (27, 34))
    same = bootstrap_comparison(loss, loss, idx, "a", "b", "crps", "all", 4)
    assert same.difference == 0 and same.ratio == 1 and same.p_value == 1.0
    better = bootstrap_comparison(loss - 0.05, loss, idx, "a", "b", "crps", "all", 4)
    assert better.high < 0 and better.p_value < 0.01 and better.ratio_high < 1
    verdicts = holm_verdicts([better, same])
    assert verdicts[0].verdict == "a better" and verdicts[1].verdict == "not distinguishable"
    assert verdicts[0].p_holm == pytest.approx(min(1.0, 2 * better.p_value))


def _fake_fit(x: np.ndarray, y: np.ndarray, spec: Any, settings: Any,
              rhat: float = 1.0) -> Any:
    from eu_okun.bayes_fit import Diagnostics
    from eu_okun.forecast_bayes import ForecastFit

    n, _, k = x.shape
    rng = np.random.default_rng(int(settings.warmup))
    samples = {"alpha": rng.normal(0, 0.05, (2, 5, n)), "beta": rng.normal(0, 0.1, (2, 5, n, k)),
               "sigma": rng.uniform(0.1, 0.3, (2, 5, n)), "nu": rng.uniform(4, 9, (2, 5))}
    if spec.common_factor:
        samples["lam"] = rng.uniform(0, 0.2, (2, 5, n))
    diagnostics = Diagnostics(rhat, "x", 900.0, 900.0, "x", 0, 0.8, 0, 4000)
    return ForecastFit(samples, diagnostics, settings, 1.0)


def _patch(monkeypatch: pytest.MonkeyPatch, module: Any, panel: Panel) -> None:
    monkeypatch.setattr(module, "audit_snapshot", lambda d: {
        "snapshot_manifest_sha256": "m", "processed_csv_sha256": "c",
        "analysis_period": {"end": "2026-Q2"}})
    monkeypatch.setattr(module, "_read_panel", lambda path: panel)


def test_refit_loop_retries_once_and_resumes(tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    import eu_okun.forecast_bayes as fb

    calls: list[int] = []

    def fake(x, y, spec, settings):  # type: ignore[no-untyped-def]
        calls.append(settings.warmup)
        return _fake_fit(x, y, spec, settings, 1.05 if len(calls) == 1 else 1.0)

    _patch(monkeypatch, fb, synthetic())
    monkeypatch.setattr(fb, "fit_forecast_model", fake)
    monkeypatch.setattr(fb, "environment_versions", lambda: {"python": "test"})
    records = fb.run_forecast_refits(tmp_path, tmp_path / "out", ["bayes_gdp_ar1_factor"],
                                     targets=["2018-Q1", "2018-Q2"])
    assert calls == [1000, 3000, 1000]
    quarters = build_forecast_panel(synthetic(), "2026-Q2").quarters
    assert records[0]["training_last"] == "2017-Q4"
    assert records[0]["training_quarters"] == quarters.index("2018-Q1")
    again = fb.run_forecast_refits(tmp_path, tmp_path / "out", ["bayes_gdp_ar1_factor"],
                                   targets=["2018-Q1", "2018-Q2"])
    assert calls == [1000, 3000, 1000] and again == records


@pytest.mark.integration
def test_forecast_model_smoke_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A tiny real refit that only checks the pipeline end to end; not a convergence test."""
    pytest.importorskip("numpyro")
    import eu_okun.forecast_bayes as fb
    from eu_okun.bayes_fit import SamplerSettings
    from eu_okun.predictive import read_predictions

    _patch(monkeypatch, fb, synthetic())
    records = fb.run_forecast_refits(tmp_path, tmp_path / "out", ["bayes_gdp_ar1"],
                                     SamplerSettings(chains=2, warmup=60, draws=60, seed=1),
                                     retry_warmup=None, targets=["2026-Q2"])
    quarters = build_forecast_panel(synthetic(), "2026-Q2").quarters
    assert records[0]["training_quarters"] == quarters.index("2026-Q2") == len(quarters) - 1
    rows = read_predictions(tmp_path / "out" / "bayes_gdp_ar1" / "2026-Q2" / "predictions.csv")
    assert len(rows) == 27 and all(np.isfinite(r.crps) and 0 <= r.pit <= 1 for r in rows)


@pytest.mark.integration
def test_report_renders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import eu_okun.audit
    import eu_okun.baseline
    import eu_okun.forecast_comparison as fc

    panel = synthetic()
    monkeypatch.setattr(eu_okun.audit, "audit_snapshot", lambda d: {
        "snapshot_manifest_sha256": "m", "processed_csv_sha256": "c",
        "analysis_period": {"end": "2026-Q2"}})
    monkeypatch.setattr(eu_okun.baseline, "_read_panel", lambda path: panel)
    fpanel = build_forecast_panel(panel, "2026-Q2")
    base = classical_predictions(fpanel, "gdp_unemployment_ar1")
    for k, model in enumerate(BAYES_MODELS):
        for quarter in fpanel.target_quarters:
            rows = [replace(p, model=model, crps=p.crps * (0.95 + 0.02 * k))
                    for p in base if p.quarter == quarter]
            directory = tmp_path / "fits" / model / quarter
            write_predictions(directory / "predictions.csv", rows)
            accepted = not (k == 0 and quarter == "2019-Q2")
            record = {"model": model, "target": quarter, "accepted": accepted,
                      "attempts": [{"accepted": accepted, "max_rhat": 1.0 if accepted else 1.02,
                                    "min_ess_bulk": 800.0, "min_ess_tail": 700.0,
                                    "divergences": 0, "runtime_seconds": 20.0,
                                    "failures": [] if accepted else ["R-hat 1.02 > 1.01 (mu_s)"]}],
                      "environment": {"python": "test"},
                      "snapshot": {"manifest_sha256": "m", "processed_csv_sha256": "c"}}
            (directory / "origin.json").write_text(json.dumps(record), encoding="utf-8")
    comparison = fc.write_report(tmp_path, tmp_path / "fits", tmp_path / "out", figures=True)
    assert len(comparison.family) == len(FAMILY)
    assert set(comparison.sensitivity) == {"block 8", "excluding pandemic",
                                           "without failed refits"}
    assert all(t.p_holm >= t.p_value for t in comparison.family)
    report = (tmp_path / "out" / "FORECAST_COMPARISON.md").read_text(encoding="utf-8")
    assert "1 of 102 refits failed" in report and "`mu_s`" in report
    for heading in ("## Summary", "## 1. Design and computation",
                    "## 2. Pre-registered comparisons", "## 3. All models",
                    "## 4. Countries (descriptive)", "Retrospective, not real-time"):
        assert heading in report
    for name in ("forecasts.csv", "comparisons.csv", "countries.csv", "refits.json",
                 "skill.png", "quarterly_differences.png", "calibration.png", "countries.png"):
        assert (tmp_path / "out" / name).exists()
    again = fc.write_report(tmp_path, tmp_path / "fits", tmp_path / "out2", figures=False,
                            replication_dir=tmp_path / "out")
    assert again.replication is not None
    assert again.replication.failed["bayes_ar1"] == ["2019-Q2"]
    text = (tmp_path / "out2" / "FORECAST_COMPARISON.md").read_text(encoding="utf-8")
    assert "## 5. Replication on a second platform" in text and "## 7. Reproduction" in text
    assert "the same verdicts for all five comparisons" in text and "at most 0.0000" in text
