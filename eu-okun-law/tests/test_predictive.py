"""Tests of the predictive comparison: training slices, no look-ahead, predictive
distributions, proper scores, the panel loss-differential test, refits and the report.

Sampling is limited to one tiny smoke run; the real refits belong to the separate
predictive workflow.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import integrate, special, stats

from eu_okun.bayes_data import PRIMARY_WINDOWS, build_bayes_data
from eu_okun.contract import EU27, quarter_sequence
from eu_okun.predictive import (
    BAYES_MODELS,
    CLASSICAL_MODELS,
    EVALUATION_SETS,
    FAMILY,
    PanelTest,
    PredictiveComponents,
    PredictiveData,
    StudentT,
    adjust_family,
    build_predictive_data,
    classical_predictions,
    crps_samples,
    crps_student_t,
    loss_matrix,
    newey_west_variance,
    ols_predictive,
    panel_test,
    predictive_components,
    predictive_log_density,
    predictive_samples,
    read_predictions,
    target_inputs,
    write_predictions,
    zero_change_predictive,
)

Panel = dict[tuple[str, str], dict[str, float | None]]
Flags = dict[tuple[str, str], dict[str, str]]
COUNTRIES = ("ES", "LU", "NL", "PT")


def synthetic(countries: tuple[str, ...] = COUNTRIES, seed: int = 3) -> Panel:
    rng = np.random.default_rng(seed)
    panel: Panel = {}
    for geo in countries:
        u, p, g_prev, du = 8.0, 70.0, 0.4, 0.0
        for quarter in quarter_sequence("2008-Q1", "2026-Q2"):
            g = float(rng.normal(0.4, 0.8))
            du = 0.3 * du - 0.3 * g - 0.1 * g_prev + float(rng.normal(0, 0.2))
            u, p = u + du, p + float(rng.normal(0, 0.2))
            panel[(geo, quarter)] = {"gdp": g, "unemployment": u, "participation": p}
            g_prev = g
    return panel


def small(evaluation: str = "E1", flags: Flags | None = None) -> PredictiveData:
    return build_predictive_data(synthetic(), flags or {}, EVALUATION_SETS[evaluation],
                                 COUNTRIES)


# ----------------------------------------------------------------------------------
# Data: targets, training slices and scored cells
# ----------------------------------------------------------------------------------


def test_targets_and_training_slices_stop_before_the_target() -> None:
    pdata = small("E1")
    assert pdata.target_quarters == quarter_sequence("2016-Q1", "2019-Q4")
    first = pdata.targets[0]
    train = pdata.training(first)
    assert train.quarters[-1] == "2015-Q4" and train.n_quarters == 27
    reference = build_bayes_data(synthetic(), {}, (("2009-Q2", "2015-Q4"),), 2, COUNTRIES)
    for name in ("y", "g", "z", "dp", "mask", "segment_start"):
        np.testing.assert_array_equal(getattr(train, name), getattr(reference, name))
    e2 = small("E2")
    assert e2.target_quarters == quarter_sequence("2022-Q3", "2026-Q2")
    segment_start = e2.training(e2.targets[0])
    assert segment_start.quarters == build_bayes_data(
        synthetic(), {}, PRIMARY_WINDOWS, 2, COUNTRIES).quarters
    later = e2.training(e2.targets[3])
    assert later.quarters[-1] == "2023-Q1" and int(later.segment_start.sum()) == 2


def test_scored_cells_follow_the_flag_rules() -> None:
    flags: Flags = {("NL", "2017-Q1"): {"lfs": "b", "gdp": ""},
                    ("PT", "2022-Q2"): {"lfs": "b", "gdp": ""}}
    e1 = small("E1", flags)
    nl, j = COUNTRIES.index("NL"), e1.target_quarters.index("2017-Q1")
    assert not e1.scored[nl, j] and not e1.scored[nl, j + 1] and e1.scored[nl, j + 2]
    assert e1.scored.sum() == e1.scored.size - 2
    e2 = small("E2", flags)
    pt = COUNTRIES.index("PT")
    # 2022-Q3 starts a segment; its previous change (into 2022-Q2) is flagged.
    assert e2.data.mask[pt, e2.targets[0]] and not e2.prev_ok[pt, e2.targets[0]]
    assert not e2.scored[pt, 0] and e2.scored[pt, 1]
    assert e2.prev_ok[COUNTRIES.index("ES"), e2.targets[0]]


def _future_changed(pdata: PredictiveData, column: int) -> PredictiveData:
    """The same data with the target's and every later outcome replaced."""
    y = pdata.data.y.copy()
    y[:, column:] += 5.0
    return replace(pdata, data=replace(pdata.data, y=y), targets=(column,))


@pytest.mark.parametrize("evaluation", ["E1", "E2"])
def test_classical_predictions_do_not_use_the_target_or_later_outcomes(evaluation: str) -> None:
    pdata = small(evaluation)
    column = pdata.targets[2]
    original = replace(pdata, targets=(column,))
    changed = _future_changed(pdata, column)
    for model in CLASSICAL_MODELS:
        before = classical_predictions(original, model)
        after = classical_predictions(changed, model)
        assert len(before) == len(COUNTRIES)
        for a, b in zip(before, after, strict=True):
            assert (a.mean, a.q05, a.q95) == (b.mean, b.q05, b.q95)
            assert b.actual == pytest.approx(a.actual + 5.0)


def _draws(n: int, k: int = 2, s: int = 5, t: int = 30, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {"alpha": rng.normal(0, 0.1, (s, n)), "theta": rng.normal(-0.3, 0.1, (s, n)),
            "b": rng.normal(0, 0.1, (s, n, k)), "phi": rng.uniform(-0.6, 0.6, (s, n)),
            "sigma": rng.uniform(0.1, 0.5, (s, n)), "lam": rng.uniform(0.0, 0.3, (s, n)),
            "f": rng.normal(size=(s, t)), "nu": rng.uniform(3, 30, s)}


def test_bayesian_inputs_do_not_use_the_target_or_later_outcomes() -> None:
    from eu_okun.bayes_model import SPECS

    pdata = small("E1")
    column = pdata.targets[2]
    a, b = target_inputs(pdata, column), target_inputs(_future_changed(pdata, column), column)
    for name in ("g", "z", "y_last", "g_last", "z_last"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    draws = _draws(len(COUNTRIES))
    ca, cb = (predictive_components(draws, SPECS["primary"], x) for x in (a, b))
    np.testing.assert_array_equal(ca.centre, cb.centre)


# ----------------------------------------------------------------------------------
# Predictive distributions and scores
# ----------------------------------------------------------------------------------


def test_ols_predictive_matches_the_textbook_simple_regression() -> None:
    rng = np.random.default_rng(1)
    x1 = rng.normal(size=30)
    y = 0.5 - 0.4 * x1 + rng.normal(0, 0.3, 30)
    x0 = 1.7
    dist = ols_predictive(np.column_stack([np.ones(30), x1]), y, np.array([1.0, x0]))
    slope, intercept = np.polyfit(x1, y, 1)
    residual = y - (intercept + slope * x1)
    s = math.sqrt(residual @ residual / 28)
    sxx = float(((x1 - x1.mean()) ** 2).sum())
    assert dist.loc == pytest.approx(intercept + slope * x0)
    assert dist.scale == pytest.approx(s * math.sqrt(1 + 1 / 30 + (x0 - x1.mean()) ** 2 / sxx))
    assert dist.df == 28
    with pytest.raises(ValueError, match="Insufficient"):
        ols_predictive(np.ones((5, 2)), np.ones(5), np.ones(2))
    zero = zero_change_predictive(y)
    assert zero.loc == 0 and zero.scale == pytest.approx(math.sqrt(np.mean(y**2)))


@pytest.mark.parametrize(("y", "loc", "scale", "df"),
                         [(0.3, 0.0, 0.5, 3.0), (-1.2, 0.2, 0.3, 8.0), (0.05, -0.1, 0.2, 200.0)])
def test_closed_form_crps_equals_its_integral_definition(y: float, loc: float, scale: float,
                                                         df: float) -> None:
    dist = StudentT(loc, scale, df)

    def integrand(x: float) -> float:
        return float((stats.t.cdf(x, df, loc, scale) - (x >= y)) ** 2)

    below, _ = integrate.quad(integrand, -np.inf, y, limit=200)
    above, _ = integrate.quad(integrand, y, np.inf, limit=200)
    assert crps_student_t(y, dist) == pytest.approx(below + above, rel=1e-6)


def test_sample_crps_equals_the_pairwise_definition_and_the_closed_form() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(60, 3))
    y = np.array([0.1, -2.0, 1.0])
    brute = [np.mean(np.abs(x[:, i] - y[i]))
             - 0.5 * np.mean(np.abs(x[:, i][:, None] - x[:, i][None, :])) for i in range(3)]
    np.testing.assert_allclose(crps_samples(x, y), brute, rtol=1e-12)
    many = stats.t.rvs(5, loc=0.2, scale=0.4, size=(200_000, 1), random_state=3)
    assert crps_samples(many, np.array([0.5]))[0] == pytest.approx(
        crps_student_t(0.5, StudentT(0.2, 0.4, 5)), rel=0.01)


def test_predictive_components_follow_the_ar1_and_segment_rules() -> None:
    from eu_okun.bayes_model import SPECS

    n = 2
    draws = _draws(n)
    target = target_inputs(small("E1"), small("E1").targets[0])
    target = replace(target, g=target.g[:n], z=target.z[:n], y_last=target.y_last[:n],
                     g_last=target.g_last[:n], z_last=target.z_last[:n])
    c = predictive_components(draws, SPECS["primary"], target)
    d = 3
    mean_now = (draws["alpha"][d] + draws["theta"][d] * target.g
                + (target.z * draws["b"][d]).sum(axis=1))
    mean_last = (draws["alpha"][d] + draws["theta"][d] * target.g_last
                 + (target.z_last * draws["b"][d]).sum(axis=1)
                 + draws["lam"][d] * draws["f"][d, -1])
    expected = mean_now + draws["phi"][d] * (target.y_last - mean_last)
    np.testing.assert_allclose(c.centre[d], expected)
    np.testing.assert_allclose(c.scale, draws["sigma"])
    start = predictive_components(draws, SPECS["primary"], replace(target, segment_start=True))
    np.testing.assert_allclose(start.centre[d], mean_now)
    np.testing.assert_allclose(start.scale, draws["sigma"] / np.sqrt(1 - draws["phi"] ** 2))
    iid = predictive_components(draws, SPECS["gaussian_iid"], target)
    assert iid.loading is None and iid.nu is None
    with pytest.raises(ValueError, match="Participation"):
        predictive_components(draws, SPECS["participation"], target)


def test_predictive_samples_have_the_model_moments() -> None:
    s, n = 4000, 2
    centre = np.tile([0.2, -0.1], (s, 1))
    scale = np.tile([0.3, 0.5], (s, 1))
    gaussian = PredictiveComponents(centre, scale, None, None)
    x = predictive_samples(gaussian, 4, np.random.default_rng(0))
    assert x.shape == (4 * s, n)
    np.testing.assert_allclose(x.mean(axis=0), [0.2, -0.1], atol=0.01)
    np.testing.assert_allclose(x.std(axis=0), [0.3, 0.5], rtol=0.02)
    loading = np.tile([0.4, 0.4], (s, 1))
    factor = predictive_samples(PredictiveComponents(centre, scale, loading, None), 4,
                                np.random.default_rng(1))
    np.testing.assert_allclose(factor.var(axis=0), [0.09 + 0.16, 0.25 + 0.16], rtol=0.03)
    # One factor value per draw, shared by the countries: positive correlation.
    expected = 0.16 / math.sqrt(0.25 * 0.41)
    assert np.corrcoef(factor.T)[0, 1] == pytest.approx(expected, abs=0.03)


def _reference_log_density(y: float, c: float, s: float, lam: float, nu: float) -> float:
    """log of the integral over f of t(y; c + lam*f, s) N(f; 0, 1), by adaptive quadrature
    on the log-scaled integrand, split at the factor value that centres the t density."""
    peak = (y - c) / lam
    lo, hi = min(-14.0, peak - 2.0), max(14.0, peak + 2.0)
    grid = np.linspace(lo, hi, 20001)
    top = float(np.max(stats.t.logpdf(y, nu, c + lam * grid, s) + stats.norm.logpdf(grid)))

    def scaled(f: float) -> float:
        return math.exp(float(stats.t.logpdf(y, nu, c + lam * f, s) + stats.norm.logpdf(f)) - top)

    points = sorted({0.0, min(max(peak, lo + 1e-9), hi - 1e-9)})
    value = integrate.quad(scaled, lo, hi, points=points, limit=2000, epsabs=0, epsrel=1e-12)[0]
    return top + math.log(value)


def test_log_density_integrates_the_factor_correctly() -> None:
    rng = np.random.default_rng(4)
    s, n = 4, 3
    centre, scale = rng.normal(0, 0.2, (s, n)), rng.uniform(0.1, 0.5, (s, n))
    loading, nu = rng.uniform(0.05, 0.4, (s, n)), rng.uniform(3, 20, s)
    y = np.array([0.3, -0.4, 0.0])
    ours = predictive_log_density(PredictiveComponents(centre, scale, loading, nu), y)
    for i in range(n):
        per_draw = [_reference_log_density(y[i], centre[d, i], scale[d, i], loading[d, i], nu[d])
                    for d in range(s)]
        assert ours[i] == pytest.approx(special.logsumexp(per_draw) - math.log(s), abs=1e-9)
    # Hard cases: innovation scale small relative to the loading, and far-tail outcomes.
    for y0, c0, s0, lam0, nu0 in [(1.5, 0.0, 0.1, 0.3, 3.0), (0.2, 0.0, 0.02, 0.4, 2.5),
                                  (-1.95, 0.0, 0.03, 0.183, 26.4), (0.05, 0.0, 0.05, 0.35, 10.0)]:
        one = PredictiveComponents(np.array([[c0]]), np.array([[s0]]), np.array([[lam0]]),
                                   np.array([nu0]))
        assert predictive_log_density(one, np.array([y0]))[0] == pytest.approx(
            _reference_log_density(y0, c0, s0, lam0, nu0), abs=1e-9)
    plain = predictive_log_density(PredictiveComponents(centre, scale, None, None), y)
    expected = np.log(np.mean(stats.norm.pdf(y[None, :], centre, scale), axis=0))
    np.testing.assert_allclose(plain, expected, rtol=1e-10)
    gaussian_factor = predictive_log_density(PredictiveComponents(centre, scale, loading, None), y)
    expected = np.log(np.mean(stats.norm.pdf(y[None, :], centre,
                                             np.sqrt(scale**2 + loading**2)), axis=0))
    np.testing.assert_allclose(gaussian_factor, expected, rtol=1e-10)


# ----------------------------------------------------------------------------------
# Panel loss-differential test
# ----------------------------------------------------------------------------------


def test_newey_west_variance_and_the_panel_test() -> None:
    rng = np.random.default_rng(5)
    d = rng.normal(size=16)
    assert newey_west_variance(d, 0) == pytest.approx(np.var(d))
    x = d - d.mean()
    manual = (x @ x + 2 * (2 / 3) * (x[1:] @ x[:-1]) + 2 * (1 / 3) * (x[2:] @ x[:-2])) / 16
    assert newey_west_variance(d, 2) == pytest.approx(manual)
    scored = np.ones((5, 16), dtype=bool)
    scored[0, 3] = False
    loss = rng.uniform(0.1, 0.5, (5, 16))
    same = panel_test("E1", "a", "b", loss, loss, scored)
    assert same.statistic == 0 and same.p_value == 1 and same.ratio == 1
    better = panel_test("E1", "a", "b", loss - 0.05 + rng.normal(0, 0.005, (5, 16)), loss, scored)
    assert better.mean_difference == pytest.approx(-0.05, abs=0.005)
    assert better.p_value < 1e-6 and better.ratio < 1
    # The HLN-corrected statistic: mean / sqrt(V / T) * sqrt((T - 1) / T).
    dd = np.array([np.mean((loss - 0.05)[scored[:, t], t] - loss[scored[:, t], t])
                   for t in range(16)])
    shifted = panel_test("E1", "a", "b", loss - 0.05 + np.arange(16) * 1e-3, loss, scored)
    dd = dd + np.arange(16) * 1e-3
    expected = dd.mean() / math.sqrt(newey_west_variance(dd, 2) / 16) * math.sqrt(15 / 16)
    assert shifted.statistic == pytest.approx(expected)


def test_holm_family_verdicts() -> None:
    def make(p: float, diff: float) -> PanelTest:
        return PanelTest("E1", "a", "b", "crps", 16, 400, 1.0, 1.0, 1.0, diff, 0.1, 1.0, p)

    tests = adjust_family([make(0.001, -0.1), make(0.02, 0.1), make(0.04, -0.1)])
    assert [t.p_holm for t in tests] == pytest.approx([0.003, 0.04, 0.04])
    assert [t.verdict for t in tests] == ["a better", "b better", "a better"]
    assert adjust_family([make(0.03, -0.1), make(0.04, -0.1)])[0].verdict == "not distinguishable"


def test_loss_matrix_requires_exact_coverage_and_predictions_round_trip(tmp_path: Path) -> None:
    pdata = small("E1")
    rows = classical_predictions(pdata, "ols_dl")
    write_predictions(tmp_path / "p.csv", rows)
    again = read_predictions(tmp_path / "p.csv")
    assert again == rows
    matrix = loss_matrix(again, pdata, "crps")
    assert np.isfinite(matrix).all() and matrix.shape == (4, 16)
    with pytest.raises(ValueError, match="exactly"):
        loss_matrix(again[:-1], pdata, "crps")
    with pytest.raises(ValueError, match="Duplicate"):
        loss_matrix([*again, again[0]], pdata, "crps")


# ----------------------------------------------------------------------------------
# Refit loop, a tiny real sampling run and the report
# ----------------------------------------------------------------------------------


def _patch_snapshot(monkeypatch: pytest.MonkeyPatch, module: Any, panel: Panel) -> None:
    monkeypatch.setattr(module, "audit_snapshot", lambda d: {
        "snapshot_manifest_sha256": "m", "processed_csv_sha256": "c"})
    monkeypatch.setattr(module, "_read_panel", lambda path: panel)
    monkeypatch.setattr(module, "read_flags", lambda path: {})


def test_refit_loop_retries_once_and_resumes(tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    import eu_okun.bayes_predictive as bp
    from eu_okun.bayes_fit import Diagnostics, FitResult

    calls: list[int] = []

    def fake_fit(data, spec, settings):  # type: ignore[no-untyped-def]
        calls.append(settings.warmup)
        rhat = 1.05 if len(calls) == 1 else 1.0
        n, t = data.y.shape
        samples = {k: v.reshape(2, 5, *v.shape[1:]) for k, v in _draws(n, s=10, t=t).items()}
        diagnostics = Diagnostics(rhat, "phi[0]", 900.0, 900.0, "x", 0, 0.8, 0, 4000)
        return FitResult(spec, data, settings, samples, diagnostics, 1.0)

    _patch_snapshot(monkeypatch, bp, synthetic(EU27))
    monkeypatch.setattr(bp, "fit_specification", fake_fit)
    monkeypatch.setattr(bp, "environment_versions", lambda: {"python": "test"})
    records = bp.run_predictive_refits(tmp_path, tmp_path / "out", ["E1"], ["bayes_primary"],
                                       targets=["2016-Q1", "2016-Q2"])
    assert calls == [1000, 3000, 1000]
    assert [len(r["attempts"]) for r in records] == [2, 1]
    assert records[0]["accepted"] and records[0]["training_segments"] == [["2009-Q2",
                                                                           "2015-Q4"]]
    rows = read_predictions(tmp_path / "out" / "E1" / "bayes_primary" / "2016-Q1" /
                            "predictions.csv")
    assert len(rows) == len(EU27) and all(r.q05 <= r.median <= r.q95 for r in rows)
    again = bp.run_predictive_refits(tmp_path, tmp_path / "out", ["E1"], ["bayes_primary"],
                                     targets=["2016-Q1", "2016-Q2"])
    assert calls == [1000, 3000, 1000] and again == records      # resumed, nothing refitted


def test_refit_smoke_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A tiny real refit that only checks the pipeline end to end; not a convergence test."""
    pytest.importorskip("numpyro")
    import eu_okun.bayes_predictive as bp
    from eu_okun.bayes_fit import SamplerSettings

    _patch_snapshot(monkeypatch, bp, synthetic(EU27))
    records = bp.run_predictive_refits(
        tmp_path, tmp_path / "out", ["E2"], ["bayes_primary"],
        SamplerSettings(chains=2, warmup=40, draws=40, seed=1), retry_warmup=None,
        targets=["2022-Q4"])
    assert records[0]["training_segments"] == [["2009-Q2", "2019-Q4"], ["2022-Q3", "2022-Q3"]]
    rows = read_predictions(tmp_path / "out" / "E2" / "bayes_primary" / "2022-Q4" /
                            "predictions.csv")
    assert len(rows) == len(EU27)
    assert all(np.isfinite([r.crps, r.log_score]).all() and 0 <= r.pit <= 1 for r in rows)


def _write_fake_refits(fits: Path, pdata: dict[str, PredictiveData], failed: str) -> None:
    """Bayesian origin directories in the refit format, filled from shifted OLS forecasts."""
    for name, data in pdata.items():
        base = classical_predictions(data, "ols_dl")
        for k, model in enumerate(BAYES_MODELS):
            for quarter in data.target_quarters:
                rows = [replace(p, model=model, crps=p.crps * (0.9 + 0.05 * k),
                                mean=p.mean + 0.01 * k)
                        for p in base if p.quarter == quarter]
                directory = fits / name / model / quarter
                write_predictions(directory / "predictions.csv", rows)
                accepted = not (name == "E1" and quarter == failed and k == 0)
                record = {"evaluation": name, "model": model, "target": quarter,
                          "accepted": accepted,
                          "attempts": [{"accepted": accepted, "max_rhat": 1.0 if accepted else 1.02,
                                        "runtime_seconds": 30.0}],
                          "environment": {"python": "test"},
                          "snapshot": {"manifest_sha256": "m", "processed_csv_sha256": "c"}}
                (directory / "origin.json").write_text(json.dumps(record), encoding="utf-8")


def test_report_renders_with_the_family_and_a_failed_refit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import eu_okun.predictive_report as pr

    panel = synthetic(EU27)
    _patch_snapshot(monkeypatch, pr, panel)
    pdata = {name: build_predictive_data(panel, {}, s) for name, s in EVALUATION_SETS.items()}
    _write_fake_refits(tmp_path / "fits", pdata, failed="2017-Q2")
    comparison = pr.write_report(tmp_path, tmp_path / "fits", tmp_path / "out", figures=True)
    assert len(comparison.family) == 2 * len(FAMILY)
    assert all(t.p_holm >= t.p_value for t in comparison.family)
    # The fixtures scale the OLS CRPS by 0.9 (primary) and 1.0 (gaussian_iid).
    primary, gaussian = (next(t for t in comparison.family if t.evaluation == "E1"
                              and t.model_a == a and t.model_b == "ols_dl")
                         for a in ("bayes_primary", "bayes_gaussian_iid"))
    assert primary.ratio == pytest.approx(0.9) and primary.verdict == "bayes_primary better"
    assert gaussian.ratio == pytest.approx(1.0) and gaussian.verdict == "not distinguishable"
    assert comparison.failed_quarters == {"E1": ["2017-Q2"], "E2": []}
    assert len(comparison.sensitivity) == 2 * len(FAMILY)
    assert next(t for t in comparison.sensitivity if t.evaluation == "E1").quarters == 15
    report = (tmp_path / "out" / "PREDICTIVE_COMPARISON.md").read_text(encoding="utf-8")
    for heading in ("## Summary", "## 2. Bayesian refits and convergence",
                    "## 3. Pre-registered comparisons", "## 7. Countries (descriptive)",
                    "## 8. Sensitivity and secondary losses"):
        assert heading in report
    assert "2017-Q2" in report
    for name in ("predictions.csv", "comparisons.csv", "quarterly_differentials.csv",
                 "refits.json", "crps_differentials.png", "quarterly_differentials.png",
                 "calibration.png"):
        assert (tmp_path / "out" / name).exists()
    assert "## 9. Interpretation and limitations" in report
    # A replication with the same tables: identical verdicts and predictions.
    again = pr.write_report(tmp_path, tmp_path / "fits", tmp_path / "out2", figures=False,
                            replication_dir=tmp_path / "out")
    assert again.replication is not None and again.replication.failed["E1"] == ["2017-Q2"]
    text = (tmp_path / "out2" / "PREDICTIVE_COMPARISON.md").read_text(encoding="utf-8")
    assert "## 9. Replication on a second platform" in text and "## 11. Reproduction" in text
    assert "same verdicts for all ten comparisons" in text and "at most 0.0000" in text
    with pytest.raises(FileNotFoundError, match="no refit"):
        import shutil
        shutil.rmtree(tmp_path / "fits" / "E2" / "bayes_primary" / "2023-Q1")
        pr.build_comparison(tmp_path, tmp_path / "fits")
