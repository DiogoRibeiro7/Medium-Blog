"""Tests of the hierarchical Bayesian data, likelihood, diagnostics and reporting rules.

Sampling is limited to one tiny smoke run; the real fits belong to the separate
Bayesian workflow.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

numpyro = pytest.importorskip("numpyro")
numpyro.enable_x64()   # the real fits run in float64; compare at that precision

import jax.numpy as jnp  # noqa: E402
from numpyro import handlers  # noqa: E402

from eu_okun.bayes_data import (  # noqa: E402
    PANDEMIC_EXCLUDED_WINDOWS,
    build_bayes_data,
    window_quarters,
)
from eu_okun.bayes_fit import (  # noqa: E402
    SamplerSettings,
    compute_diagnostics,
    fit_specification,
    predictive_checks,
)
from eu_okun.bayes_model import (  # noqa: E402
    SPECS,
    model_inputs,
    okun_model,
    pointwise_log_likelihood,
)
from eu_okun.bayes_report import (  # noqa: E402
    LoadedFit,
    claim_verdict,
    evaluate_claims,
    pairwise_probability,
    rank_distribution,
    sign_class,
)
from eu_okun.contract import EurostatDataError, quarter_sequence  # noqa: E402

Panel = dict[tuple[str, str], dict[str, float | None]]
COUNTRIES = ("ES", "LU", "NL", "PT")


def synthetic(countries: tuple[str, ...] = COUNTRIES, seed: int = 3) -> Panel:
    rng = np.random.default_rng(seed)
    panel: Panel = {}
    for geo in countries:
        u, p, g_prev = 8.0, 70.0, 0.4
        for quarter in quarter_sequence("2008-Q1", "2026-Q2"):
            g = float(rng.normal(0.4, 0.8))
            u = u - 0.3 * g - 0.1 * g_prev + float(rng.normal(0, 0.2))
            p = p + float(rng.normal(0, 0.2))
            panel[(geo, quarter)] = {"gdp": g, "unemployment": u, "participation": p}
            g_prev = g
    return panel


# ----------------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------------


def test_reparameterised_design_and_shapes() -> None:
    panel = synthetic()
    data = build_bayes_data(panel, {}, lags=2, countries=COUNTRIES)
    assert data.y.shape == (4, 43) and data.z.shape == (4, 43, 2)
    assert data.quarters[0] == "2009-Q2" and data.quarters[-1] == "2019-Q4"
    i, j = 0, 5
    quarter = data.quarters[j]
    year, q = int(quarter[:4]), int(quarter[-1])
    previous = f"{year - 1}-Q4" if q == 1 else f"{year}-Q{q - 1}"
    assert data.z[i, j, 0] == pytest.approx(panel[("ES", previous)]["gdp"] - data.g[i, j])
    assert data.y[i, j] == pytest.approx(
        panel[("ES", quarter)]["unemployment"] - panel[("ES", previous)]["unemployment"])
    np.testing.assert_allclose(data.gdp_design()[i, j, 2], panel[("ES", previous)]["gdp"])


def test_flags_mask_the_change_and_the_following_ar_term() -> None:
    flags = {("NL", "2015-Q1"): {"lfs": "b", "gdp": ""}}
    data = build_bayes_data(synthetic(), flags, lags=2, countries=COUNTRIES)
    nl = COUNTRIES.index("NL")
    j = data.quarters.index("2015-Q1")
    assert not data.mask[nl, j] and not data.mask[nl, j + 1]
    assert data.mask[nl, j + 2] and data.mask[nl, j - 1]
    assert data.mask[[0, 1, 3]].all()
    assert data.masked_reasons == {"NL": ["2015-Q1"]}


def test_pandemic_excluded_windows_are_two_segments_without_2020_2021() -> None:
    data = build_bayes_data(synthetic(), {}, windows=PANDEMIC_EXCLUDED_WINDOWS, lags=2,
                            countries=COUNTRIES)
    assert int(data.segment_start.sum()) == 2
    second = data.quarters.index("2022-Q3")
    assert data.segment_start[second]
    assert not any("2020" in q or "2021" in q for q in data.quarters)


def test_excluded_inputs_and_missing_values_raise() -> None:
    with pytest.raises(EurostatDataError, match="excluded period"):
        build_bayes_data(synthetic(), {}, windows=(("2022-Q1", "2023-Q4"),), lags=2,
                         countries=COUNTRIES)
    panel = synthetic()
    panel[("PT", "2012-Q3")]["unemployment"] = None
    with pytest.raises(EurostatDataError, match="nothing is imputed"):
        build_bayes_data(panel, {}, lags=2, countries=COUNTRIES)
    with pytest.raises(ValueError, match="ordered"):
        window_quarters((("2015-Q1", "2016-Q1"), ("2010-Q1", "2011-Q1")))


# ----------------------------------------------------------------------------------
# Likelihood against an independent SciPy implementation
# ----------------------------------------------------------------------------------


def random_params(data_shape: tuple[int, int, int], rng: np.random.Generator,
                  participation: bool = False) -> dict[str, np.ndarray]:
    n, t, k = data_shape
    params = {
        "theta": rng.normal(-0.3, 0.2, n), "alpha": rng.normal(0, 0.1, n),
        "b": rng.normal(0, 0.1, (n, k)), "phi": rng.uniform(-0.6, 0.6, n),
        "sigma": rng.uniform(0.1, 0.6, n), "lam": rng.uniform(0, 0.2, n),
        "f": rng.normal(size=t), "nu": np.array(rng.uniform(3, 30)),
    }
    if participation:
        params["gamma"] = rng.normal(0, 0.2, n)
    return params


def scipy_loglik(data_y: np.ndarray, g: np.ndarray, z: np.ndarray, dp: np.ndarray,
                 starts: np.ndarray, p: dict[str, np.ndarray], student_t: bool, ar1: bool,
                 factor: bool) -> np.ndarray:
    """Element-by-element reference written independently of the JAX code."""
    n, t = data_y.shape
    out = np.empty((n, t))
    for i in range(n):
        residual = np.empty(t)
        for j in range(t):
            mean = p["alpha"][i] + p["theta"][i] * g[i, j] + float(z[i, j] @ p["b"][i])
            if "gamma" in p:
                mean += p["gamma"][i] * dp[i, j]
            if factor:
                mean += p["lam"][i] * p["f"][j]
            residual[j] = data_y[i, j] - mean
        for j in range(t):
            if ar1 and not starts[j]:
                u, scale = residual[j] - p["phi"][i] * residual[j - 1], p["sigma"][i]
            elif ar1:
                u, scale = residual[j], p["sigma"][i] / np.sqrt(1 - p["phi"][i] ** 2)
            else:
                u, scale = residual[j], p["sigma"][i]
            out[i, j] = (stats.t.logpdf(u, df=float(p["nu"]), scale=scale) if student_t
                         else stats.norm.logpdf(u, scale=scale))
    return out


@pytest.mark.parametrize("name", ["primary", "gaussian_iid", "participation",
                                  "pandemic_excluded"])
def test_likelihood_matches_scipy_reference(name: str) -> None:
    spec = SPECS[name]
    data = build_bayes_data(synthetic(), {}, windows=spec.windows, lags=spec.lags,
                            countries=COUNTRIES)
    rng = np.random.default_rng(11)
    params = random_params(data.z.shape, rng, spec.participation)
    used = {k: v for k, v in params.items()
            if (k != "lam" and k != "f") or spec.common_factor}
    ours = np.asarray(pointwise_log_likelihood(
        model_inputs(data), {k: jnp.asarray(v) for k, v in used.items()}, spec))
    reference = scipy_loglik(data.y, data.g, data.z, data.dp, data.segment_start, used,
                             spec.student_t, spec.ar1, spec.common_factor)
    np.testing.assert_allclose(ours, reference, rtol=1e-9, atol=1e-9)


def test_model_factor_is_the_masked_likelihood_sum() -> None:
    spec = SPECS["primary"]
    flags = {("LU", "2013-Q2"): {"lfs": "b", "gdp": ""}}
    data = build_bayes_data(synthetic(), flags, lags=2, countries=COUNTRIES)
    inputs = model_inputs(data)
    seeded = handlers.seed(okun_model, rng_seed=0)
    trace = handlers.trace(seeded).get_trace(inputs, spec)
    params = {name: trace[name]["value"] for name in ("theta", "alpha", "b", "phi", "sigma",
                                                       "lam", "f", "nu")}
    terms = np.asarray(pointwise_log_likelihood(inputs, params, spec))
    factor = float(np.asarray(trace["log_likelihood"]["fn"].log_factor))
    assert factor == pytest.approx(float(terms[data.mask].sum()), rel=1e-10)
    assert not data.mask.all()
    # beta0 is theta minus the lag weights, so theta is the coefficient sum.
    np.testing.assert_allclose(np.asarray(trace["beta0"]["value"]),
                               np.asarray(params["theta"]) - np.asarray(params["b"]).sum(1))


# ----------------------------------------------------------------------------------
# Diagnostics, sampling smoke test and reporting rules
# ----------------------------------------------------------------------------------


def test_diagnostics_accept_mixed_chains_and_reject_offset_chains() -> None:
    rng = np.random.default_rng(0)
    good = {"theta": rng.normal(size=(4, 1000, 3))}
    extra = {"energy": rng.normal(size=(4, 1000)), "diverging": np.zeros((4, 1000), bool),
             "num_steps": np.full((4, 1000), 15)}
    assert compute_diagnostics(good, extra, 10).accepted
    bad = {"theta": good["theta"] + np.arange(4)[:, None, None]}
    diagnostics = compute_diagnostics(bad, extra, 10)
    assert not diagnostics.accepted
    assert any("R-hat" in failure for failure in diagnostics.failures())
    extra["diverging"][0, 3] = True
    assert compute_diagnostics(good, extra, 10).divergences == 1


def test_sampling_smoke_run() -> None:
    """A tiny run that only checks the pipeline end to end; not a convergence test."""
    spec = replace(SPECS["primary"], windows=(("2012-Q1", "2017-Q4"),))
    data = build_bayes_data(synthetic(), {}, windows=spec.windows, lags=2,
                            countries=COUNTRIES)
    fit = fit_specification(data, spec, SamplerSettings(chains=1, warmup=100, draws=100,
                                                       seed=1))
    assert fit.samples["theta"].shape == (1, 100, 4)
    assert np.isfinite(fit.samples["theta"]).all()
    assert not fit.diagnostics.accepted          # one chain: R-hat undefined
    check = predictive_checks(fit, max_draws=20)
    assert 0.0 <= check.coverage_50 <= check.coverage_90 <= 1.0


def test_rank_and_pairwise_probabilities() -> None:
    rng = np.random.default_rng(5)
    theta = rng.normal([-0.8, -0.2, 0.0], 0.1, size=(4000, 3))
    ranks = rank_distribution(theta)
    np.testing.assert_allclose(ranks.sum(axis=1), 1.0)
    np.testing.assert_allclose(ranks.sum(axis=0), 1.0)
    assert ranks[0, 0] > 0.99
    pairs = pairwise_probability(theta)
    np.testing.assert_allclose(pairs + pairs.T + np.eye(3), 1.0)
    assert pairs[0, 2] > 0.99
    assert sign_class(theta[:, 0]) == "negative" and sign_class(theta[:, 2]) == "uncertain"


def _fit(theta: np.ndarray, beta0: np.ndarray, accepted: bool = True) -> LoadedFit:
    summary = {"countries": list(COUNTRIES), "diagnostics": {"accepted": accepted}}
    return LoadedFit("primary", summary, {"theta": theta, "beta0": beta0,
                                          "mu_theta": theta.mean(axis=1)})


def test_claim_rules() -> None:
    rng = np.random.default_rng(2)
    draws = 4000
    # ES strongly negative, LU near zero, NL negative with most weight on lags, PT average.
    theta = np.column_stack([rng.normal(-0.9, 0.05, draws), rng.normal(0.0, 0.05, draws),
                             rng.normal(-0.4, 0.05, draws), rng.normal(-0.4, 0.05, draws)])
    beta0 = theta.copy()
    beta0[:, 2] = theta[:, 2] * 0.2
    claims = {c.claim: c for c in evaluate_claims(_fit(theta, beta0))}
    assert claims["Luxembourg weak"].holds
    assert claims["Netherlands delayed"].holds
    assert claims["Spain large"].holds
    assert claim_verdict(claims["Spain large"], [claims["Spain large"]]) == "supported"
    reversed_theta = theta[:, [1, 0, 2, 3]]
    flipped = {c.claim: c for c in evaluate_claims(_fit(reversed_theta, reversed_theta))}
    assert flipped["Spain large"].contradicted
    assert claim_verdict(flipped["Spain large"], []) == "contradicted"


def _write_fixture_fit(directory: Path, rng: np.random.Generator, pooled: bool = True) -> None:
    """A fit directory in the format bayes-fit writes, filled with synthetic draws."""
    import json

    from eu_okun.contract import EU27

    n, chains, draws = len(EU27), 2, 60
    theta = rng.normal(-0.3, 0.1, size=(chains, draws, n))
    samples = {"theta": theta, "beta0": theta * 0.6}
    if pooled:
        samples["mu_theta"] = theta.mean(axis=2)
        samples["tau_theta"] = np.abs(rng.normal(0.1, 0.01, size=(chains, draws)))
    directory.mkdir(parents=True)
    np.savez_compressed(directory / "draws.npz", **samples)
    summary = {
        "countries": list(EU27), "quarters": ["2009-Q2", "2019-Q4"], "masked_terms": 3,
        "settings": {"chains": chains, "warmup": 10, "draws": draws, "target_accept": 0.95,
                     "max_tree_depth": 10, "seed": 1},
        "diagnostics": {"accepted": pooled, "failures": [] if pooled else ["R-hat 1.2 > 1.01"],
                        "max_rhat": 1.0 if pooled else 1.2, "min_ess_bulk": 500.0,
                        "min_ess_tail": 450.0, "divergences": 0, "min_bfmi": 0.8},
        "predictive_check": {
            "coverage_50": 0.5, "coverage_90": 0.9, "draws_used": 10,
            "observed": {"lag1_autocorrelation": 0.2, "cross_country_correlation": 0.1,
                         "mean_sd": 0.4},
            "replicated_mean": {"lag1_autocorrelation": 0.18, "cross_country_correlation": 0.1,
                                "mean_sd": 0.4},
            "p_value": {"lag1_autocorrelation": 0.4, "cross_country_correlation": 0.5,
                        "mean_sd": 0.6}},
        "classical_same_rows": {g: {"theta": -0.35, "se": 0.1, "ci_low": -0.55,
                                    "ci_high": -0.15, "n": 40.0} for g in EU27},
        "runtime_seconds": 12.0,
        "environment": {"python": "3.12", "platform": "test", "jax": "x", "numpyro": "x",
                        "arviz": "x", "numpy": "x"},
        "snapshot": {"manifest_sha256": "m" * 64, "processed_csv_sha256": "c" * 64},
    }
    (directory / "fit.json").write_text(json.dumps(summary), encoding="utf-8")


def test_report_renders_from_saved_fits(tmp_path: Path) -> None:
    from eu_okun.bayes_report import write_report

    rng = np.random.default_rng(8)
    _write_fixture_fit(tmp_path / "fits" / "primary", rng)
    _write_fixture_fit(tmp_path / "fits" / "no_pooling", rng, pooled=False)
    review = write_report(tmp_path / "fits", tmp_path / "out", figures=True)
    report = (tmp_path / "out" / "HIERARCHICAL_BAYES.md").read_text(encoding="utf-8")
    assert review.accepted() == ["primary"]
    for heading in ("## Summary", "## 2. Convergence and acceptance", "## 4. Rank uncertainty",
                    "## 5. Earlier claims", "## 7. Posterior predictive checks"):
        assert heading in report
    assert "**failed**" in report                  # the failed fit is reported, not hidden
    for name in ("country_estimates.csv", "rank_probabilities.csv", "pairwise_probabilities.csv",
                 "claims.csv", "diagnostics.json", "shrinkage.png", "pairwise.png"):
        assert (tmp_path / "out" / name).exists()


def test_factor_checks_detect_persistent_gdp_correlated_factor() -> None:
    from eu_okun.bayes_report import factor_checks

    rng = np.random.default_rng(4)
    t = 60
    gdp = rng.normal(size=t)
    persistent = np.zeros((500, t))
    for j in range(1, t):
        persistent[:, j] = 0.8 * persistent[:, j - 1] + rng.normal(size=500)
    f = persistent - 1.5 * gdp[None, :]
    fit = LoadedFit("primary", {"countries": list(COUNTRIES)}, {"f": f})
    starts = np.zeros(t, dtype=bool)
    starts[0] = True
    check = factor_checks(fit, gdp, -gdp, starts)
    assert check.corr_mean_gdp < -0.3 and check.corr_mean_unemployment_change > 0.3
    independent = LoadedFit("primary", {"countries": list(COUNTRIES)},
                            {"f": rng.normal(size=(500, t))})
    quiet = factor_checks(independent, gdp, -gdp, starts)
    assert abs(quiet.lag1_autocorrelation) < 0.1 and abs(quiet.corr_mean_gdp) < 0.1


def test_failed_fit_is_retried_once_with_longer_warmup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protocol section 6: one re-run with more warm-up; the first attempt is kept."""
    import eu_okun.bayes_fit as bf
    from eu_okun.bayes_fit import Diagnostics

    calls: list[int] = []

    def fake_fit(data, spec, settings):  # type: ignore[no-untyped-def]
        calls.append(settings.warmup)
        rhat = 1.05 if len(calls) == 1 else 1.0
        diagnostics = Diagnostics(rhat, "phi[0]", 900.0, 900.0, "x", 0, 0.8, 0, 4000)
        return bf.FitResult(spec, data, settings, {}, diagnostics, 1.0)

    saved: list[Path] = []
    monkeypatch.setattr(bf, "audit_snapshot", lambda d: {"snapshot_manifest_sha256": "m",
                                                         "processed_csv_sha256": "c"})
    monkeypatch.setattr(bf, "_read_panel", lambda path: synthetic())
    monkeypatch.setattr(bf, "read_flags", lambda path: {})
    monkeypatch.setattr(bf, "build_bayes_data", lambda *a, **k: "data")
    monkeypatch.setattr(bf, "fit_specification", fake_fit)
    monkeypatch.setattr(bf, "predictive_checks", lambda fit: None)

    def fake_save(fit, check, directory, snapshot):  # type: ignore[no-untyped-def]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "fit.json").write_text(str(fit.settings.warmup))
        saved.append(directory)

    monkeypatch.setattr(bf, "save_fit", fake_save)
    results = bf.run_specifications(tmp_path, tmp_path / "bayes", ["primary"],
                                    retry_warmup=3000)
    assert calls == [1000, 3000]
    assert results[0][1].accepted
    assert (tmp_path / "bayes" / "primary" / "fit.json").read_text() == "3000"
    assert (tmp_path / "bayes-first-attempt" / "primary" / "fit.json").read_text() == "1000"
