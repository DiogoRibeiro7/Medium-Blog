"""Tests of amendment 1: quarter effects, NetCDF artifacts, prior predictive checks,
LOO comparisons and posterior recovery on synthetic data with known parameters.

One small recovery fit samples for real (a few seconds); the full recovery runs belong
to the separate Bayesian workflow.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import stats

numpyro = pytest.importorskip("numpyro")
numpyro.enable_x64()

import jax.numpy as jnp  # noqa: E402
from numpyro import handlers  # noqa: E402

from eu_okun.bayes_artifacts import (  # noqa: E402
    build_datatree,
    pointwise_log_likelihood_draws,
    simulate_outcomes,
    write_posterior_netcdf,
)
from eu_okun.bayes_checks import (  # noqa: E402
    LooResult,
    assess_recovery,
    compare_loo,
    loo_of,
    prior_predictive_check,
    recovery_data,
)
from eu_okun.bayes_data import BayesData, build_bayes_data  # noqa: E402
from eu_okun.bayes_fit import SamplerSettings, fit_specification  # noqa: E402
from eu_okun.bayes_model import (  # noqa: E402
    AMENDMENT_1,
    PREREGISTERED,
    SPECS,
    model_inputs,
    okun_model,
    pointwise_log_likelihood,
)
from eu_okun.contract import EU27, quarter_sequence  # noqa: E402

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


def small_data(countries: tuple[str, ...] = COUNTRIES) -> BayesData:
    return build_bayes_data(synthetic(countries), {}, lags=2, countries=countries)


def fake_samples(data: BayesData, spec_name: str, chains: int = 2, draws: int = 30,
                 seed: int = 0) -> dict[str, np.ndarray]:
    """Posterior-shaped draws (chain, draw, ...) for a specification."""
    rng = np.random.default_rng(seed)
    spec = SPECS[spec_name]
    n, t, k = data.z.shape
    shape = (chains, draws)
    out = {"theta": rng.normal(-0.3, 0.1, (*shape, n)), "alpha": rng.normal(0, 0.05, (*shape, n)),
           "b": rng.normal(0, 0.05, (*shape, n, k)), "sigma": rng.uniform(0.15, 0.3, (*shape, n)),
           "mu_theta": rng.normal(-0.3, 0.05, shape), "tau_theta": rng.uniform(0.05, 0.2, shape)}
    out["beta0"] = out["theta"] - out["b"].sum(axis=-1)
    if spec.ar1:
        out["phi"] = rng.uniform(-0.3, 0.5, (*shape, n))
    if spec.student_t:
        out["nu"] = rng.uniform(4, 20, shape)
    if spec.common_factor:
        out["lam"] = rng.uniform(0, 0.2, (*shape, n))
        out["f"] = rng.normal(size=(*shape, t))
    if spec.time_effects:
        effect = rng.normal(0, 0.1, (*shape, t))
        out["quarter_effect"] = effect - effect.mean(axis=-1, keepdims=True)
        out["tau_quarter"] = rng.uniform(0.05, 0.2, shape)
    if spec.participation:
        out["gamma"] = rng.normal(0, 0.1, (*shape, n))
    return out


# ----------------------------------------------------------------------------------
# Quarter effects
# ----------------------------------------------------------------------------------


def test_amendment_specifications_are_separate_from_the_preregistered_nine() -> None:
    assert len(PREREGISTERED) == 9 and set(AMENDMENT_1).isdisjoint(PREREGISTERED)
    assert all(SPECS[n].time_effects for n in ("time_effects", "time_effects_participation"))
    assert not any(SPECS[n].common_factor for n in AMENDMENT_1)
    assert SPECS["participation_no_common_factor"].participation
    with pytest.raises(ValueError, match="never combined"):
        handlers.seed(okun_model, 0)(model_inputs(small_data()),
                                     replace(SPECS["primary"], time_effects=True))


def test_quarter_effect_likelihood_matches_an_independent_reference() -> None:
    spec = SPECS["time_effects_participation"]
    data = small_data()
    params = {k: v[0, 0] for k, v in fake_samples(data, spec.name, seed=4).items()}
    ours = np.asarray(pointwise_log_likelihood(
        model_inputs(data), {k: jnp.asarray(v) for k, v in params.items()}, spec))
    n, t = data.y.shape
    expected = np.empty((n, t))
    for i in range(n):
        resid = np.array([
            data.y[i, j] - params["alpha"][i] - params["theta"][i] * data.g[i, j]
            - data.z[i, j] @ params["b"][i] - params["gamma"][i] * data.dp[i, j]
            - params["quarter_effect"][j] for j in range(t)])
        for j in range(t):
            if data.segment_start[j]:
                u, scale = resid[j], params["sigma"][i] / np.sqrt(1 - params["phi"][i] ** 2)
            else:
                u, scale = resid[j] - params["phi"][i] * resid[j - 1], params["sigma"][i]
            expected[i, j] = stats.t.logpdf(u, df=float(params["nu"]), scale=scale)
    np.testing.assert_allclose(ours, expected, rtol=1e-9, atol=1e-9)


def test_quarter_effects_sum_to_zero_in_the_model() -> None:
    trace = handlers.trace(handlers.seed(okun_model, 1)).get_trace(
        model_inputs(small_data()), SPECS["time_effects"])
    effect = np.asarray(trace["quarter_effect"]["value"])
    assert effect.shape == (43,) and abs(effect.sum()) < 1e-10 and effect.std() > 0


# ----------------------------------------------------------------------------------
# Simulation, prior predictive checks and NetCDF artifacts
# ----------------------------------------------------------------------------------


def test_simulated_outcomes_follow_the_ar1_recursion() -> None:
    data = small_data()
    spec = SPECS["time_effects"]
    draws = {k: v[0, :3] for k, v in fake_samples(data, spec.name).items()}
    y = simulate_outcomes(spec, data, draws, np.random.default_rng(5))
    rng = np.random.default_rng(5)
    eps = rng.standard_t(np.asarray(draws["nu"]).reshape(3, 1, 1), size=y.shape)
    noise = draws["sigma"][:, :, None] * eps
    mean = (draws["alpha"][:, :, None] + draws["theta"][:, :, None] * data.g[None]
            + np.einsum("ntk,snk->snt", data.z, draws["b"]) + draws["quarter_effect"][:, None, :])
    e = noise[:, :, 0] / np.sqrt(1 - draws["phi"] ** 2)
    np.testing.assert_allclose(y[:, :, 0], mean[:, :, 0] + e)
    e1 = draws["phi"] * e + noise[:, :, 1]
    np.testing.assert_allclose(y[:, :, 1], mean[:, :, 1] + e1)


def test_prior_predictive_check_summarises_simulations() -> None:
    check = prior_predictive_check(SPECS["primary"], small_data(), draws=60, seed=1)
    assert check.draws == 60
    for low, mid, high in check.quantiles.values():
        assert low <= mid <= high
    assert 0.0 <= check.share_above_5pp <= 1.0
    assert set(check.observed) == set(check.quantiles)


@pytest.mark.integration
def test_netcdf_artifact_round_trip_and_loo(tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    import xarray as xr

    import eu_okun.bayes_artifacts as artifacts

    monkeypatch.setattr(artifacts, "PRIOR_DRAWS", 50)
    data = small_data()
    data = replace(data, mask=data.mask.copy())
    data.mask[1, 5] = False
    spec = SPECS["time_effects"]
    samples = fake_samples(data, spec.name)
    stats_ = {"diverging": np.zeros((2, 30), bool), "energy": np.ones((2, 30)),
              "num_steps": np.full((2, 30), 7)}
    path = tmp_path / "posterior.nc"
    write_posterior_netcdf(path, spec, data, samples, stats_, {"settings": {"seed": 1}})
    with xr.open_datatree(path, engine="h5netcdf") as tree:
        assert {"posterior", "sample_stats", "prior", "prior_predictive", "observed_data",
                "constant_data", "log_likelihood"} <= set(tree.children)
        assert tree["posterior"]["theta"].dims == ("chain", "draw", "country")
        assert tree["posterior"]["quarter_effect"].dims == ("chain", "draw", "quarter")
        assert tree["prior"]["theta"].shape == (1, 50, 4)
        assert tree["prior_predictive"]["y"].shape == (1, 50, 4, 43)
        assert tree["log_likelihood"]["y"].shape == (2, 30, int(data.mask.sum()))
        assert json.loads(tree.attrs["settings"]) == {"seed": 1}
        assert tree.attrs["specification"] == "time_effects"
        np.testing.assert_allclose(tree["observed_data"]["y"].values, data.y)
        expected = pointwise_log_likelihood_draws(spec, data, samples)[:, :, data.mask]
        np.testing.assert_allclose(tree["log_likelihood"]["y"].values, expected)
    result = loo_of(path, spec.name)
    assert result.cells == int(data.mask.sum()) and np.isfinite(result.elpd)
    without_stats = build_datatree(spec, data, samples)
    assert "sample_stats" not in without_stats.children


def test_loo_comparison_rule() -> None:
    a = LooResult("a", -100.0, 5.0, 3.0, 4, 0, tuple([-1.0, -1.0, -1.0, -97.0]))
    near = LooResult("b", -100.5, 5.0, 3.0, 4, 0, tuple([-1.1, -0.9, -1.2, -97.3]))
    assert compare_loo(a, near).verdict == "not distinguishable"
    far = LooResult("c", -140.0, 5.0, 3.0, 4, 1, tuple([-11.0, -11.0, -11.0, -107.0]))
    comparison = compare_loo(a, far)
    assert comparison.elpd_diff == pytest.approx(40.0)
    assert comparison.verdict == "a better supported" and not comparison.reliable
    with pytest.raises(ValueError, match="different cells"):
        compare_loo(a, LooResult("d", -1.0, 1.0, 1.0, 3, 0, (-1.0, -1.0, -1.0)))


# ----------------------------------------------------------------------------------
# Posterior recovery
# ----------------------------------------------------------------------------------


def test_coverage_range_matches_binomial_quantiles() -> None:
    from eu_okun.bayes_checks import coverage_range

    for total in (10, 27, 43, 100):
        low, high = stats.binom.ppf([0.025, 0.975], total, 0.9)
        assert coverage_range(total) == (int(low), int(high))


def test_assess_recovery_counts_coverage() -> None:
    rng = np.random.default_rng(6)
    truth = {"theta": np.array([-0.5, -0.2, 0.1])}
    samples = {"theta": truth["theta"] + rng.normal(0, 0.05, (2, 500, 3)),
               "mu_theta": rng.normal(-0.3, 0.05, (2, 500))}
    samples["theta"][:, :, 2] += 1.0                    # third country badly missed
    result = assess_recovery("x", samples, truth, True, 1.0)
    assert result.coverage["theta"][1:] == (2, 3)
    assert result.hyper["mu_theta"][3]
    assert len(result.theta) == 3 and result.theta[0][0] == -0.5


@pytest.mark.integration
def test_posterior_recovery_on_synthetic_data_with_known_parameters() -> None:
    """A small real fit: known country sums are recovered (gaussian_iid, 8 countries)."""
    countries = ("AT", "BE", "DE", "ES", "FR", "IT", "NL", "PT")
    spec = SPECS["gaussian_iid"]
    data, truth = recovery_data(spec, small_data(countries), seed=11)
    fit = fit_specification(data, spec, SamplerSettings(chains=2, warmup=300, draws=300,
                                                        seed=3))
    result = assess_recovery(spec.name, fit.samples, truth, fit.diagnostics.accepted,
                             fit.diagnostics.max_rhat)
    in50, in90, total = result.coverage["theta"]
    assert total == 8 and in90 >= 6
    median = np.median(fit.samples["theta"].reshape(-1, 8), axis=0)
    assert np.max(np.abs(median - truth["theta"])) < 0.15
    assert result.hyper["mu_theta"][3]


# ----------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------


def _write_fake_fit(directory: Path, data: BayesData, name: str, seed: int) -> None:
    import eu_okun.bayes_artifacts as artifacts

    samples = fake_samples(data, name, seed=seed)
    directory.mkdir(parents=True)
    np.savez_compressed(directory / "draws.npz", **samples)
    check = {"coverage_50": 0.5, "coverage_90": 0.9, "draws_used": 10,
             "observed": {"lag1_autocorrelation": 0.2, "cross_country_correlation": 0.3,
                          "mean_sd": 0.4},
             "replicated_mean": {"lag1_autocorrelation": 0.1, "cross_country_correlation": 0.25,
                                 "mean_sd": 0.4},
             "p_value": {"lag1_autocorrelation": 0.1, "cross_country_correlation": 0.3,
                         "mean_sd": 0.5}}
    summary: dict[str, Any] = {
        "spec": {"name": name}, "countries": list(data.countries),
        "quarters": list(data.quarters), "predictive_check": check,
        "diagnostics": {"accepted": True, "failures": [], "max_rhat": 1.004,
                        "min_ess_bulk": 800.0, "min_ess_tail": 700.0, "divergences": 0}}
    (directory / "fit.json").write_text(json.dumps(summary), encoding="utf-8")
    artifacts.write_posterior_netcdf(directory / "posterior.nc", SPECS[name], data, samples)


@pytest.mark.integration
def test_checks_report_renders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import eu_okun.bayes_artifacts as artifacts
    import eu_okun.bayes_checks as checks

    panel = synthetic(EU27)
    monkeypatch.setattr(artifacts, "PRIOR_DRAWS", 20)
    monkeypatch.setattr(checks, "PRIOR_CHECK_DRAWS", 20)
    monkeypatch.setattr(checks, "_read_panel", lambda path: panel)
    monkeypatch.setattr(checks, "read_flags", lambda path: {})
    data = build_bayes_data(panel, {}, lags=2)
    for k, name in enumerate(("primary", "no_common_factor", "participation", *AMENDMENT_1)):
        _write_fake_fit(tmp_path / "fits" / name, data, name, seed=k)
    recovery = tmp_path / "recovery" / "primary"
    recovery.mkdir(parents=True)
    rec = assess_recovery("primary", {"theta": fake_samples(data, "primary")["theta"],
                                      "mu_theta": fake_samples(data, "primary")["mu_theta"]},
                          {"theta": np.full(27, -0.3)}, True, 1.003)
    from dataclasses import asdict
    (recovery / "recovery.json").write_text(json.dumps(asdict(rec)), encoding="utf-8")
    review = checks.write_checks_report(tmp_path / "fits", tmp_path, tmp_path / "out",
                                        recovery_dir=tmp_path / "recovery", figures=True)
    assert len(review.comparisons) == 4 and len(review.prior) == len(SPECS)
    report = (tmp_path / "out" / "BAYES_MODEL_CHECKS.md").read_text(encoding="utf-8")
    for heading in ("## 1. Additive shared quarter effects",
                    "## 2. With versus without shared quarter effects",
                    "## 3. Prior predictive checks", "## 4. Posterior recovery on synthetic data"):
        assert heading in report
    for name in ("model_checks.json", "quarter_effects.png", "theta_shared_effects.png",
                 "recovery.png"):
        assert (tmp_path / "out" / name).exists()
    assert "## 6. Limitations" in report
    # A replication with the same results: identical verdicts, same acceptance.
    import shutil

    replication = tmp_path / "replication"
    replication.mkdir()
    shutil.copy(tmp_path / "out" / "model_checks.json", replication / "model_checks.json")
    for fit_dir in (tmp_path / "fits").iterdir():
        (replication / "fits" / fit_dir.name).mkdir(parents=True)
        shutil.copy(fit_dir / "fit.json", replication / "fits" / fit_dir.name / "fit.json")
    again = checks.write_checks_report(tmp_path / "fits", tmp_path, tmp_path / "out2",
                                       recovery_dir=tmp_path / "recovery", figures=False,
                                       replication_dir=replication)
    assert again.replication is not None
    text = (tmp_path / "out2" / "BAYES_MODEL_CHECKS.md").read_text(encoding="utf-8")
    assert "## 6. Replication on a second platform" in text and "## 8. Reproduction" in text
    assert "gives the same LOO verdicts" in text
    assert "acceptance is the same for every specification" in text
