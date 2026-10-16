"""Tests of amendment 3: the AR(1) common factor, the ARDL specifications and the
dynamics report (``BAYES_PROTOCOL.md``, section 11).

One small fit samples for real (a few seconds); the full fits belong to the separate
Bayesian workflow.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

numpyro = pytest.importorskip("numpyro")
numpyro.enable_x64()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from numpyro import handlers  # noqa: E402

from eu_okun.bayes_artifacts import (  # noqa: E402
    pointwise_log_likelihood_draws,
    simulate_outcomes,
    write_posterior_netcdf,
)
from eu_okun.bayes_data import (  # noqa: E402
    PANDEMIC_EXCLUDED_WINDOWS,
    BayesData,
    ar1_path,
    ardl_path,
    build_bayes_data,
)
from eu_okun.bayes_dynamics import (  # noqa: E402
    COMPARISONS,
    FactorDynamics,
    Persistence,
    cell_labels,
    common_cells,
    dynamic_claims,
    loo_on_cells,
    total_association,
)
from eu_okun.bayes_fit import (  # noqa: E402
    FitResult,
    SamplerSettings,
    compute_diagnostics,
    fit_specification,
    predictive_checks,
)
from eu_okun.bayes_model import (  # noqa: E402
    AMENDMENT_1,
    AMENDMENT_2,
    AMENDMENT_3,
    PREREGISTERED,
    SPECS,
    SegmentAR1,
    likelihood_mask,
    model_inputs,
    okun_model,
    pointwise_log_likelihood,
)
from eu_okun.bayes_report import FactorCheck, LoadedFit  # noqa: E402
from eu_okun.contract import quarter_sequence  # noqa: E402

Panel = dict[tuple[str, str], dict[str, float | None]]
COUNTRIES = ("ES", "LU", "NL", "PT")


def synthetic(countries: tuple[str, ...] = COUNTRIES, seed: int = 3, psi: float = 0.0,
              ) -> Panel:
    """Unemployment changes with an optional lagged-change coefficient ``psi``."""
    rng = np.random.default_rng(seed)
    panel: Panel = {}
    for geo in countries:
        u, p, g_prev, du = 8.0, 70.0, 0.4, 0.0
        for quarter in quarter_sequence("2008-Q1", "2026-Q2"):
            g = float(rng.normal(0.4, 0.8))
            du = -0.3 * g - 0.1 * g_prev + psi * du + float(rng.normal(0, 0.2))
            u += du
            p = p + float(rng.normal(0, 0.2))
            panel[(geo, quarter)] = {"gdp": g, "unemployment": u, "participation": p}
            g_prev = g
    return panel


def small_data(countries: tuple[str, ...] = COUNTRIES, seed: int = 3, psi: float = 0.0,
               ) -> BayesData:
    return build_bayes_data(synthetic(countries, seed, psi), {}, lags=2, countries=countries)


def fake_samples(data: BayesData, spec_name: str, chains: int = 2, draws: int = 30,
                 seed: int = 0) -> dict[str, np.ndarray]:
    """Posterior-shaped draws (chain, draw, ...) for an amendment 3 specification."""
    rng = np.random.default_rng(seed)
    spec = SPECS[spec_name]
    n, t, k = data.z.shape
    shape = (chains, draws)
    out = {"theta": rng.normal(-0.3, 0.1, (*shape, n)), "alpha": rng.normal(0, 0.05, (*shape, n)),
           "b": rng.normal(0, 0.05, (*shape, n, k)), "sigma": rng.uniform(0.15, 0.3, (*shape, n)),
           "mu_theta": rng.normal(-0.3, 0.05, shape), "tau_theta": rng.uniform(0.05, 0.2, shape),
           "nu": rng.uniform(4, 20, shape), "lam": rng.uniform(0, 0.2, (*shape, n)),
           "f": rng.normal(size=(*shape, t))}
    out["beta0"] = out["theta"] - out["b"].sum(axis=-1)
    if spec.ar1:
        out["phi"] = rng.uniform(-0.3, 0.5, (*shape, n))
    if spec.lagged_outcome:
        out["psi"] = rng.uniform(-0.2, 0.6, (*shape, n))
        out["mu_psi"] = rng.normal(0.2, 0.1, shape)
        out["tau_psi"] = rng.uniform(0.05, 0.3, shape)
    if spec.factor_ar:
        out["rho_f"] = rng.uniform(0.3, 0.9, shape)
    return out


# ----------------------------------------------------------------------------------
# Specifications and likelihood
# ----------------------------------------------------------------------------------


def test_amendment_3_specifications_are_separate_and_consistent() -> None:
    assert set(AMENDMENT_3).isdisjoint({*PREREGISTERED, *AMENDMENT_1, *AMENDMENT_2})
    assert SPECS["dynamic_factor"].factor_ar and SPECS["dynamic_factor"].ar1
    for name in ("ardl", "ardl_dynamic_factor"):
        spec = SPECS[name]
        assert spec.lagged_outcome and not spec.ar1 and spec.student_t and spec.common_factor
    assert not SPECS["ardl"].factor_ar and SPECS["ardl_dynamic_factor"].factor_ar
    inputs = model_inputs(small_data())
    with pytest.raises(ValueError, match="never both"):
        handlers.seed(okun_model, 0)(inputs, replace(SPECS["ardl"], ar1=True))
    with pytest.raises(ValueError, match="needs the common factor"):
        handlers.seed(okun_model, 0)(inputs, replace(SPECS["dynamic_factor"],
                                                     common_factor=False))


def test_lagged_change_is_zero_at_segment_starts() -> None:
    data = build_bayes_data(synthetic(), {}, PANDEMIC_EXCLUDED_WINDOWS, 2, COUNTRIES)
    lagged = data.y_lag()
    starts = np.flatnonzero(data.segment_start)
    assert starts.size == 2 and np.all(lagged[:, starts] == 0.0)
    inside = ~data.segment_start[1:]
    np.testing.assert_array_equal(lagged[:, 1:][:, inside], data.y[:, :-1][:, inside])


def test_ardl_likelihood_matches_scipy_reference_and_conditions_on_segment_starts() -> None:
    spec = SPECS["ardl_dynamic_factor"]
    data = build_bayes_data(synthetic(), {}, PANDEMIC_EXCLUDED_WINDOWS, 2, COUNTRIES)
    params = {k: v[0, 0] for k, v in fake_samples(data, spec.name, seed=4).items()}
    ours = np.asarray(pointwise_log_likelihood(
        model_inputs(data), {k: jnp.asarray(v) for k, v in params.items()}, spec))
    n, t = data.y.shape
    for i in range(n):
        for j in range(t):
            if data.segment_start[j]:
                continue
            resid = (data.y[i, j] - params["alpha"][i] - params["theta"][i] * data.g[i, j]
                     - data.z[i, j] @ params["b"][i] - params["lam"][i] * params["f"][j]
                     - params["psi"][i] * data.y[i, j - 1])
            expected = stats.t.logpdf(resid, df=float(params["nu"]), scale=params["sigma"][i])
            assert ours[i, j] == pytest.approx(expected, rel=1e-10)
    used = likelihood_mask(data.mask, data.segment_start, spec)
    assert used.sum() == data.mask.sum() - 2 * n
    # The model's likelihood factor sums exactly the used cells.
    trace = handlers.trace(handlers.seed(okun_model, 0)).get_trace(model_inputs(data), spec)
    sampled = {name: trace[name]["value"] for name in ("theta", "alpha", "b", "sigma", "lam",
                                                       "f", "nu", "psi")}
    terms = np.asarray(pointwise_log_likelihood(model_inputs(data), sampled, spec))
    factor = float(trace["log_likelihood"]["fn"].log_factor)
    assert factor == pytest.approx(float(terms[used].sum()), rel=1e-10)


def test_segment_ar1_density_matches_scipy_reference() -> None:
    starts = np.zeros(9, dtype=bool)
    starts[[0, 5]] = True
    f = np.random.default_rng(2).normal(size=9)
    rho = 0.7
    expected = 0.0
    for j in range(9):
        if starts[j]:
            expected += stats.norm.logpdf(f[j])
        else:
            expected += stats.norm.logpdf(f[j], rho * f[j - 1], np.sqrt(1 - rho**2))
    ours = float(SegmentAR1(jnp.asarray(rho), starts).log_prob(jnp.asarray(f)))
    assert ours == pytest.approx(expected, rel=1e-12)


def test_segment_ar1_samples_are_stationary_and_restart_at_segments() -> None:
    starts = np.zeros(12, dtype=bool)
    starts[[0, 6]] = True
    draws = np.asarray(SegmentAR1(jnp.asarray(0.8), starts).sample(
        jax.random.PRNGKey(0), (40_000,)))
    assert draws.shape == (40_000, 12)
    np.testing.assert_allclose(draws.var(axis=0), 1.0, atol=0.03)
    assert np.corrcoef(draws[:, 3], draws[:, 4])[0, 1] == pytest.approx(0.8, abs=0.01)
    assert abs(np.corrcoef(draws[:, 5], draws[:, 6])[0, 1]) < 0.02
    # The NumPy recursion used for replications draws from the same process.
    numpy_paths = ar1_path(np.random.default_rng(1).standard_normal((40_000, 12)), 0.8, starts)
    np.testing.assert_allclose(numpy_paths.var(axis=0), 1.0, atol=0.03)
    assert abs(np.corrcoef(numpy_paths[:, 5], numpy_paths[:, 6])[0, 1]) < 0.02


def test_dynamic_factor_prior_draws_follow_the_ar1() -> None:
    from eu_okun.bayes_artifacts import prior_draws

    prior = prior_draws(SPECS["dynamic_factor"], small_data(), draws=400, seed=3)
    assert {"rho_f", "f", "phi"} <= set(prior) and prior["f"].shape == (400, 43)
    assert np.all(np.abs(prior["rho_f"]) < 1)
    # Conditional on rho_f, consecutive values follow the AR(1): with rho_f known the
    # standardised innovations are N(0, 1).
    rho = prior["rho_f"][:, None]
    innovations = (prior["f"][:, 1:] - rho * prior["f"][:, :-1]) / np.sqrt(1 - rho**2)
    assert innovations.std() == pytest.approx(1.0, abs=0.03)


# ----------------------------------------------------------------------------------
# Simulation, predictive checks and artifacts
# ----------------------------------------------------------------------------------


def test_ardl_path_starts_from_observed_values() -> None:
    starts = np.array([True, False, False, True, False])
    y = np.arange(10.0).reshape(2, 5)
    x = np.ones((3, 2, 5))
    psi = np.array([[0.5, -0.5]] * 3)
    out = ardl_path(x, psi, y, starts)
    np.testing.assert_allclose(out[:, :, 0], np.broadcast_to(y[:, 0], (3, 2)))
    np.testing.assert_allclose(out[0, 0, 1:3], [1 + 0.5 * 0, 1 + 0.5 * 1])
    np.testing.assert_allclose(out[0, 1, 1], 1 - 0.5 * 5)
    np.testing.assert_allclose(out[:, :, 3], np.broadcast_to(y[:, 3], (3, 2)))


def test_simulated_ardl_outcomes_follow_the_recursion() -> None:
    data = small_data()
    spec = SPECS["ardl"]
    draws = {k: v[0, :3] for k, v in fake_samples(data, spec.name).items()}
    y = simulate_outcomes(spec, data, draws, np.random.default_rng(5))
    rng = np.random.default_rng(5)
    eps = rng.standard_t(np.asarray(draws["nu"]).reshape(3, 1, 1), size=y.shape)
    mean = (draws["alpha"][:, :, None] + draws["theta"][:, :, None] * data.g[None]
            + np.einsum("ntk,snk->snt", data.z, draws["b"])
            + draws["lam"][:, :, None] * draws["f"][:, None, :])
    x = mean + draws["sigma"][:, :, None] * eps
    np.testing.assert_allclose(y[:, :, 0], np.broadcast_to(data.y[:, 0], (3, 4)))
    np.testing.assert_allclose(y[:, :, 1], x[:, :, 1] + draws["psi"] * data.y[:, 0])
    np.testing.assert_allclose(y[:, :, 2], x[:, :, 2] + draws["psi"] * y[:, :, 1])


def _fake_fit(data: BayesData, name: str, seed: int = 0, **overrides: float) -> FitResult:
    samples = fake_samples(data, name, seed=seed)
    for key, value in overrides.items():
        samples[key] = np.full_like(samples[key], value)
    extra = {"energy": np.random.default_rng(seed).normal(size=(2, 30)),
             "diverging": np.zeros((2, 30), bool), "num_steps": np.full((2, 30), 7)}
    diagnostics = compute_diagnostics({"theta": samples["theta"]}, extra, 10)
    return FitResult(SPECS[name], data, SamplerSettings(chains=2, draws=30), samples,
                     diagnostics, 1.0, extra)


def test_ardl_replications_carry_the_persistence_of_psi() -> None:
    data = small_data()
    low = predictive_checks(_fake_fit(data, "ardl", psi=0.0, lam=0.0), max_draws=60)
    high = predictive_checks(_fake_fit(data, "ardl", psi=0.7, lam=0.0), max_draws=60)
    assert high.replicated_mean["lag1_autocorrelation"] > (
        low.replicated_mean["lag1_autocorrelation"] + 0.3)
    assert 0.0 <= high.coverage_90 <= 1.0


def test_dynamic_factor_replications_are_persistent_across_countries() -> None:
    data = small_data()
    iid = predictive_checks(_fake_fit(data, "dynamic_factor", rho_f=0.0, lam=0.3, phi=0.0),
                            max_draws=60)
    persistent = predictive_checks(
        _fake_fit(data, "dynamic_factor", rho_f=0.95, lam=0.3, phi=0.0), max_draws=60)
    assert persistent.replicated_mean["lag1_autocorrelation"] > (
        iid.replicated_mean["lag1_autocorrelation"] + 0.1)


def test_ardl_netcdf_log_likelihood_excludes_segment_starts(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import xarray as xr

    import eu_okun.bayes_artifacts as artifacts

    monkeypatch.setattr(artifacts, "PRIOR_DRAWS", 40)
    data = small_data()
    spec = SPECS["ardl_dynamic_factor"]
    samples = fake_samples(data, spec.name)
    path = tmp_path / "posterior.nc"
    write_posterior_netcdf(path, spec, data, samples)
    used = likelihood_mask(data.mask, data.segment_start, spec)
    with xr.open_datatree(path, engine="h5netcdf") as tree:
        cells = list(tree["log_likelihood"]["cell"].values)
        assert len(cells) == int(used.sum()) and not any(c.endswith("2009-Q2") for c in cells)
        assert {"psi", "rho_f", "mu_psi", "tau_psi"} <= set(tree["posterior"].data_vars)
        expected = pointwise_log_likelihood_draws(spec, data, samples)[:, :, used]
        np.testing.assert_allclose(tree["log_likelihood"]["y"].values, expected)
        assert np.all(np.isfinite(tree["prior_predictive"]["y"].values))


def test_ardl_sampling_recovers_the_lagged_change() -> None:
    data = small_data(psi=0.5, seed=11)
    fit = fit_specification(data, SPECS["ardl_dynamic_factor"],
                            SamplerSettings(chains=2, warmup=300, draws=300, seed=1))
    psi = fit.samples["psi"].reshape(-1, 4)
    assert fit.diagnostics.divergences <= 5
    assert np.all(np.abs(psi) < 1) and np.all(np.abs(fit.samples["rho_f"]) < 1)
    assert 0.3 < float(np.median(psi)) < 0.7


# ----------------------------------------------------------------------------------
# The dynamics report (section 11.4)
# ----------------------------------------------------------------------------------


def _loaded(name: str, draws: dict[str, np.ndarray], countries: list[str]) -> LoadedFit:
    return LoadedFit(name, {"countries": countries, "diagnostics": {"accepted": True}}, draws)


def test_long_run_multiplier_keeps_the_sign_of_theta() -> None:
    rng = np.random.default_rng(0)
    theta = rng.normal(-0.2, 0.1, (500, 3))
    psi = rng.uniform(-0.9, 0.9, (500, 3))
    estimand, total = total_association(_loaded("ardl", {"theta": theta, "psi": psi},
                                                ["A", "B", "C"]))
    assert estimand == "theta_LR"
    np.testing.assert_allclose(total, theta / (1 - psi))
    assert np.all(np.sign(total) == np.sign(theta))
    estimand, same = total_association(_loaded("dynamic_factor", {"theta": theta},
                                               ["A", "B", "C"]))
    assert estimand == "theta" and same is theta


def test_persistence_and_factor_rules() -> None:
    def check(p: float) -> Persistence:
        return Persistence("x", 0.27, 0.2, p, 0.5, 0.9, {})

    assert check(0.05).fixed and check(0.95).fixed and check(0.5).fixed
    assert not check(0.049).fixed and not check(0.951).fixed
    factor = FactorCheck("x", 0.5, 0.4, 0.6, -0.5, -0.6, -0.4, 0.1)
    assert FactorDynamics("x", True, 0.5, 0.01, 0.9, factor).supported
    assert FactorDynamics("x", True, -0.5, -0.9, -0.01, factor).supported
    assert not FactorDynamics("x", True, 0.1, -0.01, 0.4, factor).supported
    assert not FactorDynamics("x", False, np.nan, np.nan, np.nan, factor).supported


def test_ardl_claims_use_the_long_run_multiplier_for_luxembourg_and_spain() -> None:
    from eu_okun.contract import EU27

    rng = np.random.default_rng(1)
    countries = list(EU27)
    n = len(countries)
    theta = rng.normal(-0.2, 0.02, (400, n))
    psi = np.full((400, n), 0.3)
    lu, es, nl = countries.index("LU"), countries.index("ES"), countries.index("NL")
    psi[:, lu] = -0.6           # shrinks the long-run multiplier of LU towards zero
    theta[:, lu] = rng.normal(-0.2, 0.01, 400)
    theta[:, es] = rng.normal(-0.5, 0.01, 400)
    beta0 = theta * 0.8
    beta0[:, nl] = theta[:, nl] * 0.2
    draws = {"theta": theta, "beta0": beta0, "psi": psi,
             "mu_theta": rng.normal(-0.2, 0.002, 400)}
    claims = dynamic_claims(_loaded("ardl", draws, countries))
    assert [c.claim for c in claims] == ["Luxembourg weak", "Netherlands delayed", "Spain large"]
    assert claims[0].probability == pytest.approx(1.0) and "theta_LR_LU" in claims[0].detail
    assert "mean theta_LR" in claims[2].detail and claims[2].holds
    # The Netherlands rule keeps the short-run share (theta - beta0) / theta = 0.8.
    assert claims[1].holds and "median lag share 0.80" in claims[1].detail
    # Without a lagged change the section 7 rules are unchanged.
    plain = dynamic_claims(_loaded("dynamic_factor", draws, countries))
    assert "mu_theta" in plain[0].detail and plain[0].probability < 0.95


def test_loo_on_cells_matches_full_loo_and_restricts_to_common_cells(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import eu_okun.bayes_artifacts as artifacts
    from eu_okun.bayes_checks import loo_of

    monkeypatch.setattr(artifacts, "PRIOR_DRAWS", 20)
    data = small_data()
    paths = {}
    for k, name in enumerate(("primary", "ardl")):
        paths[name] = tmp_path / f"{name}.nc"
        write_posterior_netcdf(paths[name], SPECS[name], data, fake_samples(data, name, seed=k))
    labels = {name: cell_labels(path) for name, path in paths.items()}
    assert len(labels["primary"]) == len(labels["ardl"]) + 4
    common = common_cells(list(labels.values()))
    assert common == labels["ardl"] and common_cells([]) == []
    full = loo_of(paths["primary"], "primary")
    same = loo_on_cells(paths["primary"], "primary", labels["primary"])
    assert same.elpd == pytest.approx(full.elpd) and same.cells == full.cells
    restricted = loo_on_cells(paths["primary"], "primary", common)
    assert restricted.cells == len(common)
    starts = [i for i, c in enumerate(labels["primary"]) if c.endswith("2009-Q2")]
    expected = np.delete(np.asarray(full.pointwise), starts)
    np.testing.assert_allclose(restricted.pointwise, expected, rtol=1e-6)


def _write_fit(directory: Path, data: BayesData, name: str, seed: int, p_lag1: float) -> None:
    import json

    samples = fake_samples(data, name, seed=seed)
    directory.mkdir(parents=True)
    np.savez_compressed(directory / "draws.npz", **samples)  # type: ignore[arg-type]
    check = {"coverage_50": 0.5, "coverage_90": 0.9, "draws_used": 10,
             "observed": {"lag1_autocorrelation": 0.27, "cross_country_correlation": 0.3,
                          "mean_sd": 0.4},
             "replicated_mean": {"lag1_autocorrelation": 0.1, "cross_country_correlation": 0.25,
                                 "mean_sd": 0.4},
             "p_value": {"lag1_autocorrelation": p_lag1, "cross_country_correlation": 0.3,
                         "mean_sd": 0.5}}
    summary = {
        "spec": {"name": name}, "countries": list(data.countries),
        "quarters": list(data.quarters), "predictive_check": check,
        "settings": {"warmup": 1000}, "runtime_seconds": 60.0,
        "classical_same_rows": {g: {"theta": -0.2, "ci_low": -0.3, "ci_high": -0.1}
                                for g in data.countries},
        "diagnostics": {"accepted": True, "failures": [], "max_rhat": 1.004,
                        "max_rhat_parameter": "theta[0]", "min_ess_bulk": 800.0,
                        "min_ess_tail": 700.0, "divergences": 0, "min_bfmi": 0.8}}
    (directory / "fit.json").write_text(json.dumps(summary), encoding="utf-8")
    write_posterior_netcdf(directory / "posterior.nc", SPECS[name], data, samples)


def test_dynamics_report_renders_and_replicates(tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    import eu_okun.bayes_artifacts as artifacts
    import eu_okun.bayes_dynamics as dynamics
    from eu_okun.contract import EU27

    panel = synthetic(EU27)
    monkeypatch.setattr(artifacts, "PRIOR_DRAWS", 20)
    monkeypatch.setattr(dynamics, "_read_panel", lambda path: panel)
    monkeypatch.setattr(dynamics, "read_flags", lambda path: {})
    monkeypatch.setattr(dynamics, "PRIOR_DRAWS", 30)
    data = build_bayes_data(panel, {}, lags=2)
    for k, (name, p) in enumerate((("primary", 0.0), ("dynamic_factor", 0.03),
                                   ("ardl", 0.2), ("ardl_dynamic_factor", 0.5))):
        _write_fit(tmp_path / "fits" / name, data, name, k, p)
    with pytest.raises(FileNotFoundError, match="primary"):
        dynamics.build_dynamics(tmp_path / "empty", tmp_path)
    review = dynamics.write_dynamics_report(tmp_path / "fits", tmp_path, tmp_path / "out")
    assert [(c.model_a, c.model_b) for c in review.comparisons] == list(COMPARISONS)
    assert all(r.cells == int(data.mask.sum()) - 27 for r in review.loo.values())
    assert not review.persistence["dynamic_factor"].fixed and review.persistence["ardl"].fixed
    report = (tmp_path / "out" / "BAYES_DYNAMICS.md").read_text(encoding="utf-8")
    for heading in ("## 1. Convergence", "## 2. Prior predictive check",
                    "## 3. The persistence misfit", "## 4. The common factor",
                    "## 5. PSIS-LOO on common cells", "## 6. Country estimates",
                    "## 7. The three earlier claims", "## 8. What this changes"):
        assert heading in report
    assert set(review.prior) == {"primary", *AMENDMENT_3}
    assert all(p.draws == 30 for p in review.prior.values())
    assert "`dynamic_factor` does not fix the persistence misfit (p = 0.030)" in report
    assert "`ardl` fixes the persistence misfit (p = 0.200)" in report
    for name in ("dynamics.json", "persistence.png", "dynamics_parameters.png",
                 "country_estimates.png"):
        assert (tmp_path / "out" / name).exists()
    replication = tmp_path / "replication"
    replication.mkdir()
    shutil.copy(tmp_path / "out" / "dynamics.json", replication / "dynamics.json")
    (replication / "README.md").write_text("# Linux replication, run 1\n", encoding="utf-8")
    again = dynamics.write_dynamics_report(tmp_path / "fits", tmp_path, tmp_path / "out2",
                                           figures=False, replication_dir=replication)
    assert again.replication is not None and again.replication.label == "Linux replication, run 1"
    text = (tmp_path / "out2" / "BAYES_DYNAMICS.md").read_text(encoding="utf-8")
    assert "## 9. Replication on a second platform" in text
    assert "| `ardl` | yes / yes | 1000 / 1000 | 0.200 / 0.200 | 0.000 | yes | yes |" in text
    assert "Replication (Linux replication, run 1): every verdict is the same." in text
    # A replication that crosses a threshold: the verdicts are flagged as not stable.
    import json

    summary = json.loads((replication / "dynamics.json").read_text(encoding="utf-8"))
    summary["specifications"]["ardl"].update(lag1_p_value=0.03, persistence_fixed=False)
    first = summary["comparisons"][0]
    first["verdict"] = ("not distinguishable" if first["verdict"] != "not distinguishable"
                        else "dynamic_factor better supported")
    (replication / "dynamics.json").write_text(json.dumps(summary), encoding="utf-8")
    flipped = dynamics.write_dynamics_report(tmp_path / "fits", tmp_path, tmp_path / "out3",
                                             figures=False, replication_dir=replication)
    unstable = dynamics.unstable_verdicts(flipped)
    assert len(unstable) == 2 and "persistence verdict of `ardl`" in unstable[0]
    assert "PSIS-LOO `dynamic_factor` against `primary`" in unstable[1]
    text = (tmp_path / "out3" / "BAYES_DYNAMICS.md").read_text(encoding="utf-8")
    assert "2 verdicts differ and are not treated as established" in text
    assert "the replication gives p = 0.030, so this verdict is not stable" in text
    assert "; not stable)" in text
