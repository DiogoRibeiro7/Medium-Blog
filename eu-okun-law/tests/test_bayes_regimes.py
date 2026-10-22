"""Tests of amendment 4: regime shifts, the drifting association and the regimes report
(``BAYES_PROTOCOL.md``, section 12).

One small fit samples for real (a few seconds); the full fits belong to the separate
Bayesian workflow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import stats

numpyro = pytest.importorskip("numpyro")
numpyro.enable_x64()

import jax.numpy as jnp  # noqa: E402
from numpyro import handlers  # noqa: E402

from eu_okun.bayes_artifacts import simulate_outcomes  # noqa: E402
from eu_okun.bayes_data import (  # noqa: E402
    PANDEMIC_EXCLUDED_WINDOWS,
    BayesData,
    build_bayes_data,
    quarter_index,
    regime_terms,
)
from eu_okun.bayes_fit import SamplerSettings, fit_specification  # noqa: E402
from eu_okun.bayes_model import (  # noqa: E402
    AMENDMENT_1,
    AMENDMENT_2,
    AMENDMENT_3,
    AMENDMENT_4,
    PREREGISTERED,
    SPECS,
    design_for,
    model_inputs,
    okun_model,
    pointwise_log_likelihood,
)
from eu_okun.bayes_report import LoadedFit  # noqa: E402
from eu_okun.contract import quarter_sequence  # noqa: E402

Panel = dict[tuple[str, str], dict[str, float | None]]
COUNTRIES = ("ES", "LU", "NL", "PT")


def synthetic(countries: tuple[str, ...] = COUNTRIES, seed: int = 3, shift: float = 0.0,
              ) -> Panel:
    """Unemployment changes whose GDP sum moves by ``shift`` from 2022."""
    rng = np.random.default_rng(seed)
    panel: Panel = {}
    for geo in countries:
        u, p, g_prev = 8.0, 70.0, 0.4
        for quarter in quarter_sequence("2008-Q1", "2026-Q2"):
            g = float(rng.normal(0.4, 0.8))
            extra = shift if quarter >= "2022-Q1" else 0.0
            u = u + (-0.3 + extra) * g - 0.1 * g_prev + float(rng.normal(0, 0.15))
            p = p + float(rng.normal(0, 0.2))
            panel[(geo, quarter)] = {"gdp": g, "unemployment": u, "participation": p}
            g_prev = g
    return panel


def regime_data(countries: tuple[str, ...] = COUNTRIES, seed: int = 3, shift: float = 0.0,
                ) -> BayesData:
    return build_bayes_data(synthetic(countries, seed, shift), {}, PANDEMIC_EXCLUDED_WINDOWS,
                            2, countries)


def fake_draw(data: BayesData, spec_name: str, seed: int = 0) -> dict[str, np.ndarray]:
    """One parameter draw for an amendment 4 specification."""
    rng = np.random.default_rng(seed)
    spec = SPECS[spec_name]
    n, t, k = data.z.shape
    out = {"theta": rng.normal(-0.3, 0.1, n), "alpha": rng.normal(0, 0.05, n),
           "b": rng.normal(0, 0.05, (n, k)), "sigma": rng.uniform(0.15, 0.3, n),
           "nu": np.asarray(rng.uniform(4, 20)), "lam": rng.uniform(0, 0.2, n),
           "f": rng.normal(size=t)}
    if spec.ar1:
        out["phi"] = rng.uniform(-0.3, 0.5, n)
    if spec.regime_shift:
        out["delta"] = rng.normal(0.2, 0.1, n)
    if spec.post_shifts:
        out["alpha_shift"] = rng.normal(0, 0.05, n)
        out["log_sigma_shift"] = np.asarray(rng.normal(0, 0.3))
    if spec.drift:
        walk = np.cumsum(rng.normal(0, 0.05, t))
        out["kappa"] = walk - walk.mean()
    if spec.lagged_outcome:
        out["psi"] = rng.uniform(-0.2, 0.6, n)
    return out


# ----------------------------------------------------------------------------------
# Specifications, data and likelihood
# ----------------------------------------------------------------------------------


def test_amendment_4_specifications_are_separate_and_on_the_pandemic_excluded_windows() -> None:
    assert set(AMENDMENT_4).isdisjoint({*PREREGISTERED, *AMENDMENT_1, *AMENDMENT_2,
                                        *AMENDMENT_3})
    for name in AMENDMENT_4:
        assert SPECS[name].windows == PANDEMIC_EXCLUDED_WINDOWS and SPECS[name].lags == 2
    assert SPECS["regime_shift"].regime_shift and SPECS["regime_shift"].post_shifts
    assert SPECS["regime_shift_final_gdp"].final_gdp_only
    assert SPECS["drifting_association"].drift and not SPECS["drifting_association"].regime_shift
    for name in ("pandemic_excluded_dynamic", "regime_shift_dynamic"):
        assert SPECS[name].lagged_outcome and SPECS[name].factor_ar and not SPECS[name].ar1
    assert not SPECS["pandemic_excluded_dynamic"].post_shifts
    from dataclasses import replace

    with pytest.raises(ValueError, match="partial pooling"):
        handlers.seed(okun_model, 0)(model_inputs(regime_data()),
                                     replace(SPECS["regime_shift"], pooling=False))


def test_post_indicator_and_calendar_gaps() -> None:
    data = regime_data()
    post, gaps = data.post(), data.calendar_gaps()
    first_post = list(data.quarters).index("2022-Q3")
    assert not post[:first_post].any() and post[first_post:].all()
    assert gaps[0] == 0 and gaps[first_post] == 11 and set(gaps[1:]) == {1.0, 11.0}
    assert quarter_index("2022-Q3") - quarter_index("2019-Q4") == 11


def test_final_gdp_only_masks_terms_with_provisional_gdp_inputs() -> None:
    flags = {("LU", "2024-Q1"): {"gdp": "p"}}
    spec = SPECS["regime_shift_final_gdp"]
    data = design_for(spec, synthetic(), flags, COUNTRIES)  # type: ignore[arg-type]
    plain = design_for(SPECS["regime_shift"], synthetic(), flags,  # type: ignore[arg-type]
                       COUNTRIES)
    lu, quarters = 1, list(data.quarters)
    masked = {quarters[j] for j in np.flatnonzero(~data.mask[lu])}
    # GDP of 2024-Q1 enters the terms of 2024-Q1 (lag 0), Q2 (lag 1) and Q3 (lag 2); the
    # AR(1) term of 2024-Q4 follows a masked term.
    assert masked == {"2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4"}
    assert plain.mask.all() and data.mask[[0, 2, 3]].all()


def test_regime_likelihood_matches_scipy_reference() -> None:
    spec = SPECS["regime_shift"]
    data = regime_data()
    params = fake_draw(data, spec.name, seed=4)
    ours = np.asarray(pointwise_log_likelihood(
        model_inputs(data), {k: jnp.asarray(v) for k, v in params.items()}, spec))
    post = data.post()
    n, t = data.y.shape
    for i in range(n):
        theta = params["theta"][i] + params["delta"][i] * post
        alpha = params["alpha"][i] + params["alpha_shift"][i] * post
        sigma = params["sigma"][i] * np.exp(params["log_sigma_shift"] * post)
        resid = (data.y[i] - alpha - theta * data.g[i] - data.z[i] @ params["b"][i]
                 - params["lam"][i] * params["f"])
        for j in range(t):
            if data.segment_start[j]:
                u, scale = resid[j], sigma[j] / np.sqrt(1 - params["phi"][i] ** 2)
            else:
                u, scale = resid[j] - params["phi"][i] * resid[j - 1], sigma[j]
            expected = stats.t.logpdf(u, df=float(params["nu"]), scale=scale)
            assert ours[i, j] == pytest.approx(expected, rel=1e-10)


def test_drift_is_a_centred_random_walk_scaled_by_calendar_gaps() -> None:
    data = regime_data()
    t = data.n_quarters
    steps = np.random.default_rng(1).normal(size=t - 1)
    substituted = handlers.substitute(okun_model, data={
        "tau_kappa": jnp.asarray(0.04), "kappa_raw": jnp.asarray(steps)})
    trace = handlers.trace(handlers.seed(substituted, 0)).get_trace(
        model_inputs(data), SPECS["drifting_association"])
    kappa = np.asarray(trace["kappa"]["value"])
    walk = np.concatenate([[0.0], np.cumsum(0.04 * np.sqrt(data.calendar_gaps()[1:]) * steps)])
    np.testing.assert_allclose(kappa, walk - walk.mean(), atol=1e-12)
    assert abs(kappa.sum()) < 1e-10
    assert "delta" not in trace and "alpha_shift" in trace and "log_sigma_shift" in trace


def test_regime_terms_are_exact_without_shifts_and_shift_after_2021() -> None:
    data = regime_data()
    base = fake_draw(data, "pandemic_excluded")
    theta, alpha, sigma = regime_terms(data, base)
    assert theta.shape == (4, 1) and np.array_equal(theta[:, 0], base["theta"])
    assert np.array_equal(sigma[:, 0], base["sigma"])
    shifted = fake_draw(data, "regime_shift")
    theta, alpha, sigma = regime_terms(data, shifted)
    post = data.post()
    np.testing.assert_allclose(theta[:, post], (shifted["theta"] + shifted["delta"])[:, None]
                               * np.ones(post.sum()))
    np.testing.assert_allclose(theta[:, ~post], shifted["theta"][:, None]
                               * np.ones((~post).sum()))
    np.testing.assert_allclose(sigma[:, post][:, 0],
                               shifted["sigma"] * np.exp(shifted["log_sigma_shift"]))


def test_simulated_regime_outcomes_use_the_shifted_terms() -> None:
    data = regime_data()
    spec = SPECS["regime_shift"]
    one = fake_draw(data, spec.name, seed=2)
    draws = {k: np.asarray(v)[None] for k, v in one.items()}
    y = simulate_outcomes(spec, data, draws, np.random.default_rng(5))
    rng = np.random.default_rng(5)
    eps = rng.standard_t(np.asarray(draws["nu"]).reshape(1, 1, 1), size=y.shape)
    theta, alpha, sigma = regime_terms(data, draws)
    mean = (alpha + theta * data.g[None] + np.einsum("ntk,snk->snt", data.z, draws["b"])
            + draws["lam"][:, :, None] * draws["f"][:, None, :])
    noise = sigma * eps
    phi = draws["phi"]
    j = list(data.quarters).index("2022-Q3")          # a segment start in the post regime
    np.testing.assert_allclose(y[:, :, j], mean[:, :, j] + noise[:, :, j] / np.sqrt(1 - phi**2))
    e_prev = y[:, :, j] - mean[:, :, j]
    np.testing.assert_allclose(y[:, :, j + 1], mean[:, :, j + 1] + phi * e_prev
                               + noise[:, :, j + 1])


@pytest.mark.integration
def test_regime_sampling_detects_a_known_shift() -> None:
    data = regime_data(shift=0.35, seed=11)
    fit = fit_specification(data, SPECS["regime_shift"],
                            SamplerSettings(chains=2, warmup=300, draws=300, seed=1))
    mu_delta = fit.samples["mu_delta"].reshape(-1)
    assert fit.diagnostics.divergences <= 5
    assert 0.1 < float(np.median(mu_delta)) < 0.6 and np.mean(mu_delta > 0) > 0.9
    theta_post = fit.samples["theta_post"].reshape(-1, 4)
    np.testing.assert_allclose(theta_post, (fit.samples["theta"]
                                            + fit.samples["delta"]).reshape(-1, 4))


# ----------------------------------------------------------------------------------
# The regimes report (section 12.5)
# ----------------------------------------------------------------------------------


def test_shift_verdict_thresholds() -> None:
    from eu_okun.bayes_regimes import shift_verdict

    def draws(share_positive: float) -> np.ndarray:
        k = int(round(1000 * share_positive))
        return np.concatenate([np.full(k, 1.0), np.full(1000 - k, -1.0)])

    assert shift_verdict(draws(0.95)) == "rises" and shift_verdict(draws(0.949)) == (
        "no credible change")
    assert shift_verdict(draws(0.05)) == "falls" and shift_verdict(draws(0.051)) == (
        "no credible change")


def test_classical_weaker_rule_is_recomputed_from_the_review_tables(tmp_path: Path) -> None:
    from eu_okun.bayes_regimes import classical_weaker

    (tmp_path / "grid.csv").write_text(
        "sample,geo,gdp_lags,participation,ci_high\n"
        "pre_2020,AT,1,False,-0.10\npre_2020,BE,1,False,-0.10\n"
        "pre_2020,AT,2,False,9\npre_2020,AT,1,True,9\nfull,BE,1,False,9\n", encoding="utf-8")
    (tmp_path / "stability_review.csv").write_text(
        "geo,participation,post_loo_min\nAT,False,-0.05\nBE,False,-0.20\nAT,True,-0.30\n",
        encoding="utf-8")
    assert classical_weaker(tmp_path) == {"AT": True, "BE": False}


def test_drift_contrasts_find_a_known_step() -> None:
    from eu_okun.bayes_regimes import drift_summary

    data = regime_data()
    quarters = list(data.quarters)
    rng = np.random.default_rng(0)
    step = np.where(data.post(), 0.2, 0.0)
    kappa = step - step.mean() + rng.normal(0, 0.01, (400, len(quarters)))
    fit = LoadedFit("drifting_association", {"countries": list(COUNTRIES)},
                    {"kappa": kappa, "theta": np.full((400, 4), -0.3),
                     "tau_kappa": np.abs(rng.normal(0, 0.05, 400))})
    drift = drift_summary(fit, quarters)
    d1, d2 = drift.contrasts["D1"], drift.contrasts["D2"]
    assert d1[2] == "no credible change" and abs(d1[0].median) < 0.01
    assert d2[2] == "rises" and d2[0].median == pytest.approx(0.2, abs=0.01)
    assert len(drift.path) == len(quarters)
    assert drift.path[-1][2] == pytest.approx(-0.3 + 0.2 - step.mean(), abs=0.01)


def test_regime_recovery_truth_and_assessment() -> None:
    from eu_okun.bayes_regimes import (
        RECOVERY_SHIFT,
        assess_regime_recovery,
        regime_recovery_data,
    )

    design = regime_data()
    data, truth = regime_recovery_data(SPECS["regime_shift"], design, seed=5)
    again, same = regime_recovery_data(SPECS["regime_shift"], design, seed=5)
    np.testing.assert_array_equal(data.y, again.y)
    np.testing.assert_allclose(truth["theta_post"], truth["theta"] + truth["delta"])
    assert float(truth["log_sigma_shift"]) == 0.0 and truth["delta"].shape == (4,)
    rng = np.random.default_rng(1)
    samples = {"delta": (truth["delta"] + rng.normal(0, 0.05, (2, 200, 4))),
               "theta": (truth["theta"] + rng.normal(0, 0.05, (2, 200, 4))),
               "mu_delta": rng.normal(RECOVERY_SHIFT["mu_delta"], 0.05, (2, 200))}
    result = assess_regime_recovery(samples, truth, [True, True, False, True], True, 1.004)
    assert result.delta_total == 3 and result.delta_inside_90 == 3 and result.detected
    assert result.theta_inside_90 == 4


def _fake_samples(data: BayesData, spec_name: str, seed: int, shift: float = 0.3,
                  chains: int = 2, draws: int = 30) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    spec = SPECS[spec_name]
    n, t, k = data.z.shape
    shape = (chains, draws)
    out = {"theta": rng.normal(-0.3, 0.03, (*shape, n)), "alpha": rng.normal(0, 0.05, (*shape, n)),
           "b": rng.normal(0, 0.05, (*shape, n, k)), "sigma": rng.uniform(0.15, 0.3, (*shape, n)),
           "mu_theta": rng.normal(-0.3, 0.05, shape), "tau_theta": rng.uniform(0.05, 0.2, shape),
           "nu": rng.uniform(4, 20, shape), "lam": rng.uniform(0, 0.2, (*shape, n)),
           "f": rng.normal(size=(*shape, t))}
    out["beta0"] = out["theta"] - out["b"].sum(axis=-1)
    if spec.ar1:
        out["phi"] = rng.uniform(-0.3, 0.5, (*shape, n))
    if spec.lagged_outcome:
        out["psi"] = rng.uniform(-0.2, 0.4, (*shape, n))
        out["mu_psi"] = rng.normal(0.1, 0.05, shape)
        out["tau_psi"] = rng.uniform(0.05, 0.3, shape)
    if spec.factor_ar:
        out["rho_f"] = rng.uniform(0.7, 0.95, shape)
    if spec.regime_shift:
        out["delta"] = rng.normal(shift, 0.03, (*shape, n))
        out["mu_delta"] = rng.normal(shift, 0.03, shape)
        out["tau_delta"] = rng.uniform(0.05, 0.1, shape)
        out["theta_post"] = out["theta"] + out["delta"]
    if spec.post_shifts:
        out["alpha_shift"] = rng.normal(0, 0.05, (*shape, n))
        out["log_sigma_shift"] = rng.normal(0, 0.1, shape)
    if spec.drift:
        walk = np.cumsum(rng.normal(0, 0.03, (*shape, t)), axis=-1)
        out["kappa"] = walk - walk.mean(axis=-1, keepdims=True)
        out["tau_kappa"] = rng.uniform(0.01, 0.05, shape)
    return out


def _write_fit(directory: Path, data: BayesData, name: str, seed: int) -> None:
    import json

    from eu_okun.bayes_artifacts import write_posterior_netcdf

    samples = _fake_samples(data, name, seed)
    directory.mkdir(parents=True)
    np.savez_compressed(directory / "draws.npz", **samples)  # type: ignore[arg-type]
    stats_ = {"lag1_autocorrelation": 0.1, "cross_country_correlation": 0.3, "mean_sd": 0.5}
    summary = {
        "spec": {"name": name}, "countries": list(data.countries),
        "quarters": list(data.quarters), "settings": {"warmup": 1000},
        "predictive_check": {"coverage_50": 0.5, "coverage_90": 0.9, "observed": stats_,
                             "replicated_mean": stats_, "p_value": stats_},
        "diagnostics": {"accepted": True, "failures": [], "max_rhat": 1.004,
                        "max_rhat_parameter": "theta[0]", "min_ess_bulk": 800.0,
                        "min_ess_tail": 700.0, "divergences": 0, "min_bfmi": 0.8}}
    (directory / "fit.json").write_text(json.dumps(summary), encoding="utf-8")
    write_posterior_netcdf(directory / "posterior.nc", SPECS[name], data, samples)


@pytest.mark.integration
def test_regimes_report_renders_and_flags_unstable_verdicts(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    import shutil

    import eu_okun.bayes_artifacts as artifacts
    import eu_okun.bayes_regimes as regimes
    from eu_okun.contract import EU27

    panel = synthetic(EU27)
    # Spain: definition flag on every post-2021 quarter, as in the snapshot.
    flags = {("ES", q): {"lfs": "d"} for q in quarter_sequence("2022-Q1", "2026-Q2")}
    monkeypatch.setattr(artifacts, "PRIOR_DRAWS", 20)
    monkeypatch.setattr(regimes, "PRIOR_DRAWS", 20)
    monkeypatch.setattr(regimes, "_read_panel", lambda path: panel)
    monkeypatch.setattr(regimes, "read_flags", lambda path: flags)
    for k, name in enumerate(regimes.COMPARED):
        data = design_for(SPECS[name], panel, flags)  # type: ignore[arg-type]
        _write_fit(tmp_path / "fits" / name, data, name, k)
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    (review_dir / "grid.csv").write_text(
        "sample,geo,gdp_lags,participation,ci_high\n"
        + "".join(f"pre_2020,{g},1,False,-0.1\n" for g in EU27), encoding="utf-8")
    (review_dir / "stability_review.csv").write_text(
        "geo,participation,post_loo_min\n"
        + "".join(f"{g},False,{0.0 if g in ('AT', 'BE') else -0.5}\n" for g in EU27),
        encoding="utf-8")
    review = regimes.write_regimes_report(tmp_path / "fits", tmp_path, tmp_path / "out",
                                          review_dir)
    es = next(r for r in review.shifts["regime_shift"] if r.geo == "ES")
    assert es.post_terms == 0 and es.verdict == "not evaluable"
    assert review.averages["regime_shift"].verdict == "rises"
    assert [(c.model_a, c.model_b) for c in review.comparisons] == list(regimes.COMPARISONS)
    report = (tmp_path / "out" / "BAYES_REGIMES.md").read_text(encoding="utf-8")
    for heading in ("## 1. Convergence", "## 2. Prior and posterior predictive checks",
                    "## 3. The average shift", "## 4. Country shifts",
                    "## 5. Provisional GDP and dynamics", "## 6. A gradual or an abrupt change?",
                    "## 7. PSIS-LOO", "## 8. Recovery of a known shift", "## 9. What this changes"):
        assert heading in report
    assert "The classical rule marks 2 countries (AT and BE)" in report
    for name in ("regimes.json", "regime_countries.png", "regime_deltas.png", "drift.png"):
        assert (tmp_path / "out" / name).exists()
    replication = tmp_path / "replication"
    replication.mkdir()
    shutil.copy(tmp_path / "out" / "regimes.json", replication / "regimes.json")
    (replication / "README.md").write_text("# Linux replication, run 2\n", encoding="utf-8")
    same = regimes.write_regimes_report(tmp_path / "fits", tmp_path, tmp_path / "out2",
                                        review_dir, figures=False, replication_dir=replication)
    assert regimes.unstable_verdicts(same) == []
    text = (tmp_path / "out2" / "BAYES_REGIMES.md").read_text(encoding="utf-8")
    assert "## 10. Replication on a second platform" in text
    assert "Replication (Linux replication, run 2): every verdict is the same." in text
    summary = json.loads((replication / "regimes.json").read_text(encoding="utf-8"))
    summary["specifications"]["regime_shift"]["average_verdict"] = "no credible change"
    summary["specifications"]["regime_shift"]["countries"]["AT"]["verdict"] = "falls"
    (replication / "regimes.json").write_text(json.dumps(summary), encoding="utf-8")
    flipped = regimes.write_regimes_report(tmp_path / "fits", tmp_path, tmp_path / "out3",
                                           review_dir, figures=False,
                                           replication_dir=replication)
    unstable = regimes.unstable_verdicts(flipped)
    assert len(unstable) == 2 and unstable[0].startswith("the average shift in `regime_shift`")
    assert unstable[1].startswith("AT in `regime_shift` (rises here")
