"""Tests of the diagnostics of the reported Bayesian results
(``BAYES_DIAGNOSTICS_PROTOCOL.md``).

Two small fits sample for real (a few seconds each); the full refits run outside the
test suite. The report is regenerated from its committed summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

numpyro = pytest.importorskip("numpyro")
numpyro.enable_x64()

from eu_okun.bayes_artifacts import (  # noqa: E402
    pointwise_log_likelihood_draws,
    write_posterior_netcdf,
)
from eu_okun.bayes_checks import COMPARISONS as CHECKS_COMPARISONS  # noqa: E402
from eu_okun.bayes_data import BayesData, build_bayes_data  # noqa: E402
from eu_okun.bayes_diagnostics import (  # noqa: E402
    COMPARISON_SETS,
    DECOMPOSITION,
    FITS,
    LADDER,
    Pointwise,
    _corrected,
    cell_label,
    choose_scale,
    decomposition,
    exact_elpd,
    leave_out,
    psis_pointwise,
    render,
)
from eu_okun.bayes_dynamics import COMPARISONS as DYNAMICS_COMPARISONS  # noqa: E402
from eu_okun.bayes_fit import SamplerSettings, fit_specification  # noqa: E402
from eu_okun.bayes_model import SPECS  # noqa: E402
from eu_okun.bayes_regimes import COMPARISONS as REGIMES_COMPARISONS  # noqa: E402
from eu_okun.contract import quarter_sequence  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "docs" / "bayes-diagnostics"
REPORT = ROOT / "docs" / "BAYES_DIAGNOSTICS.md"
COUNTRIES = ("ES", "LU", "NL", "PT")


def small_data(countries: tuple[str, ...] = COUNTRIES, seed: int = 3) -> BayesData:
    rng = np.random.default_rng(seed)
    panel: dict[tuple[str, str], dict[str, float | None]] = {}
    for geo in countries:
        u, p, g_prev = 8.0, 70.0, 0.4
        for quarter in quarter_sequence("2008-Q1", "2026-Q2"):
            g = float(rng.normal(0.4, 0.8))
            u += -0.3 * g - 0.1 * g_prev + float(rng.normal(0, 0.2))
            p += float(rng.normal(0, 0.2))
            panel[(geo, quarter)] = {"gdp": g, "unemployment": u, "participation": p}
            g_prev = g
    return build_bayes_data(panel, {}, lags=2, countries=countries)


def test_the_scope_is_the_registered_comparisons() -> None:
    assert COMPARISON_SETS == {"9.2": CHECKS_COMPARISONS, "11.4": DYNAMICS_COMPARISONS,
                               "12.5": REGIMES_COMPARISONS}
    assert sum(len(v) for v in COMPARISON_SETS.values()) == 12
    assert len(FITS) == 13 and len(set(FITS)) == 13 and "participation" not in FITS


def test_leaving_out_a_cell_keeps_its_value_as_conditioning_data() -> None:
    data = small_data()
    label = cell_label(data, 1, 10)
    left = leave_out(data, [label])
    assert not left.mask[1, 10] and left.mask.sum() == data.mask.sum() - 1
    np.testing.assert_array_equal(left.y, data.y)
    spec = SPECS["primary"]
    rng = np.random.default_rng(0)
    n, t, k = data.z.shape
    samples = {"theta": rng.normal(-0.3, 0.1, (1, 5, n)), "alpha": np.zeros((1, 5, n)),
               "b": np.zeros((1, 5, n, k)), "sigma": np.full((1, 5, n), 0.2),
               "phi": np.full((1, 5, n), 0.5), "nu": np.full((1, 5), 8.0),
               "lam": np.zeros((1, 5, n)), "f": np.zeros((1, 5, t))}
    # The AR(1) term of the next quarter still uses the left-out value.
    np.testing.assert_allclose(pointwise_log_likelihood_draws(spec, left, samples),
                               pointwise_log_likelihood_draws(spec, data, samples))
    with pytest.raises(ValueError, match="not in the likelihood"):
        leave_out(left, [label])


def test_exact_value_agrees_with_psis_for_a_cell_with_low_pareto_k(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import eu_okun.bayes_artifacts as artifacts

    monkeypatch.setattr(artifacts, "PRIOR_DRAWS", 20)
    data = small_data(seed=5)
    spec = SPECS["no_common_factor"]
    settings = SamplerSettings(chains=2, warmup=300, draws=600, seed=1)
    fit = fit_specification(data, spec, settings)
    path = tmp_path / "posterior.nc"
    write_posterior_netcdf(path, spec, data, fit.samples)
    point = psis_pointwise(path, spec.name)
    assert point.elpd.size == int(data.mask.sum())
    label = point.labels[int(np.argmin(point.pareto_k))]
    assert float(point.pareto_k.min()) < 0.5
    refit = fit_specification(leave_out(data, [label]), spec, settings)
    exact = exact_elpd(refit, data, label)
    assert exact == pytest.approx(float(point.elpd[point.labels.index(label)]), abs=0.1)


def test_corrected_comparison_replaces_only_the_refitted_cells() -> None:
    labels = ("AT 2010-Q1", "AT 2010-Q2", "AT 2010-Q3")
    local = {"pointwise": {
        "a": Pointwise("a", labels, np.array([-1.0, -1.0, -1.0]), np.array([0.1, 0.9, 0.1])),
        "b": Pointwise("b", labels, np.array([-2.0, -2.0, -2.0]), np.array([0.1, 0.1, 0.1]))}}
    exact = {("a", "AT 2010-Q2"): {"elpd": -4.0, "accepted": True}}
    corrected, replaced = _corrected("a", "b", list(labels), local, exact)
    assert replaced == 1 and corrected is not None
    # PSIS gives A - B = 3; the exact value of A's refitted cell brings it to 0.
    assert corrected["elpd_diff"] == pytest.approx(0.0)
    untouched, none = _corrected("a", "b", list(labels), local, {})
    assert none == 0 and untouched is not None
    assert untouched["elpd_diff"] == pytest.approx(3.0)
    rejected = {("a", "AT 2010-Q2"): {"elpd": -4.0, "accepted": False}}
    assert _corrected("a", "b", list(labels), local, rejected) == (None, 1)


def test_the_calibrated_scale_is_the_widest_passing_step() -> None:
    steps = [{"scale": s, "share_above_5pp": v} for s, v in
             zip(LADDER, (0.04, 0.015, 0.006, 0.003, 0.002), strict=True)]
    assert choose_scale(steps) == (0.5, True)
    failing = [{**s, "share_above_5pp": 0.05} for s in steps]
    assert choose_scale(failing) == (LADDER[-1], False)


def test_decomposition_neutralises_one_component_at_a_time() -> None:
    data = small_data()
    out = decomposition(SPECS["primary"], data, draws=200, seed=4)
    assert list(out) == list(DECOMPOSITION)
    assert all(0.0 <= v["share_above_5pp"] <= 1.0 for v in out.values())
    # Without Ireland in the design, leaving Ireland out changes nothing.
    assert out[DECOMPOSITION[4]] == out[DECOMPOSITION[0]]
    # With the GDP terms at zero, the largest simulated change cannot come from GDP.
    assert out[DECOMPOSITION[1]]["median_largest_change"] <= out[DECOMPOSITION[0]][
        "median_largest_change"]


def test_report_regenerates_from_the_committed_summary() -> None:
    if not (COMMITTED / "summary.json").exists():
        pytest.skip("No committed results.")
    study = json.loads((COMMITTED / "summary.json").read_text(encoding="utf-8"))
    assert render(study, study["command"]) == REPORT.read_text(encoding="utf-8")
