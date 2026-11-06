import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import evidence  # noqa: E402
import simulate as sim  # noqa: E402


@pytest.fixture(scope="module")
def u():
    return sim.load()


def test_country_demand_calibration():
    d = sim.CountryDemand(0.08, sim.gini_to_sigma(0.30))
    assert float(d.priced_out(1.0)) == pytest.approx(0.08, abs=1e-12)
    assert float(d.d(np.array([1.0]))[0]) == pytest.approx(1.0, abs=1e-12)
    for q in (0.8, 0.95, 1.1):
        assert float(d.d(d.inverse(q))) == pytest.approx(q, rel=1e-3)


def test_union_normalisation(u):
    assert float(np.sum(u.w)) == pytest.approx(1.0)
    assert float(np.sum(u.w * u.s)) == pytest.approx(1.0)
    assert float(u.eu_demand(np.array([1.0]))[0]) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("target", [0.0, 0.1, 0.5, 0.9])
def test_correlation_family(u, target):
    R = u.corr_at(target)
    assert np.linalg.eigvalsh(R).min() > -1e-9
    assert u.mean_corr(R) == pytest.approx(target, abs=1e-9)


def test_pooling_reduces_bad_year_risk_at_empirical_correlation(u):
    R0 = 0.9 * u.R_emp + 0.1 * np.eye(len(u.geo))
    x, f = u.draw(R0, 5000)
    _, po_p, _, po_u = u.outcomes(x, f, 0.0)
    a, b = u.summarise(po_p), u.summarise(po_u)
    assert a["national_cvar95_pp"] < b["national_cvar95_pp"]
    assert a["mean_excess_pp"] < b["mean_excess_pp"]


def test_spreading_a_common_loss_does_not_change_the_pooled_outcome(u):
    R0 = 0.9 * u.R_emp + 0.1 * np.eye(len(u.geo))
    x, f = u.draw(R0, 3000)
    loss = np.linspace(0.0, 0.2, len(u.geo))
    even = np.full_like(loss, float(np.sum(u.w * u.s * loss)))
    p1, _, _, _ = u.outcomes(x, f, 0.0, capacity=1 - loss)
    p2, _, _, _ = u.outcomes(x, f, 0.0, capacity=1 - even)
    # same expected EU supply loss; pooled prices close (only the shock weights differ)
    assert np.median(p1) == pytest.approx(np.median(p2), rel=0.01)


def test_evidence_headline_numbers():
    prod = evidence.cereal_production()
    s, s_eu = evidence.shocks(prod)
    w = evidence.production_weights(prod)
    assert s.shape == (25, 26)
    assert evidence.pairwise(s.corr().values).mean() == pytest.approx(0.231, abs=0.001)
    assert evidence.diversification(s, w)["n_eff"] == pytest.approx(3.17, abs=0.01)
    assert s_eu.std() == pytest.approx(0.0594, abs=0.0005)
