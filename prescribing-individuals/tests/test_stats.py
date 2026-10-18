import numpy as np
import pytest

from prescribing import stats


def test_survey_mean_reduces_to_srs():
    rng = np.random.default_rng(0)
    y = rng.normal(10, 3, 500)
    est = stats.survey_mean(y, np.ones(500), np.zeros(500), np.arange(500))
    assert est.est == pytest.approx(y.mean())
    assert est.se == pytest.approx(y.std(ddof=1) / np.sqrt(500), rel=1e-9)


def test_survey_domain_keeps_design():
    y = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    w = np.array([1.0, 2.0, 1.0, 1.0, 2.0, 1.0])
    strata = np.array([0, 0, 0, 1, 1, 1])
    psu = np.array([0, 1, 2, 0, 1, 2])
    dom = np.array([True, True, False, True, True, False])
    e = stats.survey_mean(y, w, strata, psu, dom)
    assert e.est == pytest.approx((1 + 0 + 1 + 0) / (1 + 2 + 1 + 2))
    assert e.n == 4
    assert e.se > 0


def test_ln_vr_zero_when_equal():
    y, v = stats.ln_vr(5.0, 50, 5.0, 50)
    assert y == pytest.approx(0.0)
    assert v == pytest.approx(2 / (2 * 49))


def test_dersimonian_laird_homogeneous():
    p = stats.dersimonian_laird([0.2, 0.2, 0.2], [0.01, 0.02, 0.04])
    assert p.est == pytest.approx(0.2)
    assert p.tau2 == 0.0
    assert p.i2 == 0.0


def test_dersimonian_laird_heterogeneous_widens():
    fixed = stats.dersimonian_laird([0.0, 1.0], [0.01, 0.01])
    assert fixed.tau2 > 0
    assert fixed.se > np.sqrt(1 / 200)


def test_ror():
    r = stats.reporting_odds_ratio(20, 80, 100, 1900)
    assert r.ror == pytest.approx((20 / 80) / (100 / 1900))
    assert r.lo < r.ror < r.hi


def test_within_person_sd():
    r = np.array([[100, 102, 98], [120, 118, np.nan], [140, np.nan, np.nan]])
    sd, n = stats.within_person_sd(r)
    # person 1: deviations 0, 2, -2 -> ss 8, df 2; person 2: 1, -1 -> ss 2, df 1
    assert n == 2
    assert sd == pytest.approx(np.sqrt(10 / 3))


def test_readings_needed_monotone():
    assert stats.readings_needed(10, 5) < stats.readings_needed(5, 5)


def test_log_risk_ratio_zero_cell():
    y, v = stats.log_risk_ratio(0, 50, 5, 50)
    assert np.isfinite(y) and np.isfinite(v)
