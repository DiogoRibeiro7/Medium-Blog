import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import did  # noqa: E402


def panel(effect, trends=False, noise=0.0, seed=0):
    """Staggered adoption ending in a common year, like the suffrage laws.

    effect(e) gives the true effect at event time e >= 0."""
    rng = np.random.default_rng(seed)
    cohorts = {1905: 3, 1908: 4, 1911: 4, 1914: 3, 1920: 8}
    rows = []
    k = 0
    for g, n in cohorts.items():
        for _ in range(n):
            a, slope = rng.normal(0, 1), (rng.normal(0, 0.02) if trends else 0.0)
            for t in range(1900, 1931):
                e = t - g
                y = a + 0.03 * (t - 1900) ** 1.2 + slope * (t - 1918) + (effect(e) if e >= 0 else 0.0)
                rows.append(dict(state=f"s{k}", year=t, g=g, y=y + rng.normal(0, noise)))
            k += 1
    return pd.DataFrame(rows)


def true_att(df, effect, last=1919):
    d = df[(df.year >= df.g) & (df.year <= last)]
    return np.mean([effect(e) for e in d.year - d.g])


def test_constant_effect_is_recovered_by_every_estimator():
    df = panel(lambda e: -0.1)
    assert did.overall(did.att_gt(df, "state", "y")) == pytest.approx(-0.1, abs=1e-10)
    assert did.impute(df, "state", "y")[0] == pytest.approx(-0.1, abs=1e-10)
    assert did.twfe(df, "y")[0] == pytest.approx(-0.1, abs=1e-8)
    assert did.twfe(df, "y", window=1919)[0] == pytest.approx(-0.1, abs=1e-8)


def test_unit_trends_need_the_trend_versions():
    df = panel(lambda e: -0.1, trends=True)
    assert did.twfe(df, "y", trend="state")[0] == pytest.approx(-0.1, abs=1e-7)
    assert did.impute(df, "state", "y", trend=True)[0] == pytest.approx(-0.1, abs=1e-8)
    assert abs(did.overall(did.att_gt(df, "state", "y")) + 0.1) > 1e-3


def test_dynamic_effects_bias_twfe_but_not_the_clean_estimators():
    eff = lambda e: -0.02 * (e + 1)  # noqa: E731
    df = panel(eff)
    target = true_att(df, eff)
    a = did.att_gt(df, "state", "y")
    cells = a[a.e >= 0]
    weighted = np.average([eff(e) for e in cells.e], weights=cells.n_treat)
    assert did.overall(a) == pytest.approx(weighted, abs=1e-10)
    assert did.impute(df, "state", "y")[0] == pytest.approx(target, abs=1e-10)
    assert abs(did.twfe(df, "y")[0] - target) > 0.01


def test_event_study_is_zero_before_adoption_without_anticipation():
    df = panel(lambda e: -0.1)
    ev = did.event(did.att_gt(df, "state", "y"), -5, 5)
    assert np.allclose(ev.loc[ev.index < 0, "att"], 0, atol=1e-10)
    assert np.allclose(ev.loc[ev.index >= 0, "att"], -0.1, atol=1e-10)


def test_anticipation_moves_the_base_year():
    eff = lambda e: -0.1  # noqa: E731
    df = panel(eff)
    df.loc[df.year == df.g - 1, "y"] -= 0.05  # effect starts a year early
    assert did.overall(did.att_gt(df, "state", "y")) != pytest.approx(-0.1, abs=1e-3)
    a = did.att_gt(df, "state", "y", anticipation=1)
    assert did.overall(a) == pytest.approx(-0.1, abs=1e-10)


def test_twfe_weights_sum_to_one_and_fully_treated_years_net_to_zero():
    df = panel(lambda e: -0.1)
    for trend in (None, "state"):
        w = did.twfe_weights(df, "y", trend=trend)
        assert w.w.sum() == pytest.approx(1.0, abs=1e-8)
        for t in range(1920, 1931):
            assert w.loc[w.year == t, "w"].sum() == pytest.approx(0.0, abs=1e-8)


def test_twfe_equals_weighted_sum_of_cell_effects():
    eff = lambda e: -0.02 * (e + 1)  # noqa: E731
    df = panel(eff)
    w = did.twfe_weights(df, "y")
    implied = (w.w * w.e.map(eff)).sum()
    assert did.twfe(df, "y")[0] == pytest.approx(implied, abs=1e-8)


def test_bootstrap_copies_are_separate_units():
    df = panel(lambda e: -0.1)
    rng = np.random.default_rng(1)
    r = did.resample(df, "state", rng)
    assert r.state.nunique() == df.state.nunique()
    assert len(r) == len(df)
    draws = did.bootstrap(df, lambda d: did.overall(did.att_gt(d, "state", "y")), reps=5)
    assert np.allclose(draws, -0.1, atol=1e-10)


def test_cluster_se_shrinks_with_less_noise():
    eff = lambda e: -0.1  # noqa: E731
    se_big = did.twfe(panel(eff, noise=0.2, seed=3), "y")[1]
    se_small = did.twfe(panel(eff, noise=0.02, seed=3), "y")[1]
    assert se_small < se_big / 5
