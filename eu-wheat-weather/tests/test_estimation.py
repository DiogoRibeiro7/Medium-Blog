"""Tests of the estimation machinery on simulated data, run before any real estimate."""

from __future__ import annotations

import numpy as np
import pytest

from eu_wheat_weather.estimation import (
    cluster_vcov,
    dummies,
    fit,
    holm,
    minimum_detectable,
    singleton_mask,
    t_test,
    twoway_vcov,
    twoway_wild_bootstrap,
    wald_test,
    wild_bootstrap,
)


def panel(seed: int, beta: tuple[float, float] = (0.0, 0.0), countries: int = 12,
          regions: int = 4, years: int = 15) -> dict[str, np.ndarray]:
    """Regions within countries; country-year shocks and region effects; clustered noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(countries):
        country_shock = rng.normal(0, 1, years)
        for r in range(regions):
            region_effect = rng.normal()
            for t in range(years):
                x1, x2 = rng.normal() + country_shock[t], rng.normal()
                rows.append((f"C{c}", f"C{c}R{r}", str(2000 + t), x1, x2,
                             region_effect + country_shock[t]))
    country, region, year = (np.array([r[i] for r in rows]) for i in range(3))
    x = np.array([[r[3], r[4]] for r in rows])
    # Errors correlated within country (the bootstrap's cluster dimension).
    shock = {c: rng.normal(0, 0.5) for c in set(country.tolist())}
    noise = np.array([shock[c] for c in country]) * x[:, 0] * 0.3 + rng.normal(0, 1, len(rows))
    y = np.array([r[5] for r in rows]) + x @ np.array(beta) + noise
    return {"country": country, "region": region, "year": year, "x": x, "y": y}


def controls(d: dict[str, np.ndarray]) -> np.ndarray:
    cy = [f"{c}|{t}" for c, t in zip(d["country"], d["year"], strict=True)]
    return np.column_stack([dummies(list(d["region"])), dummies(cy)])


def m1(d: dict[str, np.ndarray]):  # type: ignore[no-untyped-def]
    return fit(["x1", "x2"], d["x"], d["y"], controls(d), list(d["country"]), list(d["year"]))


def test_projection_route_equals_explicit_dummies() -> None:
    d = panel(1, beta=(0.5, -0.2))
    f = m1(d)
    coef, *_ = np.linalg.lstsq(np.column_stack([d["x"], controls(d)]), d["y"], rcond=None)
    assert np.allclose(f.beta, coef[:2], atol=1e-10)


def test_weighted_fit_equals_scaled_least_squares() -> None:
    d = panel(2, beta=(0.5, -0.2))
    w = np.random.default_rng(3).uniform(0.5, 2.0, d["y"].size)
    f = fit(["x1", "x2"], d["x"], d["y"], controls(d), list(d["country"]), list(d["year"]), w)
    root = np.sqrt(w)
    design = np.column_stack([d["x"], controls(d)]) * root[:, None]
    coef, *_ = np.linalg.lstsq(design, d["y"] * root, rcond=None)
    assert np.allclose(f.beta, coef[:2], atol=1e-10)


def test_singletons_are_dropped_until_none_remain() -> None:
    # "c" is alone in group B; dropping it leaves "b" alone in group A's "y".
    a = ["x", "x", "y", "y"]
    b = ["p", "p", "q", "r"]
    assert singleton_mask([a, b]).tolist() == [True, True, False, False]


def test_cluster_vcov_matches_the_sandwich_formula() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(40, 2))
    e = rng.normal(size=40)
    codes = np.array([f"g{i % 5}" for i in range(40)])
    v, g = cluster_vcov(x, e, codes, 2)
    bread = np.linalg.inv(x.T @ x)
    meat = sum(np.outer(x[codes == c].T @ e[codes == c], x[codes == c].T @ e[codes == c])
               for c in sorted(set(codes)))
    assert g == 5
    assert np.allclose(v, 5 / 4 * 39 / 38 * bread @ meat @ bread)


def test_twoway_vcov_combines_three_clusterings() -> None:
    f = m1(panel(5, beta=(0.3, 0.0)))
    v, df = twoway_vcov(f)
    vc, gc = cluster_vcov(f.x, f.resid, f.country, f.k)
    vt, gt = cluster_vcov(f.x, f.resid, f.year, f.k)
    cell = np.char.add(np.char.add(f.country, "|"), f.year)
    vct, _ = cluster_vcov(f.x, f.resid, cell, f.k)
    raw = vc + vt - vct
    assert df == min(gc, gt) - 1 == 11
    if np.all(np.linalg.eigvalsh(raw) >= 0):
        assert np.allclose(v, raw)
    assert np.all(np.linalg.eigvalsh(v) >= -1e-12)


def test_bootstrap_statistic_is_the_country_clustered_t_and_wald() -> None:
    f = m1(panel(6, beta=(0.3, 0.1)))
    vc, gc = cluster_vcov(f.x, f.resid, f.country, f.k)
    t_cr1, _, _ = t_test(f.beta, vc, 0, gc - 1)
    w_cr1, _ = wald_test(f.beta, vc, [0, 1], gc - 1)
    t_boot, _ = wild_bootstrap(f, [0], draws=99)
    w_boot, _ = wild_bootstrap(f, [0, 1], draws=99)
    assert t_boot == pytest.approx(t_cr1, rel=1e-10)
    assert w_boot == pytest.approx(w_cr1, rel=1e-10)


def test_bootstrap_rejects_a_real_effect_and_holds_its_size_under_the_null() -> None:
    strong = m1(panel(7, beta=(0.5, 0.0)))
    assert wild_bootstrap(strong, [0], draws=499)[1] < 0.01
    rejections = []
    for seed in range(120):
        f = m1(panel(100 + seed, beta=(0.0, 0.0), countries=10, regions=3, years=10))
        rejections.append(wild_bootstrap(f, [0], draws=199, seed=seed)[1] < 0.05)
    assert 0.0 <= float(np.mean(rejections)) <= 0.11


def test_twoway_bootstrap_statistic_is_the_two_way_t_and_wald() -> None:
    """Amendment A1, section 8.3: the bootstrapped statistic is the registered two-way one."""
    f = m1(panel(8, beta=(0.3, 0.1)))
    v, df = twoway_vcov(f)
    t_two, _, _ = t_test(f.beta, v, 0, df)
    w_two, _ = wald_test(f.beta, v, [0, 1], df)
    for by in ("country", "year"):
        t_boot, p = twoway_wild_bootstrap(f, [0], by, draws=99)
        w_boot, _ = twoway_wild_bootstrap(f, [0, 1], by, draws=99)
        assert t_boot == pytest.approx(t_two, rel=1e-10)
        assert w_boot == pytest.approx(w_two, rel=1e-10)
        assert 0.0 <= p <= 1.0
    # the same seed gives the same draws
    assert twoway_wild_bootstrap(f, [1], "year", draws=99, seed=3) == \
        twoway_wild_bootstrap(f, [1], "year", draws=99, seed=3)


def test_twoway_bootstrap_rejects_a_real_effect_and_holds_its_size_under_the_null() -> None:
    strong = m1(panel(9, beta=(0.5, 0.0)))
    assert twoway_wild_bootstrap(strong, [0], "country", draws=499)[1] < 0.01
    for by in ("country", "year"):
        rejections = []
        for seed in range(100):
            f = m1(panel(300 + seed, beta=(0.0, 0.0), countries=10, regions=3, years=10))
            rejections.append(twoway_wild_bootstrap(f, [0], by, draws=199, seed=seed)[1] < 0.05)
        assert 0.0 <= float(np.mean(rejections)) <= 0.12, by


def test_a_draw_that_reproduces_the_statistic_is_a_tie_and_counts() -> None:
    from eu_wheat_weather.estimation import _at_least_as_extreme

    original = 2.0
    noisy = original * (1 - 4e-16)        # the identity draw, off by rounding
    draws = np.array([noisy, -noisy, 1.9, -2.5])
    assert _at_least_as_extreme(draws, original, two_sided=True) == 3
    assert _at_least_as_extreme(np.array([noisy, 1.0, 3.0]), original, two_sided=False) == 2
    assert _at_least_as_extreme(np.array([original * (1 - 1e-6)]), original, True) == 0


def test_holm_and_minimum_detectable() -> None:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.001]) == pytest.approx([0.5, 0.002])
    assert minimum_detectable(1.0, 10_000) == pytest.approx(1.96 + 0.8416, abs=1e-3)
