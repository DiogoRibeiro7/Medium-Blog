"""Tests of the small-sample inference study (``SMALL_SAMPLE_PROTOCOL.md``)."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from eu_okun.baseline import fit_hac_covariance
from eu_okun.contract import quarter_sequence
from eu_okun.small_sample import (
    Z_975,
    binomial_range,
    block_errors,
    bootstrap_p,
    change_contrast,
    fit_sieve,
    hac_design,
    hac_t,
    null_sieve_test,
    restricted_fit,
    segment_starts,
    sieve_errors,
)

PRE = quarter_sequence("2009-Q2", "2019-Q4")
POST = quarter_sequence("2022-Q2", "2026-Q2")


def interacted_design(seed: int = 0, width: int = 3) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    def block(n: int) -> np.ndarray:
        columns = [np.ones(n), rng.normal(0.4, 0.8, n), rng.normal(0.4, 0.8, n)]
        columns += [rng.normal(0, 0.3, n) for _ in range(width - 3)]
        return np.column_stack(columns)
    x_pre, x_post = block(len(PRE)), block(len(POST))
    x = np.vstack((np.hstack((x_pre, np.zeros_like(x_pre))), np.hstack((x_post, x_post))))
    return x, PRE + POST


@pytest.mark.parametrize("bandwidth", [0, 2, 4, 8])
@pytest.mark.parametrize("width", [3, 4])
def test_batched_hac_t_matches_the_published_covariance(bandwidth: int, width: int) -> None:
    x, periods = interacted_design(width=width)
    rng = np.random.default_rng(1)
    ys = rng.normal(size=(x.shape[0], 3)) + x @ rng.normal(size=x.shape[1])[:, None]
    c = change_contrast(width)
    design = hac_design(x, periods, c, bandwidth)
    estimate, se, t = hac_t(design, ys)
    for j in range(ys.shape[1]):
        beta, covariance, _, _ = fit_hac_covariance(x, ys[:, j], bandwidth, periods)
        assert estimate[j] == pytest.approx(float(c @ beta), rel=1e-10)
        assert se[j] == pytest.approx(math.sqrt(float(c @ covariance @ c)), rel=1e-9)
        assert t[j] == pytest.approx(estimate[j] / se[j], rel=1e-12)


def test_calendar_gaps_are_respected_by_the_hac_pairs() -> None:
    x, periods = interacted_design()
    design = hac_design(x, periods, change_contrast(3), 1)
    (weight, current, previous), = design.pairs
    assert weight == 0.5
    # 2019-Q4 -> 2022-Q2 is not a one-quarter step: no pair crosses the gap.
    assert len(current) == len(periods) - 2
    assert all(periods[c] > periods[p] for c, p in zip(current, previous, strict=True))


def test_restricted_fit_imposes_the_null() -> None:
    x, periods = interacted_design()
    c = change_contrast(3)
    design = hac_design(x, periods, c, 4)
    y = np.random.default_rng(2).normal(size=x.shape[0])
    beta_r, resid = restricted_fit(design, y)
    assert float(c @ beta_r) == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(resid, y - x @ beta_r)
    # The intercepts stay free, so the residuals sum to zero in each period.
    assert abs(resid[:len(PRE)].sum()) < 1e-10 and abs(resid[len(PRE):].sum()) < 1e-10


def test_sieve_follows_its_recursion_and_restarts_at_each_period() -> None:
    periods = PRE + POST
    starts = segment_starts(periods)
    assert list(np.flatnonzero(starts)) == [0, len(PRE)]
    rng = np.random.default_rng(3)
    resid = rng.normal(size=len(periods))
    sieve = fit_sieve(resid, starts)
    assert len(sieve.period_rho) == 2 and all(abs(r) <= 0.95 for r in sieve.period_rho)
    lagged, current = resid[:len(PRE) - 1], resid[1:len(PRE)]
    assert sieve.period_rho[0] == pytest.approx(float(lagged @ current / (lagged @ lagged)))
    errors = sieve_errors(sieve, resid, 5, np.random.default_rng(4))
    v = np.random.default_rng(4).choice(np.asarray([-1.0, 1.0]), size=(len(periods), 5))
    np.testing.assert_allclose(errors[0], v[0] * resid[0])
    np.testing.assert_allclose(errors[1], sieve.rho[1] * errors[0]
                               + v[1] * sieve.innovations[1])
    np.testing.assert_allclose(errors[len(PRE)], v[len(PRE)] * resid[len(PRE)])


def test_block_errors_stay_within_each_period() -> None:
    periods = PRE + POST
    starts = segment_starts(periods)
    resid = np.concatenate([np.arange(len(PRE)), 1000 + np.arange(len(POST))]).astype(float)
    errors = block_errors(resid, starts, 50, np.random.default_rng(5))
    assert np.all(errors[:len(PRE)] < 1000) and np.all(errors[len(PRE):] >= 1000)
    # Blocks of four consecutive residuals.
    assert np.all(np.diff(errors[:4], axis=0) == 1.0)


def test_bootstrap_p_value_counts_ties_and_adds_one() -> None:
    simulated = np.asarray([-3.0, -1.0, 0.5, 2.0, 2.0])
    assert bootstrap_p(2.0, simulated) == pytest.approx((1 + 3) / 6)
    assert bootstrap_p(10.0, simulated) == pytest.approx(1 / 6)


def test_binomial_range_matches_exact_quantiles() -> None:
    stats = pytest.importorskip("scipy.stats")
    for total in (200, 10_800):
        low, high = binomial_range(total)
        assert low == pytest.approx(stats.binom.ppf(0.025, total, 0.05) / total)
        assert high == pytest.approx(stats.binom.ppf(0.975, total, 0.05) / total)


def test_sieve_bootstrap_test_has_roughly_nominal_size_on_null_data() -> None:
    """Under a true null with AR(1) errors, rejections at 5% stay near 5%."""
    x, periods = interacted_design(seed=6)
    c = change_contrast(3)
    design = hac_design(x, periods, c, 4)
    starts = segment_starts(periods)
    rng = np.random.default_rng(7)
    beta = rng.normal(size=6)
    beta[4] = -beta[5]                  # c'beta = 0: no change in the GDP sum
    rejections_boot = rejections_normal = 0
    replications = 120
    for _ in range(replications):
        e = np.empty(len(periods))
        for t in range(len(periods)):
            shock = rng.normal(0, 0.3)
            e[t] = shock if starts[t] else 0.5 * e[t - 1] + shock
        y = x @ beta + e
        t_obs, t_star, _ = null_sieve_test(design, y, starts, 199, rng)
        rejections_boot += bootstrap_p(t_obs, t_star) <= 0.05
        rejections_normal += abs(t_obs) > Z_975
    assert rejections_boot / replications < 0.12
    assert rejections_normal >= rejections_boot


@pytest.mark.reproduction
def test_committed_small_sample_tables_reproduce(tmp_path: Path) -> None:
    """The committed tables regenerate from the frozen snapshot (validation gate)."""
    from eu_okun.small_sample import write_report

    root = Path(__file__).resolve().parents[1]
    committed = root / "docs" / "small-sample"
    if not (committed / "change_tests.csv").exists():
        pytest.skip("No committed small-sample results.")
    write_report(root / "snapshots" / "eurostat-2026-10-15", tmp_path, review_dir=root / "docs"
                 / "review", figures=False)

    def read(path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    for name in ("change_tests.csv", "size_study.csv", "post_sums.csv"):
        old, new = read(committed / name), read(tmp_path / name)
        assert len(old) == len(new), name
        for a, b in zip(old, new, strict=True):
            for key, value in a.items():
                try:
                    number = float(value)
                except ValueError:
                    assert value == b[key], (name, key)
                    continue
                # A bootstrap p-value may move by one draw if a floating-point tie flips
                # on another platform; every other number must agree to rounding error.
                tolerance = 0.0011 if key.startswith("p_") else 1e-9
                assert number == pytest.approx(float(b[key]), rel=1e-7, abs=tolerance), (
                    name, a["geo"], key)
