"""The exact dual interval (METHODS.md, sections 1 and 4) and the HiGHS check."""

from __future__ import annotations

import numpy as np
import pytest

from iberian_lp.programme import Steps, exact, solve


def steps(sell: list[tuple[int, int]], buy: list[tuple[int, int]], r: int = 0) -> Steps:
    """Steps from (EUR/MWh, MW) pairs, in whole euros and MW for readability."""
    def arrays(pairs: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
        return (np.array([100 * p for p, _ in pairs], dtype=np.int64),
                np.array([10 * q for _, q in pairs], dtype=np.int64))
    return Steps(*arrays(sell), *arrays(buy), 10 * r)


def test_a_partly_accepted_step_sets_the_price() -> None:
    e, _ = solve(steps([(10, 100), (20, 100)], [(30, 150)]))
    assert (e.lo, e.hi) == (2000, 2000)
    assert e.welfare == 1000 * (30 * 150 - 10 * 100 - 20 * 50)
    assert (e.buy_min, e.buy_max) == (1500, 1500)


def test_no_partly_accepted_step_leaves_an_interval() -> None:
    e, _ = solve(steps([(10, 100), (40, 100)], [(30, 100)]))
    assert (e.lo, e.hi) == (1000, 3000)
    assert (e.buy_min, e.buy_max) == (1000, 1000)


def test_steps_at_the_same_price_leave_the_volume_open() -> None:
    e, _ = solve(steps([(20, 100)], [(20, 60), (50, 30)]))
    assert (e.lo, e.hi) == (2000, 2000)
    assert (e.buy_min, e.buy_max) == (300, 900)


def test_fixed_volume_shifts_the_balance() -> None:
    # Exports of 50 MW are fixed demand: the 150 MW of buy steps and the 50 MW must be met.
    e, _ = solve(steps([(10, 100), (20, 150)], [(30, 150)], r=50))
    assert (e.lo, e.hi) == (2000, 2000)
    # Fixed supply covering the whole demand leaves every price below the cheapest sell step.
    e, _ = solve(steps([(10, 100)], [(30, 50)], r=-50))
    assert (e.lo, e.hi) == (-np.inf, 1000)


def test_an_infeasible_programme_stops() -> None:
    with pytest.raises(ValueError, match="infeasible"):
        exact(steps([(10, 100)], [(30, 50)], r=200))


def in_interval_by_definition(s: Steps, lam: float) -> bool:
    """Section 1: λ is an optimal dual exactly when the steps it puts in the money can
    balance with those it puts out of it, the steps priced at λ taking any share."""
    sell_lt = int(s.sell_qty[s.sell_price < lam].sum())
    sell_le = int(s.sell_qty[s.sell_price <= lam].sum())
    buy_gt = int(s.buy_qty[s.buy_price > lam].sum())
    buy_ge = int(s.buy_qty[s.buy_price >= lam].sum())
    return sell_lt - buy_ge <= s.r <= sell_le - buy_gt


def test_random_programmes_against_the_definition_and_highs() -> None:
    rng = np.random.default_rng(1)
    for _ in range(300):
        n_sell, n_buy = rng.integers(1, 12, size=2)
        s = Steps(rng.integers(-500, 500, n_sell) * 10, rng.integers(1, 2000, n_sell),
                  rng.integers(-500, 500, n_buy) * 10, rng.integers(1, 2000, n_buy),
                  int(rng.integers(-1000, 1000)))
        try:
            e = exact(s)
        except ValueError:
            continue
        prices = np.unique(np.concatenate((s.sell_price, s.buy_price)))
        grid = np.concatenate((prices, prices + 0.5, prices - 0.5, [-1e9, 1e9]))
        for lam in grid:
            assert in_interval_by_definition(s, lam) == (e.lo <= lam <= e.hi), (s, lam, e)
        solve(s)
