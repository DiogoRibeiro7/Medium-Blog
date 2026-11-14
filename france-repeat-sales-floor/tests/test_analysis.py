import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis as A  # noqa: E402


def test_no_floor_no_jump():
    c, ex = A.simulate()
    t, se = A.density_jump(c, 0.0, drop_exact=ex)
    td, sd = A.donut_jump(c, 0.0)
    assert abs(t) < 4 * se + 0.02
    assert abs(td) < 4 * sd + 0.02


@pytest.mark.parametrize("kw", [{"q": 0.3}, {"withdraw": 0.3}, {"q": 0.3, "r": 0.5},
                                {"q": 0.1, "withdraw": 0.2, "r": 0.5}])
def test_both_estimators_recover_a_floor(kw):
    """A floor that removes 30% of the small would-be losses below the old price, whether they
    sell at the old price or do not sell, with or without rounding to the old price."""
    c, ex = A.simulate(**kw)
    assert A.missing_share(A.density_jump(c, 0.0, drop_exact=ex)[0]) == pytest.approx(0.3, abs=0.04)
    assert A.missing_share(A.donut_jump(c, 0.0)[0]) == pytest.approx(0.3, abs=0.05)


def test_donut_ignores_rounding_to_the_old_price():
    c, _ = A.simulate(r=0.5)
    t, se = A.donut_jump(c, 0.0)
    assert abs(t) < 4 * se + 0.02


def test_bunching_tells_conversion_from_withdrawal():
    conv = A.bunching(A.simulate(q=0.3)[0], **A.BUNCH)
    gone = A.bunching(A.simulate(withdraw=0.3)[0], **A.BUNCH)
    assert conv["missing"] == pytest.approx(0.3, abs=0.04)
    assert conv["excess"] == pytest.approx(0.3, abs=0.04)
    assert gone["missing"] == pytest.approx(0.3, abs=0.04)
    assert abs(gone["excess"]) < 0.03


def test_scan_finds_no_jump_away_from_zero_and_finds_the_floor_edge():
    """The simulated floor acts on would-be losses of up to 10%, so the density drops at -10% and
    nowhere else away from zero."""
    c, _ = A.simulate(q=0.3, r=0.5)
    for cut in (-0.05, -0.03, 0.03, 0.05, 0.075, 0.10):
        t, se = A.donut_jump(c, cut, inner=0.0, mask=A.scan_mask(cut))
        assert abs(t) < 4 * se + 0.03, cut
    t, _ = A.donut_jump(c, -0.10, inner=0.0, mask=A.scan_mask(-0.10))
    assert t == pytest.approx(np.log(0.7), abs=0.05)


def test_repeat_sales_index_recovers_a_known_path():
    rng = np.random.default_rng(3)
    true = np.cumsum(np.r_[0.0, rng.normal(0.01, 0.02, 11)])
    rows = []
    for q0 in range(12):
        for q1 in range(q0 + 2, 12):
            n = 50
            lr = true[q1] - true[q0] + rng.normal(0, 0.05, n)
            rows.append({"q0": q0, "q1": q1, "n": n, "sum_lr": lr.sum()})
    est = A.bmn_index(pd.DataFrame(rows))
    assert np.allclose(est.to_numpy(), true - true[0], atol=0.01)


def test_wls_matches_ols_with_equal_weights():
    rng = np.random.default_rng(4)
    X = np.column_stack([np.ones(200), rng.normal(size=200)])
    y = X @ np.array([0.2, 0.5]) + rng.normal(0, 0.1, 200)
    b, se = A.wls(y, X, np.ones(200))
    assert np.allclose(b, np.linalg.lstsq(X, y, rcond=None)[0])
    assert (se > 0).all()
