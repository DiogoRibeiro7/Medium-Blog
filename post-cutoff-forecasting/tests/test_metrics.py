import numpy as np
import pytest

from postcutoff.metrics import crossing_rate, pinball, score, seasonal_scale, summarise

Q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def test_pinball_by_hand():
    y = np.array([[10.0]])
    f = np.full((1, 1, 2), 8.0)
    loss = pinball(y, f, [0.1, 0.9])
    assert loss[0, 0, 0] == pytest.approx(0.1 * 2)
    assert loss[0, 0, 1] == pytest.approx(0.9 * 2)


def test_seasonal_scale():
    x = np.tile(np.arange(24.0), 10)[None, :]
    assert seasonal_scale(x, 24)[0] == 0.0
    x2 = np.arange(200.0)[None, :]
    assert seasonal_scale(x2, 168)[0] == pytest.approx(168.0)


def test_score_sums_and_coverage():
    rng = np.random.default_rng(1)
    y = rng.normal(100, 5, (2, 48))
    med = y + 2.0
    f = np.stack([med + (q - 0.5) * 20 for q in Q], axis=2)
    s = score(y, f, Q, scale=np.array([4.0, 2.0]), holiday=np.zeros((2, 48)))
    assert len(s) == 4
    assert s["sum_abs_err"].sum() == pytest.approx(2.0 * 96)
    assert s["sum_err"].sum() == pytest.approx(-2.0 * 96)
    frame = s.assign(origin=s["origin_idx"])
    m = summarise(frame)
    assert m["mae"] == pytest.approx(2.0)
    assert m["bias"] == pytest.approx(-2.0)
    assert m["mase"] == pytest.approx(np.mean([2.0 / 4.0, 2.0 / 2.0]))
    assert m["coverage80"] == 1.0  # the band is +/- 8 around a median 2 off


def test_score_sorts_crossed_quantiles():
    y = np.full((1, 24), 10.0)
    f = np.stack([np.full((1, 24), v) for v in [9, 8, 7, 6, 10, 14, 13, 12, 11]], axis=2).astype(float)
    assert crossing_rate(f) == 1.0
    s = score(y, f, Q, scale=np.ones(1), holiday=np.zeros((1, 24)))
    # After sorting the median (5th of 9) is 10, so the point error is zero.
    assert s["sum_abs_err"].iloc[0] == 0.0


def test_score_rejects_bad_shapes():
    with pytest.raises(ValueError):
        score(np.zeros((1, 24)), np.zeros((1, 24, 3)), Q, np.ones(1), np.zeros((1, 24)))
    with pytest.raises(ValueError):
        score(np.zeros((1, 25)), np.zeros((1, 25, 9)), Q, np.ones(1), np.zeros((1, 25)))
