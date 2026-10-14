import numpy as np
import pytest

from postcutoff.baselines import column_for, forecast, levels_for

Q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def test_level_mapping():
    assert levels_for(Q) == [20, 40, 60, 80]
    assert column_for(0.1) == "m-lo-80" and column_for(0.9) == "m-hi-80" and column_for(0.5) == "m"


def test_seasonal_naive_week_repeats_the_last_week():
    rng = np.random.default_rng(0)
    X = rng.normal(6000, 300, (3, 2048))
    f = forecast("SeasonalNaive(168)", X, 168, Q, n_jobs=1)
    assert f.shape == (3, 168, 9)
    np.testing.assert_allclose(f[:, :, 4], X[:, -168:])
    assert (np.diff(f, axis=2) >= 0).all()


def test_unknown_spec():
    with pytest.raises(ValueError):
        forecast("Prophet()", np.zeros((1, 400)), 24, Q)
