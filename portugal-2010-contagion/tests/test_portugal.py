"""Unit tests for the model, contagion and programme helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contagion import REST, components, responses  # noqa: E402
from warranted import PANEL, fit, in_window, matrix  # noqa: E402


def test_fit_recovers_coefficients():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(size=500), rng.normal(size=500), np.ones(500)])
    y = X @ np.array([0.02, -0.1, 0.3])
    assert np.allclose(fit(X, y), [0.02, -0.1, 0.3])


def test_window_excludes_the_hole():
    q = pd.Series(pd.period_range("2009Q1", "2013Q4", freq="Q"))
    keep = in_window(q, "2009Q1", "2013Q4", ("2010Q2", "2012Q4"))
    assert keep.sum() == len(q) - 11
    assert not keep[q == pd.Period("2011Q1", "Q")].any()


def test_greece_gets_the_average_country_effect():
    rows = [dict(geo=g, debt_gap=0.0) for g in PANEL] + [dict(geo="EL", debt_gap=0.0)]
    X = matrix(pd.DataFrame(rows), ["debt_gap"], bund=False, fe=True)
    effects = X[:, -len(PANEL):]
    assert np.allclose(effects[:-1].sum(axis=1), 1.0)
    assert np.allclose(effects[-1], 1 / len(PANEL))


def _simulated(n: int, seed: int, b_gr: float, b_ie: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = 0.05 + 0.3 * rng.normal(size=n)
    g = 0.4 + 3.0 * rng.normal(size=n)
    i = 0.1 + 1.5 * rng.normal(size=n)
    cols = {c: common + 0.1 * rng.normal(size=n) for c in REST}
    cols["EL"] = common + g
    cols["IE"] = common + 0.2 * g + i
    cols["PT"] = common + b_gr * g + b_ie * i + 0.3 * rng.normal(size=n)
    return pd.DataFrame(cols)


def test_contagion_recovers_known_responses():
    est = responses(_simulated(2000, 3, 0.3, 0.5), "PT")
    assert abs(est["b_greek"] - 0.3) < 0.03 and abs(est["b_irish"] - 0.5) < 0.04


def test_decomposition_adds_up():
    d = _simulated(40, 4, 0.2, 0.4)
    est = responses(d, "PT")
    assert abs(est["part_common"] + est["part_greek"] + est["part_irish"] + est["part_own"] - est["total"]) < 1e-9


def test_components_reuse_given_coefficients():
    a, b = _simulated(60, 5, 0.2, 0.4), _simulated(30, 6, 0.2, 0.4)
    fitted = components(a, "PT")
    applied = components(b, "PT", fitted.attrs["coefs"])
    assert applied.attrs["coefs"][0] == fitted.attrs["coefs"][0]
    assert not np.allclose(applied["greek"].to_numpy(), components(b, "PT")["greek"].to_numpy())
