"""Unit tests for the pricing, IMF-charge and retail-rate helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exchanges import price  # noqa: E402
from imf import marginal_rate, rate_of_charge, schedule, threshold  # noqa: E402
from retail import e_rate, f_rate  # noqa: E402


def test_par_bond_prices_at_par():
    assert abs(price(5.0, 5.0, 3.0) - 100.0) < 1e-9
    assert abs(price(2.0, 2.0, 7.0) - 100.0) < 1e-9


def test_published_exchange_price():
    # OT 2.125% October 2028 sold on 6 July 2022 at 102.88 with a yield of 1.64%
    years = (pd.Timestamp("2028-10-17") - pd.Timestamp("2022-07-06")).days / 365.25
    assert abs(price(2.125, 1.64, years) - 102.88) < 0.05


def test_eff_schedule_twelve_instalments():
    p = pd.Series([12.0], index=[pd.Period("2011-05", "M")])
    s = schedule(p)
    assert len(s) == 12 and np.isclose(s.sdr.sum(), 12.0)
    assert s.due.min() == pd.Period("2015-11", "M") and s.due.max() == pd.Period("2021-05", "M")


def test_threshold_changes_with_the_2016_rules():
    assert np.isclose(threshold(pd.Period("2015-12", "M")), 3.0 * 1.0297)
    assert np.isclose(threshold(pd.Period("2016-03", "M")), 1.875 * 2.0601)


def test_rate_of_charge_level_surcharge():
    m = pd.Period("2016-12", "M")
    credit = 12.78
    r = rate_of_charge(m, credit, 0.0, 0.5, None)
    expected = 0.5 + 1.0 + 2.0 * (credit - threshold(m)) / credit
    assert np.isclose(r, expected)
    assert np.isclose(marginal_rate(m, credit, 0.5), 3.5)
    assert np.isclose(marginal_rate(m, 1.0, 0.5), 1.5)


def test_savings_certificate_formulas():
    assert e_rate(3.0, 0) == 3.5 and e_rate(1.2, 0) == 2.2 and e_rate(1.2, 24) == 2.7 and e_rate(-0.5, 0) == 0.5
    assert f_rate(3.0, 0) == 2.5 and f_rate(2.0, 13) == 2.25 and f_rate(-0.3, 0) == 0.0
