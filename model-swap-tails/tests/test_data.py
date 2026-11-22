from datetime import date

import numpy as np
import pandas as pd
import pytest

from swaptails.data import daily_mean, fill_from_earlier, parse_forecast, parse_public_power, to_hourly


def test_parse_public_power():
    payload = {
        "unix_seconds": [0, 900, 1800],
        "production_types": [{"name": "Load", "data": [1.0, None, 3.0]}, {"name": "Solar", "data": [0, 0, 0]}],
    }
    s = parse_public_power(payload, "Load")
    assert s.isna().tolist() == [False, True, False]
    with pytest.raises(ValueError):
        parse_public_power(payload, "Wind")


def test_parse_forecast_both_formats():
    columnar = {"unix_seconds": [0, 3600], "forecast_values": [10.0, 11.0]}
    assert parse_forecast(columnar).tolist() == [10.0, 11.0]
    envelope = {
        "data": [
            {"timestamp": "2025-01-01T01:00:00+01:00", "values": [5.0]},
            {"timestamp": "2025-01-01T02:00:00+01:00", "values": [6.0]},
        ]
    }
    s = parse_forecast(envelope)
    assert s.index[0] == pd.Timestamp("2025-01-01T00:00:00Z")
    assert s.tolist() == [5.0, 6.0]
    with pytest.raises(ValueError):
        parse_forecast({"something": 1})


def test_to_hourly_needs_complete_hours():
    idx = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")
    s = pd.Series([1, 2, 3, 4, 5, np.nan, 7, 8], index=idx, dtype=float)
    out, res = to_hourly(s)
    assert out.iloc[0] == 2.5
    assert np.isnan(out.iloc[1])
    assert res == {"15 min": 2}


def test_daily_mean_on_daylight_saving_days():
    idx = pd.date_range("2025-03-29T23:00Z", "2025-03-31T21:00Z", freq="h")  # Lisbon: 30 March has 23 hours
    s = pd.Series(1.0, index=idx)
    out = daily_mean(s, "Europe/Lisbon")
    assert out[date(2025, 3, 30)] == 1.0
    s.iloc[5] = np.nan
    assert np.isnan(daily_mean(s, "Europe/Lisbon")[date(2025, 3, 30)])


def test_fill_never_uses_later_values():
    x = np.arange(400, dtype=float)
    x[[10, 200, 399]] = np.nan
    out, filled = fill_from_earlier(x)
    assert np.isnan(out[10])  # nothing a week or a day earlier
    assert out[200] == 200 - 168 and out[399] == 399 - 168
    assert filled.sum() == 2
