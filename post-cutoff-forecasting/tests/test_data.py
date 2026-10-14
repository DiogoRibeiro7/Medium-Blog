import numpy as np
import pandas as pd
import pytest

from postcutoff.data import anomalies, load_snapshot, parse_public_power, prepare, regularise, write_snapshot


def payload(values, start=1767225600):
    return {
        "unix_seconds": [start + 3600 * i for i in range(len(values))],
        "production_types": [
            {"name": "Solar", "data": [0.0] * len(values)},
            {"name": "Load", "data": values},
        ],
        "deprecated": False,
    }


def test_parse_public_power_takes_the_named_series_and_keeps_nulls():
    s = parse_public_power(payload([5827.3, None, 5200.9]), "Load")
    assert s.index[0] == pd.Timestamp("2026-01-01T00:00Z")
    assert s.index.freq is None and (s.index[1] - s.index[0]) == pd.Timedelta(hours=1)
    assert np.isnan(s.iloc[1]) and s.iloc[2] == pytest.approx(5200.9)


def test_parse_public_power_rejects_missing_or_misaligned_series():
    with pytest.raises(ValueError):
        parse_public_power(payload([1.0, 2.0]), "Residual load")
    bad = payload([1.0, 2.0])
    bad["production_types"][1]["data"] = [1.0]
    with pytest.raises(ValueError):
        parse_public_power(bad, "Load")


def test_regularise_fills_the_grid_with_nan():
    idx = pd.to_datetime(["2026-01-01T00:00Z", "2026-01-01T03:00Z"])
    s = regularise(pd.Series([1.0, 4.0], index=idx))
    assert len(s) == 4 and s.isna().sum() == 2


def test_snapshot_round_trip_and_hash(cfg):
    idx = pd.date_range("2022-01-01", periods=500, freq="h", tz="UTC")
    s = pd.Series(np.linspace(5000, 6000, 500), index=idx)
    s.iloc[10] = np.nan
    manifest = write_snapshot(s, cfg, {"requests": []})
    back = load_snapshot(cfg)
    assert back.index.equals(idx)
    assert np.isnan(back.iloc[10]) and back.iloc[11] == pytest.approx(round(s.iloc[11], 1))
    assert manifest["missing_hours"] == 1
    cfg.snapshot.write_text(cfg.snapshot.read_text().replace("5000.0", "5000.1"))
    with pytest.raises(ValueError, match="hash"):
        load_snapshot(cfg)


def test_prepare_replaces_masked_hours_with_last_week_and_never_scores_them(cfg):
    idx = pd.date_range("2025-04-01", "2025-05-15", freq="h", tz="UTC", inclusive="left")
    y = 6000 + 500 * np.sin(np.arange(idx.size) / 24 * 2 * np.pi)
    s = pd.Series(y, index=idx)
    p = prepare(s, cfg)
    blackout = (idx >= pd.Timestamp("2025-04-28T10:00Z")) & (idx < pd.Timestamp("2025-04-28T22:00Z"))
    assert blackout.sum() == 12
    assert p.masked[blackout].all() and not p.scorable[blackout].any()
    i = int(np.flatnonzero(blackout)[0])
    assert p.inputs[i] == y[i - 168]
    assert p.scorable[~blackout].all()


def test_anomalies_flag_collapses_and_report_whether_masked(cfg):
    idx = pd.date_range("2025-03-01", "2025-05-15", freq="h", tz="UTC", inclusive="left")
    y = np.full(idx.size, 6000.0)
    blackout = (idx >= pd.Timestamp("2025-04-28T10:00Z")) & (idx < pd.Timestamp("2025-04-28T22:00Z"))
    y[blackout] = 90.0
    y[np.flatnonzero(idx == pd.Timestamp("2025-05-10T12:00Z"))] = 100.0
    flagged = anomalies(pd.Series(y, index=idx), cfg)
    assert len(flagged) == 13
    assert flagged["masked"].sum() == 12


def test_to_hourly_averages_quarter_hours_and_rejects_incomplete_hours():
    from postcutoff.data import to_hourly

    idx = pd.date_range("2025-10-01T00:00Z", periods=12, freq="15min")
    values = [100, 200, 300, 400, 1000, 1000, 1000, 1000, 5, np.nan, 5, 5]
    hourly, resolution = to_hourly(pd.Series(values, index=idx, dtype=float))
    assert list(hourly.index.hour) == [0, 1, 2]
    assert hourly.iloc[0] == pytest.approx(250.0) and hourly.iloc[1] == pytest.approx(1000.0)
    assert np.isnan(hourly.iloc[2])
    assert resolution == {"15 min": 3}


def test_to_hourly_handles_a_switch_from_hourly_to_quarter_hourly():
    from postcutoff.data import to_hourly

    a = pd.Series([1.0, 2.0], index=pd.date_range("2025-09-30T22:00Z", periods=2, freq="h"))
    b = pd.Series([4.0, 4.0, 8.0, 8.0], index=pd.date_range("2025-10-01T00:00Z", periods=4, freq="15min"))
    hourly, resolution = to_hourly(pd.concat([a, b]))
    assert list(hourly) == [1.0, 2.0, 6.0]
    assert resolution == {"60 min": 2, "15 min": 1}


def test_regularise_refuses_timestamps_off_the_hour():
    idx = pd.date_range("2025-10-01T00:00Z", periods=4, freq="15min")
    with pytest.raises(ValueError, match="on the hour"):
        regularise(pd.Series([1.0, 2.0, 3.0, 4.0], index=idx))


def test_prepare_fills_long_gaps_from_earlier_weeks_only(cfg):
    idx = pd.date_range("2024-01-01", "2024-03-01", freq="h", tz="UTC", inclusive="left")
    y = 6000 + np.arange(idx.size, dtype=float)  # every hour distinct, rising
    y[24 * 30 : 24 * 55] = np.nan  # 25 days missing
    p = prepare(pd.Series(y, index=idx), cfg)
    gap = np.arange(24 * 30, 24 * 55)
    assert p.imputed[gap].all() and not p.scorable[gap].any()
    # Every filled value is a copy of an hour before the gap: nothing comes from after it.
    assert (p.inputs[gap] < y[24 * 30 - 1] + 1).all()
    assert np.isin(p.inputs[gap], y[: 24 * 30]).all()


def test_prepare_refuses_a_hole_with_no_earlier_week(cfg):
    idx = pd.date_range("2024-01-01", periods=400, freq="h", tz="UTC")
    y = np.full(idx.size, 6000.0)
    y[5] = np.nan
    with pytest.raises(ValueError, match="cannot be filled"):
        prepare(pd.Series(y, index=idx), cfg)
