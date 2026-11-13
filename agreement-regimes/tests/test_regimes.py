import filecmp
from pathlib import Path

import numpy as np
import pytest

from feed import detect as D
from feed import regimes as R

HERE = Path(__file__).resolve().parents[1]


def months(n, start=2015):
    return [f"{start + i // 12:04d}-{i % 12 + 1:02d}" for i in range(n)]


def test_detector_is_the_second_count_prediction_one_unchanged():
    assert filecmp.cmp(
        HERE / "feed" / "detect.py", HERE.parent / "second-count-prediction" / "feed" / "detect.py", shallow=False
    )


def test_split_finds_a_level_shift_and_measures_it():
    rng = np.random.default_rng(3)
    ms = months(100)
    r = {m: (-0.07 if i < 40 else 0.0) + rng.normal(0, 0.001) for i, m in enumerate(ms)}
    sp = R.split(r)
    assert sp["month"] == ms[40] and sp["reduction"] > 0.9
    assert sp["median_left"] == pytest.approx(-0.07, abs=0.002)


def test_split_respects_the_minimum_segments_and_flags_no_regime_in_noise():
    rng = np.random.default_rng(4)
    ms = months(70)
    r = {m: rng.normal(0, 0.01) for m in ms}
    sp = R.split(r)
    assert R.MIN_LEFT <= sp["left"] and sp["right"] >= R.MIN_RIGHT
    assert sp["reduction"] < R.REGIME


def test_no_split_when_too_short():
    r = {m: 0.0 for m in months(50)}
    assert R.split(r)["month"] is None


def test_add_months():
    assert R.add_months("2024-11", 3) == "2025-02" and R.add_months("2024-01", -36) == "2021-01"


def test_forward_windows_on_a_quiet_series():
    rng = np.random.default_rng(5)
    ms = months(110)
    r = {m: rng.normal(0, 0.002) for m in ms}
    ws = R.forward_windows(r)
    assert ws and all(w["start"].endswith("-01") for w in ws)
    assert all(w["n_trail"] == 36 and w["n_ahead"] == 24 for w in ws)
    assert all(w["observed"] is not None and w["observed"] < 0.05 for w in ws)
    assert R.add_months(ws[-1]["start"], 24) <= R.add_months(ms[-1], 1)


def test_window_curve_matches_detect_on_the_whole_series():
    rng = np.random.default_rng(6)
    ms = months(60)
    r = {m: rng.normal(0, 0.01) for m in ms}
    cur, n = R.window_curve(r, ms[12], "2099-01")
    assert n == 48
    assert cur[D.GRID[20]] == pytest.approx(D.detection(r, D.GRID[20]))
