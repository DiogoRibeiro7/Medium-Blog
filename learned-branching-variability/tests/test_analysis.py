"""Checks on the evaluation statistics."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import e2_train  # noqa: E402
import e4_analyse  # noqa: E402


def test_shifted_geometric_mean():
    assert e4_analyse.sgm([1, 1, 1], 1) == pytest.approx(1.0)
    assert e4_analyse.sgm([0, 3], 1) == pytest.approx(np.sqrt(1 * 4) - 1)


def test_decision_metrics_perfect_and_random():
    node = np.repeat(np.arange(50), 10)
    y = np.random.default_rng(0).random(500) + 0.01
    perfect = e2_train.decision_metrics(y, y, node)
    assert perfect["top1"] == 1.0 and perfect["mean_score_ratio"] == pytest.approx(1.0)
    rand = e2_train.decision_metrics(None, y, node, rng=np.random.default_rng(1))
    assert rand["mean_score_ratio"] < 1.0


def _runs(effect):
    rows = []
    rng = np.random.default_rng(0)
    for i in range(30):
        base = rng.lognormal(2, 0.5)
        for s in range(5):
            noise = rng.lognormal(0, 0.1)
            rows.append(dict(instance=i, seed=s, policy="relpscost", cpu=base * noise, nodes=100))
            rows.append(dict(instance=i, seed=s, policy="learned", cpu=base * noise * effect,
                             nodes=100))
    return pd.DataFrame(rows)


def test_paired_ratio_recovers_a_known_effect():
    df = _runs(0.8)
    r = e4_analyse.paired(df, "learned", "cpu", [0, 1, 2, 3, 4], np.random.default_rng(0))
    assert r["lo"] <= r["ratio"] <= r["hi"]
    assert r["ratio"] == pytest.approx(0.8, abs=0.03)


def test_placebo_is_symmetric():
    p = e4_analyse.placebo(_runs(1.0), "cpu", [0, 1, 2])
    assert len(p["ratios"]) == 6
    assert p["min"] == pytest.approx(1 / p["max"])
