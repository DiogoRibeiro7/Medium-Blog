"""Contracts for part5_volatility.py helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import part5_volatility as p5  # noqa: E402
from part4_threshold_crossings import ChangeDist  # noqa: E402


def benchmark(seed: int = 1) -> tuple[ChangeDist, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.clip(0.02 + 0.12 * rng.standard_t(4, 200000), -1, 1)
    lows = np.floor(np.round(x / 0.02, 9)) * 0.02
    v, c = np.unique(np.round(lows, 2), return_counts=True)
    return ChangeDist(v, c.astype(float), 0.02), x


def cells_from(x: np.ndarray, floor: float, n: int) -> dict:
    """A group whose changes are the benchmark draws, with a share ``floor`` leaving for other reasons."""
    keep = x[: int(n * (1 - floor))]
    keys = (np.floor(np.round(keep / 0.02, 9)) * 20).astype(int)
    h = pd.Series(keys).value_counts().sort_index()
    return {"stock": n, "exit": n - len(keep), "hist": h}


class ScaleTests(unittest.TestCase):
    def test_recovers_scale_one_with_a_floor(self) -> None:
        d, x = benchmark()
        m = d.quantile(0.5)
        s = p5.fit_scale({0: cells_from(x[100000:], 0.03, 60000)}, {0: d}, 0.03)
        self.assertAlmostEqual(s, 1.0, delta=0.04)
        wide = m + 1.3 * (x[100000:] - m)
        s2 = p5.fit_scale({0: cells_from(np.clip(wide, -1, 1), 0.03, 60000)}, {0: d}, 0.03)
        self.assertAlmostEqual(s2, 1.3, delta=0.06)


class ExitProbTests(unittest.TestCase):
    def test_rules_order_and_scale(self) -> None:
        d, _ = benchmark()
        p_down = p5.exit_prob(1000, d, 1.0, "down", 1.0)
        p_950 = p5.exit_prob(1000, d, 1.0, "nearest_950", 1.0)
        p_1000 = p5.exit_prob(1000, d, 1.0, "nearest_1000", 1.0)
        self.assertGreater(p_1000, p_down)           # listed only from a true $1bn: closer to the line
        self.assertGreater(p_down, 0.2)
        self.assertGreater(p5.exit_prob(1400, d, 1.5, "down", 1.0), p5.exit_prob(1400, d, 1.0, "down", 1.0))
        self.assertLess(abs(p_950 - p_down), 0.05)


class NewcomerTests(unittest.TestCase):
    def test_excess_over_incumbents(self) -> None:
        rows = [(2011, "stock", 1000, "incumbent", 100), (2011, "exit", 1000, "incumbent", 20),
                (2011, "stock", 1000, "new", 50), (2011, "exit", 1000, "new", 20)]
        t = pd.DataFrame(rows, columns=["year", "table", "band_lo_musd", "tenure", "n"])
        self.assertAlmostEqual(p5.newcomer_excess(t, [1000], [2011]), 10.0)


if __name__ == "__main__":
    unittest.main()
