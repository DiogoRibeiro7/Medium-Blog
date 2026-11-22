"""Checks of services_core and jsonstat against answers known by construction."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import services_core as sc  # noqa: E402
from jsonstat import read_jsonstat, to_number  # noqa: E402


class JsonStat(unittest.TestCase):
    def test_durations_become_minutes(self):
        self.assertEqual(to_number("05:04"), 304.0)
        self.assertEqual(to_number(1.5), 1.5)

    def test_row_major_positions_and_missing_values(self):
        doc = {
            "version": "2.0",
            "class": "dataset",
            "id": ["freq", "geo", "time"],
            "size": [1, 2, 3],
            "dimension": {
                "freq": {"category": {"index": {"A": 0}}},
                "geo": {"category": {"index": {"PT": 1, "ES": 0}}},
                "time": {"category": {"index": {"2000": 0, "2001": 1, "2002": 2}}},
            },
            "value": {"0": 1.0, "2": 3.0, "4": 5.0},
            "status": {"4": "p"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.json"
            path.write_text(json.dumps(doc))
            df = read_jsonstat(path)
        self.assertEqual(len(df), 3)
        got = {(r.geo, r.time): (r.value, r.flag) for r in df.itertuples()}
        self.assertEqual(got[("ES", 2000)], (1.0, ""))
        self.assertEqual(got[("ES", 2002)], (3.0, ""))
        self.assertEqual(got[("PT", 2001)], (5.0, "p"))


class Aggregation(unittest.TestCase):
    def test_tornqvist_equals_common_growth_when_all_industries_grow_alike(self):
        years = [2000, 2001, 2002]
        vol = pd.DataFrame({"a": [100, 110, 121], "b": [50, 55, 60.5]}, index=years)
        nom = pd.DataFrame({"a": [100, 120, 140], "b": [50, 50, 70]}, index=years)
        idx = sc.tornqvist_index(vol, nom)
        np.testing.assert_allclose(idx.to_numpy(), [1.0, 1.1, 1.21])

    def test_tornqvist_weights_by_mean_nominal_share(self):
        years = [2000, 2001]
        vol = pd.DataFrame({"a": [100.0, 200.0], "b": [100.0, 100.0]}, index=years)
        nom = pd.DataFrame({"a": [25.0, 25.0], "b": [75.0, 75.0]}, index=years)
        idx = sc.tornqvist_index(vol, nom)
        self.assertAlmostEqual(idx.iloc[1], math.exp(0.25 * math.log(2)))


class GrowthAccounting(unittest.TestCase):
    def test_decomposition_adds_up(self):
        o, i, p = sc.decompose(100, 150, 80, 100, 10)
        self.assertAlmostEqual(o, i + p)
        self.assertAlmostEqual(o, 100 * math.log(1.5) / 10)

    def test_cagr_and_required_rate(self):
        self.assertAlmostEqual(sc.cagr(1, 1.02**30, 30), 0.02)
        self.assertAlmostEqual(sc.required_productivity_growth(0.02, 0.0), 0.02)
        self.assertAlmostEqual(1.01 ** sc.years_to_factor(10, 0.01), 10)

    def test_classification(self):
        self.assertEqual(sc.classify_decoupling(1.5, 0.9), "absolute")
        self.assertEqual(sc.classify_decoupling(1.5, 1.2), "relative")
        self.assertEqual(sc.classify_decoupling(1.2, 1.5), "none")
        self.assertEqual(sc.classify_decoupling(0.9, 0.8), "none")


class Comparisons(unittest.TestCase):
    def test_ols_recovers_an_exact_line(self):
        x = np.arange(8.0)
        slope, se, r2 = sc.ols_slope(x, 1 - 0.5 * x)
        self.assertAlmostEqual(slope, -0.5)
        self.assertAlmostEqual(se, 0.0, places=9)
        self.assertAlmostEqual(r2, 1.0)

    def test_ranges(self):
        self.assertAlmostEqual(sc.ratio_range([2, 4, 8, np.nan]), 4.0)
        self.assertAlmostEqual(sc.coefficient_of_variation([1, 1, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
