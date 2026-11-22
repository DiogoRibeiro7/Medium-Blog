"""Checks of the debt model, the JSON-stat parser and the data identity.

python -m unittest discover -s tests -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import build_data as bd  # noqa: E402
import debt_model as M  # noqa: E402


class TestModel(unittest.TestCase):
    def test_2027_matches_the_budget(self):
        r = M.run(M.scenarios()["A"])
        self.assertAlmostEqual(r.loc[2027, "debt"], 84.5, delta=0.05)

    def test_identity_holds_inside_the_model(self):
        r = M.run(M.scenarios()["R2"])
        prev = M.D0
        for y, row in r.iterrows():
            g = row["nominal_growth"] / 100
            expected = prev / (1 + g) + row["interest"] - row["primary_balance"] + row["sfa"]
            self.assertAlmostEqual(row["debt"], expected, places=10, msg=str(y))
            prev = row["debt"]

    def test_more_surplus_means_less_debt(self):
        low = M.run(M.scenarios(pb_level=2.0)["A"]).loc[2035, "debt"]
        high = M.run(M.scenarios(pb_level=3.0)["A"]).loc[2035, "debt"]
        self.assertLess(high, low)

    def test_required_surplus_hits_the_target(self):
        sc = M.scenarios()["A"]
        pb = M.required_primary_surplus(sc)
        pbs = {y: (sc.pbs[y] if y == 2027 else pb) for y in M.YEARS}
        self.assertAlmostEqual(M.run(sc.copy(pbs=pbs)).loc[2035, "debt"], 60.0, places=6)

    def test_excessive_deficit_rule_tightens(self):
        sc = M.scenarios()["R2"].copy(eps=0.84)
        r = M.run(sc)
        breach = r.index[r["balance"] < -3]
        self.assertTrue(len(breach) > 0)
        self.assertEqual(r.loc[breach[0] + 1, "edp_adjustment"], 0.5)

    def test_recession_raises_2035_debt(self):
        base = M.run(M.scenarios()["A"]).loc[2035, "debt"]
        for key in ("R1", "R2", "R3"):
            self.assertGreater(M.run(M.scenarios()[key]).loc[2035, "debt"], base)


class TestJsonStat(unittest.TestCase):
    def test_flat_index_order(self):
        js = {
            "id": ["geo", "time"],
            "size": [1, 2],
            "dimension": {
                "geo": {"category": {"index": {"PT": 0}}},
                "time": {"category": {"index": {"2024": 0, "2025": 1}}},
            },
            "value": {"0": 1.5, "1": 2.5},
        }
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            p.write_text(json.dumps(js))
            df = bd.jsonstat_to_frame(p)
        self.assertEqual(df.set_index("time")["value"].to_dict(), {"2024": 1.5, "2025": 2.5})


@unittest.skipUnless((HERE / "data" / "panel.csv").exists(), "run fetch_data.py and build_data.py first")
class TestData(unittest.TestCase):
    def test_debt_identity_closes(self):
        import pandas as pd

        import run_analysis as ra

        h = ra.history(pd.read_csv(HERE / "data" / "panel.csv", index_col="year"))
        self.assertLess(h["residual"].abs().max(), 1e-9)
        self.assertAlmostEqual(h.loc[2025, "debt"], 89.2, delta=0.1)


if __name__ == "__main__":
    unittest.main()
