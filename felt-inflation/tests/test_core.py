"""The methods on cases with a known answer."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import felt_core as fc  # noqa: E402


def months(y0, y1):
    return [f"{y}-{m:02d}" for y in range(y0, y1 + 1) for m in range(1, 13)]


def toy_country():
    """Two items with prices that grow at fixed monthly rates, weights that change each year."""
    idx = ["2000-12"] + months(2001, 2003)
    t = np.arange(len(idx))
    P = pd.DataFrame({"A": 100 * 1.01**t, "B": 100 * 0.995**t}, index=idx)
    W = pd.DataFrame({"A": [600.0, 500.0, 400.0], "B": [400.0, 500.0, 600.0]}, index=["2001", "2002", "2003"])
    return P, W


class JsonStat(unittest.TestCase):
    def test_rows_follow_row_major_order(self):
        d = {
            "id": ["freq", "coicop", "geo", "time"],
            "size": [1, 2, 2, 2],
            "dimension": {
                "freq": {"category": {"index": {"M": 0}}},
                "coicop": {"category": {"index": {"CP00": 0, "CP01": 1}}},
                "geo": {"category": {"index": {"PT": 0, "ES": 1}}},
                "time": {"category": {"index": {"2020-01": 0, "2020-02": 1}}},
            },
            "value": {"0": 1.0, "3": 4.0, "5": 6.0, "7": 8.0},
        }
        rows = sorted(fc.jsonstat_rows(d))
        self.assertEqual(
            rows,
            [
                ("ES", "CP00", "2020-02", 4.0),
                ("ES", "CP01", "2020-02", 8.0),
                ("PT", "CP00", "2020-01", 1.0),
                ("PT", "CP01", "2020-02", 6.0),
            ],
        )

    def test_wide_csv_round_trip_and_conflicts(self):
        rows = [("PT", "X", "2020-02", 2.5), ("PT", "X", "2020-01", 1.0), ("ES", "X", "2020-01", 3.0)]
        text = fc.wide_csv(rows)
        self.assertEqual(text.splitlines()[0], "geo,item,2020-01,2020-02")
        want = {("ES", "X", "2020-01"): "3.0", ("PT", "X", "2020-01"): "1.0", ("PT", "X", "2020-02"): "2.5"}
        self.assertEqual(fc.cells(text), want)
        self.assertEqual(text, fc.wide_csv(list(reversed(rows))))
        with self.assertRaises(ValueError):
            fc.wide_csv(rows + [("PT", "X", "2020-01", 9.0)])


class ChainLinking(unittest.TestCase):
    def test_chain_is_the_product_of_annual_laspeyres_links(self):
        P, W = toy_country()
        out = fc.chain(P, W, fc.fixed_items(["A", "B"]), 2001, 2003)
        level = 100.0
        for y in (2001, 2002, 2003):
            dec = f"{y - 1}-12"
            w = W.loc[str(y)]
            for m in range(1, 13):
                mo = f"{y}-{m:02d}"
                rel = (w.A * P.at[mo, "A"] / P.at[dec, "A"] + w.B * P.at[mo, "B"] / P.at[dec, "B"]) / (w.A + w.B)
                self.assertAlmostEqual(out.at[mo, "index"], level * rel, places=10)
            level = out.at[f"{y}-12", "index"]
        self.assertTrue(np.allclose(out["weight"], 1000.0))

    def test_complement_recovers_the_other_item(self):
        P, W = toy_country()
        total = fc.chain(P, W, fc.fixed_items(["A", "B"]), 2001, 2003)
        a = fc.chain(P, W, fc.fixed_items(["A"]), 2001, 2003)
        b = fc.chain(P, W, fc.fixed_items(["B"]), 2001, 2003)
        rest = fc.complement(total, [a])
        self.assertTrue(np.allclose(rest["index"], b["index"], rtol=1e-12))
        self.assertTrue(np.allclose(rest["weight"], b["weight"]))

    def test_mix_with_the_true_weights_reproduces_the_total(self):
        P, W = toy_country()
        total = fc.chain(P, W, fc.fixed_items(["A", "B"]), 2001, 2003)["index"]
        a = pd.concat([pd.Series({"2000-12": 100.0}), fc.chain(P, W, fc.fixed_items(["A"]), 2001, 2003)["index"]])
        b = pd.concat([pd.Series({"2000-12": 100.0}), fc.chain(P, W, fc.fixed_items(["B"]), 2001, 2003)["index"]])
        w = {y: W.at[str(y), "A"] / 1000 for y in (2001, 2002, 2003)}
        mixed = fc.mix(a, b, w)
        self.assertTrue(np.allclose(mixed.values, total.reindex(mixed.index).values, rtol=1e-12))

    def test_pairs_fall_back_to_the_combined_class(self):
        idx = ["2019-12", "2020-01"]
        codes = fc.FRO + ["CP1212_1213"]
        P = pd.DataFrame(100.0, index=idx, columns=codes)
        W = pd.DataFrame(1.0, index=["2020"], columns=codes)
        P.loc[:, "CP1212"] = np.nan
        got = fc.frequent_items(2020, P, W)
        self.assertIn("CP1212_1213", got)
        self.assertNotIn("CP1213", got)
        P.loc[:, "CP1212"] = 100.0
        got = fc.frequent_items(2020, P, W)
        self.assertIn("CP1212", got)
        self.assertNotIn("CP1212_1213", got)

    def test_rates(self):
        s = pd.Series(100 * 1.01 ** np.arange(40), index=months(2001, 2004)[:40])
        self.assertAlmostEqual(fc.rate(s).iloc[-1], (1.01**12 - 1) * 100, places=10)
        self.assertAlmostEqual(fc.rate(s, 36).iloc[-1], (1.01**12 - 1) * 100, places=10)
        self.assertAlmostEqual(fc.annualised(s, "2001-01", "2003-01"), (1.01**12 - 1) * 100, places=10)


class Regression(unittest.TestCase):
    def panel(self, noise=0.0, seed=1):
        rng = np.random.default_rng(seed)
        rows = []
        for g, a in zip("ABCDE", [5.0, -3.0, 10.0, 0.0, 2.0], strict=True):
            for t in range(60):
                x1, x2 = rng.normal(2, 1), rng.normal(1, 2)
                y = a + 3 * x1 + 1 * x2 + noise * rng.normal()
                rows.append({"geo": g, "month": f"m{t:03d}", "x1": x1, "x2": x2, "y": y})
        return pd.DataFrame(rows)

    def test_fixed_effects_recover_the_slopes(self):
        beta, r2 = fc.fe_fit(self.panel(), "y", ["x1", "x2"])
        self.assertTrue(np.allclose(beta, [3, 1], atol=1e-10))
        self.assertAlmostEqual(r2, 1.0, places=10)
        self.assertTrue(np.allclose(fc.implied_weights(beta), [0.75, 0.25]))

    def test_two_way_fixed_effects_absorb_a_common_shock(self):
        p = self.panel()
        shock = {m: 7 * math.sin(i) for i, m in enumerate(sorted(p.month.unique()))}
        p["y"] = p.y + p.month.map(shock)
        beta, _ = fc.fe_fit(p, "y", ["x1", "x2"], two_way=True)
        self.assertTrue(np.allclose(beta, [3, 1], atol=1e-8))

    def test_solver_matches_numpy(self):
        rng = np.random.default_rng(3)
        A = rng.normal(size=(5, 5)) + 5 * np.eye(5)
        b = rng.normal(size=5)
        self.assertTrue(np.allclose(fc._solve(A.tolist(), b.tolist()), np.linalg.solve(A, b)))

    def test_bootstrap_is_reproducible_and_brackets_the_truth(self):
        p = self.panel(noise=1.0)

        def stat(q):
            return fc.fe_fit(q, "y", ["x1", "x2"])[0][0]

        a = fc.bootstrap(p, stat, 50, seed=4)
        b = fc.bootstrap(p, stat, 50, seed=4)
        self.assertTrue(np.array_equal(a, b))
        lo, hi = fc.interval(a)
        self.assertLess(lo, 3)
        self.assertGreater(hi, 3)

    def test_out_of_sample_prediction_uses_training_effects(self):
        p = self.panel()
        train, test = p[p.month < "m040"], p[p.month >= "m040"]
        pred, beta = fc.fe_predict(train, test, "y", ["x1", "x2"])
        self.assertTrue(np.allclose(pred, test.y))
        self.assertAlmostEqual(fc.rmse(pred, test.y), 0.0, places=8)

    def test_exact_mean_and_correlation(self):
        x = np.array([1e16, 1.0, -1e16, 1.0])
        self.assertEqual(fc.mean(x), 0.5)
        self.assertAlmostEqual(fc.corr([1, 2, 3, 4], [2, 4, 6, 8]), 1.0)
        self.assertAlmostEqual(fc.corr([1, 2, 3, 4], [8, 6, 4, 2]), -1.0)


if __name__ == "__main__":
    unittest.main()
