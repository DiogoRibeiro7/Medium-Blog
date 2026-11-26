"""The methods on cases with a known answer. No YRBS file is needed."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yrbs_core as Y  # noqa: E402


def toy_design(n_strata=6, psu_per=4, per_psu=60, seed=0):
    rng = np.random.default_rng(seed)
    stratum = np.repeat(np.arange(n_strata), psu_per * per_psu).astype(float)
    psu = np.tile(np.repeat(np.arange(psu_per), per_psu), n_strata).astype(float) + 100 * stratum
    n = len(stratum)
    u_psu = rng.normal(0, 0.3, n_strata * psu_per)
    cid, s_of, s_cnt = Y.psu_index(stratum, psu)
    x = rng.normal(0, 1, n) + u_psu[cid]
    y = 0.4 * x + rng.normal(0, 1, n) + u_psu[cid]
    w = rng.uniform(0.5, 2.0, n)
    return x, y, w, stratum, psu, (cid, s_of, s_cnt)


class SasLayout(unittest.TestCase):
    def test_column_and_pointer_input(self):
        text = """
        DATA yrbs; INFILE 'x.dat' LRECL=500;
        INPUT
          Q1 $ 17-17
          Q5 $ 21-28
          weight 431-440
          @441 stratum 3.
          @444 PSU $6. ;
        LABEL Q1 = 'How old are you';
        """
        lay = Y.parse_sas_input(text)
        self.assertEqual(lay["q1"], (17, 17))
        self.assertEqual(lay["q5"], (21, 28))
        self.assertEqual(lay["weight"], (431, 440))
        self.assertEqual(lay["stratum"], (441, 443))
        self.assertEqual(lay["psu"], (444, 449))

    def test_read_fixed(self):
        lay = {"a": (1, 2), "b": (3, 5)}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.dat"
            p.write_bytes(b"1 2.5\n  .  \n12  7\n")
            got = Y.read_fixed(p, lay, ["a", "b"])
        np.testing.assert_array_equal(got["a"], [1.0, np.nan, 12.0])
        np.testing.assert_array_equal(got["b"], [2.5, np.nan, 7.0])


class Estimators(unittest.TestCase):
    def test_wls_matches_scaled_least_squares(self):
        x, y, w, stratum, psu, design = toy_design()
        X = np.column_stack([np.ones_like(x), x])
        coef, *_ = Y.wls(X, y, w, design[0], design[1], design[2], len(design[1]))
        sw = np.sqrt(w)
        ref, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        np.testing.assert_allclose(coef, ref, rtol=1e-10)

    def test_design_variance_by_hand(self):
        x, y, w, stratum, psu, design = toy_design(seed=3)
        cid, s_of, s_cnt = design
        X = np.column_stack([np.ones_like(x), x])
        coef, se_d, _, _ = Y.wls(X, y, w, cid, s_of, s_cnt, len(s_of))
        # Explicit sandwich: (X'WX)^-1 [sum_h n_h/(n_h-1) sum_i (g_hi - gbar_h)(...)'] (X'WX)^-1
        A = (X.T * w) @ X
        e = y - X @ coef
        G = np.zeros((len(s_of), 2))
        for i in range(len(y)):
            G[cid[i]] += w[i] * e[i] * X[i]
        meat = np.zeros((2, 2))
        for h in range(len(s_cnt)):
            rows = np.where(s_of == h)[0]
            gh = G[rows] - G[rows].mean(axis=0)
            meat += len(rows) / (len(rows) - 1) * gh.T @ gh
        V = np.linalg.inv(A) @ meat @ np.linalg.inv(A)
        self.assertAlmostEqual(se_d, float(np.sqrt(V[1, 1])), places=10)

    def test_clustering_inflates_design_se(self):
        x, y, w, stratum, psu, design = toy_design(per_psu=150, seed=5)
        X = np.column_stack([np.ones_like(x), x])
        _, se_d, se_n, _ = Y.wls(X, y, np.ones_like(w), design[0], design[1], design[2], len(design[1]))
        self.assertGreater(se_d, se_n)

    def test_multiplier_bootstrap_reproduces_design_covariance(self):
        x, y, w, stratum, psu, design = toy_design(seed=7)
        cid, s_of, s_cnt = design
        y2 = 0.5 * y + np.random.default_rng(1).normal(0, 1, len(y))
        X = np.column_stack([np.ones_like(x), x])
        _, se1, _, z1 = Y.wls(X, y, w, cid, s_of, s_cnt, len(s_of))
        _, se2, _, z2 = Y.wls(X, y2, w, cid, s_of, s_cnt, len(s_of))
        Z = np.vstack([z1, z2])
        rng = np.random.default_rng(11)
        b, t = Y.joint_null({2023: (np.array([0, 1]), Z)}, np.array([se1, se2]), np.ones(2), 40000, rng)
        np.testing.assert_allclose(b.std(axis=0), [se1, se2], rtol=0.02)
        target = float(z1 @ z2) / (se1 * se2)
        self.assertAlmostEqual(float(np.corrcoef(b.T)[0, 1]), target, delta=0.02)
        np.testing.assert_allclose(t.std(axis=0), [1.0, 1.0], rtol=0.02)

    def test_domain_estimate_keeps_all_psus(self):
        x, y, w, stratum, psu, design = toy_design(seed=9)
        cid, s_of, s_cnt = design
        sub = cid % 4 != 0  # drop every fourth PSU from the subgroup
        X = np.column_stack([np.ones(sub.sum()), x[sub]])
        _, _, _, z = Y.wls(X, y[sub], w[sub], cid[sub], s_of, s_cnt, len(s_of))
        self.assertEqual(len(z), len(s_of))


class Summaries(unittest.TestCase):
    def test_shapley_sums_to_r2_and_splits_symmetric_factors(self):
        rng = np.random.default_rng(2)
        a = rng.integers(0, 3, 3000)
        b = rng.integers(0, 2, 3000)
        c = rng.integers(0, 4, 3000)
        y = 1.0 * (a == 1) + 1.0 * (b == 1) + 0.01 * rng.normal(size=3000)
        phi, r2 = Y.shapley_r2(y, {"a": a, "b": b, "c": c})
        self.assertAlmostEqual(sum(phi.values()), r2, places=10)
        self.assertLess(phi["c"], 0.005)
        self.assertGreater(phi["a"], 0.3)
        self.assertGreater(phi["b"], 0.3)

    def test_e_value(self):
        # VanderWeele and Ding (2017): RR = 3.9 gives E = 7.26.
        self.assertAlmostEqual(float(Y.e_value(3.9)), 7.26, places=2)
        self.assertAlmostEqual(float(Y.e_value(1.0)), 1.0)
        self.assertAlmostEqual(float(Y.e_value(1 / 3.9)), float(Y.e_value(3.9)))

    def test_besd(self):
        lo, hi = Y.besd(0.32)
        self.assertAlmostEqual(lo, 0.34)
        self.assertAlmostEqual(hi, 0.66)


class Recoding(unittest.TestCase):
    def setUp(self):
        n = 8
        base = {k: np.full(n, np.nan) for k in set().union(*[set(v.values()) for v in Y.ITEMS.values()])}
        base["q2"] = np.array([1, 2, 1, 2, 1, 2, np.nan, 1.0])
        base["q3"] = np.array([1, 2, 3, 4, 5, 1, 2, 3.0])
        base["raceeth"] = np.array([5, 3, 6, 7, 2, 8, 1, 4.0])
        self.raw = base

    def test_social_media_and_thresholds(self):
        raw = dict(self.raw)
        it = Y.ITEMS[2023]
        raw[it["social"]] = np.array([1, 2, 3, 4, 5, 6, 7, 8.0])
        raw[it["attempt"]] = np.array([1, 2, 3, 4, 5, 1, 1, np.nan])
        raw[it["poor_mh"]] = np.array([1, 2, 3, 4, 5, np.nan, 1, 1.0])
        raw[it["sexid"]] = np.array([1, 2, 3, 4, 5, 6, 1, 1.0])
        d = Y.recode(raw, 2023)
        np.testing.assert_array_equal(d["social"], np.arange(8.0))
        np.testing.assert_array_equal(d["attempt"], [0, 1, 1, 1, 1, 0, 0, np.nan])
        np.testing.assert_array_equal(d["poor_mh"], [0, 0, 0, 1, 1, np.nan, 0, 0])
        np.testing.assert_array_equal(d["minority"], [0, 1, 1, 1, 1, np.nan, 0, 0])
        np.testing.assert_array_equal(d["grade"], [1, 2, 3, 4, np.nan, 1, 2, 3])
        x, ok = Y.exposure(d, "social", "high_vs_light")
        # several times a day or more against a few times a week to once a day
        np.testing.assert_array_equal(ok, [False, False, False, True, True, True, True, True])
        np.testing.assert_array_equal(x[ok], [0, 0, 1, 1, 1])
        x, ok = Y.exposure(d, "social", "top_vs_rest")
        np.testing.assert_array_equal(x, [0, 0, 0, 0, 0, 0, 1, 1])

    def test_tv_hours(self):
        raw = dict(self.raw)
        it = Y.ITEMS[2019]
        raw[it["tv"]] = np.array([1, 2, 3, 4, 5, 6, 7, np.nan])
        d = Y.recode(raw, 2019)
        np.testing.assert_array_equal(d["tv"], [0, 0.5, 1, 2, 3, 4, 5, np.nan])
        x, ok = Y.exposure(d, "tv", "top_vs_light")
        np.testing.assert_array_equal(ok, [False, False, True, True, False, False, True, False])
        np.testing.assert_array_equal(x[ok], [0, 0, 1])


class Grid(unittest.TestCase):
    def test_size(self):
        avail = {2017: set(Y.OUTCOMES[:4]), 2019: set(Y.OUTCOMES[:4]), 2021: set(Y.OUTCOMES), 2023: set(Y.OUTCOMES)}
        specs = Y.grid(avail)
        per = len(Y.CODINGS) * len(Y.CONTROLS) * len(Y.SAMPLES) * len(Y.WEIGHTINGS)
        self.assertEqual(len(specs), (2 * 4 + 2 * 4 + 1 * 5 + 1 * 5) * per)
        self.assertEqual(len({s.key() for s in specs}), len(specs))


if __name__ == "__main__":
    unittest.main()
