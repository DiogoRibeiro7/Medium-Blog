"""Tests for the vintage reader, the revision statistics and the measurement-error extension of the rules."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_data as B  # noqa: E402
import revisions as V  # noqa: E402
import rules as R  # noqa: E402

J = {y: t for t, y in enumerate(R.YEARS)}


def frame(rows):
    return pd.DataFrame(rows, columns=["vintage_year", "season", "country", "year", "var", "value"])


class TestReader(unittest.TestCase):
    def test_flat_and_nested_releases_give_the_same_rows(self):
        text = "CODE;COUNTRY;SUB-CHAPTER;TITLE;UNIT;2019;2020;\nPRT.1.0.0.0.AVGDGP;Portugal;x;y;z;1.5;NA;\n"
        with tempfile.TemporaryDirectory() as d:
            flat, nested = Path(d) / "ameco_autumn2011.zip", Path(d) / "ameco_autumn2021.zip"
            with zipfile.ZipFile(flat, "w") as z:
                z.writestr("AMECO1.TXT", text)
            inner = io.BytesIO()
            with zipfile.ZipFile(inner, "w") as z:
                z.writestr("AMECO1.TXT", text)
            with zipfile.ZipFile(nested, "w") as z:
                z.writestr("ameco0.zip", inner.getvalue())
            codes = {"PRT.1.0.0.0.AVGDGP": ("PRT", "gap")}
            self.assertEqual(B.read_release(flat, codes), [("PRT", 2019, "gap", 1.5)])
            self.assertEqual(B.read_release(nested, codes), B.read_release(flat, codes))
            self.assertEqual(B.release_label(nested), (2021, "autumn"))


class TestRevisions(unittest.TestCase):
    def test_estimates_pick_the_age_and_the_latest_value(self):
        v = frame(
            [
                (2015, "autumn", "PRT", 2016, "gap", -1.0),  # age -1
                (2015, "autumn", "PRT", 2015, "gap", -2.0),  # age 0
                (2015, "spring", "PRT", 2015, "gap", -9.0),  # other season
                (2026, "autumn", "PRT", 2015, "gap", -1.5),
                (2026, "autumn", "PRT", 2016, "gap", -0.5),
            ]
        )
        e0 = V.estimates(v, "gap", 0)
        self.assertEqual(len(e0), 1)
        self.assertAlmostEqual(e0.loc[0, "error"], -0.5)
        e1 = V.estimates(v, "gap", -1)
        self.assertAlmostEqual(e1.loc[0, "error"], -0.5)
        self.assertEqual(int(e1.loc[0, "year"]), 2016)

    def test_miss_share_recovers_a_known_error_process(self):
        rng = np.random.default_rng(3)
        innov = rng.normal(0, 2, 400)
        latest = innov + rng.normal(0, 1, 400)
        df = pd.DataFrame(
            {
                "country": "X",
                "year": np.arange(400),
                "latest": latest,
                "innovation": innov,
                "estimate": latest - 0.6 * innov + 0.2,
            }  # fmt: skip
        )
        m = V.miss_share(df)
        self.assertAlmostEqual(m["c"], 0.6, places=6)
        self.assertAlmostEqual(-m["const"], 0.2, places=6)
        self.assertLess(m["resid_sd"], 1e-8)

    def test_attenuation_slope(self):
        x = np.linspace(-4, 4, 50)
        df = pd.DataFrame({"latest": x, "estimate": 0.5 * x - 0.3})
        a = V.attenuation(df)
        self.assertAlmostEqual(a["lam"], 0.5, places=8)
        self.assertAlmostEqual(a["a"], -0.3, places=8)

    def test_structural_effort_flags_a_changed_verdict(self):
        v = frame(
            [
                (2015, "autumn", "PRT", 2014, "sb", -2.0),
                (2015, "autumn", "PRT", 2015, "sb", -1.4),  # real time +0.6: meets 0.5
                (2026, "autumn", "PRT", 2014, "sb", -2.0),
                (2026, "autumn", "PRT", 2015, "sb", -1.8),  # now +0.2: does not
            ]
        )
        s = V.structural_effort(v, "sb")
        self.assertEqual(len(s), 1)
        self.assertAlmostEqual(s.loc[0, "realtime"], 0.6)
        self.assertAlmostEqual(s.loc[0, "now"], 0.2)
        self.assertTrue(bool(s.loc[0, "flip"]))

    def test_decomposition_splits_the_change_in_the_gap(self):
        rows = []
        for vy, y, Y, Yp, gap in ((2015, 2014, 100.0, 101.0, -1.0), (2015, 2015, 101.0, 102.0, -0.98)):
            rows += [(vy, "autumn", "PRT", y, "Y", Y), (vy, "autumn", "PRT", y, "Ypot", Yp),
                     (vy, "autumn", "PRT", y, "gap", gap)]  # fmt: skip
        for y, Y, Yp, gap in ((2014, 100.0, 101.0, -1.0), (2015, 102.0, 102.5, -0.49)):
            rows += [(2026, "autumn", "PRT", y, "Y", Y), (2026, "autumn", "PRT", y, "Ypot", Yp),
                     (2026, "autumn", "PRT", y, "gap", gap)]  # fmt: skip
        d = V.decomposition(frame(rows))
        self.assertEqual(len(d), 1)
        self.assertAlmostEqual(d.loc[0, "growth_rev"], 1.0, places=8)
        self.assertAlmostEqual(d.loc[0, "potential_rev"], 100 * (102.5 / 101 - 1) - 100 * (102 / 101 - 1), places=8)


class TestMeasurementInRules(unittest.TestCase):
    def test_exact_measurement_reproduces_the_earlier_fixed_path(self):
        s = R.simulate(R.Rule("f", "f", "fixed", level=2.5), R.Macro(rho_gap=0.467), R.zero_draws())
        self.assertAlmostEqual(s["debt"][0, J[2035]], 63.828, places=2)

    def test_a_bias_shifts_a_cycle_rule_and_not_the_rigid_one(self):
        m = R.Macro(gap_bias=-0.7, mu=0.0)
        fixed = R.simulate(R.Rule("f", "f", "fixed", level=2.5), m, R.zero_draws())
        rigid = R.simulate(R.Rule("r", "r", "fixed", level=2.5, eps=0.0), m, R.zero_draws())
        self.assertAlmostEqual(fixed["pb"][0, J[2030]], 2.5 - 0.35, places=10)
        self.assertAlmostEqual(rigid["pb"][0, J[2030]], 2.5, places=10)

    def test_noise_is_stationary_and_reproducible(self):
        n = 20_000
        z = np.random.default_rng(5).standard_normal((n, R.H))
        m = R.Macro(gap_miss=0.0, gap_noise_sd=1.2, gap_noise_rho=0.8, mu=0.0, edp=False)
        a = R.simulate(R.Rule("f", "f", "fixed", level=2.5), m, R.zero_draws(n), z)
        b = R.simulate(R.Rule("f", "f", "fixed", level=2.5), m, R.zero_draws(n), z)
        np.testing.assert_array_equal(a["pb"], b["pb"])
        implied = (a["pb"][:, 1:] - 2.5) / 0.5  # eps * noise
        self.assertAlmostEqual(float(implied[:, 3].std()), 1.2, delta=0.03)
        self.assertAlmostEqual(float(implied[:, -1].std()), 1.2, delta=0.03)
        r = np.corrcoef(implied[:, 5], implied[:, 6])[0, 1]
        self.assertAlmostEqual(float(r), 0.8, delta=0.02)


if __name__ == "__main__":
    unittest.main()
