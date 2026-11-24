"""The estimators on cases with a known answer, and the source pins."""

import re
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fetch_data  # noqa: E402
import ownership_core as oc  # noqa: E402


def simulated_countries(n, rng, beta=(-1.0, -0.5), noise=0.3):
    """Countries with one covariate and a logit-normal share: share = expit(b0 + b1 x + e)."""
    x = rng.normal(size=n)
    share = expit(beta[0] + beta[1] * x + rng.normal(scale=noise, size=n))
    return pd.DataFrame({"iso3": [f"C{i:02d}" for i in range(n)], "x": x, "share": share})


class FractionalLogit(unittest.TestCase):
    def test_recovers_coefficients(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=5000)
        y = rng.binomial(200, expit(0.5 - 1.0 * x)) / 200
        res = oc.fit_fractional_logit(pd.DataFrame({"x": x, "y": y}), "y", ["x"])
        self.assertAlmostEqual(res.params["const"], 0.5, delta=0.03)
        self.assertAlmostEqual(res.params["x"], -1.0, delta=0.03)

    def test_rejects_shares_outside_unit_interval(self):
        with self.assertRaises(ValueError):
            oc.fit_fractional_logit(pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.1, 1.2, 0.3]}), "y", ["x"])

    def test_clustered_errors_run(self):
        rng = np.random.default_rng(2)
        d = pd.DataFrame({"g": np.repeat(np.arange(20), 5), "x": rng.normal(size=100)})
        d["y"] = expit(-0.5 + 0.3 * d["x"] + np.repeat(rng.normal(scale=0.5, size=20), 5))
        res = oc.fit_fractional_logit(d, "y", ["x"], groups="g")
        self.assertTrue(np.all(np.isfinite(res.bse)))


class HoldOut(unittest.TestCase):
    def test_target_outcome_does_not_enter_its_own_prediction(self):
        rng = np.random.default_rng(3)
        d = simulated_countries(30, rng)
        _, p1 = oc.holdout(d, "share", ["x"], "C05")
        d.loc[d["iso3"] == "C05", "share"] = 0.999
        _, p2 = oc.holdout(d, "share", ["x"], "C05")
        self.assertAlmostEqual(p1, p2, places=12)

    def test_planted_outlier_ranks_first(self):
        rng = np.random.default_rng(4)
        d = simulated_countries(30, rng, noise=0.1)
        d.loc[d["iso3"] == "C07", "share"] = expit(np.log(d.loc[d["iso3"] == "C07", "share"] / 0.5) + 2.0).clip(
            upper=0.95
        )
        loo = oc.leave_one_out(d, "share", ["x"])
        self.assertEqual(loo.iloc[0]["iso3"], "C07")
        self.assertEqual(int(loo.loc[loo["iso3"] == "C07", "rank"].iloc[0]), 1)


class JackknifePlus(unittest.TestCase):
    def test_coverage_is_at_least_one_minus_two_alpha(self):
        """Barber et al. (2021): coverage >= 1 - 2 alpha for exchangeable units; usually near 1 - alpha."""
        rng = np.random.default_rng(5)
        alpha, reps, hits = 0.1, 200, 0
        for _ in range(reps):
            d = simulated_countries(26, rng)
            lo, hi = oc.jackknife_plus(d, "share", ["x"], "C25", alpha=alpha)
            hits += lo <= d.loc[d["iso3"] == "C25", "share"].iloc[0] <= hi
        self.assertGreaterEqual(hits / reps, 1 - 2 * alpha)

    def test_interval_contains_point_prediction(self):
        rng = np.random.default_rng(6)
        d = simulated_countries(25, rng)
        lo, hi = oc.jackknife_plus(d, "share", ["x"], "C00")
        _, pred = oc.holdout(d, "share", ["x"], "C00")
        self.assertLess(lo, pred)
        self.assertGreater(hi, pred)


class Shares(unittest.TestCase):
    def test_pooling_weights_years_by_counts(self):
        panel = pd.DataFrame(
            {
                "iso3": ["A", "A", "B"],
                "year": [2010, 2011, 2010],
                "inventions": [100.0, 900.0, 50.0],
                "foreign": [90.0, 90.0, 10.0],
                "us": [10.0, 45.0, 5.0],
            }
        )
        s = oc.pooled_shares(panel, 2010, 2011).set_index("iso3")
        self.assertAlmostEqual(s.loc["A", "foreign_share"], 180 / 1000)  # not the mean of 0.9 and 0.1
        self.assertAlmostEqual(s.loc["A", "other_share"], (180 - 55) / 1000)
        self.assertAlmostEqual(s.loc["A", "log_inventions"], np.log(500))
        s2 = oc.pooled_shares(panel, 2010, 2011, min_inventions=100)
        self.assertEqual(list(s2["iso3"]), ["A"])

    def test_covariates(self):
        s = pd.DataFrame({"iso3": ["GBR", "POL", "DEU"]})
        c = oc.add_covariates(s)
        self.assertEqual(list(c["english"]), [1.0, 0.0, 0.0])
        self.assertEqual(list(c["cee"]), [0.0, 1.0, 0.0])


class SourcePins(unittest.TestCase):
    def test_every_source_has_a_sha256(self):
        for name, (_, digest) in fetch_data.SOURCES.items():
            self.assertRegex(digest, r"^[0-9a-f]{64}$", name)

    def test_request_formats(self):
        for name in fetch_data.SOURCES:
            url = fetch_data.url_for(name)
            self.assertTrue(re.search(r"\?format=(csvfile|SDMX-CSV)$", url), url)


if __name__ == "__main__":
    unittest.main()
