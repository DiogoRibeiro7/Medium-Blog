"""Contracts for part1_poverty_lines.py and stats_tools.py."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import part1_poverty_lines as p1  # noqa: E402
import stats_tools as st  # noqa: E402


class StatsToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.normal(size=60)
        z = rng.normal(size=60)
        share = 1 / (1 + np.exp(-(0.3 + 0.8 * x - 0.4 * z + rng.normal(scale=0.3, size=60))))
        self.frame = pd.DataFrame({"x": x, "z": z, "y": 100 * share})

    def test_loo_ols_equals_explicit_refits(self) -> None:
        errors = []
        for i in range(len(self.frame)):
            train = self.frame.drop(index=i)
            fit = sm.OLS(train.y, sm.add_constant(train[["x", "z"]])).fit()
            errors.append(self.frame.y[i] - (fit.params["const"] + fit.params["x"] * self.frame.x[i]
                                             + fit.params["z"] * self.frame.z[i]))
        self.assertAlmostEqual(st.loo_rmse_ols(self.frame, "y", ["x", "z"]), float(np.sqrt(np.mean(np.square(errors)))), 9)

    def test_ame_matches_finite_difference(self) -> None:
        res = st.fractional_logit(self.frame, "y", ["x", "z"], "x")
        fit = sm.GLM(self.frame.y / 100, sm.add_constant(self.frame[["x", "z"]]), family=sm.families.Binomial()).fit()
        X = sm.add_constant(self.frame[["x", "z"]]).to_numpy()
        shifted = X.copy()
        shifted[:, 1] += 1e-6
        expit = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
        numeric = 100 * np.mean(expit(shifted @ fit.params.to_numpy()) - expit(X @ fit.params.to_numpy())) / 1e-6
        self.assertAlmostEqual(res["ame"], numeric, places=4)
        self.assertLess(res["ci95"][0], res["ame"])
        self.assertGreater(res["ci95"][1], res["ame"])


class Part1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.panel = p1.load_panel(p1.PANEL)
        cls.surveys, cls.heads, cls.manifest = p1.load_pip()
        cls.base, cls.wide = p1.build_panel(cls.panel, cls.surveys, cls.heads)

    def test_every_country_matches_the_first_article_survey(self) -> None:
        self.assertEqual(len(self.base), 54)
        gap = (self.wide[30.0] - self.panel.set_index("iso3").loc[self.base.iso3, "poverty_pct"].to_numpy()).abs()
        self.assertLess(float(gap.max()), 1e-4)
        self.assertTrue((self.base.reporting_level == "national").all())

    def test_headcounts_are_monotone_in_the_line(self) -> None:
        self.assertTrue((np.diff(self.wide.to_numpy(), axis=1) >= -1e-9).all())
        self.assertEqual(list(self.wide.columns), sorted(self.wide.columns))

    def test_data_hashes_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("pip_surveys.csv.gz", "pip_headcounts.csv.gz", "pip_manifest.json"):
                (Path(tmp) / name).write_bytes((p1.DATA / name).read_bytes())
            data = bytearray((Path(tmp) / "pip_surveys.csv.gz").read_bytes())
            data[-1] ^= 1
            (Path(tmp) / "pip_surveys.csv.gz").write_bytes(bytes(data))
            with self.assertRaises(ValueError):
                p1.load_pip(Path(tmp))

    def test_thirty_dollar_line_reproduces_the_first_article(self) -> None:
        f = p1.at_line(self.base, self.wide, 30.0)
        fit = st.ols_hc3(f, "poverty_pct", [p1.B, p1.G])
        self.assertAlmostEqual(fit["coef"][p1.B], -7.912, places=3)
        self.assertAlmostEqual(fit["r_squared"], 0.8113, places=4)

    def test_committed_numbers_reproduce(self) -> None:
        numbers = json.loads(json.dumps(p1.compute(self.panel, self.surveys, self.heads, self.manifest)))
        committed = json.loads((p1.RESULTS / "numbers.json").read_text(encoding="utf-8"))
        problems: list[str] = []
        p1._compare(numbers, committed, "numbers", problems)
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
