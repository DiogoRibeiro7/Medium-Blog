"""Contracts for part2_wealth_shares.py."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import part2_wealth_shares as p2  # noqa: E402


class PartialCorrelationTests(unittest.TestCase):
    def test_matches_regression_identity(self) -> None:
        rng = np.random.default_rng(11)
        z = rng.normal(size=80)
        x = 0.7 * z + rng.normal(size=80)
        y = -0.4 * z + 0.3 * x + rng.normal(size=80)
        frame = pd.DataFrame({"x": x, "y": y, "z": z})
        r = p2.pcorr(frame, "x", "y", ["z"])
        # Partial correlation from the correlation matrix.
        c = frame.corr()
        expected = (c.x.y - c.x.z * c.y.z) / np.sqrt((1 - c.x.z ** 2) * (1 - c.y.z ** 2))
        self.assertAlmostEqual(r["r"], expected, 10)
        self.assertLess(r["ci95"][0], r["r"])
        self.assertGreater(r["ci95"][1], r["r"])
        self.assertEqual(r["n"], 80)

    def test_raw_correlation_without_controls(self) -> None:
        frame = pd.DataFrame({"x": [1.0, 2, 3, 4, 5], "y": [2.0, 1, 4, 3, 6]})
        self.assertAlmostEqual(p2.pcorr(frame, "x", "y", [])["r"], float(np.corrcoef(frame.x, frame.y)[0, 1]), 12)


class InputTests(unittest.TestCase):
    def test_forbes_hash_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name in ("SHA256SUMS.txt", "counts_citizenship.csv", "counts_api_country.csv"):
                shutil.copy(p2.EVIDENCE / name, tmp / name)
            p2.load_forbes(tmp)  # unchanged copies load
            with (tmp / "counts_api_country.csv").open("a") as handle:
                handle.write("Atlantis,1,api_country,country\n")
            with self.assertRaises(ValueError):
                p2.load_forbes(tmp)

    def test_wid_hash_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name in ("wid_manifest.json", "wid_series.csv.gz", "wid_metadata.csv.gz"):
                shutil.copy(p2.DATA / name, tmp / name)
            p2.load_wid(tmp)
            manifest = json.loads((tmp / "wid_manifest.json").read_text())
            manifest["wid_series_sha256"] = "0" * 64
            (tmp / "wid_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                p2.load_wid(tmp)


class PanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        panel = p2.load_panel(p2.p1.PANEL)
        surveys, heads, _ = p2.load_pip()
        series, _, _ = p2.load_wid()
        citizenship, api = p2.load_forbes()
        cls.base, cls.wide = p2.build(panel, surveys, heads, series, citizenship, api)

    def test_every_country_has_shares_and_counts(self) -> None:
        self.assertEqual(len(self.base), 54)
        self.assertFalse(self.base[["wT1", "wT10", "wB50", "iT10", "n_cit"]].isna().any().any())
        # Nested top groups: top 0.01% <= top 0.1% <= top 1% <= top 10%.
        self.assertTrue((self.base.wT001 <= self.base.wT01).all())
        self.assertTrue((self.base.wT01 <= self.base.wT1).all())
        self.assertTrue((self.base.wT1 <= self.base.wT10).all())

    def test_status_split_and_zero_residence(self) -> None:
        self.assertEqual(int(self.base.observed.sum()), 30)
        self.assertEqual(sorted(self.base.loc[self.base.n_res == 0, "iso3"]), ["BGR", "EST", "HRV"])
        self.assertTrue(self.base.loc[self.base.n_res == 0, "B_res"].isna().all())

    def test_secondary_density_is_the_panel_column(self) -> None:
        np.testing.assert_allclose(self.base.B_sec, self.base[p2.B])


if __name__ == "__main__":
    unittest.main()
