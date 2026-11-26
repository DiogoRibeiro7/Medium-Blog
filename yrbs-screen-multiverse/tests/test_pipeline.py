"""End to end on synthetic files with a known effect: read, recode, fit, and recover it."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import synthetic  # noqa: E402

import yrbs_core as Y  # noqa: E402


class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        synthetic.write(cls.tmp.name)
        cls.waves = {w: Y.load_wave(w, cls.tmp.name) for w in Y.WAVES}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_recovers_known_risk_difference(self):
        for w, d in self.waves.items():
            design = Y.psu_index(d["stratum"], d["psu"])
            m = Y.MEASURES[w][0]
            got = []
            for o in ("sad", "consider", "plan"):
                s = Y.Spec(w, m, "top_vs_rest", o, "demographic_identity", "all", "survey")
                f = Y.fit_spec(d, s, design)
                got.append((f.b, f.se_design))
            est = np.mean([g[0] for g in got])
            self.assertLess(abs(est - synthetic.TRUE_EFFECT), 0.05, msg=f"{w} {m}: {got}")

    def test_layout_reads_every_item(self):
        for d in self.waves.values():
            self.assertGreater(np.isfinite(d["sad"]).mean(), 0.9)
            self.assertTrue(set(np.unique(d["female"][np.isfinite(d["female"])])) <= {0.0, 1.0})


if __name__ == "__main__":
    unittest.main()
