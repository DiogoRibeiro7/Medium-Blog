"""Checks on the real files: sample sizes and answer codes that CDC documents. Skipped without data/.

A wrong question number in yrbs_core.ITEMS would read another item; these checks catch that through
the documented sample sizes and the number of answer options each item has.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yrbs_core as Y

HAVE_DATA = all((Y.DATA / rel.split("/")[1]).exists() for pair in Y.FILES.values() for rel in pair)

# Usable questionnaires, from each year's Data User's Guide.
N_STUDENTS = {2017: 14765, 2019: 13677, 2021: 17232, 2023: 20103}
# Highest answer code of each raw item (A = 1, B = 2, ...).
MAX_CODE = {
    "sex": 2,
    "sad": 2,
    "consider": 2,
    "plan": 2,
    "attempt": 5,
    "bullied": 2,
    "ebullied": 2,
    "active": 8,
    "sleep": 7,
    "tv": 7,
    "games": 7,
    "screen": 6,
    "social": 8,
    "poor_mh": 5,
}
SEXID_MAX = {2017: 4, 2019: 4, 2021: 6, 2023: 6}


@unittest.skipUnless(HAVE_DATA, "YRBS files not downloaded (python fetch_data.py)")
class RealFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = {}
        for w, (dat, sas) in Y.FILES.items():
            layout = Y.parse_sas_input((Y.DATA / sas.split("/")[1]).read_text(encoding="latin-1"))
            names = sorted(set(Y.ITEMS[w].values()))
            cls.raw[w] = Y.read_fixed(Y.DATA / dat.split("/")[1], layout, names)

    def test_sample_sizes(self):
        for w, raw in self.raw.items():
            self.assertEqual(len(raw["weight"]), N_STUDENTS[w], msg=str(w))

    def test_answer_codes(self):
        for w, raw in self.raw.items():
            for item, var in Y.ITEMS[w].items():
                if item not in MAX_CODE:
                    continue
                v = raw[var][np.isfinite(raw[var])]
                self.assertEqual(v.min(), 1, msg=f"{w} {item} ({var})")
                self.assertEqual(v.max(), MAX_CODE[item], msg=f"{w} {item} ({var})")
            v = raw[Y.ITEMS[w]["sexid"]]
            self.assertEqual(np.nanmax(v), SEXID_MAX[w], msg=f"{w} sexid")

    def test_social_media_item_missing_as_cdc_reports(self):
        # Young et al. (2024): 15,203 answered, 4,900 missing.
        v = self.raw[2023][Y.ITEMS[2023]["social"]]
        self.assertEqual(int(np.isfinite(v).sum()), 15203)

    def test_design_variables(self):
        for w, raw in self.raw.items():
            for k in ("weight", "stratum", "psu"):
                self.assertTrue(np.isfinite(raw[k]).all(), msg=f"{w} {k}")
            self.assertTrue((raw["weight"] > 0).all())


if __name__ == "__main__":
    unittest.main()
