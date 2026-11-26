"""The numbers and figures scripts run end to end on a synthetic cohort with a planted difference."""

import json
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import exits_core as ec  # noqa: E402
import exits_numbers  # noqa: E402
import figures  # noqa: E402

CENSOR = date(2026, 11, 25)
RATES = {  # events per year after the first allotment
    "foreign_acquisition": {"British": 0.030, "non-British": 0.015, "mixed": 0.02},
    "flip": {"British": 0.004, "non-British": 0.010, "mixed": 0.006},
    "uk_acquisition": {"British": 0.015, "non-British": 0.012, "mixed": 0.012},
    "offshore": {"British": 0.002, "non-British": 0.003, "mixed": 0.002},
    "closure": {"British": 0.08, "non-British": 0.10, "mixed": 0.08},
}


def synthetic_cohort(n=2500, seed=0):
    rng = np.random.default_rng(seed)
    groups = rng.choice(["British", "non-British", "mixed"], size=n, p=[0.6, 0.3, 0.1])
    origins = {"British": ["British"], "mixed": ["mixed"], "non-British": ["EU and EEA", "North America", "India"]}
    rows = []
    for i, g in enumerate(groups):
        inc = date(2016, 7, 1) + timedelta(days=int(rng.integers(0, 1645)))
        entry = inc + timedelta(days=int(rng.integers(0, 1095)))
        ev = {}
        for cause, rate in RATES.items():
            t = rng.exponential(1 / rate[g])
            ev[cause] = (entry + timedelta(days=int(365.25 * t))).isoformat() if t < 12 else ""
        nxt = entry + timedelta(days=366 + int(rng.exponential(700)))
        rows.append(
            {
                "company_number": f"{i:08d}",
                "incorporated": inc.isoformat(),
                "censor": CENSOR.isoformat(),
                "entry": entry.isoformat(),
                "round_entry": entry.isoformat() if rng.uniform() < 0.4 else "",
                **ev,
                "status": "active",
                "cohort_year": inc.year,
                "sic_group": rng.choice(["software", "data and web", "research", "hardware and health"]),
                "region": rng.choice(["London", "rest of UK", "Oxford and Cambridge"], p=[0.5, 0.4, 0.1]),
                "founders": int(rng.integers(1, 4)),
                "founder_group": g,
                "founder_origin": rng.choice(origins[g]),
                "founder_abroad": int(g == "non-British" and rng.uniform() < 0.3),
                "allotments_first_year": int(rng.integers(1, 4)),
                "next_allotment": nxt.isoformat() if nxt <= CENSOR and rng.uniform() < 0.6 else "",
            }
        )
    return pd.DataFrame(rows)


class EndToEnd(unittest.TestCase):
    def test_numbers_and_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            synthetic_cohort().to_csv(tmp / "cohort.csv", index=False)
            report = {
                "snapshot": "persons-with-significant-control-snapshot-2026-11-25.zip",
                "censor_date": CENSOR.isoformat(),
                "searched_companies": 9000,
                "capped_search_chunks": 0,
                "excluded": {"no allotment of shares within three years": 6500},
            }
            (tmp / "report.json").write_text(json.dumps(report))
            args = ["--cohort", str(tmp / "cohort.csv"), "--report", str(tmp / "report.json")]
            args += ["--out", str(tmp / "output.txt"), "--figdata", str(tmp / "figdata.json")]
            args += ["--bootstrap", "20", "--fg-bootstrap", "5", "--curve-bootstrap", "10"]
            self.assertEqual(exits_numbers.main(args), 0)
            text = (tmp / "output.txt").read_text()
            for heading in ("Cohort", "Cumulative incidence by founder group", "Fine-Gray", "Sensitivity"):
                self.assertIn(heading, text)
            figdata = json.loads((tmp / "figdata.json").read_text())
            self.assertEqual(set(figdata["n"]), {"British", "non-British", "mixed"})
            self.assertEqual(figures.main(["--figdata", str(tmp / "figdata.json"), "--out", str(tmp / "fig")]), 0)
            self.assertGreaterEqual(len(list((tmp / "fig").glob("*.png"))), 4)

    def test_planted_difference_is_found(self):
        out = ec.outcomes(synthetic_cohort(n=6000, seed=1))
        s = ec.cause_specific_cox(out, "foreign_acquisition", exits_numbers.COVARIATES, exits_numbers.REFERENCE)
        self.assertGreater(s.loc["founder_group[British]", "low"], 1.2)  # planted hazard ratio is 2


if __name__ == "__main__":
    unittest.main()
