"""End-to-end test against the simulated models: the runner survives daily quotas, short rate limits,
refusals and restarts without duplicating or losing a response, and the analysis recovers the
simulation's error correlations and anchoring rates.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

os.environ["BLIND_TESTER_CONFIG"] = str(HERE / "config_test.json")

import common  # noqa: E402
from fake_models import MODELS, FakeModels  # noqa: E402

N_SYNTH = 3000


def synthetic_questions(path: Path) -> list[dict]:
    import numpy as np
    rng = np.random.default_rng(7)
    rows = [{"qid": i, "subject": "synthetic", "question": f"Synthetic question {i}: which option?",
             "choices": [f"option {j}" for j in range(4)], "answer": int(rng.integers(0, 4))}
            for i in range(N_SYNTH)]
    path.write_text(json.dumps(rows))
    return rows


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.qfile = cls.tmp / "questions.json"
        rows = synthetic_questions(cls.qfile)
        cls.fake = FakeModels(rows, common.SYSTEM, quota=1300)
        cls.url = cls.fake.start()
        cls.runs = cls.tmp / "runs"
        env = dict(os.environ, GITHUB_TOKEN="test-token")
        cls.statuses = []
        for day in range(8):
            out = subprocess.run([sys.executable, "run.py", "--runs-dir", str(cls.runs), "--questions-file",
                                  str(cls.qfile), "--endpoint", cls.url, "--status", str(cls.tmp / "status.json")],
                                 cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
            if out.returncode != 0:
                raise AssertionError(out.stdout[-3000:] + out.stderr[-3000:])
            st = json.loads((cls.tmp / "status.json").read_text())
            cls.statuses.append(st)
            cls.fake.new_day()
            if all(m["complete"] for m in st["models"]):
                break
        cls.summary_path = cls.tmp / "summary.json"
        out = subprocess.run([sys.executable, "analyse.py", "--runs-dir", str(cls.runs), "--questions-file",
                              str(cls.qfile), "--fig-dir", str(cls.tmp / "figs"), "--summary", str(cls.summary_path)],
                             cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
        if out.returncode != 0:
            raise AssertionError(out.stdout[-3000:] + out.stderr[-3000:])
        cls.report = out.stdout
        cls.summary = json.loads(cls.summary_path.read_text())

    @classmethod
    def tearDownClass(cls):
        cls.fake.stop()

    def test_quota_stops_and_resumes(self):
        first = {m["key"]: m for m in self.statuses[0]["models"]}
        self.assertEqual(first["gpt-4o-mini"]["stopped"], "quota")
        self.assertGreater(len(self.statuses), 1)
        last = {m["key"]: m for m in self.statuses[-1]["models"]}
        self.assertTrue(all(m["complete"] for m in last.values()), last)
        self.assertEqual(self.statuses[-1]["missing_models"], ["phi-4"])

    def test_no_duplicates_and_followups_where_needed(self):
        cfg = common.load_config()
        for key in ("gpt-4o-mini", "gpt-4o", "llama", "mistral-small"):
            recs = common.read_runs(self.runs / f"{key}.jsonl")
            pairs = [(r["qid"], r["role"]) for r in recs]
            self.assertEqual(len(pairs), len(set(pairs)), key)
            done = {(r["qid"], r["role"]): r for r in recs}
            for (qid, role), r in done.items():
                if role != "coder":
                    continue
                b = done[(qid, "blind")]
                need = common.in_check_set(qid, cfg) or r["answer"] is None or b["answer"] != r["answer"]
                self.assertEqual((qid, "anchored") in done, need, (key, qid))
                self.assertEqual((qid, "warned") in done, need, (key, qid))
        n_low = cfg["tiers"]["low"]["n_questions"]
        n_high = cfg["tiers"]["high"]["n_questions"]
        self.assertEqual(self.summary["models"]["gpt-4o-mini"]["complete"], n_low)
        self.assertEqual(self.summary["models"]["gpt-4o"]["complete"], n_high)

    def test_refusals_recorded(self):
        recs = common.read_runs(self.runs / "gpt-4o-mini.jsonl")
        refused = [r for r in recs if r.get("error")]
        self.assertTrue(refused)
        self.assertTrue(all(r["qid"] % 97 == 0 and r["role"] == "coder" for r in refused))

    def test_recovers_error_correlation(self):
        for key, mid in (("gpt-4o-mini", "openai/gpt-4o-mini"), ("llama", "meta/Llama-3.3-70B-Instruct")):
            same = self.summary["pairs"][f"{key}|{key}"]["rho"]
            self.assertAlmostEqual(same, MODELS[mid][2], delta=0.1, msg=key)
        cross = self.summary["pairs"]["gpt-4o-mini|llama"]["rho"]
        self.assertAlmostEqual(cross, 0.40, delta=0.15)

    def test_recovers_anchoring(self):
        for key, mid in (("gpt-4o-mini", "openai/gpt-4o-mini"), ("mistral-small", "mistral-ai/mistral-small-2503")):
            an = self.summary["models"][key]["anchoring"]
            self.assertAlmostEqual(an["anchored"]["a"], MODELS[mid][4], delta=0.06, msg=key)
            self.assertAlmostEqual(an["warned"]["a"], MODELS[mid][5], delta=0.08, msg=key)
            # reading the answer never turns an agreement into a disagreement, up to unparseable replies
            self.assertLessEqual(an["anchored"]["check_broken"], 0.03 * an["anchored"]["check_agreements"] + 2)

    def test_designs_order(self):
        d = self.summary["models"]["gpt-4o-mini"]["designs"]
        self.assertGreater(d["anchored"]["pass"], d["blind (same model)"]["pass"])
        self.assertGreater(d["blind (same model)"]["pass"], d["blind llama"]["pass"])

    def test_figures_drawn(self):
        for f in ("fig1_designs", "fig2_error_correlation", "fig3_anchoring"):
            self.assertGreater((self.tmp / "figs" / f"{f}.png").stat().st_size, 10_000, f)

    def test_parse_answer(self):
        p = common.parse_answer
        self.assertEqual(p("blah\nANSWER: C"), "C")
        self.assertEqual(p("ANSWER: (b)"), "B")
        self.assertEqual(p("**ANSWER:** D"), "D")
        self.assertEqual(p("Answer - A\nmore\nANSWER: B"), "B")
        self.assertIsNone(p("I would pick the second one"))
        self.assertIsNone(p(None))


if __name__ == "__main__":
    unittest.main()
