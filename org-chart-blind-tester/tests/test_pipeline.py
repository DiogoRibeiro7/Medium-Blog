"""End-to-end test against simulated models behind a stand-in Ollama server: the runner survives
interrupted jobs, restarts and a server error without duplicating or losing a response, shards cover
every question exactly once, and the analysis recovers the simulation's error correlations (within a
model, within a family, across families) and anchoring rates.

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
from fake_models import MODELS, RHO_CROSS, FakeOllama  # noqa: E402

N_SYNTH = 3000
MAX_CALLS = 700   # each job stops early, as a timed-out Actions job would, and the next one resumes


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
        cls.fake = FakeOllama(rows, common.SYSTEM)
        cls.url = cls.fake.start()
        cls.runs = cls.tmp / "runs"
        cls.cfg = common.load_config()
        cls.jobs = []
        for spec in common.model_specs(cls.cfg):
            for shard in range(spec["shards"]):
                for attempt in range(10):
                    st_path = cls.tmp / f"status-{spec['key']}-{shard}.json"
                    out = subprocess.run([sys.executable, "run.py", "--runs-dir", str(cls.runs), "--models",
                                          spec["key"], "--shard", str(shard), "--questions-file", str(cls.qfile),
                                          "--base-url", cls.url, "--max-calls", str(MAX_CALLS),
                                          "--status", str(st_path)],
                                         cwd=ROOT, capture_output=True, text=True, timeout=900)
                    if out.returncode != 0:
                        raise AssertionError(out.stdout[-3000:] + out.stderr[-3000:])
                    st = json.loads(st_path.read_text())["shards"][0]
                    cls.jobs.append(st)
                    if st["complete"]:
                        break
        out = subprocess.run([sys.executable, "analyse.py", "--runs-dir", str(cls.runs), "--questions-file",
                              str(cls.qfile), "--fig-dir", str(cls.tmp / "figs"), "--summary",
                              str(cls.tmp / "summary.json")],
                             cwd=ROOT, capture_output=True, text=True, timeout=900)
        if out.returncode != 0:
            raise AssertionError(out.stdout[-3000:] + out.stderr[-3000:])
        cls.report = out.stdout
        cls.summary = json.loads((cls.tmp / "summary.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.fake.stop()

    def test_interrupted_jobs_resume_to_completion(self):
        self.assertTrue(any(j["stopped"] == "max_calls" for j in self.jobs))
        last = {}
        for j in self.jobs:
            last[(j["key"], j["shard"])] = j
        self.assertTrue(all(j["complete"] for j in last.values()), last)
        self.assertTrue(self.fake.flaked)

    def test_shards_cover_each_question_once_without_duplicates(self):
        n = self.cfg["n_questions"]
        for spec in common.model_specs(self.cfg):
            recs = common.read_model_runs(self.runs, spec["key"])
            pairs = [(r["qid"], r["role"]) for r in recs]
            self.assertEqual(len(pairs), len(set(pairs)), spec["key"])
            coder = {r["qid"] for r in recs if r["role"] == "coder"}
            self.assertEqual(len(coder), n, spec["key"])
            done = {(r["qid"], r["role"]): r for r in recs}
            for qid in coder:
                c, b = done[(qid, "coder")], done[(qid, "blind")]
                need = common.in_check_set(qid, self.cfg) or c["answer"] is None or b["answer"] != c["answer"]
                self.assertEqual((qid, "anchored") in done, need, (spec["key"], qid))
                self.assertEqual((qid, "warned") in done, need, (spec["key"], qid))
            self.assertEqual(self.summary["models"][spec["key"]]["complete"], n)
            prov = sorted(self.runs.glob(f"{spec['key']}.shard*.provenance.json"))
            self.assertEqual(len(prov), spec["shards"])

    def test_recovers_error_correlation(self):
        for key, tag in (("llama3.1-8b", "llama3.1:8b"), ("ministral-3-8b", "ministral-3:8b")):
            self.assertAlmostEqual(self.summary["pairs"][f"{key}|{key}"]["rho"], MODELS[tag][2], delta=0.1, msg=key)
        self.assertAlmostEqual(self.summary["pairs"]["qwen3-8b|qwen3.5-9b"]["rho"], MODELS["qwen3:8b"][3], delta=0.12)
        self.assertAlmostEqual(self.summary["pairs"]["llama3.1-8b|ministral-3-8b"]["rho"], RHO_CROSS, delta=0.12)
        self.assertEqual(self.summary["pairs"]["qwen3-8b|qwen3.5-9b"]["relation"], "same family")
        self.assertEqual(self.summary["pairs"]["qwen3-8b|llama3.1-8b"]["relation"], "other family")

    def test_recovers_anchoring(self):
        for key, tag in (("llama3.1-8b", "llama3.1:8b"), ("qwen3.5-9b", "qwen3.5:9b")):
            an = self.summary["models"][key]["anchoring"]
            self.assertAlmostEqual(an["anchored"]["a"], MODELS[tag][4], delta=0.06, msg=key)
            self.assertAlmostEqual(an["warned"]["a"], MODELS[tag][5], delta=0.08, msg=key)
            # reading the answer never turns an agreement into a disagreement, up to unparseable replies
            self.assertLessEqual(an["anchored"]["check_broken"], 0.03 * an["anchored"]["check_agreements"] + 2)

    def test_designs_order(self):
        d = self.summary["models"]["llama3.1-8b"]["designs"]
        self.assertGreater(d["anchored"]["pass"], d["blind (same model)"]["pass"])
        self.assertGreater(d["blind (same model)"]["pass"], d["blind ministral-3-8b"]["pass"])

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
