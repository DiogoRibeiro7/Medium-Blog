"""Contracts for New Deal source transcriptions, illustrations and artifacts."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis import (
    FIGURE_NAMES,
    bank_data,
    gdp_data,
    gdp_index,
    generate_all,
    liquidity_gap,
    monthly_disposable_income,
    realized_relief,
    relief_classification_gap,
    relief_data,
    unemployment_data,
)


class SourceDataTests(unittest.TestCase):
    def test_bls_retro_annual_rates_are_transcribed(self) -> None:
        d=unemployment_data()
        self.assertEqual(d["year"].astype(int).tolist(),list(range(1929,1940)))
        np.testing.assert_allclose(
            d["unemployment_rate_percent"],
            [3.2,8.7,15.9,23.6,24.9,21.7,20.1,16.9,14.3,19.0,17.2],
        )
        self.assertEqual(d["unemployed_thousands"][4],12_830)

    def test_bea_real_gdp_and_base_index(self) -> None:
        d=gdp_data()
        self.assertEqual(d["year"][0],1929)
        self.assertEqual(d["year"][-1],1940)
        self.assertEqual(d["real_gdp_billion_chained_2017_usd"][4],877.4)
        idx=gdp_index(d)
        self.assertAlmostEqual(idx[0],100)
        self.assertAlmostEqual(idx[4],100*877.4/1191.1)
        self.assertGreater(idx[-1],110)
        with self.assertRaises(ValueError):
            gdp_index(d,base_year=1928)

    def test_relief_worker_counting_conventions_are_distinct(self) -> None:
        d=relief_data()
        gap=relief_classification_gap(d)
        self.assertEqual(gap[0],0)
        self.assertEqual(gap[7],3744)
        self.assertEqual(gap[-1],3255)
        self.assertTrue(np.all(gap>=0))

    def test_bank_suspensions_explicitly_omit_1933(self) -> None:
        d=bank_data()
        self.assertEqual(d["year"].astype(int).tolist(),
                         [1929,1930,1931,1932,1934,1935])
        self.assertEqual(d["number_suspended"].tolist(),
                         [659,1352,2294,1456,57,34])
        self.assertNotIn(1933,d["year"])


class MathematicalIllustrationTests(unittest.TestCase):
    def test_no_insurance_high_withdrawals_create_shortfall(self) -> None:
        self.assertAlmostEqual(liquidity_gap(0.,1.),.8)
        self.assertEqual(liquidity_gap(1.,1.),0.)
        self.assertEqual(liquidity_gap(0.,.1),0.)

    def test_insurance_share_reduces_stress_under_assumptions(self) -> None:
        self.assertGreater(liquidity_gap(.2,.8),liquidity_gap(.8,.8))

    def test_invalid_liquidity_parameters(self) -> None:
        for args in [(-.1,.4),(.3,1.01),(.3,.4,-.1),
                     (.3,.4,.2,math.nan)]:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    liquidity_gap(*args)
        with self.assertRaises(TypeError):
            liquidity_gap(True,.4)

    def test_income_smoothing_exact_experiment(self) -> None:
        status=np.array([1,1,0,0,1])
        np.testing.assert_allclose(
            monthly_disposable_income(status,100.,.35),
            [100.,100.,35.,35.,100.],
        )
        self.assertEqual(
            monthly_disposable_income(status,100.,0.)[2],0.,
        )
        self.assertEqual(
            monthly_disposable_income(status,100.,1.)[2],100.,
        )

    def test_invalid_income_states_and_rates(self) -> None:
        for states in (np.array([]),np.array([0,2]),np.array([[1,0]])):
            with self.subTest(states=repr(states)):
                with self.assertRaises(ValueError):
                    monthly_disposable_income(states,100.,.5)
        with self.assertRaises(ValueError):
            monthly_disposable_income(np.array([1,0]),0.,.5)
        with self.assertRaises(ValueError):
            monthly_disposable_income(np.array([1,0]),100.,1.1)

    def test_realized_relief_is_multiplicative(self) -> None:
        harms=np.array([100.,65.,35.])
        coverage=np.array([.3,.65,.95])
        access=np.array([.6,.75,.9])
        np.testing.assert_allclose(
            realized_relief(harms,coverage,access,.6),
            harms*coverage*access*.6,
        )
        self.assertGreater(
            realized_relief(harms,np.array([.85,.9,.95]),access,.6)[0],
            realized_relief(harms,coverage,access,.6)[0],
        )

    def test_invalid_relief_shapes(self) -> None:
        with self.assertRaises(ValueError):
            realized_relief(np.array([1.,2.]),np.array([.5]),np.array([.5]))
        with self.assertRaises(ValueError):
            realized_relief(np.array([-1.]),np.array([.5]),np.array([.5]))
        with self.assertRaises(ValueError):
            realized_relief(np.array([1.]),np.array([1.2]),np.array([.5]))
        with self.assertRaises(ValueError):
            realized_relief(np.array([1.]),np.array([.5]),np.array([.5]),2.)


class PublicationArtifactTests(unittest.TestCase):
    def test_article_has_eight_referenced_figures_and_length(self) -> None:
        article=(Path(__file__).resolve().parents[1]/"article.md").read_text(
            encoding="utf-8"
        )
        self.assertGreaterEqual(len(article.split()),8000)
        for i,name in enumerate(FIGURE_NAMES,start=1):
            self.assertEqual(article.count(f"](figures/{name})"),1)
            self.assertIn(f"Figure {i} —",article)

    def test_notebook_json_schema_and_code_reuse(self) -> None:
        path=Path(__file__).resolve().parents[1]/"model_walkthrough.ipynb"
        notebook=json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"],4)
        ids=[cell["id"] for cell in notebook["cells"]]
        self.assertEqual(len(ids),len(set(ids)))
        source="".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertIn("from analysis import",source)
        self.assertIn("generate_all(",source)

    def test_all_outputs_generate_with_correct_evidence_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            generate_all(out)
            self.assertEqual(len(FIGURE_NAMES),8)
            for name in FIGURE_NAMES:
                img=out/"figures"/name
                self.assertTrue(img.is_file())
                self.assertGreater(img.stat().st_size,10000)
            for name,expected,source in [
                ("source_based_summary.csv",29,"historical"),
                ("hypothetical_bank_stress.csv",441,"hypothetical"),
                ("hypothetical_income_smoothing.csv",36,"hypothetical"),
            ]:
                with (out/"results"/name).open("r",encoding="utf-8",
                                                newline="") as f:
                    rows=list(csv.DictReader(f))
                self.assertEqual(len(rows),expected)
                self.assertTrue(all(row["evidence_class"]==source
                                    for row in rows))


if __name__=="__main__":
    unittest.main()
