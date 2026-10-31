"""Historical transcription checks, mathematical contracts and end-to-end output."""

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
    Implementation,
    detection_probability,
    effective_access,
    generate_all,
    group_benefits,
    regulatory_reach,
    tax_data,
    union_data,
    urban_data,
)


class HistoricalEvidenceTests(unittest.TestCase):
    def test_urban_series_uses_contemporary_census_definition(self) -> None:
        d = urban_data()
        self.assertEqual(d["year"].astype(int).tolist(), [1900, 1910, 1920])
        np.testing.assert_array_equal(d["urban_percent"], [40.0, 45.8, 51.4])
        self.assertEqual(d["urban_population"][2], 54_804_003)
        self.assertAlmostEqual(
            100 * d["urban_population"][0]
            / (d["urban_population"][0] + d["rural_population"][0]),
            40.0, delta=0.06
        )

    def test_union_counts_match_wolman_table38(self) -> None:
        d = union_data()
        years = d["year"].astype(int).tolist()
        self.assertEqual(d["members"][years.index(1900)], 868_500)
        self.assertEqual(d["members"][years.index(1915)], 2_607_700)
        self.assertEqual(d["members"][years.index(1920)], 5_110_800)
        self.assertEqual(len(years), 9)
        self.assertTrue(np.all(d["members"] > 0))

    def test_tax_rates_do_not_conflate_marginal_and_effective(self) -> None:
        d = tax_data()
        years = d["year"].astype(int).tolist()
        for year, expected in [(1913, 7), (1916, 15), (1917, 67),
                               (1918, 77), (1920, 73)]:
            self.assertEqual(d["highest_bracket_percent"][years.index(year)], expected)
        self.assertEqual(d["highest_bracket_threshold_usd"][years.index(1918)], 1e6)
        self.assertTrue(np.all(
            d["highest_bracket_percent"] > d["lowest_bracket_percent"]
        ))


class HypotheticalModelTests(unittest.TestCase):
    def test_serial_implementation_chain(self) -> None:
        chain = Implementation(.9, .7, .6, .8)
        self.assertAlmostEqual(chain.realized_fraction, .3024)
        self.assertAlmostEqual(Implementation(1., 1., 1., 1.).realized_fraction, 1.)
        self.assertEqual(Implementation(1., 0., 1., 1.).realized_fraction, 0.)

    def test_implementation_rejects_bad_inputs(self) -> None:
        for invalid in (-.01, 1.01, math.nan, math.inf):
            with self.subTest(value=invalid):
                with self.assertRaises(ValueError):
                    Implementation(invalid, .5, .5, .5)
        with self.assertRaises(TypeError):
            Implementation(True, .5, .5, .5)

    def test_formal_equal_access_can_diverge(self) -> None:
        self.assertAlmostEqual(effective_access(1, .95, .95), .9025)
        self.assertAlmostEqual(effective_access(1, .55, .45), .2475)

    def test_monitoring_formula_and_limits(self) -> None:
        self.assertEqual(detection_probability(25, 0), 0)
        self.assertAlmostEqual(detection_probability(0, .5), 0)
        self.assertAlmostEqual(detection_probability(50, 1),
                               1-math.exp(-.5))
        self.assertAlmostEqual(regulatory_reach(50, 1),
                               detection_probability(50, 1))
        self.assertLessEqual(regulatory_reach(25, .8), .8)

    def test_detection_decreases_with_broader_coverage_at_fixed_budget(self) -> None:
        self.assertGreater(
            detection_probability(25, .2), detection_probability(25, 1.)
        )
        self.assertGreater(
            regulatory_reach(25, 1.), regulatory_reach(25, .2)
        )

    def test_invalid_monitoring_arguments(self) -> None:
        for args in [(-1., .5), (1., -1), (1., 2.),
                     (math.inf, .5), (1., .5, 0)]:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    detection_probability(*args)

    def test_group_benefits_ordering_and_exact_values(self) -> None:
        h = np.array([100., 65., 30.])
        access = np.array([.35, .65, .9])
        np.testing.assert_allclose(
            group_benefits(h, access, .25), [8.75, 10.5625, 6.75]
        )
        np.testing.assert_allclose(
            group_benefits(h, access, 0.), [0., 0., 0.]
        )
        self.assertGreater(
            group_benefits(h, np.array([.85, .85, .9]), .75)[0],
            group_benefits(h, access, .75)[0]
        )

    def test_invalid_group_shapes_and_bounds(self) -> None:
        with self.assertRaises(ValueError):
            group_benefits(np.array([10., 20.]), np.array([.5]), .5)
        with self.assertRaises(ValueError):
            group_benefits(np.array([-1.]), np.array([.5]), .5)
        with self.assertRaises(ValueError):
            group_benefits(np.array([10.]), np.array([1.1]), .5)
        with self.assertRaises(ValueError):
            group_benefits(np.array([10.]), np.array([.5]), 1.1)


class ArtifactTests(unittest.TestCase):
    def test_article_links_all_eight_figures_once(self) -> None:
        article = (Path(__file__).resolve().parents[1] / "article.md").read_text(
            encoding="utf-8"
        )
        for i, name in enumerate(FIGURE_NAMES, start=1):
            self.assertEqual(article.count(f"]" + f"(figures/{name})"), 1)
            self.assertIn(f"Figure {i} —", article)
        self.assertGreaterEqual(len(article.split()), 8_000)

    def test_notebook_is_valid_json_with_same_analysis_import(self) -> None:
        notebook = json.loads(
            (Path(__file__).resolve().parents[1] / "model_walkthrough.ipynb").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(notebook["nbformat"], 4)
        source = "".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn("from analysis import", source)
        self.assertIn("generate_all(", source)

    def test_regeneration_creates_all_named_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            generate_all(target)
            for name in FIGURE_NAMES:
                with self.subTest(name=name):
                    figure = target / "figures" / name
                    self.assertTrue(figure.is_file())
                    self.assertGreater(figure.stat().st_size, 10_000)
            rows_expected = {
                "source_based_summary.csv": 20,
                "hypothetical_capacity.csv": 288,
                "hypothetical_incidence.csv": 9,
            }
            for name, count in rows_expected.items():
                with (target / "results" / name).open(
                    newline="", encoding="utf-8"
                ) as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), count)
                if name.startswith("hypothetical"):
                    self.assertTrue(
                        all(row["evidence_class"] == "hypothetical" for row in rows)
                    )
                else:
                    self.assertTrue(
                        all(row["evidence_class"] == "historical_transcription"
                            for row in rows)
                    )


if __name__ == "__main__":
    unittest.main()
