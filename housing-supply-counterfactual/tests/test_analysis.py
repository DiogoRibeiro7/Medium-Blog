"""Offline regression tests for the frozen descriptive article companion."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import (
    COUNTRIES,
    FIGURES,
    counterfactual_price_growth_pct,
    load_inputs,
    run,
)


class FrozenSourceTests(unittest.TestCase):
    """Protect the observed source anchors and conceptual definitions."""

    def test_bulgaria_chart_and_flow_scale(self) -> None:
        inputs = load_inputs()
        self.assertEqual(
            inputs.bulgaria.completed_dwellings.tolist(),
            [17868, 19847, 22649, 20998, 37571],
        )
        self.assertAlmostEqual(100 * 37571 / 4_360_805, 0.86158, places=4)
        self.assertAlmostEqual(100 * (37571 / 17868 - 1), 110.27, delta=0.05)

    def test_portugal_census_and_completions(self) -> None:
        inputs = load_inputs()
        self.assertEqual(int(inputs.portugal_occupancy.dwellings.sum()), 5_970_677)
        self.assertEqual(
            int(
                inputs.portugal_builds.loc[
                    inputs.portugal_builds.year == 2025, "completed_new_dwellings"
                ].iloc[0]
            ),
            27_301,
        )

    def test_descriptive_panel_scope(self) -> None:
        inputs = load_inputs()
        self.assertEqual(set(COUNTRIES), {"PT", "ES", "FR", "DE"})
        self.assertEqual(len(inputs.permits), 55)
        self.assertEqual(len(inputs.annual), 55)
        self.assertEqual(len(inputs.associations), 15)
        pt = inputs.associations.loc[
            (inputs.associations.geo == "PT")
            & (inputs.associations.permit_lead_years == 0)
        ].iloc[0]
        self.assertEqual(int(pt.observed_pairs), 10)
        self.assertEqual(int(pt.unflagged_pairs), 5)
        self.assertAlmostEqual(float(pt.pearson_r), 0.24897236124827316)
        self.assertAlmostEqual(float(pt.unflagged_pearson_r), 0.8908573013622829)


class ScenarioTests(unittest.TestCase):
    """Scenarios are mathematical demonstrations, not fitted estimates."""

    def test_extra_supply_moderates_fixed_demand_shock(self) -> None:
        without = counterfactual_price_growth_pct(0.12, 0.0, 1.5)
        with_supply = counterfactual_price_growth_pct(0.12, math.log1p(0.009), 1.5)
        self.assertGreater(without, with_supply)
        self.assertGreater(with_supply, 0.0)

    def test_parameter_validation(self) -> None:
        for args in ((0.12, -0.01, 1.5), (0.12, 0.01, -1.0), (float("nan"), 0.1, 1.5)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                counterfactual_price_growth_pct(*args)

    def test_full_offline_reproduction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            summary = run(folder)
            self.assertFalse(summary["causal_effect_estimated"])
            generated_files: object = summary["figure_files"]
            self.assertIsInstance(generated_files, list)
            if not isinstance(generated_files, list):
                self.fail("Figure file names must be stored as a list")
            self.assertEqual(len(generated_files), 10)
            self.assertEqual(len(FIGURES), 10)
            for filename, _ in FIGURES:
                self.assertTrue((folder / filename).is_file())
                self.assertGreater((folder / filename).stat().st_size, 20_000)
            parsed = json.loads((folder / "results.json").read_text(encoding="utf-8"))
            self.assertIn("source_sha256", parsed)
            self.assertIn(
                "Bulgarian HPI data-source break beginning Q1 2022",
                parsed["limitations"],
            )


if __name__ == "__main__":
    unittest.main()
