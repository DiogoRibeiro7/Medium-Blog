"""Mathematical, provenance and observed-data regression tests."""

from __future__ import annotations

import math
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import (  # noqa: E402
    INPUT_PANEL,
    MUNICIPALITIES,
    SOURCE_BLOB_SHA,
    commute_counterfactual,
    displacement_chain,
    load_panel,
    observed_summary,
    percent_change,
    require_float,
    run,
)


class ObservedPanelTests(unittest.TestCase):
    """Assert the exact municipality/time grid and no invented source cells."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.panel = load_panel()

    def test_panel_grid(self) -> None:
        self.assertEqual(len(self.panel), 119)
        self.assertEqual(len(MUNICIPALITIES), 17)
        self.assertTrue(all((m, y) in self.panel for m in MUNICIPALITIES for y in range(2019, 2026)))

    def test_missing_values_are_not_imputed(self) -> None:
        self.assertTrue(all(self.panel[m, 2019].rent_m2 is None for m in MUNICIPALITIES))
        self.assertTrue(all(self.panel[m, 2025].income is None for m in MUNICIPALITIES))
        with self.assertRaisesRegex(ValueError, "Missing value"):
            require_float(None, "missing")

    def test_reference_porto_numbers_from_upstream(self) -> None:
        self.assertEqual(self.panel["Porto", 2019].price_m2, 1856)
        self.assertEqual(self.panel["Porto", 2025].price_m2, 3347)
        self.assertEqual(self.panel["Porto", 2020].rent_m2, 8.97)
        self.assertEqual(self.panel["Porto", 2025].rent_m2, 14.13)
        self.assertEqual(SOURCE_BLOB_SHA, "9bade430163315e7c5da323251daa0715e1ffede")
        self.assertEqual(
            hashlib.sha256(INPUT_PANEL.read_bytes()).hexdigest(),
            "88502e15b55a4d46536eed42ddc1924d30f555498d2c4a7e7a1b41fbeed71c66",
        )

    def test_growth_is_source_derived(self) -> None:
        rows = observed_summary(self.panel)
        porto = next(row for row in rows if row["municipality"] == "Porto")
        self.assertAlmostEqual(porto["sale_price_growth_2019_2025_pct"], 100 * (3347 / 1856 - 1))
        self.assertAlmostEqual(porto["new_lease_rent_growth_2020_2025_pct"], 100 * (14.13 / 8.97 - 1))

    def test_observed_price_and_rent_endpoints_increase_for_all_17(self) -> None:
        for row in observed_summary(self.panel):
            self.assertGreater(row["sale_price_growth_2019_2025_pct"], 0)
            self.assertGreater(row["new_lease_rent_growth_2020_2025_pct"], 0)

    def test_reject_empty_or_duplicate_rows(self) -> None:
        contents = INPUT_PANEL.read_text(encoding="utf-8")
        lines = contents.splitlines()
        with tempfile.TemporaryDirectory() as directory:
            filepath = Path(directory) / "broken.csv"
            filepath.write_text("\n".join(lines + [lines[1]]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate municipality-year"):
                load_panel(filepath)

    def test_zero_growth_base_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            percent_change(0, 100)


class HypotheticalModelTests(unittest.TestCase):
    """Check the accounting, boundary cases and sensitivity of illustrations."""

    def test_conservation(self) -> None:
        for parameters in ([0, 0, 0], [0.2, 0.2, 0.25], [0.55, 0.6, 0.75], [1]):
            stages = displacement_chain(1000, parameters)
            self.assertAlmostEqual(sum(s.absorbed for s in stages) + stages[-1].onward, 1000)
            for stage in stages:
                self.assertAlmostEqual(stage.incoming, stage.absorbed + stage.onward)

    def test_full_absorption_ends_chain(self) -> None:
        stages = displacement_chain(1000, [1.0, 0.2])
        self.assertEqual(stages[0].onward, 0)
        self.assertEqual(stages[1].incoming, 0)

    def test_more_absorption_reduces_remaining(self) -> None:
        low = displacement_chain(1000, [0.2, 0.2, 0.25])[-1].onward
        high = displacement_chain(1000, [0.55, 0.6, 0.75])[-1].onward
        self.assertLess(high, low)
        self.assertAlmostEqual(low, 480)
        self.assertAlmostEqual(high, 45)

    def test_invalid_parameters_are_rejected(self) -> None:
        for invalid in (-0.1, 1.1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                displacement_chain(1000, [invalid])
        with self.assertRaises(ValueError):
            displacement_chain(-1, [0.3])
        with self.assertRaises(ValueError):
            displacement_chain(1, [])

    def test_commuting_cost_accounting(self) -> None:
        for k in (0, 10, 35, 60):
            values = commute_counterfactual(k)
            self.assertAlmostEqual(values["illustrative_total_eur_month"], sum(
                values[key] for key in (
                    "illustrative_rent_eur_month", "illustrative_transport_eur_month",
                    "illustrative_time_value_eur_month",
                )
            ))
        self.assertGreater(commute_counterfactual(60)["illustrative_total_eur_month"],
                           commute_counterfactual(25)["illustrative_total_eur_month"])

    def test_outputs_have_exact_figure_count_and_explicit_classification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            metadata = run(output)
            self.assertFalse(metadata["relocation_flow_data_present"])
            self.assertFalse(metadata["causal_effect_estimated"])
            self.assertEqual(len(list((output / "figures").glob("fig*.png"))), 8)
            self.assertTrue((output / "results" / "amp_growth_summary.csv").is_file())
            self.assertTrue((output / "results" / "hypothetical_cascade.csv").is_file())


if __name__ == "__main__":
    unittest.main()
