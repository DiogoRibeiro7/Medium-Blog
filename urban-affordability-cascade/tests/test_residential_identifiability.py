"""Partial-identification tests: never treat Census margins as directed OD."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from residential_identifiability import (  # noqa: E402
    FIGURE,
    RESULT_CSV,
    RESULT_JSON,
    DestinationMargins,
    allocate,
    contrasting_worlds,
    load_margins,
    run,
)


class PartialIdentificationTests(unittest.TestCase):
    """Mathematical bounds on possible *within-AMP* residential exchange."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.margins = load_margins()

    def test_official_census_nested_destination_counts(self) -> None:
        self.assertEqual(self.margins.porto, 15_724)
        self.assertEqual(self.margins.amp, 124_618)
        self.assertEqual(self.margins.rest_amp, 108_894)

    def test_sharp_net_interval_contains_both_signs(self) -> None:
        lower, upper = self.margins.net_outward_bounds
        self.assertEqual((lower, upper), (-15_724, 108_894))
        self.assertLess(lower, 0)
        self.assertGreater(upper, 0)

    def test_bounds_are_attained_by_constructible_worlds(self) -> None:
        smallest = allocate(self.margins, scenario="minimum", porto_to_rest_amp=0,
                            rest_amp_to_porto=self.margins.porto)
        largest = allocate(self.margins, scenario="maximum",
                           porto_to_rest_amp=self.margins.rest_amp, rest_amp_to_porto=0)
        self.assertEqual(smallest.net_porto_to_rest_amp, -self.margins.porto)
        self.assertEqual(largest.net_porto_to_rest_amp, self.margins.rest_amp)
        self.assertEqual(smallest.porto_inbound_from_elsewhere, 0)
        self.assertEqual(largest.rest_amp_inbound_from_non_porto, 0)

    def test_opposite_net_signs_reconcile_identical_observed_margins(self) -> None:
        worlds = contrasting_worlds(self.margins)
        self.assertEqual(len(worlds), 5)
        self.assertEqual(worlds[0].net_porto_to_rest_amp, 0)
        self.assertEqual(worlds[1].net_porto_to_rest_amp, 20_000)
        self.assertEqual(worlds[2].net_porto_to_rest_amp, -7_000)
        for world in worlds:
            self.assertEqual(world.porto_inbound_from_elsewhere + world.rest_amp_to_porto,
                             self.margins.porto)
            self.assertEqual(world.rest_amp_inbound_from_non_porto + world.porto_to_rest_amp,
                             self.margins.rest_amp)
            self.assertEqual(world.evidence_class, "hypothetical_feasible_allocation_not_observed")

    def test_any_integer_within_rectangle_is_feasible(self) -> None:
        for x in (0, 1, self.margins.rest_amp // 2, self.margins.rest_amp):
            for y in (0, 1, self.margins.porto // 2, self.margins.porto):
                world = allocate(self.margins, scenario="grid", porto_to_rest_amp=x,
                                 rest_amp_to_porto=y)
                self.assertEqual(world.net_porto_to_rest_amp, x - y)

    def test_reject_invalid_nested_margins(self) -> None:
        with self.assertRaises(TypeError):
            DestinationMargins(True, 10)
        with self.assertRaises(TypeError):
            DestinationMargins(5.0, 10)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            DestinationMargins(-1, 10)
        with self.assertRaises(ValueError):
            DestinationMargins(11, 10)

    def test_reject_noninteger_or_outside_rectangle_flows(self) -> None:
        for x, y in ((-1, 0), (self.margins.rest_amp + 1, 0),
                     (0, -1), (0, self.margins.porto + 1)):
            with self.assertRaises(ValueError):
                allocate(self.margins, scenario="invalid", porto_to_rest_amp=x,
                         rest_amp_to_porto=y)
        with self.assertRaises(TypeError):
            allocate(self.margins, scenario="invalid", porto_to_rest_amp=True,
                     rest_amp_to_porto=0)
        with self.assertRaises(ValueError):
            allocate(self.margins, scenario="", porto_to_rest_amp=0,
                     rest_amp_to_porto=0)

    def test_no_false_inference_from_symmetric_or_zero_margins(self) -> None:
        other = DestinationMargins(porto=50, amp=100)
        self.assertEqual(other.net_outward_bounds, (-50, 50))
        self.assertEqual(DestinationMargins(porto=0, amp=0).net_outward_bounds, (0, 0))
        self.assertEqual(DestinationMargins(porto=0, amp=5).net_outward_bounds, (0, 5))

    def test_outputs_are_deterministic_and_explicitly_hypothetical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            result = run(dest)
            expected = {"porto_to_other_amp": [0, 108894],
                        "other_amp_to_porto": [0, 15724],
                        "net_outward_porto_to_other_amp": [-15724, 108894]}
            self.assertEqual(result["sharp_bounds_given_these_margins_only"], expected)
            self.assertIn("NOT statistical confidence intervals", result["interpretation"])
            self.assertIn("Either sign", result["why_unidentified"])
            plot = dest / "figures" / FIGURE
            self.assertGreater(plot.stat().st_size, 10_000)
            data_path = dest / "results" / RESULT_JSON
            saved = json.loads(data_path.read_text(encoding="utf-8"))
            self.assertEqual(saved, result)
            with (dest / "results" / RESULT_CSV).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(r["evidence_class"].startswith("hypothetical_") for r in rows))
            csv_bytes = (dest / "results" / RESULT_CSV).read_bytes()
            json_bytes = data_path.read_bytes()
            run(dest)
            self.assertEqual((dest / "results" / RESULT_CSV).read_bytes(), csv_bytes)
            self.assertEqual(data_path.read_bytes(), json_bytes)


if __name__ == "__main__":
    unittest.main()
