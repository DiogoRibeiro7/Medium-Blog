"""Offline tests for Eurostat accounting and JSON-stat integrity."""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import eurostat_sectors as e


def fixture() -> dict:
    """Invent a small, deliberately reordered JSON-stat cube."""
    dimensions = {
        "geo": ["PT"],
        "freq": ["A"],
        "na_item": ["B1G", "D1"],
        "nace_r2": ["TOTAL", "C"],
        "time": ["2023", "2024"],
        "unit": ["CP_MEUR", "CLV20_MEUR"],
    }
    return {
        "id": list(dimensions),
        "size": [len(codes) for codes in dimensions.values()],
        "dimension": {
            dim: {"category": {"index": {code: i for i, code in enumerate(codes)}}}
            for dim, codes in dimensions.items()
        },
        "value": {"0": 100, "1": 90, "2": 110, "3": 95,
                  "4": 20, "5": 19, "6": None, "8": 999},
        "status": {"2": "p", "6": "e", "7": "b"},
    }


class TestJSONStat(unittest.TestCase):
    def test_reordered_dimensions_and_measure_filter(self) -> None:
        cells = e.decode(json.dumps(fixture()).encode(), "B1G", ("CP_MEUR", "CLV20_MEUR"))
        self.assertEqual(cells["PT", 2024, "TOTAL", "CP_MEUR"], e.Cell(110, "p", "observed"))
        self.assertEqual(cells["PT", 2023, "C", "CP_MEUR"].value, 20)
        self.assertEqual(len(cells), 8)

    def test_null_flagged_absence_and_zero_are_distinct(self) -> None:
        payload = fixture()
        payload["value"]["4"] = 0
        cells = e.decode(json.dumps(payload).encode(), "B1G", ("CP_MEUR", "CLV20_MEUR"))
        self.assertEqual(cells["PT", 2023, "C", "CP_MEUR"], e.Cell(0, "", "observed"))
        self.assertEqual(cells["PT", 2024, "C", "CP_MEUR"], e.Cell(None, "e", "explicit_null"))
        self.assertEqual(cells["PT", 2024, "C", "CLV20_MEUR"], e.Cell(None, "b", "absent"))

    def test_dense_arrays(self) -> None:
        payload = fixture()
        values = [None] * 16
        values[0] = 100
        payload["value"] = values
        cells = e.decode(json.dumps(payload).encode(), "B1G", ("CP_MEUR",))
        self.assertEqual(cells["PT", 2023, "TOTAL", "CP_MEUR"].value, 100)

    def test_noncontiguous_category_positions_fail(self) -> None:
        payload = fixture()
        payload["dimension"]["unit"]["category"]["index"]["CLV20_MEUR"] = 3
        with self.assertRaisesRegex(ValueError, "Noncontiguous"):
            e.decode(json.dumps(payload).encode(), "B1G", ("CP_MEUR",))

    def test_unexpected_measure_or_unit_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "Requested"):
            e.decode(json.dumps(fixture()).encode(), "EMP_DC", ("THS_HW",))

    def test_out_of_bounds_cell_fails(self) -> None:
        payload = fixture()
        payload["value"]["16"] = 5
        with self.assertRaisesRegex(ValueError, "position"):
            e.decode(json.dumps(payload).encode(), "B1G", ("CP_MEUR",))


class TestAccounting(unittest.TestCase):
    def test_partitions_exclude_overlays(self) -> None:
        self.assertNotIn("C", e.A10)
        self.assertNotIn("TOTAL", e.A10)
        self.assertFalse(set(e.NESTED) & set(e.A64))
        flattened = [sector for sectors in e.GROUPS.values() for sector in sectors]
        self.assertCountEqual(flattened, e.A64)

    def test_reconciled_concentration(self) -> None:
        result = e.partition_metrics([25, 25, 50], 100)
        self.assertTrue(result["complete_partition"])
        self.assertAlmostEqual(result["hhi"], 0.375)
        self.assertAlmostEqual(result["effective_sectors"], 1 / 0.375)
        self.assertEqual(result["residual_pct_total"], 0)

    def test_missing_or_unreconciled_partition_is_not_ranked(self) -> None:
        for values, total in (([25, None, 50], 100), ([20, 50], 100),
                              ([-5, 105], 100), ([25, 75], None)):
            with self.subTest(values=values, total=total):
                result = e.partition_metrics(values, total)
                self.assertFalse(result["complete_partition"])
                self.assertIsNone(result["hhi"])

    def test_partial_group_is_not_summed(self) -> None:
        gva = {
            ("PT", 2024, "TOTAL", "CP_MEUR"): e.Cell(100, "", "observed"),
            ("PT", 2024, "G", "CP_MEUR"): e.Cell(20, "", "observed"),
            ("PT", 2024, "H", "CP_MEUR"): e.Cell(5, "", "observed"),
        }
        with patch.object(e, "GEO", ("PT",)), patch.object(e, "YEARS", range(2024, 2025)):
            row = next(r for r in e.group_rows(gva)
                       if r["group"] == "trade_transport_hospitality")
        self.assertEqual(row["observed_sectors"], 2)
        self.assertIsNone(row["nominal_gva_million_eur"])
        self.assertIsNone(row["gva_share_pct"])

    def test_relative_deflator_identity(self) -> None:
        result = e.share_decomposition(10, 30, 10, 15, 100, 200, 100, 125)
        self.assertAlmostEqual(result["nominal_share_change_log_points"], e.log_change(0.1, 0.15))
        self.assertAlmostEqual(result["relative_volume_change_log_points"], e.log_change(1, 1.2))
        self.assertAlmostEqual(result["identity_residual_log_points"], 0)
        missing = e.share_decomposition(0, 30, 10, 15, 100, 200, 100, 125)
        self.assertIsNone(missing["relative_deflator_change_log_points"])

    def test_hours_units_and_real_productivity(self) -> None:
        gva = {
            ("PT", 2024, "TOTAL", "CP_MEUR"): e.Cell(100, "", "observed"),
            ("PT", 2024, "C", "CP_MEUR"): e.Cell(20, "", "observed"),
            ("PT", 2024, "C", "CLV20_MEUR"): e.Cell(20, "p", "observed"),
        }
        hours = {("PT", 2024, "C", "THS_HW"): e.Cell(1000, "", "observed")}
        with patch.object(e, "GEO", ("PT",)), patch.object(e, "YEARS", range(2024, 2025)):
            row = next(r for r in e.section_rows("a10", gva, hours) if r["sector"] == "C")
        self.assertEqual(row["gva_share_pct"], 20)
        self.assertEqual(row["real_gva_per_hour_eur"], 20)
        self.assertEqual(row["real_flag"], "p")
        self.assertFalse(row["partition_member"])


if __name__ == "__main__":
    unittest.main()
