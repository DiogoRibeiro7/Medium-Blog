"""Tests for the pure rental-market functions."""

import unittest
from typing import Any

import rental_core as C


def _doc() -> dict[str, Any]:
    # Dimensions freq (1) x tenure (2) x geo (1) x time (3); value positions are row-major.
    return {"id": ["freq", "tenure", "geo", "time"], "size": [1, 2, 1, 3],
            "dimension": {"freq": {"category": {"index": {"A": 0}}},
                          "tenure": {"category": {"index": {"OWN": 0, "RENT": 1}}},
                          "geo": {"category": {"index": {"PT": 0}}},
                          "time": {"category": {"index": {"2019": 0, "2020": 1, "2021": 2}}}},
            "value": {"0": 70.0, "1": 71.0, "3": 30.0, "4": 29.0, "5": 28.5},
            "status": {"5": "b"}}


class JsonStatTests(unittest.TestCase):
    def test_selects_one_slice_and_keeps_flags(self) -> None:
        values, flags = C.jsonstat_select(_doc(), {"tenure": "RENT"})
        self.assertEqual(values, {"2019": 30.0, "2020": 29.0, "2021": 28.5})
        self.assertEqual(flags, {"2021": "b"})
        own, _ = C.jsonstat_select(_doc(), {"tenure": "OWN", "geo": "PT"})
        self.assertEqual(own, {"2019": 70.0, "2020": 71.0})

    def test_selects_along_another_dimension(self) -> None:
        values, _ = C.jsonstat_select(_doc(), {"time": "2020"}, over="tenure")
        self.assertEqual(values, {"OWN": 71.0, "RENT": 29.0})

    def test_refuses_ambiguous_or_unknown_selections(self) -> None:
        with self.assertRaises(ValueError):
            C.jsonstat_select(_doc(), {})
        with self.assertRaises(ValueError):
            C.jsonstat_select(_doc(), {"tenure": "RENT_MKT"})
        with self.assertRaises(ValueError):
            C.jsonstat_select(_doc(), {"tenure": "RENT", "sex": "T"})

    def test_list_index_and_list_values(self) -> None:
        doc = _doc()
        doc["dimension"]["time"]["category"]["index"] = ["2019", "2020", "2021"]
        doc["value"] = [70.0, 71.0, None, 30.0, 29.0, 28.5]
        self.assertEqual(C.jsonstat_select(doc, {"tenure": "OWN"})[0], {"2019": 70.0, "2020": 71.0})


class IndexTests(unittest.TestCase):
    def test_years_rebase_relative(self) -> None:
        self.assertEqual(C.years({"2021": 1.0, "2019": 2.0}), {2019: 2.0, 2021: 1.0})
        with self.assertRaises(ValueError):
            C.years({"2019Q4": 1.0})
        self.assertEqual(C.rebase({2019: 50.0, 2020: 75.0}, 2019), {2019: 100.0, 2020: 150.0})
        with self.assertRaises(ValueError):
            C.rebase({2020: 1.0}, 2019)
        rel = C.relative_index({2019: 10.0, 2020: 12.0, 2021: 15.0}, {2019: 100.0, 2020: 100.0}, 2019)
        self.assertEqual(rel, {2019: 100.0, 2020: 120.0})


class ShiftShareTests(unittest.TestCase):
    def test_parts_add_up_and_isolate_composition(self) -> None:
        shares_a, shares_b = {"own": 80.0, "rent": 20.0}, {"own": 70.0, "rent": 30.0}
        rates = {"own": 5.0, "rent": 25.0}
        out = C.shift_share(shares_a, rates, shares_b, rates)
        self.assertAlmostEqual(out["implied_a"], 9.0)
        self.assertAlmostEqual(out["implied_b"], 11.0)
        self.assertAlmostEqual(out["within_groups"], 0.0)
        self.assertAlmostEqual(out["composition"], 2.0)
        mixed = C.shift_share(shares_a, rates, shares_b, {"own": 4.0, "rent": 30.0})
        self.assertAlmostEqual(mixed["within_groups"] + mixed["composition"], mixed["change"])
        with self.assertRaises(ValueError):
            C.shift_share(shares_a, rates, {"own": 100.0}, rates)

    def test_mean(self) -> None:
        self.assertEqual(C.mean([1.0, 2.0, 6.0]), 3.0)
        with self.assertRaises(ValueError):
            C.mean([])


if __name__ == "__main__":
    unittest.main()
