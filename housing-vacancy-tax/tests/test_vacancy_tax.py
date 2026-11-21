"""Standard-library regression and edge-case tests for reproducible calculations."""

from __future__ import annotations

import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from run import create_results
from vacancy_tax import (
    Scenario,
    census_summary,
    hypothetical_released_dwellings,
    imi_liability_eur,
    load_census,
    max_option_value_consistent_with_letting_eur,
    net_rental_flow_before_imi_eur,
)

ROOT = Path(__file__).resolve().parents[1]


class VacancyTaxTests(unittest.TestCase):
    """Check source reconciliation, tax arithmetic, and assumptions."""

    def test_census_table_reconciles(self) -> None:
        census = census_summary(load_census(ROOT / "data" / "census_2021_ihru_table11.csv"))
        self.assertEqual(census["total_classic_dwellings"], 5_970_677)
        self.assertEqual(census["total_vacant"], 723_215)
        self.assertEqual(census["vacant_for_sale_or_rent"], 348_097)
        self.assertEqual(census["vacant_for_other_reasons"], 375_118)
        self.assertEqual(census["vacant_no_or_minor_repairs"], 485_461)
        self.assertEqual(census["vacant_other_reasons_no_or_minor_repairs"], 248_534)

    def test_imi_tax_scenarios(self) -> None:
        scenario = Scenario()
        self.assertEqual(imi_liability_eur(scenario, Decimal(1)), Decimal(525))
        self.assertEqual(imi_liability_eur(scenario, Decimal(3)), Decimal(1575))
        self.assertEqual(imi_liability_eur(scenario, Decimal(10)), Decimal(5250))
        self.assertEqual(imi_liability_eur(scenario, Decimal(20)), Decimal(10500))
        with self.assertRaises(ValueError):
            imi_liability_eur(scenario, Decimal(0))

    def test_illustrative_owner_thresholds(self) -> None:
        scenario = Scenario()
        self.assertEqual(net_rental_flow_before_imi_eur(scenario), Decimal(5604))
        self.assertEqual(max_option_value_consistent_with_letting_eur(scenario, Decimal(1)), Decimal(5604))
        self.assertEqual(max_option_value_consistent_with_letting_eur(scenario, Decimal(3)), Decimal(6654))
        self.assertEqual(max_option_value_consistent_with_letting_eur(scenario, Decimal(10)), Decimal(10329))
        self.assertEqual(max_option_value_consistent_with_letting_eur(scenario, Decimal(20)), Decimal(15579))

    def test_sensitivity_is_not_forecast(self) -> None:
        self.assertEqual(hypothetical_released_dwellings(248_534, Decimal("0.25"), Decimal("0.20")), Decimal("12426.7000"))
        with self.assertRaises(ValueError):
            hypothetical_released_dwellings(248_534, Decimal("1.1"), Decimal("0.20"))

    def test_errors_on_invalid_census(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wrong.csv"
            path.write_text("market_status,repair_status,dwellings\nfor_sale_or_rent,no_or_minor_repairs,999999\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_census(path)

    def test_deterministic_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "data" / "census_2021_ihru_table11.csv").write_bytes(
                (ROOT / "data" / "census_2021_ihru_table11.csv").read_bytes()
            )
            first = create_results(root)
            first_bytes = (root / "results" / "summary.json").read_bytes()
            second = create_results(root)
            self.assertEqual(first, second)
            self.assertEqual(first_bytes, (root / "results" / "summary.json").read_bytes())
            self.assertEqual(json.loads(first_bytes)["observation_year"], 2021)


if __name__ == "__main__":
    unittest.main()
