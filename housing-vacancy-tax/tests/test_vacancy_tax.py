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
    audit_sample_size,
    break_even_additional_leases,
    census_summary,
    cost_per_additional_lease_eur,
    detection_probability_matching_rate_cut,
    expected_surcharge_eur,
    hypothetical_released_dwellings,
    imi_liability_eur,
    load_census,
    load_identification,
    load_rate_cut_cost,
    load_vancouver,
    max_option_value_consistent_with_letting_eur,
    net_rental_flow_before_imi_eur,
    rate_cut_gain_upper_bound_eur,
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
            for name in ("census_2021_ihru_table11.csv", "municipal_identification.csv",
                         "vancouver_eht.csv", "rate_cut_utao.csv"):
                (root / "data" / name).write_bytes((ROOT / "data" / name).read_bytes())
            first = create_results(root)
            first_bytes = (root / "results" / "summary.json").read_bytes()
            second = create_results(root)
            self.assertEqual(first, second)
            self.assertEqual(first_bytes, (root / "results" / "summary.json").read_bytes())
            self.assertEqual(json.loads(first_bytes)["census_2021"]["total_vacant"], 723_215)

    def test_identification_and_scale_ratio(self) -> None:
        flagged = load_identification(ROOT / "data" / "municipal_identification.csv")
        self.assertEqual(flagged.vacant_total, 12_776)
        self.assertAlmostEqual(flagged.vacant_total / 248_534, 0.0514, places=4)
        self.assertAlmostEqual(flagged.vacant_total / 723_215, 0.0177, places=4)

    def test_vancouver_counts_match_rates(self) -> None:
        years = load_vancouver(ROOT / "data" / "vancouver_eht.csv")
        self.assertEqual(years[-1].vacant_properties, 979)
        self.assertEqual(str(years[0].declared_vacancy_rate_pct), "1.18")

    def test_rate_cut_and_expected_surcharge(self) -> None:
        scenario = Scenario()
        self.assertEqual(rate_cut_gain_upper_bound_eur(scenario), Decimal("1458.00"))
        self.assertEqual(expected_surcharge_eur(scenario, Decimal(10), Decimal("0.5")), Decimal("2362.50"))
        self.assertEqual(
            max_option_value_consistent_with_letting_eur(scenario, Decimal(10), Decimal("0.1"), rate_cut=True),
            Decimal("5604") + Decimal("1458") + Decimal("472.5"),
        )
        self.assertAlmostEqual(float(detection_probability_matching_rate_cut(scenario, Decimal(10))), 0.3086, places=4)
        self.assertGreater(detection_probability_matching_rate_cut(scenario, Decimal(3)), 1)
        with self.assertRaises(ValueError):
            expected_surcharge_eur(scenario, Decimal(10), Decimal("1.2"))

    def test_cost_per_additional_lease(self) -> None:
        costs = load_rate_cut_cost(ROOT / "data" / "rate_cut_utao.csv")
        irs = costs["irs_autonomous_rate_25_to_10"]
        self.assertEqual(cost_per_additional_lease_eur(irs, 10_000), Decimal(27_910))
        self.assertEqual(round(break_even_additional_leases(irs, Decimal(10_800))), 25_843)
        with self.assertRaises(ValueError):
            cost_per_additional_lease_eur(irs, 0)

    def test_audit_sample_size(self) -> None:
        self.assertEqual(audit_sample_size(248_534, 0.02), 2_378)
        self.assertEqual(audit_sample_size(248_534, 0.05), 384)
        with self.assertRaises(ValueError):
            audit_sample_size(0, 0.02)


if __name__ == "__main__":
    unittest.main()
