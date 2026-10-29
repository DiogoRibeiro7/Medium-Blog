"""Offline regression checks for the Eurostat parser and descriptive statistics."""

import json
import unittest
from pathlib import Path

from analysis import (
    parse_hpi,
    parse_housing_overburden,
    parse_income,
    parse_leaving_age,
    parse_parental_coresidence,
    parse_tenure,
    relative_affordability,
    summarize,
)


class AnalysisTests(unittest.TestCase):
    """Validate observable source values and indexing invariants."""

    def setUp(self) -> None:
        path = Path(__file__).parent / "data" / "eurostat_prc_hpi_a_PT_2026-10-29.json"
        self.payload = json.loads(path.read_text(encoding="utf-8"))

    def test_expected_values(self) -> None:
        series = parse_hpi(self.payload)
        self.assertEqual(series[2015], 100.0)
        self.assertEqual(series[2025], 263.85)
        self.assertEqual(summarize(series)["owner_gap_percentage_points"], -19)

    def test_rejects_other_geography(self) -> None:
        self.payload["dimension"]["geo"]["category"]["index"] = {"ES": 0}
        with self.assertRaises(ValueError):
            parse_hpi(self.payload)

    def test_price_income_index(self) -> None:
        path = Path(__file__).parent / "data" / "eurostat_ilc_di03_PT_2026-10-29.json"
        income = parse_income(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(income[2015], 8435.0)
        self.assertEqual(income[2025], 14465.0)
        index = relative_affordability(parse_hpi(self.payload), income)
        self.assertEqual(index[2015], 100.0)
        self.assertGreater(index[2025], 150)

    def test_income_validation(self) -> None:
        path = Path(__file__).parent / "data" / "eurostat_ilc_di03_PT_2026-10-29.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["dimension"]["statinfo"]["category"]["index"] = {"MEAN_EI": 0}
        with self.assertRaises(ValueError):
            parse_income(raw)

    def test_age_specific_income(self) -> None:
        hpi = parse_hpi(self.payload)
        for age in ("Y18-24", "Y25-49"):
            path = Path(__file__).parent / "data" / f"eurostat_ilc_di03_PT_{age}_2026-10-29.json"
            income = parse_income(json.loads(path.read_text(encoding="utf-8")), age=age)
            self.assertGreater(income[2025], income[2015])
            self.assertEqual(relative_affordability(hpi, income)[2015], 100.0)
        with self.assertRaises(ValueError):
            parse_income({}, age="Y25-34")

    def test_age_source_mismatch_rejected(self) -> None:
        path = Path(__file__).parent / "data" / "eurostat_ilc_di03_PT_Y18-24_2026-10-29.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            parse_income(payload, age="Y25-49")

    def test_official_housing_cost_overburden(self) -> None:
        path = Path(__file__).parent / "data" / "eurostat_ilc_lvho07a_PT_2026-10-29.json"
        rates = parse_housing_overburden(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(rates["TOTAL"][2015], 9.1)
        self.assertEqual(rates["Y18-24"][2025], 6.6)
        self.assertEqual(rates["Y25-29"][2025], 8.4)

    def test_rejects_invalid_overburden_rate(self) -> None:
        path = Path(__file__).parent / "data" / "eurostat_ilc_lvho07a_PT_2026-10-29.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["value"]["0"] = 101
        with self.assertRaises(ValueError):
            parse_housing_overburden(payload)

    def test_parental_coresidence_official_values(self) -> None:
        path = Path(__file__).parent / "data" / "eurostat_ilc_lvps08_PT_2026-10-29.json"
        values = parse_parental_coresidence(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(values["Y18-34"][2015], 62.9)
        self.assertEqual(values["Y18-34"][2020], 69.8)
        self.assertEqual(values["Y18-34"][2025], 61.0)

    def test_departure_age_official_values(self) -> None:
        path = Path(__file__).parent / "data" / "eurostat_yth_demo_030_PT_2026-10-29.json"
        values = parse_leaving_age(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(values[2015], 28.9)
        self.assertEqual(values[2025], 28.8)

    def test_rejects_other_country_in_co_residence(self) -> None:
        path = Path(__file__).parent / "data" / "eurostat_ilc_lvps08_PT_2026-10-29.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["dimension"]["geo"]["category"]["index"] = {"ES": 0}
        with self.assertRaises(ValueError):
            parse_parental_coresidence(payload)

    def test_tenure_distribution_official_values(self) -> None:
        path = Path(__file__).parent / "data" / "eurostat_ilc_lvho02_PT_2026-10-29.json"
        values = parse_tenure(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(values["OWN"][2015], 74.8)
        self.assertEqual(values["OWN"][2025], 71.2)
        self.assertEqual(values["RENT_MKT"][2025], 16.3)

    def test_tenure_composition_detects_error(self) -> None:
        path = Path(__file__).parent / "data" / "eurostat_ilc_lvho02_PT_2026-10-29.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        categories = payload["dimension"]["tenure"]["category"]["index"]
        years = payload["dimension"]["time"]["category"]["index"]
        index = categories["OWN"] * len(years) + years["2025"]
        payload["value"][str(index)] = 60.0
        with self.assertRaises(ValueError):
            parse_tenure(payload)


if __name__ == "__main__":
    unittest.main()
