"""Offline regression checks for the Eurostat parser and descriptive statistics."""

import csv
import json
import math
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

from analysis import (
    annual_mean_rates,
    annuity_factor,
    construction_intensity,
    decompose_payment_change,
    parse_completed_dwellings,
    parse_demographic_totals,
    parse_dwellings_permitted,
    parse_hpi,
    parse_housing_overburden,
    parse_income,
    parse_leaving_age,
    parse_mortgage_rates,
    parse_parental_coresidence,
    parse_tenure,
    payment_to_income_index,
    relative_affordability,
    summarize,
    verify_snapshot,
)
from fetch_mortgage_rates import check_body


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

    def test_official_dwellings_permitted(self) -> None:
        path = Path(__file__).parent / "data" / "eurostat_sts_cobp_a_PT_dwellings_2026-10-29.json"
        values = parse_dwellings_permitted(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(values[2005], 73.9)
        self.assertEqual(values[2015], 8.6)
        self.assertEqual(values[2025], 40.9)
        self.assertNotIn(1992, values)  # Eurostat publishes no observation for this year.

    def test_rejects_different_permits_indicator_and_negative_values(self) -> None:
        path = Path(__file__).parent / "data" / "eurostat_sts_cobp_a_PT_dwellings_2026-10-29.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["dimension"]["indic_bt"]["category"]["index"] = {"BPRM_SQM": 0}
        with self.assertRaises(ValueError):
            parse_dwellings_permitted(payload)
        payload["dimension"]["indic_bt"]["category"]["index"] = {"BPRM_DW": 0}
        payload["value"]["33"] = -2.0
        with self.assertRaises(ValueError):
            parse_dwellings_permitted(payload)

    def test_completed_dwellings_ine_pordata(self) -> None:
        path = Path(__file__).parent / "data" / "ine_pordata_completed_new_dwellings_PT_2026-10-29.csv"
        counts, preliminary = parse_completed_dwellings(path)
        self.assertEqual(counts[2002], 125708)
        self.assertEqual(counts[2015], 7148)
        self.assertEqual(counts[2025], 27301)
        self.assertEqual(preliminary, [2021, 2022, 2023, 2024, 2025])

    def test_completed_dwellings_rejects_duplicate_year(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            path.write_text(
                "year,completed_dwellings,preliminary\n"
                "2024,200,true\n2024,300,true\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                parse_completed_dwellings(path)

    def test_official_household_and_population_totals(self) -> None:
        folder = Path(__file__).parent / "data"
        household_raw = json.loads((folder / "eurostat_lfst_hhnhtych_PT_2026-10-29.json").read_text(encoding="utf-8"))
        population_raw = json.loads((folder / "eurostat_demo_pjan_PT_2026-10-29.json").read_text(encoding="utf-8"))
        households, flags = parse_demographic_totals(household_raw, households=True)
        people, population_flags = parse_demographic_totals(population_raw, households=False)
        self.assertEqual(households[2015], 4100.8)
        self.assertEqual(households[2025], 4562.1)
        self.assertEqual(flags[2021], "b")
        self.assertEqual(people[2025], 10749635.0)
        self.assertEqual(population_flags, {})

    def test_construction_intensity_units_and_missing_years(self) -> None:
        rates = construction_intensity(
            {2024: 25000, 2025: 27301},
            {2025: 10749635.0},
            {2025: 4562.1},
        )
        self.assertEqual(list(rates), [2025])
        self.assertEqual(rates[2025]["completed_per_1000_residents"], 2.54)
        self.assertEqual(rates[2025]["completed_per_1000_households"], 5.984)
        with self.assertRaises(ValueError):
            construction_intensity({2025: 1}, {2025: 100.0}, {2025: 0.0})

    def test_household_parser_rejects_wrong_unit(self) -> None:
        path = Path(__file__).parent / "data" / "eurostat_lfst_hhnhtych_PT_2026-10-29.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["dimension"]["unit"]["category"]["index"] = {"NR": 0}
        with self.assertRaises(ValueError):
            parse_demographic_totals(raw, households=True)


RATE_CSV = Path(__file__).parent / "data" / "ecb_mir_PT_house_purchase_cost_of_borrowing_2026-10-29.csv"
RATE_RECORD = Path(__file__).parent / "data" / "ecb_mir_PT_house_purchase_cost_of_borrowing_2026-10-29.meta.json"


def _rewrite_rate_csv(directory: str, edit: Callable[[list[dict[str, str]]], None]) -> Path:
    """Copy the pinned ECB file with ``edit`` applied to its rows."""
    with RATE_CSV.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    edit(rows)
    path = Path(directory) / "rates.csv"
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


class MortgageCostTests(unittest.TestCase):
    """The ECB cost-of-borrowing snapshot and the payment-to-income index built on it."""

    def test_snapshot_matches_retrieval_record(self) -> None:
        record = verify_snapshot(RATE_CSV, RATE_RECORD)
        self.assertEqual(record["series_key"], "MIR.M.PT.B.A2C.AM.R.A.2250.EUR.N")
        self.assertEqual(record["http_status"], 200)
        self.assertEqual(check_body(RATE_CSV.read_bytes()), 284)

    def test_snapshot_detects_line_ending_normalisation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "rates.csv"
            copy.write_bytes(RATE_CSV.read_bytes().replace(b"\r\n", b"\n"))
            with self.assertRaises(ValueError):
                verify_snapshot(copy, RATE_RECORD)

    def test_official_monthly_and_annual_values(self) -> None:
        monthly, flags = parse_mortgage_rates(RATE_CSV)
        self.assertEqual(len(monthly), 284)
        self.assertEqual(monthly["2003-01"], 4.4)
        self.assertEqual(min(monthly), "2003-01")
        self.assertEqual(max(monthly), "2026-08")
        self.assertEqual(flags, {})
        annual, incomplete = annual_mean_rates(monthly)
        self.assertEqual(annual[2021], 0.83)
        self.assertAlmostEqual(annual[2023], 4.1308, places=4)
        self.assertEqual(incomplete, {2026: 8})
        self.assertNotIn(2026, annual)

    def test_rate_parser_rejects_other_series_unit_gaps_and_duplicates(self) -> None:
        def other_country(rows: list[dict[str, str]]) -> None:
            rows[0]["REF_AREA"] = "ES"

        def other_unit(rows: list[dict[str, str]]) -> None:
            rows[5]["UNIT"] = "PC"

        def gap(rows: list[dict[str, str]]) -> None:
            del rows[10]

        def duplicate(rows: list[dict[str, str]]) -> None:
            rows.insert(3, dict(rows[2]))

        def implausible(rows: list[dict[str, str]]) -> None:
            rows[7]["OBS_VALUE"] = "-0.5"

        for edit in (other_country, other_unit, gap, duplicate, implausible):
            with self.subTest(edit=edit.__name__), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    parse_mortgage_rates(_rewrite_rate_csv(directory, edit))

    def test_missing_month_is_not_filled(self) -> None:
        def blank_march_2010(rows: list[dict[str, str]]) -> None:
            for row in rows:
                if row["TIME_PERIOD"] == "2010-03":
                    row["OBS_VALUE"] = ""

        with tempfile.TemporaryDirectory() as directory:
            monthly, _ = parse_mortgage_rates(_rewrite_rate_csv(directory, blank_march_2010))
        self.assertNotIn("2010-03", monthly)
        annual, incomplete = annual_mean_rates(monthly)
        self.assertNotIn(2010, annual)
        self.assertEqual(incomplete[2010], 11)

    def test_annuity_factor(self) -> None:
        self.assertAlmostEqual(annuity_factor(0.0, 30), 1 / 360)
        self.assertAlmostEqual(annuity_factor(6.0, 30), 0.00599551, places=8)
        self.assertLess(annuity_factor(1.0, 30), annuity_factor(4.0, 30))
        self.assertGreater(annuity_factor(3.0, 25), annuity_factor(3.0, 40))
        for rate, term in ((-1.0, 30), (3.0, 0)):
            with self.assertRaises(ValueError):
                annuity_factor(rate, term)

    def test_constant_rate_reduces_to_price_income_index(self) -> None:
        hpi = {2015: 100.0, 2020: 150.0, 2025: 250.0}
        income = {2015: 10.0, 2020: 12.0, 2025: 15.0}
        flat = {2015: 2.0, 2020: 2.0, 2025: 2.0}
        self.assertEqual(payment_to_income_index(hpi, income, flat), relative_affordability(hpi, income))

    def test_income_alignment_and_missing_years(self) -> None:
        hpi = {2015: 100.0, 2016: 110.0, 2017: 120.0}
        income = {2015: 10.0, 2016: 11.0, 2017: 12.0}
        rates = {2015: 2.0, 2016: 2.0, 2017: 2.0}
        same_year = payment_to_income_index(hpi, income, rates)
        next_survey = payment_to_income_index(hpi, income, {2015: 2.0, 2016: 2.0, 2017: 2.0}, income_survey_offset=1)
        self.assertEqual(same_year[2016], 100.0)
        self.assertEqual(next_survey[2015], 100.0)
        self.assertEqual(next_survey[2016], round(100 * 1.1 / (12 / 11), 2))
        self.assertNotIn(2017, next_survey)  # survey 2018 is not available; nothing is extrapolated
        del rates[2016]
        self.assertNotIn(2016, payment_to_income_index(hpi, income, rates))

    def test_decomposition_is_an_exact_identity(self) -> None:
        hpi = parse_hpi(json.loads((Path(__file__).parent / "data" / "eurostat_prc_hpi_a_PT_2026-10-29.json").read_text(encoding="utf-8")))
        income = parse_income(json.loads((Path(__file__).parent / "data" / "eurostat_ilc_di03_PT_2026-10-29.json").read_text(encoding="utf-8")))
        rates, _ = annual_mean_rates(parse_mortgage_rates(RATE_CSV)[0])
        index = payment_to_income_index(hpi, income, rates)
        parts = decompose_payment_change(hpi, income, rates, 2021, 2023)
        self.assertAlmostEqual(parts["total"], parts["price"] + parts["financing_cost"] - parts["income"], places=1)
        self.assertAlmostEqual(parts["total"], 100 * math.log(index[2023] / index[2021]), places=1)
        self.assertGreater(parts["financing_cost"], parts["price"])  # 2021-2023 was driven by the rate rise
        with self.assertRaises(ValueError):
            decompose_payment_change(hpi, income, rates, 2015, 2026)

    def test_fetch_check_rejects_other_series(self) -> None:
        body = RATE_CSV.read_bytes().replace(b"MIR.M.PT.", b"MIR.M.ES.", 1)
        with self.assertRaises(ValueError):
            check_body(body)


if __name__ == "__main__":
    unittest.main()
