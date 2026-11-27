"""Synthetic checks for matched-sample robustness and influence statistics."""
from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import eurostat_sectors as euro
import historical_shocks as shocks
import pipeline as p
import robustness as r


def metric(country: str, model: str, value: float | None) -> dict:
    """Invent one pandemic shortfall record, never an observed data value."""
    return {
        "country": country, "metric": "real_gdp_per_capita", "shock": "pandemic",
        "benchmark": model, "complete_common_window": value is not None,
        "cumulative_shortfall_pct_benchmark": value,
    }


class TestPairedStatistics(unittest.TestCase):
    def test_midranks(self) -> None:
        self.assertEqual(r.midranks([9, 1, 1, 5]), [4, 1.5, 1.5, 3])

    def test_rank_correlation(self) -> None:
        self.assertAlmostEqual(r.spearman([1, 1, 3, 4], [4, 4, 2, 1]), -1)
        self.assertIsNone(r.spearman([1, 1, 1], [1, 2, 3]))
        self.assertIsNone(r.spearman([1, 2], [3, 4]))

    def test_leave_one_out_influence(self) -> None:
        result = r.paired_statistics([
            ("A", 1, 1), ("B", 2, 2), ("C", 3, 3), ("D", 100, -100),
        ])
        self.assertEqual(result["countries"], "A;B;C;D")
        self.assertLess(result["pearson_r"], 0)
        self.assertAlmostEqual(result["loo_pearson_max"], 1)
        self.assertEqual(result["loo_max_omits"], "D")
        self.assertTrue(result["loo_sign_reversal"])
        self.assertEqual(result["loo_defined"], 4)

    def test_unavailable_statistics_not_zero(self) -> None:
        result = r.paired_statistics([])
        self.assertEqual(result["n"], 0)
        self.assertIsNone(result["pearson_r"])
        self.assertIsNone(result["spearman_r"])
        self.assertIsNone(result["loo_sign_reversal"])

    def test_duplicate_and_nonfinite_pairs_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            r.paired_statistics([("A", 1, 2), ("A", 2, 3)])
        with self.assertRaisesRegex(ValueError, "finite"):
            r.paired_statistics([("A", math.nan, 2)])

    def test_annual_log_growth(self) -> None:
        self.assertAlmostEqual(r.annual_log_growth(100, 10), 10 * math.log(2))
        self.assertAlmostEqual(r.annual_log_growth(-50, 10), 10 * math.log(0.5))
        for invalid in (None, -100, -101, math.inf):
            self.assertIsNone(r.annual_log_growth(invalid, 10))
        with self.assertRaises(ValueError):
            r.annual_log_growth(10, 0)


class TestMatchedSamples(unittest.TestCase):
    def test_wealth_windows_and_bulgaria_exclusion(self) -> None:
        countries = ("BGR", "CYP", "PRT", "USA")
        rows = []
        for start, end in ((1995, 2020), (1996, 2020)):
            for country in countries:
                available = country != "CYP" and not (country == "BGR" and start == 1995)
                rows.append({
                    "country": country, "start": start, "end": end,
                    "real_gdp_pc_growth_pct": 50,
                    "real_wealth_pc_growth_pct": (
                        -20 if country == "PRT" else 30
                    ) if available else None,
                })
        with patch.object(p, "COUNTRIES", countries), patch.object(p, "wealth_rows", return_value=rows):
            sensitivity, stability = r.wealth_sensitivity(p.Snapshot({}), p.Snapshot({}))
        common = [row for row in sensitivity if row["sample"] == "common_all_windows"]
        self.assertEqual([row["countries"] for row in common], ["PRT;USA", "PRT;USA"])
        available = next(row for row in sensitivity if row["start"] == 1996 and row["sample"] == "available")
        excluded = next(row for row in sensitivity if row["start"] == 1996 and row["sample"] == "exclude_bulgaria")
        self.assertEqual(available["n"], 3)
        self.assertEqual(excluded["n"], 2)
        portugal = next(row for row in stability if row["country"] == "PRT")
        self.assertTrue(portugal["divergent_in_every_requested_window"])
        cyprus = next(row for row in stability if row["country"] == "CYP")
        self.assertFalse(cyprus["divergent_in_every_available_window"])

    def test_benchmark_samples_are_common_across_all_three_models(self) -> None:
        records = [
            metric(country, model, None if country == "USA" and model == "flat" else
                   (-1 if model == "flat" else 2))
            for country in ("DEU", "PRT", "USA") for model in shocks.MODELS
        ]
        rows = [row for row in r.benchmark_sensitivity(records)
                if row["shock"] == "pandemic" and row["metric"] == "real_gdp_per_capita"]
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["countries"] == "DEU;PRT" for row in rows))
        flat = next(row for row in rows if row["first_benchmark"] == "trend_5y"
                    and row["second_benchmark"] == "flat")
        self.assertEqual(flat["strict_sign_reversal_count"], 2)
        self.assertEqual(flat["median_absolute_difference_pp"], 3)

    def test_duplicate_benchmark_record_rejected(self) -> None:
        row = metric("PRT", "flat", 1)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            r.benchmark_panel([row, row], "pandemic", "real_gdp_per_capita")

    def test_structure_samples_fixed_by_feature(self) -> None:
        countries = ("DEU", "FRA", "PRT", "USA")
        records = [
            metric(country, model, None if country == "USA" and model == "flat" else
                   float(index + 1))
            for index, country in enumerate(countries) for model in shocks.MODELS
        ]
        data = {(code, country, 2019): value
                for country in countries
                for code, value in ((p.AGR, 10), (p.IND, 30), (p.SRV, 50), (p.MAN, 20))}
        del data[p.MAN, "FRA", 2019]
        with patch.object(p, "COUNTRIES", countries), patch.object(p, "EU27", countries[:3]):
            rows = r.structure_sensitivity(p.Snapshot(data), records)
        manufacturing = [row for row in rows if row["shock"] == "pandemic"
                         and row["structure_feature"] == "manufacturing_pct_gdp"
                         and row["sample"] == "all_countries"]
        self.assertEqual(len(manufacturing), 3)
        self.assertTrue(all(row["countries"] == "DEU;PRT" for row in manufacturing))
        services = [row for row in rows if row["shock"] == "pandemic"
                    and row["structure_feature"] == "services_pct_gdp"
                    and row["sample"] == "eu27_members"]
        self.assertTrue(all(row["countries"] == "DEU;FRA;PRT" for row in services))

    def test_classifications_exclude_aggregate_and_unmatched_cells(self) -> None:
        wdi_rows = [
            {"country": "PRT", "year": 2024, "hhi_three_broad_sectors": 0.5, "hhi_four_sectors": 0.3},
            {"country": "USA", "year": 2024, "hhi_three_broad_sectors": 0.4, "hhi_four_sectors": None},
        ]
        broad = [{"geo": geo, "year": 2024, "hhi": 0.2} for geo in ("EU27_2020", "PT", "DE")]
        detail = [{"geo": geo, "year": 2024, "hhi": 0.1} for geo in ("EU27_2020", "PT")]
        with patch.object(p, "sector_rows", return_value=wdi_rows), patch.object(
            euro, "concentration_rows", side_effect=[broad, detail]
        ), patch.object(r, "YEARS", (2024,)):
            rows = r.granularity_sensitivity(p.Snapshot({}), {}, {})
        self.assertEqual(rows[0]["countries"], "PRT")
        self.assertEqual(rows[1]["countries"], "PT")


if __name__ == "__main__":
    unittest.main()
