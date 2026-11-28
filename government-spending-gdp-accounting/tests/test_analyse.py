"""Known-answer and input-contract tests; no network needed."""
import unittest

from analyse import COUNTRIES, associations, build_panel, cells, common_year, correlation, decompose


def fixture() -> dict:
    """A sparse JSON-stat dataset with deliberately nonalphabetical dimensions."""
    return {
        "class": "dataset", "id": ["geo", "time", "na_item"], "size": [1, 2, 2],
        "dimension": {
            "geo": {"category": {"index": {"FR": 0}}},
            "time": {"category": {"index": {"2025": 1, "2024": 0}}},
            "na_item": {"category": {"index": {"TE": 0, "B1G": 1}}},
        },
        "value": {"0": 60, "1": 20, "2": 57.2},
        "status": {"2": "p"},
    }


class AccountingTests(unittest.TestCase):
    def test_sparse_flags_and_positions(self) -> None:
        rows = cells(fixture())
        self.assertEqual(rows[2], ({"geo": "FR", "time": "2025", "na_item": "TE"}, 57.2, "p"))
        self.assertIsNone(rows[3][1])

    def test_dense_values(self) -> None:
        payload = fixture()
        payload["value"] = [60, 20, 57.2, None]
        self.assertEqual(cells(payload), cells(fixture()))

    def test_invalid_index(self) -> None:
        payload = fixture()
        payload["value"]["4"] = 30
        with self.assertRaises(ValueError):
            cells(payload)

    def test_accounting_gap(self) -> None:
        panel = build_panel({"fiscal.json": fixture()})
        row = next(r for r in panel if r["geo"] == "FR" and r["year"] == 2024)
        self.assertEqual(row["spending_minus_s13_gva_pp"], 40)
        self.assertEqual(row["arithmetic_complement_of_spending_pct"], 40)
        self.assertEqual(row["non_s13_gva_plus_net_product_taxes_pct"], 80)
        missing = next(r for r in panel if r["geo"] == "FR" and r["year"] == 2025)
        self.assertIsNone(missing["spending_minus_s13_gva_pp"])
        self.assertEqual(missing["TE_flag"], "p")

    def test_balanced_year(self) -> None:
        rows = [{"geo": g, "year": 2024, "TE": 50, "B1G": 20} for g in COUNTRIES]
        rows.append({"geo": "FR", "year": 2025, "TE": 57, "B1G": 20})
        self.assertEqual(common_year(rows), 2024)

    def test_fixed_spending(self) -> None:
        n, d = decompose(60, 100, 60, 90)
        self.assertAlmostEqual(n, 0)
        self.assertAlmostEqual(d, 100 * 60 / 90 - 60)

    def test_decomposition_closure(self) -> None:
        n, d = decompose(60, 100, 70, 120)
        self.assertAlmostEqual(n + d, 100 * 70 / 120 - 60)
        with self.assertRaises(ValueError):
            decompose(60, 0, 70, 120)

    def test_correlations(self) -> None:
        self.assertAlmostEqual(correlation([1, 2, 3], [3, 2, 1]), -1)
        self.assertIsNone(correlation([1, 1, 1], [1, 2, 3]))

    def test_lag_alignment_and_flag_sensitivity(self) -> None:
        rows = [
            {"geo": "FR", "year": y, "TE": t, "growth": g, "TE_flag": f, "growth_flag": ""}
            for y, t, g, f in [(2020, 50, -8, ""), (2021, 55, 6, "p"), (2022, 45, 2, ""), (2023, 48, 1, "")]
        ]
        stats = associations(rows)
        lag = next(r for r in stats if r["growth_horizon_years"] == 1 and not r["exclude_flagged"] and r["method"] == "pooled")
        self.assertEqual(lag["n"], 3)
        clean = next(r for r in stats if r["growth_horizon_years"] == 1 and r["exclude_flagged"] and r["method"] == "pooled")
        self.assertEqual(clean["n"], 2)
