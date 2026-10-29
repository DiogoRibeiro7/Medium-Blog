"""Tests for the integrated assessment: panel helpers, compatibility rules and the pinned results."""

import csv
import json
import math
import unittest
from pathlib import Path
from typing import Any

import integrated as I

HERE = Path(__file__).parent


def _panel() -> dict[tuple[str, int], dict[str, float]]:
    # Four units in two regions, three years; x has unit, region-year and idiosyncratic parts.
    p: dict[tuple[str, int], dict[str, float]] = {}
    for i, u in enumerate(["a", "b", "c", "d"]):
        for t in (2019, 2020, 2021):
            p[(u, t)] = {"x": 10.0 + i + (t - 2019) * (1 + (i % 2)) + (0.3 if (u, t) == ("a", 2020) else 0.0),
                         "population": 1000.0 * (i + 1), "completions": float(t - 2019)}
    del p[("d", 2021)]["x"]
    return p


class HelperTests(unittest.TestCase):
    def test_balanced_units_and_positivity(self) -> None:
        p = _panel()
        self.assertEqual(I.balanced_units(p, ["x"], [2019, 2020, 2021], ["a", "b", "c", "d"]), ["a", "b", "c"])
        # zero completions are allowed unless the variable is logged
        self.assertEqual(I.balanced_units(p, ["completions"], [2019, 2020], ["a", "b"]), ["a", "b"])
        self.assertEqual(I.balanced_units(p, ["completions"], [2019, 2020], ["a", "b"], positive=["completions"]), [])

    def test_variables_and_transforms(self) -> None:
        p = _panel()
        units, years = ["a", "b", "c"], [2019, 2020, 2021]
        logs = I.variable(p, "log_x", units, years)
        self.assertAlmostEqual(logs[("a", 2019)], math.log(10.0))
        rate = I.variable(p, "completions_per_1000", units, years, base_pop={"a": 1000.0, "b": 2000.0, "c": 3000.0})
        self.assertAlmostEqual(rate[("b", 2021)], 1000 * 2 / 2000)
        with self.assertRaises(ValueError):
            I.variable(p, "completions_per_1000", units, years)
        with self.assertRaises(ValueError):
            I.variable(p, "x", units, years)
        region = {"a": "R1", "b": "R1", "c": "R2"}
        ry = I.transform(logs, units, years, "region_year", region)
        for t in years:
            self.assertAlmostEqual(ry[("a", t)] + ry[("b", t)], 0.0, places=12)
            self.assertAlmostEqual(ry[("c", t)], 0.0, places=12)
        fd = I.transform(logs, units, years, "fd", region)
        self.assertEqual(sorted({t for _u, t in fd}), [2020, 2021])
        for t in (2020, 2021):
            self.assertAlmostEqual(sum(fd[(u, t)] for u in units), 0.0, places=12)
        with self.assertRaises(ValueError):
            I.transform(logs, units, years, "random", region)
        self.assertEqual(I.raw_needed(["log_price_m2", "completions_per_1000"]), ["price_m2", "completions"])

    def test_growth_and_rates(self) -> None:
        p = _panel()
        self.assertAlmostEqual(I.growth(p, "x", "b", 2019, 2021) or 0, 100 * math.log(15 / 11))
        self.assertIsNone(I.growth(p, "x", "d", 2019, 2021))
        self.assertAlmostEqual(I.mean_rate(p, "completions", "a", [2019, 2020, 2021], 2019) or 0, 1000 * 1 / 1000)
        self.assertEqual(I.growth_rates({2019: 100.0, 2020: 110.0}), {2020: 100 * math.log(1.1)})
        self.assertEqual(I.growth_rates({2019: 3.0, 2020: 2.5}, log=False), {2020: -0.5})


class PinnedResultTests(unittest.TestCase):
    """Checks on results_integrated.json, integrated_panel.csv and data_audit.csv."""

    results: dict[str, Any]
    rows: list[dict[str, str]]
    audit: list[dict[str, str]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.results = json.loads((HERE / "results_integrated.json").read_text(encoding="utf-8"))
        with (HERE / "integrated_panel.csv").open(newline="", encoding="utf-8") as fh:
            cls.rows = list(csv.DictReader(fh))
        with (HERE / "data_audit.csv").open(newline="", encoding="utf-8") as fh:
            cls.audit = list(csv.DictReader(fh))

    def test_joins_and_audit(self) -> None:
        checks = self.results["join_checks"]
        self.assertTrue(checks["population_nuts2013_vs_nuts2024_2021_2023_identical"])
        self.assertLessEqual(checks["income_nuts2013_vs_nuts2024_overlap"]["max_abs_diff_eur"], 1)
        self.assertEqual([a["variable"] for a in self.audit], list(I.VARIABLES))
        by_var = {a["variable"]: a for a in self.audit}
        for v in ("population", "completions", "permits"):
            self.assertEqual(by_var[v]["missing_cells"], "0")
        self.assertEqual(by_var["price_m2"]["first_year"], "2019")
        self.assertEqual(by_var["price_m2_2018m"]["last_year"], "2020")
        for a in self.audit:
            self.assertTrue(a["breaks_and_notes"])

    def test_panel_keeps_methods_apart(self) -> None:
        dicos = {r["dico"] for r in self.rows}
        self.assertEqual(len(dicos), 308)
        for r in self.rows:
            year = int(r["year"])
            if r["population"]:
                self.assertEqual(r["population_basis"], "census-based" if year <= 2020 else "administrative-based")
            if r["price_m2"]:
                self.assertGreaterEqual(year, 2019)
            if r["price_m2_2018m"]:
                self.assertLessEqual(year, 2020)
            if r["rent_m2"]:
                self.assertGreaterEqual(year, 2020)

    def test_no_model_crosses_the_population_break(self) -> None:
        for name, m in self.results["models"].items():
            if "log_population" in m["spec"]["x"] or any(x.endswith("_per_1000") for x in m["spec"]["x"]):
                a, b = m["spec"]["years"]
                if "log_population" in m["spec"]["x"]:
                    with self.subTest(model=name):
                        self.assertTrue(b <= 2020 or a >= 2021)

    def test_findings_layers(self) -> None:
        across = self.results["across_specifications"]
        self.assertGreater(across["income"]["positive"], 0)
        self.assertGreater(across["income"]["negative"], 0)  # the income association changes sign: not identified
        self.assertEqual(across["rent"]["negative"], 0)
        comp = self.results["completions_composition_check"]
        self.assertGreater(comp["all_dwellings"]["ci95_cr1"][0], 0)
        self.assertLess(comp["existing_dwellings"]["ci95_cr1"][0], 0)
        self.assertLess(0, comp["existing_dwellings"]["ci95_cr1"][1])
        self.assertIn("requires_causal_identification", self.results["layers"])
        for v in self.results["verdicts"].values():
            self.assertIn(v["verdict"], {"robust", "fragile", "not supported"})


if __name__ == "__main__":
    unittest.main()
