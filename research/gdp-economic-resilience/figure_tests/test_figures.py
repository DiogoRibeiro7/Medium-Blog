"""Exercise evidence validation and deterministic figure generation."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt

import figures as f


class TestEvidence(unittest.TestCase):
    def test_blank_is_not_zero(self) -> None:
        self.assertIsNone(f.number({"v": ""}, "v"))
        self.assertEqual(f.number({"v": "0"}, "v"), 0)
        with self.assertRaisesRegex(ValueError, "Missing"):
            f.required({"v": ""}, "v")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            f.number({"v": "nan"}, "v")

    def test_unique_selection(self) -> None:
        rows = [{"id": "a"}, {"id": "b"}]
        self.assertEqual(f.one(rows, id="a"), rows[0])
        with self.assertRaises(ValueError):
            f.one(rows, id="c")
        with self.assertRaises(ValueError):
            f.one([*rows, rows[0]], id="a")

    def test_malformed_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.csv"
            for content in ("a,a\n1,2\n", "a,b\n1\n", "a,b\n1,2,3\n", "a,b\n"):
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    f.read_table(path)

    def test_matched_shocks(self) -> None:
        rows = [
            {"country": country, "shock": "pandemic", "metric": "real_gdp_per_capita",
             "benchmark": model, "complete_common_window": "True",
             "cumulative_shortfall_pct_benchmark": "" if country == "USA" and model == "flat" else "0"}
            for model in f.MODELS for country in ("DEU", "PRT", "USA")
        ]
        selected = f.matched_shocks(rows, "pandemic")
        self.assertEqual(len(selected), 6)
        self.assertEqual({row["country"] for row in selected}, {"DEU", "PRT"})


class TestRendering(unittest.TestCase):
    def test_gdp_uses_nonoverlapping_groups(self) -> None:
        rows = [
            {"entity": entity, "group": "fixed_eu27_comparators", "year": "2025",
             "nominal_usd": str(i * 1e12), "ppp_international_dollars": str(i * 2e12)}
            for i, entity in enumerate(f.COUNTRY_NAMES, 1)
        ]
        fig, selected, caption = f.figure_gdp([*rows, {**rows[0], "group": "countries"}])
        try:
            self.assertEqual(len(selected), 7)
            self.assertEqual(len(fig.axes), 2)
            self.assertIn("PPP", caption)
        finally:
            plt.close(fig)

    def test_manifest_and_repeatable_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            source.write_text("category,value\nA,2\n", encoding="utf-8")

            def builder(rows: list[f.Row]):
                fig, ax = plt.subplots()
                ax.bar(["A"], [f.required(rows[0], "value")])
                return fig, rows, "Synthetic figure."

            with patch.object(f, "SPECS", (("test", "source.csv", builder),)):
                f.run(root, root / "first")
                f.run(root, root / "second")
            first = json.loads((root / "first" / "figure_manifest.json").read_text())
            second = json.loads((root / "second" / "figure_manifest.json").read_text())
            self.assertEqual(first, second)
            entry = first["figures"][0]
            self.assertEqual(entry["input_sha256"], f.digest(source))
            self.assertEqual(entry["selected_rows"], 1)
            for name, sha in entry["assets"].items():
                self.assertEqual(f.digest(root / "first" / name), sha)
            self.assertEqual(f.read_table(root / "first" / "test.csv"),
                             [{"category": "A", "value": "2"}])


if __name__ == "__main__":
    unittest.main()
