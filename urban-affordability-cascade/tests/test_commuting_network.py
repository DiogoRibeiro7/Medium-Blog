"""Checks for Census-derived *undirected* commuting connectivity and null tests."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from commuting_network import (  # noqa: E402
    FIGURE_NAMES,
    RESULT_NAMES,
    growth_vectors,
    load_weights,
    morans_i,
    permutation_test,
    run,
    weighted_neighbor_growth,
    weights_for_mode,
)
from extract_commuting import (  # noqa: E402
    OUTPUT,
    PROVENANCE,
    parse_workbook_rows,
    municipality_codes_from_panel,
)


class CommutingExtractTests(unittest.TestCase):
    """Reject directed-flow interpretations and silent source data repairs."""

    def setUp(self) -> None:
        self.codes = {"0001": "Alpha", "0002": "Beta", "0003": "Gamma"}
        self.rows = [
            ("Origem", "Destino", "n"),
            ("0001: Alpha", "Alpha", 0), (None, "Beta", 10), (None, "Gamma", 3),
            ("0002: Beta", "Alpha", 10), (None, "Beta", 0), (None, "Gamma", 7),
            ("0003: Gamma", "Alpha", 3), (None, "Beta", 7), (None, "Gamma", 0),
        ]

    def test_workbook_origin_forward_fill_and_deduplication(self) -> None:
        links = parse_workbook_rows(self.rows, self.codes)
        self.assertEqual(len(links), 3)
        self.assertEqual(sum(int(r["commuting_interactions_2021"]) for r in links), 20)
        self.assertEqual(links[0]["municipality_a_code"], "0001")
        self.assertEqual(links[0]["municipality_b_code"], "0002")

    def test_reject_asymmetry_even_when_symmetric_partner_exists(self) -> None:
        rows = list(self.rows)
        rows[4] = ("0002: Beta", "Alpha", 11)
        with self.assertRaisesRegex(ValueError, "Asymmetric"):
            parse_workbook_rows(rows, self.codes)

    def test_reject_missing_or_duplicate_pair(self) -> None:
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            parse_workbook_rows(self.rows[:-1], self.codes)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_workbook_rows(self.rows + [(None, "Gamma", 0)], self.codes)

    def test_reject_invalid_diagonal_and_origin_identity(self) -> None:
        invalid = list(self.rows)
        invalid[1] = ("0001: Alpha", "Alpha", 1)
        with self.assertRaisesRegex(ValueError, "Nonzero diagonal"):
            parse_workbook_rows(invalid, self.codes)
        invalid[1] = ("0001: Other", "Alpha", 0)
        with self.assertRaisesRegex(ValueError, "name mismatch"):
            parse_workbook_rows(invalid, self.codes)

    def test_reject_noninteger_count_and_missing_origin(self) -> None:
        invalid = list(self.rows)
        invalid[2] = (None, "Beta", 1.5)
        with self.assertRaisesRegex(ValueError, "integer"):
            parse_workbook_rows(invalid, self.codes)
        invalid[1] = (None, "Alpha", 0)
        with self.assertRaisesRegex(ValueError, "no origin"):
            parse_workbook_rows(invalid, self.codes)


class ObservedNetworkTests(unittest.TestCase):
    """Regression-check the actual pinned 17-municipality interaction extract."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.names, cls.weights = load_weights()

    def test_fixed_source_provenance_and_hash(self) -> None:
        record = json.loads(PROVENANCE.read_text(encoding="utf-8"))
        self.assertEqual(record["source_commit"], "a3bd023b8eab56962cb377d1f07c6279cde67007")
        self.assertEqual(record["source_workbook_sha256"], "7d6dbb58be8356c868c9d0d548a81db27025228f662c5c09d71cd4ba1ee7dace")
        self.assertEqual(hashlib.sha256(OUTPUT.read_bytes()).hexdigest(), record["derived_csv_sha256"])
        self.assertIn("not independently", record["source_type"])
        self.assertEqual(len(municipality_codes_from_panel()), 17)

    def test_136_unique_undirected_links_and_zero_diagonal(self) -> None:
        self.assertEqual(self.weights.shape, (17, 17))
        np.testing.assert_array_equal(self.weights, self.weights.T)
        np.testing.assert_array_equal(np.diag(self.weights), np.zeros(17))
        self.assertEqual(np.count_nonzero(np.triu(self.weights, 1)), 136)
        self.assertEqual(int(np.triu(self.weights, 1).sum()), 339378)
        self.assertEqual(int(self.weights.sum()), 2 * 339378)
        i, j = self.names.index("Porto"), self.names.index("Vila Nova de Gaia")
        self.assertEqual(self.weights[i, j], 39960)
        self.assertEqual(self.weights[j, i], 39960)

    def test_duplicate_source_pair_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            data = OUTPUT.read_text(encoding="utf-8")
            path.write_text(data + data.splitlines()[1] + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_weights(path)

    def test_growth_alignment_is_finite(self) -> None:
        growth = growth_vectors(self.names)
        self.assertEqual(len(growth), 2)
        self.assertTrue(all(np.all(np.isfinite(x)) and len(x) == 17 for x in growth.values()))
        i = self.names.index("Porto")
        self.assertAlmostEqual(growth["sale_price_growth_2019_2025_pct"][i], (3347 / 1856 - 1) * 100)


class SpatialStatisticTests(unittest.TestCase):
    """Mathematical tests of weighting and fixed-seed permutation inference."""

    def setUp(self) -> None:
        self.w = np.array([[0.0, 2.0, 4.0], [2.0, 0.0, 3.0], [4.0, 3.0, 0.0]])
        self.y = np.array([10.0, 20.0, 30.0])

    def test_weighted_neighbor_growth(self) -> None:
        value = weighted_neighbor_growth(self.y, self.w)
        self.assertAlmostEqual(value[0], (2 * 20 + 4 * 30) / 6)
        self.assertAlmostEqual(value[1], (2 * 10 + 3 * 30) / 5)
        self.assertAlmostEqual(value[2], (4 * 10 + 3 * 20) / 7)

    def test_uniform_complete_graph_is_degenerate(self) -> None:
        complete = np.ones((3, 3)) - np.eye(3)
        self.assertAlmostEqual(morans_i(self.y, complete), -0.5)
        stat, p, null = permutation_test(self.y, complete, seed=1, permutations=99)
        self.assertAlmostEqual(stat, -0.5)
        self.assertEqual(p, 1.0)
        np.testing.assert_allclose(null, -0.5)

    def test_permutations_are_reproducible(self) -> None:
        a = permutation_test(self.y, self.w, seed=37, permutations=199)
        b = permutation_test(self.y, self.w, seed=37, permutations=199)
        self.assertEqual(a[:2], b[:2])
        np.testing.assert_array_equal(a[2], b[2])
        self.assertGreaterEqual(a[1], 1 / 200)

    def test_reject_bad_matrices_and_constant_outcomes(self) -> None:
        with self.assertRaisesRegex(ValueError, "symmetric"):
            morans_i(self.y, np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))
        with self.assertRaisesRegex(ValueError, "Constant"):
            morans_i(np.ones(3), self.w)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            weights_for_mode(self.w, "complete_binary")
        with self.assertRaisesRegex(ValueError, "99"):
            permutation_test(self.y, self.w, seed=1, permutations=5)

    def test_weight_transform_preserves_zero_diagonal_and_symmetry(self) -> None:
        for mode in ("raw", "sqrt", "top3_union"):
            w = weights_for_mode(self.w, mode)
            np.testing.assert_array_equal(w, w.T)
            np.testing.assert_array_equal(np.diag(w), np.zeros(3))

    def test_reproducible_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            record = run(out, permutations=99)
            self.assertEqual(record["pair_count"], 136)
            self.assertEqual(record["sum_unique_pair_interactions"], 339378)
            self.assertIn("not residential", record["interpretation"])
            for filename in FIGURE_NAMES:
                self.assertGreater((out / "figures" / filename).stat().st_size, 5000)
            for filename in RESULT_NAMES:
                self.assertGreater((out / "results" / filename).stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
