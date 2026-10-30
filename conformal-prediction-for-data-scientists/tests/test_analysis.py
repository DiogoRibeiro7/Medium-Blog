"""Unit tests for validity assumptions and synthetic numerical reproducibility."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import (  # noqa: E402
    ALPHA,
    BASE_RATE,
    FIGURE_NAMES,
    METHODS,
    RESULT_NAMES,
    SEED,
    assess,
    calibrate,
    design,
    fit_model,
    halfwidth,
    multitarget_experiment,
    plot_figures,
    predict,
    repeated_coverage,
    run_study,
    sample_batch,
    split_threshold,
    temporal_stream,
    true_mean,
    true_sd,
    weighted_threshold,
    write_results,
)


class QuantileTests(unittest.TestCase):
    """Check mathematical quantile invariants apart from figure values."""

    def test_split_threshold_uses_n_plus_one_rank(self) -> None:
        scores = np.array([9.0, 2.0, 6.0, 4.0, 1.0, 8.0, 5.0, 3.0, 7.0, 0.0])
        self.assertEqual(split_threshold(scores, 0.2), 8.0)
        self.assertEqual(split_threshold(scores, 0.1), 9.0)
        self.assertEqual(split_threshold(scores, 0.5), 5.0)

    def test_very_small_alpha_has_infinite_threshold(self) -> None:
        self.assertTrue(np.isinf(split_threshold(np.arange(5, dtype=float), 0.01)))

    def test_quantile_monotone_in_alpha(self) -> None:
        scores = np.linspace(0.0, 2.0, 100)
        self.assertGreaterEqual(split_threshold(scores, 0.05), split_threshold(scores, 0.20))

    def test_invalid_quantile_inputs(self) -> None:
        for bad in (np.array([]), np.array([-0.5]), np.array([np.nan]), np.array([[0.2]])):
            with self.subTest(bad=repr(bad)), self.assertRaises(ValueError):
                split_threshold(bad)
        for bad in (-0.1, 0, 1, float("nan"), True):
            with self.subTest(alpha=bad), self.assertRaises(ValueError):
                split_threshold(np.array([1.0]), bad)

    def test_weighted_equals_unweighted_for_identical_mixtures(self) -> None:
        scores = np.arange(1.0, 101.0)
        groups = (np.arange(len(scores)) % 2).astype(np.int64)
        ordinary = split_threshold(scores)
        for group in (0, 1):
            self.assertEqual(weighted_threshold(scores, groups, 0.2, 0.2, group), ordinary)

    def test_weighted_uses_test_atom_at_infinity(self) -> None:
        scores = np.array([0.1, 0.2, 0.3])
        groups = np.array([0, 0, 1])
        result = weighted_threshold(scores, groups, 0.2, 0.9, 1, 0.1)
        self.assertTrue(np.isinf(result))

    def test_weighted_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            weighted_threshold(np.array([1.0]), np.array([2]), 0.2, 0.6, 1)
        with self.assertRaises(ValueError):
            weighted_threshold(np.array([1.0]), np.array([0]), 0.2, 0.6, 2)
        with self.assertRaises(ValueError):
            weighted_threshold(np.array([1.0]), np.array([0]), 0, 0.6, 1)


class PopulationTests(unittest.TestCase):
    """Verify explicit mean, noise, and independent-training policy."""

    def test_design_matches_known_mean(self) -> None:
        x = np.array([0.0, 0.25, 0.7])
        g = np.array([0, 1, 0])
        expected = 1.25 * np.sin(2 * np.pi * x) + 0.5 * np.cos(4 * np.pi * x) + 0.3 * g
        np.testing.assert_allclose(true_mean(x, g), expected)
        self.assertEqual(design(x, g).shape, (3, 4))

    def test_b_group_is_noisier(self) -> None:
        x = np.linspace(0.1, 0.9, 30)
        a, b = np.zeros(30, dtype=np.int64), np.ones(30, dtype=np.int64)
        self.assertTrue(np.all(true_sd(x, b) > true_sd(x, a)))
        np.testing.assert_allclose(true_sd(x, b, 1.65), 1.65 * true_sd(x, b))
        np.testing.assert_allclose(true_sd(x, a, 1.65), true_sd(x, a))

    def test_invalid_sampling_parameters(self) -> None:
        rng = np.random.default_rng(2)
        for n in (True, 0, 1.5, -2):
            with self.assertRaises(ValueError):
                sample_batch(rng, n)
        with self.assertRaises(ValueError):
            sample_batch(rng, 10, p_b=1.0)
        with self.assertRaises(ValueError):
            sample_batch(rng, 10, b_multiplier=0)

    def test_model_predictions_are_fixed_after_fit(self) -> None:
        train = sample_batch(np.random.default_rng(4), 1500)
        cal = sample_batch(np.random.default_rng(5), 350)
        model = fit_model(train)
        before = predict(model, cal)
        altered = type(cal)(cal.x, cal.group, cal.y + 500, cal.truth, cal.noise_sd)
        np.testing.assert_array_equal(before, predict(model, altered))
        self.assertTrue(np.all(model.scale > 0))

    def test_calibration_methods_have_positive_thresholds(self) -> None:
        train = sample_batch(np.random.default_rng(4), 1500)
        val = sample_batch(np.random.default_rng(5), 250)
        model = fit_model(train)
        cal = calibrate(model, val)
        self.assertTrue(np.all(cal.group_q > 0))
        self.assertGreater(cal.normalized_q, 0)
        for method in METHODS:
            self.assertTrue(np.all(halfwidth(method, model, cal, val.group) > 0))
        with self.assertRaises(ValueError):
            halfwidth("Oracle weighted", model, cal, val.group)

    def test_bonferroni_thresholds_are_no_smaller(self) -> None:
        result = multitarget_experiment(n_test=500)
        self.assertTrue(np.all(result["bonf_q"] >= result["plain_q"]))
        self.assertGreaterEqual(float(result["bonf_joint"]), float(result["plain_joint"]))

    def test_no_future_lookahead_in_temporal_first_step(self) -> None:
        stream = temporal_stream(n=700, window=120)
        self.assertEqual(len(stream["t"]), 700)
        self.assertGreater(float(stream["rolling_q"][0]), 0)
        np.testing.assert_array_equal(stream["fixed_q"], stream["fixed_q"][0])
        self.assertTrue(np.isnan(stream["rolling_trailing"][98]))
        self.assertTrue(np.isfinite(stream["rolling_trailing"][99]))


class ReproductionTests(unittest.TestCase):
    """Keep costly complete scientific results out of each unit test."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.study = run_study(quick=True)

    def test_seeded_determinism(self) -> None:
        again = run_study(quick=True)
        self.assertEqual(again.summary, self.study.summary)
        np.testing.assert_array_equal(again.replicates, self.study.replicates)

    def test_expected_marginal_and_group_failure(self) -> None:
        outcomes = self.study.coverage
        self.assertGreater(outcomes["baseline"]["Global"]["marginal"], 0.87)
        self.assertLess(outcomes["baseline"]["Global"]["group_b"], 0.75)
        self.assertGreater(outcomes["baseline"]["Mondrian"]["group_b"], 0.83)
        self.assertLess(outcomes["mixture"]["Global"]["marginal"], 0.82)
        self.assertGreater(outcomes["mixture"]["Oracle weighted"]["marginal"], 0.85)
        self.assertLess(outcomes["conditional"]["Mondrian"]["group_b"], 0.80)

    def test_replicates_mean_is_near_target(self) -> None:
        rep = repeated_coverage(self.study.model, 30, n_test=250)
        self.assertEqual(rep.shape, (30, 3))
        self.assertTrue(0.85 < rep[:, 0].mean() < 0.95)

    def test_result_export_names_and_count(self) -> None:
        self.assertEqual(set(self.study.rows), set(RESULT_NAMES))
        with tempfile.TemporaryDirectory() as root:
            dest = Path(root)
            paths = write_results(self.study, dest)
            self.assertEqual(len(paths), 10)
            self.assertTrue(all(p.stat().st_size > 20 for p in paths))
            quick = write_results(self.study, dest, quick=True)
            self.assertTrue(all(p.name.endswith("_quick.csv") for p in quick))
            self.assertTrue(all(p.exists() for p in paths))

    def test_all_eight_figures_smoke_and_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            paths = plot_figures(self.study, Path(root), quick=True)
            self.assertEqual(len(paths), len(FIGURE_NAMES))
            self.assertTrue(all(path.name.endswith("_quick.png") for path in paths))
            self.assertTrue(all(path.stat().st_size > 5000 for path in paths))

    def test_fresh_kernel_notebook(self) -> None:
        try:
            import nbformat
            from nbclient import NotebookClient
        except ImportError:
            self.skipTest("Optional notebook execution dependencies not installed")
        notebook = ROOT / "conformal_prediction.ipynb"
        content = nbformat.read(notebook, as_version=4)
        for cell in content.cells:
            if cell.cell_type == "code":
                self.assertIsNone(cell.execution_count, "Commit an unexecuted notebook")
                self.assertEqual(cell.outputs, [], "Do not commit execution output")
        with tempfile.TemporaryDirectory() as tmp:
            result = NotebookClient(
                content, timeout=120, kernel_name="python3",
                resources={"metadata": {"path": str(ROOT)}},
            ).execute()
            self.assertTrue(any(c.cell_type == "code" and c.execution_count for c in result.cells))


if __name__ == "__main__":
    unittest.main()
