"""Numerical identities, simulation invariants, evidence, and notebook checks."""

from __future__ import annotations

import csv
import math
import os
import tempfile
import unittest
from pathlib import Path

import nbformat
import numpy as np

import analysis


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read machine-readable evidence without numeric type inference."""

    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


class PCANumericalTests(unittest.TestCase):
    def test_sample_covariance_exact_diagonal(self) -> None:
        x=np.array([[1.,0.],[-1.,0.],[0.,2.],[0.,-2.]])
        values, vectors, mean=analysis.covariance_pca(x)
        np.testing.assert_allclose(mean,0)
        np.testing.assert_allclose(values,[8/3,2/3],atol=1e-12)
        np.testing.assert_allclose(vectors.T@vectors,np.eye(2),atol=1e-12)

    def test_shift_invariance(self) -> None:
        x=np.random.default_rng(1).normal(size=(30,5))
        a=analysis.covariance_pca(x)[0]
        b=analysis.covariance_pca(x+np.arange(5)*100)[0]
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)

    def test_eigenvalue_sum_is_sample_variance(self) -> None:
        x=np.random.default_rng(3).normal(size=(60,8))
        values,_,_=analysis.covariance_pca(x)
        self.assertAlmostEqual(float(values.sum()),float(np.trace(np.cov(x,rowvar=False))))

    def test_rank_does_not_exceed_n_minus_one(self) -> None:
        x=np.random.default_rng(4).normal(size=(6,15))
        values,_,_=analysis.covariance_pca(x)
        self.assertLessEqual(int(np.count_nonzero(values>1e-10)),5)

    def test_covariance_eigenvalues_nonnegative(self) -> None:
        x=np.random.default_rng(5).normal(size=(20,100))
        values,_,_=analysis.covariance_pca(x)
        self.assertTrue(np.all(values>=0))

    def test_covariance_reconstructs_from_eigensystem(self) -> None:
        x=np.random.default_rng(6).normal(size=(50,7))
        values,vectors,_=analysis.covariance_pca(x)
        np.testing.assert_allclose(vectors@np.diag(values)@vectors.T,
                                   np.cov(x,rowvar=False),atol=1e-12)

    def test_single_observation_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.covariance_pca(np.ones((1,3)))

    def test_empty_matrix_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.covariance_pca(np.empty((0,3)))

    def test_nonfinite_matrix_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.covariance_pca(np.array([[0.,1.],[1.,np.inf]]))

    def test_nonmatrix_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.validate(np.ones(5))  # type: ignore[arg-type]

    def test_overlap_identity(self) -> None:
        basis=np.eye(6)[:,:2]
        self.assertAlmostEqual(analysis.projection_overlap(basis,basis),1.)

    def test_overlap_orthogonal(self) -> None:
        self.assertEqual(analysis.projection_overlap(np.eye(6)[:,:2],
                           np.eye(6)[:,2:4]),0.)

    def test_overlap_rotation_invariant(self) -> None:
        basis=np.eye(6)[:,:2]
        theta=.4
        rotation=np.array([[np.cos(theta),-np.sin(theta)],
                           [np.sin(theta),np.cos(theta)]])
        self.assertAlmostEqual(analysis.projection_overlap(basis@rotation,basis),1.)

    def test_overlap_rejects_nonorthonormal(self) -> None:
        with self.assertRaises(ValueError):
            analysis.projection_overlap(np.ones((5,2)),np.eye(5)[:,:2])

    def test_overlap_rejects_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            analysis.projection_overlap(np.eye(5)[:,:2],np.eye(5)[:,:3])


class SpikedModelTests(unittest.TestCase):
    def test_noise_edge(self) -> None:
        value,overlap=analysis.spiked_eigenvalue_prediction(0,.25)
        self.assertAlmostEqual(value,2.25)
        self.assertEqual(overlap,0.)

    def test_bbp_threshold_is_strict(self) -> None:
        value,overlap=analysis.spiked_eigenvalue_prediction(.5,.25)
        self.assertEqual(overlap,0.)
        self.assertAlmostEqual(value,2.25)

    def test_supercritical_formula(self) -> None:
        value,overlap=analysis.spiked_eigenvalue_prediction(2.,.25)
        self.assertAlmostEqual(value,3.*1.125)
        self.assertAlmostEqual(overlap,(1-.25/4)/(1+.25/2))

    def test_spike_alignment_between_zero_and_one(self) -> None:
        for gamma in (.1,.5,1.):
            for theta in (.1,.3,1.,2.,5.):
                _,a=analysis.spiked_eigenvalue_prediction(theta,gamma)
                self.assertGreaterEqual(a,0.)
                self.assertLessEqual(a,1.)

    def test_invalid_theory_inputs(self) -> None:
        for theta,gamma in ((-1.,.5),(1.,0.),(1.,-1.),(np.nan,.5)):
            with self.subTest(theta=theta,gamma=gamma),self.assertRaises(ValueError):
                analysis.spiked_eigenvalue_prediction(theta,gamma)

    def test_gaussian_spike_seeded(self) -> None:
        a=analysis.gaussian_spike(np.random.default_rng(9),20,7,2.)
        b=analysis.gaussian_spike(np.random.default_rng(9),20,7,2.)
        np.testing.assert_array_equal(a,b)

    def test_gaussian_spike_shape(self) -> None:
        x=analysis.gaussian_spike(np.random.default_rng(9),18,8,1.)
        self.assertEqual(x.shape,(18,8))

    def test_invalid_generation(self) -> None:
        with self.assertRaises(ValueError):
            analysis.gaussian_spike(np.random.default_rng(9),1,8,1.)

    def test_bayes_label_reference(self) -> None:
        from scipy.stats import norm
        self.assertAlmostEqual(norm.cdf(.19/.17),.8682,delta=.003)


class ClassifierAndSamplingTests(unittest.TestCase):
    def test_regularized_lda_easy_case(self) -> None:
        x=np.array([[-3.,0.],[-2.,1.],[2.,0.],[3.,1.]])
        y=np.array([0,0,1,1],dtype=np.int64)
        result=analysis.pooled_linear_discriminant(x,y,
                              np.array([[-2.5,.5],[2.5,.5]]))
        np.testing.assert_array_equal(result,[0,1])

    def test_regularized_lda_one_feature(self) -> None:
        x=np.array([[-2.],[-1.],[1.],[2.]])
        y=np.array([0,0,1,1],dtype=np.int64)
        pred=analysis.pooled_linear_discriminant(x,y,np.array([[-3.],[3.]]))
        np.testing.assert_array_equal(pred,[0,1])

    def test_lda_invalid_labels(self) -> None:
        with self.assertRaises(ValueError):
            analysis.pooled_linear_discriminant(np.ones((4,2)),
                      np.zeros(4,dtype=np.int64),np.ones((2,2)))

    def test_lda_mismatched_features(self) -> None:
        with self.assertRaises(ValueError):
            analysis.pooled_linear_discriminant(np.ones((4,2)),
                      np.array([0,0,1,1]),np.ones((2,3)))

    def test_column_permutation_preserves_marginals(self) -> None:
        rng=np.random.default_rng(12)
        x=rng.normal(size=(40,6))
        y=analysis.permute_columns(x,np.random.default_rng(8))
        for j in range(x.shape[1]):
            np.testing.assert_array_equal(np.sort(x[:,j]),np.sort(y[:,j]))

    def test_column_permutation_deterministic(self) -> None:
        x=np.arange(60,dtype=float).reshape(20,3)
        a=analysis.permute_columns(x,np.random.default_rng(11))
        b=analysis.permute_columns(x,np.random.default_rng(11))
        np.testing.assert_array_equal(a,b)

    def test_column_permutation_actually_changes_correlated_rows(self) -> None:
        x=np.column_stack((np.arange(50,dtype=float),np.arange(50,dtype=float)))
        y=analysis.permute_columns(x,np.random.default_rng(7))
        self.assertLess(abs(float(np.corrcoef(y,rowvar=False)[0,1])),.6)

    def test_design_replicate_counts(self) -> None:
        self.assertEqual(analysis.Design().repetitions["label"],36)
        self.assertEqual(analysis.Design(quick=True).repetitions["label"],4)
        self.assertEqual(analysis.Design(quick=True).suffix,"_quick")


class StatisticalSummaryTests(unittest.TestCase):
    def test_mean_mcse(self) -> None:
        rows=[{"group":1,"score":2.},{"group":1,"score":4.}]
        x=analysis.summarize(rows,("group",),("score",))[0]
        self.assertEqual(x["score_mean"],3.)
        self.assertEqual(x["score_mc_se"],1.)

    def test_multiple_group_order(self) -> None:
        rows=[{"g":2,"x":1.},{"g":1,"x":4.},{"g":2,"x":3.}]
        values=analysis.summarize(rows,("g",),("x",))
        self.assertEqual([v["g"] for v in values],[1,2])
        self.assertTrue(math.isnan(values[0]["x_mc_se"]))

    def test_empty_summary_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analysis.summarize([],("x",),("y",))

    def test_empty_csv_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as dest:
            with self.assertRaises(ValueError):
                analysis.write_csv(Path(dest)/"empty.csv",[])

    def test_rectangular_csv_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as dest:
            with self.assertRaises(ValueError):
                analysis.write_csv(Path(dest)/"bad.csv",[{"a":1},{"b":2}])

    def test_csv_float_precision_and_headers(self) -> None:
        with tempfile.TemporaryDirectory() as dest:
            p=Path(dest)/"good.csv"
            analysis.write_csv(p,[{"a":1,"b":1/3}])
            rows=read_rows(p)
            self.assertEqual(rows[0]["a"],"1")
            self.assertAlmostEqual(float(rows[0]["b"]),1/3,places=10)


class GeneratedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temp.name)
        cls.summary=analysis.run(analysis.Design(quick=True,root=cls.root))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def rows(self,name:str) -> list[dict[str,str]]:
        return read_rows(self.root/"results"/f"{name}_quick.csv")

    def test_all_nine_original_figures(self) -> None:
        for name in analysis.FIGURES:
            with self.subTest(name=name):
                file=self.root/"figures"/f"{name}_quick.png"
                self.assertEqual(file.read_bytes()[:8],b"\x89PNG\r\n\x1a\n")
                self.assertGreater(file.stat().st_size,8000)

    def test_all_eighteen_csv_tables(self) -> None:
        self.assertEqual(len(analysis.TABLES),18)
        for name in analysis.TABLES:
            with self.subTest(name=name):
                self.assertGreater(len(self.rows(name)),0)

    def test_quick_suffix_isolated(self) -> None:
        self.assertTrue(all(x.stem.endswith("_quick") for x in
                            (self.root/"results").glob("*.csv")))

    def test_full_and_quick_effort_are_distinct(self) -> None:
        for key,value in analysis.Design().repetitions.items():
            self.assertGreater(value,analysis.Design(quick=True).repetitions[key])

    def test_spectrum_replicates_archived_per_component(self) -> None:
        self.assertEqual(len(self.rows("spectrum_eigenvalues")),2*3*120)

    def test_noise_spectrum_edge_positive(self) -> None:
        self.assertGreater(self.summary["spectrum_noise_edge"],2.)

    def test_alignment_replicates_count(self) -> None:
        self.assertEqual(len(self.rows("alignment_replicates")),3*5*3)

    def test_supercritical_spike_has_alignment(self) -> None:
        self.assertGreater(self.summary["alignment_p180_theta4"],.5)

    def test_reconstruction_replicates_count(self) -> None:
        self.assertEqual(len(self.rows("reconstruction_replicates")),2*3*8)

    def test_denoising_favors_limited_rank(self) -> None:
        rows=self.rows("reconstruction_summary")
        for size in ("70","280"):
            risk5=float(next(r["latent_mse_mean"] for r in rows
                       if r["train_size"]==size and r["components"]=="5"))
            risk80=float(next(r["latent_mse_mean"] for r in rows
                        if r["train_size"]==size and r["components"]=="80"))
            self.assertLess(risk5,risk80)

    def test_observed_reconstruction_with_all_components_is_zero(self) -> None:
        rows=self.rows("reconstruction_summary")
        self.assertTrue(all(float(r["observed_mse_mean"])<1e-20
                            for r in rows if r["components"]=="80"))

    def test_label_complete_scores_better_than_top_thirteen(self) -> None:
        self.assertGreater(self.summary["label_all14_accuracy"],
                           self.summary["label_top13_accuracy"]+.15)

    def test_label_pairing(self) -> None:
        self.assertEqual(len(self.rows("label_replicates")),6*4)

    def test_scale_reverses_first_pc(self) -> None:
        self.assertGreater(self.summary["scale_small_signal_alignment"],
                           self.summary["scale_large_signal_alignment"]+.5)

    def test_outlier_turns_first_pc(self) -> None:
        self.assertGreater(self.summary["outlier_clean_alignment"],
                           self.summary["outlier_40_alignment"]+.8)

    def test_rotations_preserve_covariance(self) -> None:
        self.assertLess(self.summary["rotation_covariance_max_error"],1e-10)

    def test_rotations_preserve_data(self) -> None:
        self.assertLess(max(float(r["max_data_difference"]) for r in
                            self.rows("rotation_replicates")),1e-10)

    def test_parallel_analysis_is_not_kaiser(self) -> None:
        self.assertLess(self.summary["parallel_p70_mean"],
                        self.summary["kaiser_p70_mean"])

    def test_parallel_rows_have_three_generated_factors(self) -> None:
        self.assertTrue(all(r["latent_factors"]=="3" for r in
                            self.rows("parallel_replicates")))

    def test_anisotropic_noise_degrades_signal_subspace(self) -> None:
        self.assertGreater(self.summary["anisotropic_low_signal_overlap"],
                           self.summary["anisotropic_high_signal_overlap"])

    def test_all_headline_values_are_finite(self) -> None:
        self.assertTrue(all(math.isfinite(v) for v in self.summary.values()))


class NotebookTests(unittest.TestCase):
    def test_notebook_valid_and_independent(self) -> None:
        book=nbformat.read(ROOT/"pca_latent_factors.ipynb",as_version=4)
        nbformat.validate(book)
        source="\n".join(str(c.source) for c in book.cells)
        self.assertIn("Design(quick=True",source)
        self.assertIn("synthetic",source.lower())
        self.assertIn("latent",source.lower())

    @unittest.skipUnless(os.environ.get("RUN_NOTEBOOK_EXECUTION")=="1",
                         "Set RUN_NOTEBOOK_EXECUTION=1 for fresh-kernel validation")
    def test_fresh_kernel_executes(self) -> None:
        from nbclient import NotebookClient

        book=nbformat.read(ROOT/"pca_latent_factors.ipynb",as_version=4)
        client=NotebookClient(book,timeout=240,kernel_name="python3",
                              resources={"metadata":{"path":str(ROOT)}})
        client.execute()
        self.assertTrue(any(c.cell_type=="code" and c.get("outputs")
                            for c in book.cells))


if __name__=="__main__":
    unittest.main()
