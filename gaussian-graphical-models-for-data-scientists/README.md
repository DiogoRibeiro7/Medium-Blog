# Gaussian graphical models and conditional independence

This directory accompanies the 8,000+-word Medium essay
**[A Missing Edge Is Not a Missing Relationship](article.md)** by Diogo Ribeiro.
It is article **#14** of the long-form, original-figure series recorded in
[`../ARTICLES.md`](../ARTICLES.md).

The entire study is **synthetic**. Its deliberately known graph, Gaussian
sampling laws, and fixed seeds make the errors and interpretation limits
inspectable. The results are **not** discoveries about patients, people,
financial markets, genes, or industrial equipment. No causal effect is
identified by an edge or a missing edge.

## Environment and reproduction

Use Python 3.12 or newer:

```bash
cd gaussian-graphical-models-for-data-scientists
python -m pip install -r requirements.txt
python analysis.py
python -m unittest discover -s tests -v
```

The full seeded experiment (`20261031`) creates **nine original PNG figures**
in `figures/` and **thirteen CSV evidence tables** in `results/`. It evaluates
three sample sizes (`n=60,180,600`), three fixed graphical-lasso penalties
(`alpha=.04,.09,.16`), **70 independent train/test trials per sample size**,
and **150 nonparametric bootstraps** from a separate sample of 250.
Each held-out trial contains 400 new observations; its three penalties share
the same training/test pair for a fair within-trial comparison.

For a fast smoke test, use:

```bash
python analysis.py --quick
```

Quick mode uses eight trials per sample size and twelve bootstrap resamples.
Every quick filename has a `_quick` suffix, so it cannot overwrite the full
evidence. Override the output root with `--output /path/to/safe/directory`.

The Jupyter notebook `gaussian_graphical_models.ipynb` is executable from
a fresh Python kernel. It regenerates quick evidence in a temporary directory
and displays the results; it does not silently depend on prior notebook state.
`python notebook_builder.py` regenerates the notebook source.

## Experiments and interpretation

| Experiment | Population truth | Demonstration |
| --- | --- | --- |
| Three-node chain | Precision has `Omega[1,3] = 0` | Marginal correlation between endpoints remains nonzero despite zero partial correlation |
| Gaussian collider | Independent parents of common effect | Conditioning on their common effect creates negative association |
| Hidden factor | Observed variables independent conditional on H | Integrating out H makes all six pairwise observed precision entries nonzero |
| High dimension | `p=30`, `n=20` | Centered empirical covariance has rank 19 and is not invertible |
| Sparse network | Twelve variables, fifteen known edges | Fixed-penalty selection sensitivity to n and alpha; Gaussian held-out NLL |
| Bootstrap | 250 observed points, 150 full bootstrap samples | Selection frequencies measure resampling stability, not posterior probabilities |
| Nonlinear relationship | `Y=X^2-1+eps` | Population Pearson zero, but out-of-sample quadratic prediction succeeds |
| Orientation | Both bivariate Gaussian SEM directions match | Same joint observational covariance supports both directions |

### The actual optimization

For the centered, standardized training design, scikit-learn's graphical
lasso solves the penalized Gaussian likelihood problem

\[
\underset{\Omega\succ0}{\arg\min}\;
-\log\det\Omega+\operatorname{tr}(S\Omega)
+\alpha\sum_{i\ne j}|\Omega_{ij}|.
\]

Every fit uses the training mean and scale alone. Positive-definiteness is
checked and optimizer errors are **not** silently caught. The numerical dual
gap tolerance is `1e-3`; the experiment reports support estimates at a
fixed `1e-7` numerical threshold because graphical lasso yields exact
off-diagonal zeros. The fixed penalty values are not the result of a
data-driven tuning exercise and must not be presented as universally optimal.

In support recovery, each unordered off-diagonal pair is counted *once*.
We report precision, recall, F1, false positive rate among known null edges,
and structural Hamming distance. Monte Carlo error bars use the standard
error across independent trials, not the sample-standard deviation of
measurements within a trial. Held-out negative log likelihood is calculated
per observation in nats using the **training** standardizer. It is not a
support-identification metric, and support metrics are not log-likelihood
metrics. Bootstrap stability does not supply a calibrated edge confidence
interval or a causal edge probability.

## Evidence inventory

| CSV file | Contents |
| --- | --- |
| `chain_matrices.csv` | All 3 × 3 population covariance, correlation, precision, and partial correlation entries |
| `chain_samples.csv` | Known-Gaussian samples and regression residuals from conditioning on the middle node |
| `collider_samples.csv` | Independent simulated parents, common effect, and conditional residuals |
| `collider_metrics.csv` | Population and sample marginal/conditional parent correlations |
| `latent_matrices.csv` | Conditional and marginal covariance, observed precision and partial correlations |
| `high_dim_spectrum.csv` | Sorted population, empirical and ridge-regularized covariance eigenvalues |
| `recovery_replicates.csv` | Every known-support experiment, including held-out likelihood |
| `recovery_summary.csv` | Across-trial means and Monte Carlo standard errors per n × alpha cell |
| `edge_stability.csv` | True edge label, original estimate, bootstrap selection frequency and mean coefficient |
| `nonlinear_samples.csv` | Held-out samples and linear/quadratic predictions |
| `nonlinear_metrics.csv` | Held-out R²/RMSE, Pearson correlation and population optimum |
| `orientation_covariance.csv` | Identical observed covariance under both bivariate SEM orientations |
| `study_summary.csv` | Central full-run numerical indicators used in the article |

All nine figures are listed and captioned in `article.md`, and generated by
`analysis.py` from the CSV-producing experiments. The full outputs are
committed to the repository; quick outputs are deliberately excluded.

## Limits and quality gates

This study assumes IID sampling from finite-dimensional Gaussian laws for
the graph recovery experiments. It does not study mixed data, measurement
error, temporal drift, general missingness, interventions, or arbitrary
latent-variable discovery. Observational graph structure is interpreted as
conditional association only. A zero in an estimated sparse precision is a
regularization decision, not a proof that two processes are unrelated.

The test suite validates population identities, positive definiteness,
edge counting, residual orthogonality, data leakage avoidance, finite
outputs, design completeness, and the fresh-kernel notebook. A path-scoped
GitHub Actions job regenerates quick evidence after executing the tests.
The associated published article should be reviewed separately for the
accuracy of its mathematical interpretation, references, and claims.
