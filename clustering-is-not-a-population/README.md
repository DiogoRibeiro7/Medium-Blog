# Article #28 — A Cluster Is Not a Population

**Full title:** *A Cluster Is Not a Population: K-means, Mixtures, Stability, and the Limits of Unsupervised Labels*

This is the executable companion to [article.md](article.md), not a benchmark
of any real population. Nine independent synthetic studies produce ten original
figures and eighteen machine-readable tables. The generator labels are known
only because we construct the observations; they are never supplied to the
unsupervised estimators.

## Reproduce

Python 3.12, within this article directory:

    python -m pip install -r requirements.txt
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python analysis.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
    RUN_NOTEBOOK_EXECUTION=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v

The last command executes the notebook inside a **fresh kernel**, which
requires permission to create local Jupyter IPC sockets. The GitHub Actions
workflow performs this execution on every relevant PR.

The script offers --quick, --output DIRECTORY, and --seed INTEGER. A quick
run uses separate filenames ending in _quick.csv and _quick.png; it does
not overwrite the canonical tables or images. The notebook invokes the same
quick study from a fresh temporary output directory. To regenerate the
notebook itself, run python notebook_builder.py.

The canonical seed is **20261101**. Each independent outer repetition uses
a separate NumPy SeedSequence stream derived from the study name and repetition
index. Paired manipulations of one dataset reuse the same observations.
Scikit-learn supplies multi-start K-means, full-covariance EM Gaussian mixtures
with covariance regularization, DBSCAN, silhouette score, and adjusted Rand
index. Every fitted model receives a fixed, recorded pseudorandom seed;
no generated labels are passed into K-means, DBSCAN, or GaussianMixture.

## Evidence inventory

| Experiment / CSV prefix | Figure(s) | Design (full run) | Main target |
| --- | --- | --- | --- |
| elbow | 1 | 420 observations, 3 isotropic Gaussian components, k=1..8, 18 independent repeats | Squared-error compression versus component count |
| scale | 2 | 440 observations, 2 label-bearing groups, nuisance-unit factors 0.25..4, 24 paired repeats | Adjusted Rand index under feature scaling |
| rings | 3 and 3b | 600 points on two noisy concentric annuli; DBSCAN eps=0.08..0.70, 18 repeats | Connectivity, noise, and epsilon sensitivity |
| ellipses | 4 | 520 points from 2 strongly heteroskedastic 2-D Gaussians, 24 repeats | K-means versus full-covariance Gaussian mixtures |
| dimensions | 5 | 340 points, 2 signal dimensions plus 0..150 independent nuisance dimensions, 18 paired repeats | K-means adjusted Rand index |
| silhouette | 6 | 450 points from 3 partly overlapping Gaussian components, k=2..5, 24 repeats | Silhouette choice versus generator labels |
| bic | 7 | 300 points from 2 separated Gaussian components, Gaussian mixtures k=1..5, 18 repeats | Penalized likelihood and component selection |
| stability | 8 | 650 points from **one** elongated Gaussian; two bootstrap K-means fits, 250 common held-out anchors, 24 repeats | Bootstrap adjusted Rand index |
| posterior | 9 | 420 points from 2 Gaussian components, mean separations 0.6..3.5, 24 repeats | Soft responsibilities versus hard labels |

Each prefix corresponds to two archived CSVs under results/:
PREFIX_replicates.csv and PREFIX_summary.csv. The nine full-run studies
therefore produce **18 CSV tables**. The summary tables report independent
outer-replication means and Monte Carlo standard errors. Some conditions
reuse the same observations within a replicate; marginal MC standard errors
should not be combined as if those conditions were independent.

Ten publication PNGs live in figures/. Figure 3b is an explicit DBSCAN
epsilon-sensitivity companion to the paired ring partition illustration
in Figure 3. Short smoke-run outputs are ignored by Git; the full tables
are versioned using a directory-specific exception to the repository's
global CSV Git LFS policy.

## Interpretation limits

1. **Clustering is a choice of geometry or model**. A partition is not
   automatically a count of scientific types, causes, or populations.
2. **K-means** minimizes Euclidean within-group squared errors with
   stochastic initialization; global optimal distortion is monotone in k.
   Its clusters are Voronoi cells, not arbitrary disconnected shapes.
3. **DBSCAN** depends on a neighborhood radius and minimum-neighbor count.
   In this implementation, ARI includes the -1 noise label as an additional
   partition label. That convention can change how an external score is read.
4. **A Gaussian mixture** assumes a weighted Gaussian density family and
   fitted covariance structure. EM is initialization-dependent and covariance
   regularization limits degeneracy. Responsibilities are conditional
   model probabilities, not evidence that a category exists in nature.
5. **BIC** scores a specified candidate family using its parameter penalty.
   Mixture-model boundaries can violate simple regular asymptotic arguments,
   so BIC is neither a universal oracle nor a causal test.
6. **Silhouette** evaluates separation under a chosen metric; **adjusted
   Rand index** evaluates agreement with a given reference partition;
   **stability** evaluates reproducibility under a resampling design.
   These have different estimands and can disagree without mathematical
   contradiction.
7. The study uses **synthetic, independent Gaussian or annular draws**,
   not samples from a field population. All claims are conditional on
   the explicit sample designs. They do not estimate failure frequencies
   for a product, sensor, learned embedding, or social category.

## Testing and workflows

The 94-test suite exercises validation, algebraic identities, seed streams,
generator parameters, fitted-model behavior, CSV integrity, ten original
figures, condition-level evidence, notebook structure and, when enabled,
fresh-kernel notebook execution. The workflow is path-scoped so other articles
do not rerun these experiments unnecessarily.

Article and reference URLs are in [article.md](article.md). The manuscript
distinguishes exact optimization claims, statistical model assumptions,
finite Monte Carlo observations, and applied interpretation explicitly.
