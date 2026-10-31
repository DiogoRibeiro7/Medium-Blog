# Article #16: A Nearest Neighbor Is Not Necessarily Similar

**Full title:** *A Nearest Neighbor Is Not Necessarily Similar: Distance Concentration, Hubness, and the Geometry of High-Dimensional Data*

This directory is a reproducible scientific companion to the long-form article
[`article.md`](article.md). It separates analytical identities from finite Monte
Carlo evidence, and does not use external data, unverified benchmarks, or
claims about a particular production embedding model.

## Requirements and full reproduction

Python 3.12; tested with NumPy, SciPy, Matplotlib, nbformat, nbclient,
and ipykernel. From this directory:

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 python analysis.py
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
RUN_NOTEBOOK_EXECUTION=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
```

The last command executes the notebook in a fresh Python kernel and requires
permission to create localhost Jupyter IPC sockets. It runs in the article's
path-scoped GitHub Actions workflow as part of PR validation. A reduced
smoke run writes **separate** `*_quick.csv` and `*_quick.png` files:

```bash
OPENBLAS_NUM_THREADS=1 python analysis.py --quick
```

To rebuild the versioned notebook structure, run `python notebook_builder.py`.
Its internal experiment writes to a temporary directory so the checked-in
canonical evidence is unchanged. The CLI also supports `--output /path/to/run`
to keep replication products separate from the tracked evidence. `SEED=20261031`
is fixed in `analysis.py`; independent named random-number streams isolate
each experiment and its repeats.

## Experiment map

| Experiment | Figure(s) | Evidence tables | Design and inference target |
| --- | --- | --- | --- |
| Gaussian pair distance | 1, 2 | `distance_samples`, `distance_summary` | 2,100 independent Gaussian differences per dimension; exact chi moments |
| Finite-pool contrast | 2 | `neighbor_contrast_replicates`, `neighbor_contrast_summary` | 14 repeats; 100 queries, 260 candidates per repeat and dimension |
| Irrelevant features, 7-NN | 3 | `classification_replicates`, `classification_summary` | 36 paired datasets, 260 training / 160 independent test points, p = 2–512; Bayes rule known |
| Nuisance feature scaling | 4 | `scale_replicates`, `scale_summary` | 28 paired replicates, p=128, nuisance factors 0.1–2.0 |
| Gaussian JL projection | 5 | `jl_replicates`, `jl_summary` | 170 fixed p=256 points, 20 independent maps per k=8–128; 14,365 distinct pair distances |
| Nearest-neighbor hubs | 6, 7 | `hub_replicates`, `hub_summary`, `hub_example_counts` | 12 independent n=350 isotropic Gaussian sets per p; k=10, self-neighbors excluded |
| Spectral effective rank | 8 | `intrinsic_replicates`, `intrinsic_summary` | 14 repeats; 5 latent factors, ambient p=120, 7 noise levels |
| Metric depends on task | 9 | `metric_replicates`, `metric_summary` | 36 repeats per angular/radial task, three dissimilarities, independent tests |

The 18th table, `study_summary.csv`, collects prespecified headline metrics.
**All 18 canonical CSVs and nine PNGs** are committed along with source code.
Every numeric manuscript statement can be audited against the table schema
and code that produced it. Figures report across-replicate means with
`mean ± 1.96 × Monte Carlo standard error` when applicable. Those whiskers are
descriptive simulation uncertainty, not a substitute for uncertainty about
how well the design represents a target population.

## Critical design distinctions

1. **Distances have different denominators.** The coefficient of variation of
   independent pair distances is not the same statistic as the relative
   farthest-to-nearest contrast for a finite candidate pool. The latter uses
   the median across queries *within* a replication; its across-replication
   mean is graphed to avoid a small-denominator tail dominating presentation.
2. **The class experiment has a fixed Bayes boundary.** Only the first feature
   carries class information. Adding independent noise dimensions does not
   alter the Bayes decision or accuracy `Phi(0.85) ≈ 0.8023`; it degrades this
   particular fixed-sample, all-features Euclidean 7-NN algorithm.
3. **Features and units matter.** The scaling experiment changes only the 127
   nuisance coordinates, so differences in classification arise from the
   Euclidean geometry rather than a changed outcome mechanism.
4. **JL preservation is a finite-set geometry statement.** Random projection
   preserves pairwise distances approximately at sufficient dimension and
   with controlled failure probability, but it does not establish semantic
   similarity, label recovery, or a universal k for every dataset.
5. **Hubness is a counting phenomenon.** For n points and k self-excluded
   neighbors per point, total k-occurrence is exactly n×k. A hub is a point
   overrepresented in others' lists, not necessarily a good semantic exemplar.
6. **Effective dimension is definition-dependent.** Participation ratio
   `(tr(Sigma))^2/tr(Sigma^2)` is a spectral summary; it differs from latent
   generative rank, top-eigenvalue counts, manifold dimension, and the
   entropy-based effective rank of Roy and Vetterli.
7. **Norm difference is a pseudodistance.** `abs(||x|| - ||y||)` is zero for
   some distinct vectors. It is a deliberate radial-task baseline, not a
   metric on Euclidean vectors. Cosine dissimilarity also is not universally
   metric and is undefined for zero vectors.

## Tests and CI

The test suite checks exact identities, invalid input handling, seed
reproducibility, count conservation, coherent paired runs, theoretical moments,
full inventories of 9 figures/18 tables, statistical directionality under
the *specified synthetic designs*, notebook structure, and fresh-kernel
execution when enabled. CI is path-scoped to this article and its workflow.

Scientific limitations, alternative explanations, a possible applied
validation design, and DOI-linked academic references appear in `article.md`.
The work is educational simulation and methodological analysis; it is not
an empirical evaluation of deployed retrieval systems or a causal claim.
