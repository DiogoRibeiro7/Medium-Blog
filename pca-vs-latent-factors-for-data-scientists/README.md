# Article #24 — A Principal Component Is Not a Latent Factor

**Full title:** *A Principal Component Is Not a Latent Factor: Eigenvectors,
Noise, and the Limits of PCA*

This is the executable companion to [article.md](article.md). All evidence
is **synthetic**, with mathematical identities, asymptotic approximations,
and finite Monte Carlo results distinguished. It contains nine original
figures, eighteen canonical CSV tables, typed Python, tests, and a notebook.

## Reproduction

Use Python 3.12 and run these commands from this article directory:

    python -m pip install -r requirements.txt
    OPENBLAS_NUM_THREADS=1 python analysis.py
    OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
    RUN_NOTEBOOK_EXECUTION=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v

The last command executes the notebook in a fresh Jupyter kernel and needs
permission to open localhost IPC sockets. GitHub Actions runs it. To
generate isolated smoke results, use *python analysis.py --quick*. That mode
generates separate names ending in *_quick.csv* and *_quick.png*, without
overwriting canonical evidence. The CLI accepts *--output* for a separate
directory. Regenerate the versioned notebook with
*python notebook_builder.py*. The notebook uses a new temporary directory.
All independent streams derive from the fixed seed 20261101.

## Evidence inventory

| Experiment | Figure | CSV evidence | Full experiment design |
| --- | ---: | --- | --- |
| Gaussian covariance spectra | 1 | spectrum_eigenvalues, spectrum_summary | n=320, p=120, spike excess θ=3, 18 clouds per scenario, all eigenvalues per cloud |
| Spiked eigenstructure | 2 | alignment_replicates, alignment_summary | n=280, p=30/100/180, θ=.2/.5/1/2/4, 18 independent repeats |
| Reconstruction and denoising | 3 | reconstruction_replicates, reconstruction_summary | p=80, q=5, n_train=70/280, n_test=240, 18 independent repeats |
| Low-variance class signal | 4 | label_replicates, label_summary | p=14, 13 high-variance nuisances, n_train=240, n_test=180, 36 repeats |
| Changing feature units | 5 | scale_replicates, scale_summary | p=18, n=300, five nuisance scales, 24 paired repeats |
| Leverage of one outlier | 6 | outlier_replicates, outlier_summary | 180 original 2D points plus one controlled outlier, six heights, 24 paired repeats |
| Factor rotation | 7 | rotation_replicates, rotation_summary | p=10, q=2, n=200, six angles, 15 repeats |
| Parallel analysis | 8 | parallel_replicates, parallel_summary | p=30/70, n=150, q=3; 12 column-permutation nulls per repeat, 18 repeats |
| Anisotropic noise | 9 | anisotropic_replicates, anisotropic_summary | p=50, q=3, n=260, five noise scales, 24 paired repeats |

Figures live in *figures/* and tables in *results/*. The 18 full-run CSVs
are versioned in Git using a path-specific exception to the repository's
global CSV LFS policy. Disposable quick artifacts are ignored.

## Interpretation rules

1. **PCA is a variance optimization**, not a latent causal interpretation.
   Eigenvector signs are arbitrary and repeated-eigenvalue eigenspaces admit
   arbitrary orthogonal basis choices.
2. **Spiked-covariance limits are asymptotic**, using the isotropic Gaussian
   one-spike model and an aspect ratio converging to γ. Finite samples do
   not exactly attain these limits.
3. **Reconstruction differs from latent denoising.** Adding components
   reduces reconstruction error of the observed vector, while attempting
   to recover a hidden noiseless signal can deteriorate after including
   noise directions. Only synthetic ground truth makes the latter measurable.
4. **Low marginal variance does not imply low predictive relevance.**
   The classifier fits PCA and regularized discrimination on training
   observations and labels only; all accuracies use independent test sets.
5. **Units and contamination change covariance eigenstructure.** A
   leading eigenvector is not evidence that a variable caused an outcome.
6. **Orthogonal factor rotations are observationally indistinguishable**
   without identifying restrictions under isotropic latent factors.
7. **Horn-style parallel analysis is a calibrated heuristic**, not an
   oracle for latent dimension. Its rank-wise 95% permutation thresholds
   and simulation count are explicit.
8. **Monte Carlo standard errors** summarize independent repetitions,
   not uncertainty over applied population models. Error whiskers show
   mean ± 1.96 times the independent-run MC-SE where displayed.

## Tests and CI

The source has typed and documented matrix validation, centered PCA,
spiked-model references, subspace alignment, a regularized pooled linear
classifier, column permutations, and reproducible aggregation. Tests
exercise known spectral identities, invalid input handling, deterministic
sampling, exact factor rotation, generated figures and tables, and the
notebook. The path-scoped GitHub Actions workflow executes the notebook
in a fresh kernel and separately regenerates quick evidence.

The study supports methodological understanding under explicitly chosen
synthetic distributions. It does not constitute a field evaluation of an
embedding system or a causal claim about observed populations.
