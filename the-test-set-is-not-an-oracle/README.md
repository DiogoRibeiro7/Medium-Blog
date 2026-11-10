# The Test Set Is Not an Oracle — reproducible companion

Article #33 of the Medium-Blog long-form series studies the gap between an
evaluation number and a defensible generalization claim. The complete
[editorial manuscript](https://github.com/DiogoRibeiro7/research-articles-medium/blob/main/themes/statistics-for-data-scientists/the-test-set-is-not-an-oracle/article.md)
is maintained privately; this public computational companion retains nine
original seeded experiments, nine original figures, and eighteen CSV evidence
tables. **Every observation, target, prediction, and experiment is synthetic.**
The results demonstrate controlled mathematical mechanisms; they are not
empirical estimates of real-world benchmark contamination or deployment risk.

## Reproduce

Use Python 3.12 and the supplied requirements. The scientific versions used
to generate the archived results are NumPy 2.3.5, SciPy 1.17.0, Matplotlib
3.10.8, and scikit-learn 1.8.0. Notebook dependencies have bounded versions.
Numerical least-significant digits may vary with hardware and BLAS libraries.

~~~bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg python analysis.py
RUN_NOTEBOOK_EXECUTION=1 python -m unittest discover -s tests -v
~~~

The canonical seed is **20261102**. Each experiment/repetition uses a
deterministic independent SeedSequence keyed by its stable study name. The
full run creates one replicate CSV, one summary CSV, and one PNG per study:
18 canonical CSVs and 9 publication PNGs. There are 20–70 independent
outer repetitions, depending on experiment cost. Each summary contains
the across-repetition mean, standard deviation, Monte Carlo standard error,
and 1.96-standard-error interval for the *simulation average*. These are not
confidence intervals for a real dataset's unknown performance. Empty
theory cells mean that no analytic reference was provided.

For a smoke test that cannot overwrite canonical outputs:

~~~bash
python analysis.py --quick
~~~

This writes only files with the suffix _quick, using three repetitions and
smaller synthetic samples. If you choose a noncanonical seed for a full run,
pass a separate output directory, for example:

~~~bash
python analysis.py --seed 42 --output /tmp/validation-study-seed42
~~~

Running a full noncanonical seed against the article directory is refused.
The Jupyter notebook is regenerated deterministically with the
notebook_builder.py command and is committed without cached output.
Its quick experiments use a temporary output directory. It can execute
with a kernel working directory at the article or repository root; when
the kernel starts elsewhere, set the environment variable
VALIDATION_ARTICLE_DIR to this article's absolute directory. The fresh-kernel
tests exercise both root-start and external-directory execution without
relying on notebook resource metadata to set the working directory.

The dedicated [GitHub Actions workflow](../.github/workflows/the-test-set-is-not-an-oracle.yml)
is restricted to the article and workflow paths, uses contents-read
permissions, and validates numeric contracts and the fresh notebook.

## Experimental designs

| Figure | Study key | Full repetitions | Mechanism deliberately isolated |
| --- | --- | ---: | --- |
| fig1_iid_validation_baseline.png | iid | 30 | IID logistic population, stratified five-fold CV, fresh test, and resubstitution at three training sizes |
| fig2_model_search_optimism.png | search | 36 | 1–500 unrelated noise-scoring candidates evaluated on one validation set; selected minimum versus fresh test |
| fig3_nested_cross_validation.png | nested | 20 | Six logistic regularization values; re-used inner CV, independent outer selection folds, fresh test |
| fig4_supervised_preprocessing_leak.png | preprocess | 25 | Pure-noise features; selecting features globally before CV versus within each training fold |
| fig5_temporal_validation_drift.png | temporal | 30 | Gradually changing logistic coefficient; random historical folds, forward historical folds, later period |
| fig6_lookahead_feature_leakage.png | lookahead | 25 | AR(1) horizon prediction; features available at forecast time versus future values revealed after forecast |
| fig7_hierarchical_group_validation.png | groups | 25 | Site-specific intercepts; mixing records of known sites versus holding out entire sites |
| fig8_fold_standard_error.png | uncertainty | 70 | Five correlated fold losses; naive fold-based SE, across-dataset spread, and independent-test inclusion |
| fig9_adaptive_holdout_reconstruction.png | adaptive | 30 | Synthetic exact-accuracy oracle reveals 120 fixed binary labels through single-coordinate queries |

All conditions and metric names are explicitly locked in tests, for both
regenerated study rows and the archived replicate/summary CSVs. Numeric
fields and optional theoretical references are checked for finiteness;
duplicate or missing repetitions and malformed CSV structure fail tests.
Quick-mode tests preseed canonical file names and verify they remain
byte-for-byte unchanged.

## Numerical cross-checks

- At 900 iid training records, stratified five-fold CV Brier loss is
  **0.1842** versus **0.1848** on a separate synthetic test draw.
- Searching 500 equally uninformative candidates reduces the selected
  validation Brier to **0.2398** while fresh test Brier stays **0.2651**.
  This is search optimism, not an improvement in the true null prediction.
- The six-value nested logistic study obtains **0.2055** on the reused
  inner CV, **0.2070** on nested outer CV, and **0.2000** on fresh test.
  Different training sizes and Monte Carlo variation mean the latter
  need not equal either CV estimate in a particular experiment.
- With 600 wholly uninformative features, global feature selection yields
  apparent CV Brier **0.1998**. Fold-safe CV yields **0.2962**, close to
  fresh test Brier **0.2995** for the overfitted selected model.
- Under strong temporal coefficient drift, random historical CV gives
  **0.2356**, forward historical CV **0.2557**, and later-period
  evaluation **0.3083**. Forward CV cannot anticipate an unseen regime.
- At a fourteen-step forecast horizon, causal features give Brier **0.2458**;
  illegally including the future process value gives **0.1578**.
- At site-intercept SD 1.8, record-mixed CV gives **0.1755**, versus
  **0.2535** for entirely unseen sites. At SD zero, group CV can do better;
  grouping is about the deployment target, not a guaranteed higher loss.
- The average naive fold-based SE is **0.0091**, while the empirical
  SD of CV means across new synthetic datasets is **0.0113**. A naive
  five-fold interval contains independent-test loss in only **78.6%**
  of repetitions under this design, not the nominal 95%.
- After 120 single-coordinate queries plus one baseline exact-score query,
  the deliberately exposed public labels are recovered completely and
  public accuracy reaches **100%**. Accuracy on fresh independent labels
  stays approximately **49.8%**.

## Scope, assumptions, and limitations

The figures are demonstrations of specified generators. IID validation does
not establish validity under shift. The model-search experiment intentionally
uses independent uninformative scores; its fresh test vector follows the
same null distribution, not the exact same fitted classifier. The nested
procedure evaluates tuning under smaller outer training sets; it is not
guaranteed to predict the risk of the final full-data fit exactly. Selecting
features using every target is knowingly invalid. The temporal study's
specified coefficient shift need not resemble actual drift. The look-ahead
experiment deliberately supplies a future observation unavailable at forecast
time, even though its test split respects chronology. Group CV addresses
unseen sites, which is different from deployment at known sites. Fold
standard errors do not account for all relevant variability; the nominal
inclusion diagnostic is against an independent synthetic test estimate from
a model fit on all training data, so training-size mismatch also matters.
Finally, the adaptive experiment uses a **hypothetical oracle exposing exact
accuracy after each query**. It does not access or attack a real benchmark,
and coarsened, restricted, or private feedback behaves differently.

The complete [private manuscript](https://github.com/DiogoRibeiro7/research-articles-medium/blob/main/themes/statistics-for-data-scientists/the-test-set-is-not-an-oracle/article.md)
provides the extended derivations, historical source audit, and publication
text. This public repository retains the code, original figures, evidence
tables, notebook, and tests needed to reproduce the scientific findings.
