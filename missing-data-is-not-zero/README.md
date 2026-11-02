# Article #30 — A Missing Value Is Not a Zero

**Full title:** *A Missing Value Is Not a Zero: MCAR, MAR, MNAR, Imputation, and the Limits of Complete Cases*

This is the reproducible companion to the [10,700+ word manuscript](article.md).
Nine independent synthetic studies produce **nine original publication
figures** and **eighteen machine-readable CSV tables**. Their generators are
fully specified. They do **not** estimate the distribution of missingness
mechanisms in any field population.

## Reproduce

Use Python 3.12 from this directory:

    python -m pip install -r requirements.txt
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python analysis.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
    RUN_NOTEBOOK_EXECUTION=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v

The final command runs the notebook in a **fresh Jupyter kernel** and
requires local IPC-socket access. CI installs notebook dependencies and
executes this full test sequence for relevant pull requests.

The generator supports these additional commands:

    python analysis.py --quick
    python analysis.py --quick --output /tmp/missing-data-smoke
    python analysis.py --seed 73 --output /tmp/missing-data-sensitivity
    python notebook_builder.py

Quick mode uses three independent outer replications per experiment,
except ten for the interval-coverage study. Its output names end in
_quick.csv or _quick.png, preventing it from overwriting canonical results.
The notebook runs quick mode inside a temporary output directory and
displays all nine newly generated figures. The committed notebook JSON
is deterministically built by notebook_builder.py.

The canonical seed is **20261101**. SeedSequence streams are keyed by the
base seed, a stable CRC32 experiment tag, and the outer replicate number.
Independent repetitions are genuinely independent; multiple conditions
within one repetition may reuse paired complete populations.

## Evidence inventory

| Experiment / CSV prefix | Figure | Full-run design | Main question |
| --- | --- | --- | --- |
| zero | 1 | n=1,800, MCAR q=.1/.3/.6, 24 repetitions | Is a missing marker a numerical zero? |
| mechanisms | 2 | n=2,100, MCAR/MAR/MNAR, 24 repetitions | When does weighting on observed Z repair complete cases? |
| mar_recovery | 3 | n=1,600, selection on fully observed Z, 24 repetitions | Complete-case mean vs oracle IPW and outcome standardization |
| variance | 4 | n=1,800, MCAR q=.1/.3/.5/.7, 24 repetitions | Mean-fill variance attenuation |
| intervals | 5 | n=120, MCAR q=.2/.5/.75, **300 repetitions**, 20 imputations each | True coverage of nominal 95% confidence intervals |
| conditional | 6 | n=1,800, four selection intensities, 24 repetitions | Conditional regression slope versus marginal mean |
| sensitivity | 7 | n=2,100, outcome-dependent MNAR and six delta assumptions, 24 repetitions | Pattern-mixture tipping points |
| indicator | 8 | 650 train/1,200 test, 24 repetitions | Missingness-flag prediction across policy changes |
| positivity | 9 | n=2,600, four selection intensities, 24 repetitions | Inverse-weight RMSE and effective sample size |

Each prefix has a pair of files under results/:

- PREFIX_replicates.csv: raw statistics, replicate ID, condition, metric, and
  any generator-known benchmark.
- PREFIX_summary.csv: independent-replication count, mean, sample SD,
  MCSE, ±1.96-MCSE interval, and optional benchmark.

The nine canonical figures under figures/ are derived from the same
summary objects and follow the order of the table above.

## Critical assumptions and distinctions

1. **R=1 means observed.** The Boolean mask is not the missingness flag M
   used in the predictor workflow experiment; in that experiment, M=1
   means the separate sensor was missing.
2. **MCAR** implies the observation indicator is independent of the
   complete values in the stated setup. A complete-case mean can be valid
   but less precise.
3. **MAR** requires conditional independence of missing Y and R given
   fully observed auxiliaries Z. It does not mean unconditional random
   missingness. With correct models and support, outcome standardization
   or inverse-probability weighting can estimate marginal targets.
4. **MNAR** remains dependent on unseen outcomes after conditioning on
   the relevant observed information. The data alone generally cannot
   recover the missing-group distribution; the pattern-mixture delta
   is therefore a **sensitivity assumption**, not an estimated fact.
5. **Hájek weighting** is self-normalized and uses observed outcomes
   only. This implementation accepts NaN in unobserved entries but
   verifies all observed values and probabilities; the floor on
   probabilities is an explicit bias-variance intervention.
6. **Multiple imputation** in the notebook is a narrowly defined normal
   model with *known variance*, a flat prior for an unknown mean, twenty
   posterior predictive completions, and Rubin's within/between pooling.
   It is not a general-purpose MICE library. MCAR complete-case inference
   remains valid in this deliberately simple example.
7. **Prediction** can benefit from missingness flags while a changed
   measurement protocol breaks the learned relation. This does not make
   the flag a structural cause of outcomes or identify missing targets.
8. **Monte Carlo errors** quantify finite simulation replication
   uncertainty; they are not confidence in MAR or intervals for a
   real population. For paired scenario contrasts use replicate-level
   differences, not independence assumptions.

## Validation and provenance

The **111-test suite** exercises typed input and mask checks, stream
independence, analytic identities, self-normalized weighting, effective
sample sizes, normal-model MI, study conditions, 18 CSV files,
9 PNG figures, numerical evidence claims, notebook structure, and
fresh-kernel notebook execution. When notebook dependencies or execution
are disabled locally, two notebook-specific tests skip; CI enables them.

The manuscript's source-audit table distinguishes original academic
literature from the nine synthetic numerical results. The results are
not transcribed from publications. All formulas, numeric claims, and
figure placements are documented in article.md and the companion source.

The repository's global CSV Git LFS filter is overridden for **this
article directory only** using a dedicated .gitattributes exception;
the article-specific GitHub Actions workflow is path-scoped.
