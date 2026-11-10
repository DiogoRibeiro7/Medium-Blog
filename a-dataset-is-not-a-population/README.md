# A Dataset Is Not a Population — public research companion

This directory contains the reproducible synthetic experiments supporting
**A Dataset Is Not a Population**. The full [editorial manuscript](https://github.com/DiogoRibeiro7/research-articles-medium/blob/main/themes/statistics-for-data-scientists/a-dataset-is-not-a-population/article.md)
is maintained privately, while [`article.md`](article.md) preserves the
historical public path as a compatibility notice.

The [research plan](RESEARCH_PLAN.md) specifies nine implemented studies.
Their code, canonical evidence tables, figures, notebook and scientific audit
remain in this public repository.

## Implemented studies

1. `srs`: simple random sampling without replacement, with a fixed finite population.
2. `strata`: disproportionate allocation, with unweighted, poststratified,
   Horvitz–Thompson, and Hájek estimates. Here HT and Hájek coincide because
   the design weights sum to the known population count.
3. `nonresponse`: outcome-dependent response and an explicitly **oracle**
   known-propensity weighted comparison. Actual researchers would not know
   the latent outcome shock driving selection. A covariate-only, deliberately
   misspecified response model demonstrates residual bias.
4. `coverage`: a sample from a frame that excludes the high-outcome 20%
   segment entirely. Its inclusion probability is zero, preventing design-
   based identification of the full-population mean.
5. `volunteer`: a large opt-in panel with outcome-related selection compared
   with a smaller probability sample from the same fixed population.
6. `calibration`: matching the marginal shares of two binary covariates while
   reversing their joint-cell frequencies. Marginal raking is the identity
   in this design; externally known joint counts permit poststratification.
7. `weighting`: unequal Poisson inclusion with known probabilities, comparing
   unweighted, raw inverse-weight and 90th-percentile capped-weight Hájek
   estimators. The cap is a sensitivity choice, not a principled optimum.
8. `predictive`: a deliberately misspecified constant predictor, estimated
   under training covariates centred at −1 and deployed to target covariates
   centred at +1. The conditional outcome law stays invariant; exact density
   ratios allow target-weighted fitting. Reported losses are compared with
   the irreducible Bayes squared-error risk of 0.25, not a survey mean.
9. `drift`: historical versus updated group shares, distinguishing a stable
   group outcome law from a subsequently changed group outcome law. Known
   target composition alone cannot fix a change in conditional means.

All populations and responses are synthetic. No empirical survey is used.
Each outer repetition has a study-specific stable random stream. The SRS and
stratified studies share a fixed population across repetitions, while the
nonresponse study draws a new independent finite population each repetition.
The `mean_target` in the latter study is averaged over replicate-specific
finite-population means; the archived errors always use their corresponding
replication's finite-population target.

## Run

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python analysis.py --quick
python analysis.py --output /tmp/population-experiments --repetitions 300
```

Quick mode produces eighteen `*_quick.csv` files in `data/`, which are not
publication outputs. Full provisional runs must select an explicit output
directory to avoid accidentally presenting unreviewed numbers as canonical.
The summary interval is a normal-approximation 95% Monte Carlo interval for
the **mean simulation bias**, not a survey confidence interval for a
population parameter. A future experiment will evaluate design interval
coverage directly.

## Verification scope

The unittest suite (including an opt-in notebook execution test)
verifies deterministic streams,
complete methods and finite rows, algebraic archive identities, known-weight
correction of stratified oversampling, and an oracle-weight improvement over
the intentionally misspecified response mechanism, undercoverage, opt-in
selection bias, marginal-versus-joint calibration, weight clipping,
covariate-shift prediction risk and concept shift. They are regression
checks, not substitutes for the remaining experimental, theoretical and
source audit requirements. The public checks also validate the nine canonical PNG
headers and dimensions without relying on manuscript formatting.

## Full synthetic evidence and figures

The deterministic full-run protocol uses 300 independent repetitions per
study, with the default seed 20261102. Each study writes a replication-level
CSV and a summary CSV. The plotting script reads the archived replications
and creates one PNG per study, with the mean signed error and ±1.96 Monte
Carlo standard errors. For predictive risk, signed error means excess
squared-error loss above the theoretical Bayes noise risk 0.25.

```bash
python analysis.py --output /tmp/article34-full --repetitions 300
python figures.py --data /tmp/article34-full --output /tmp/article34-figures
```

These outputs are synthetic experiment evidence, not representative empirical
survey observations. Cross-repetition intervals quantify simulation means.
The plot colors are descriptive visual markers and must not be interpreted as
formal accept/reject decisions. Editorial source review takes place in the
private manuscript repository; public CI validates the generated evidence.

## Executable notebook

The notebook is built deterministically from `notebook_builder.py`. It has
no committed execution outputs and writes exclusively to a temporary
directory. The execution test sets `POPULATION_ARTICLE_DIR`, starts a fresh
Python kernel from an unrelated working directory, and verifies that all
canonical CSVs remain unchanged.

```bash
python notebook_builder.py
RUN_NOTEBOOK_EXECUTION=1 python -m unittest discover -s tests -v
```

The notebook runs three replications per study, generates eighteen temporary
CSVs and nine temporary figures, and displays explicit interpretation limits.
It does not rerun 300-repetition publication evidence on every execution.

## Scientific audit

See [SCIENTIFIC_AUDIT.md](SCIENTIFIC_AUDIT.md) for a study-by-study audit of
the 300-repetition results, the realised Monte Carlo deviations, the
limits of the identifying assumptions, and the primary reference checks.
The test suite regenerates all nine full studies in a temporary directory
and compares every canonical replicate and summary CSV row, with exact
identifiers and a tight 1e-12 numeric tolerance for cross-platform rounding.

