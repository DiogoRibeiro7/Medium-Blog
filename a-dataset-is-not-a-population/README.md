# A Dataset Is Not a Population — research companion (in progress)

This directory is the staged, synthetic research companion for Article #34.
The [research plan](RESEARCH_PLAN.md) specifies nine studies. **Only the
first six are implemented.** There is no complete article, final figure
set, publication dataset, or notebook yet; this branch remains a draft.

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

Quick mode produces twelve `*_quick.csv` files in `data/`, which are not
publication outputs. Full provisional runs must select an explicit output
directory to avoid accidentally presenting unreviewed numbers as canonical.
The summary interval is a normal-approximation 95% Monte Carlo interval for
the **mean simulation bias**, not a survey confidence interval for a
population parameter. A future experiment will evaluate design interval
coverage directly.

## Verification scope

The nine standard-library unittest cases verify deterministic streams,
complete methods and finite rows, algebraic archive identities, known-weight
correction of stratified oversampling, and an oracle-weight improvement over
the intentionally misspecified response mechanism, undercoverage, opt-in
selection bias and marginal-versus-joint calibration. They are regression
checks, not substitutes for the remaining experimental, theoretical and
source audit requirements.
