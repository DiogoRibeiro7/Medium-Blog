# Okun's law revisited: an EU-27 empirical study

> Status: analysis complete; the article awaits the author's review. The manuscript
> (LaTeX), its Medium version, the manifests and the audit are in
> [publication/](publication/README.md). Every number in the article is read by code from
> the committed results. There are no country rankings and no causal effects.

## Research question

How stable is the link between quarterly real GDP growth and changes in unemployment
across the 27 EU countries, and to what extent can differences reflect labour-market
participation, macroeconomic regimes, measurement or institutions?

A reported 2009–2026 hierarchical Bayesian analysis motivated this study.
Its estimates for Spain, the Netherlands, Luxembourg and other countries are
hypotheses for independent examination, not conclusions to replicate.

We distinguish:

1. Descriptive conditional associations.
2. Out-of-sample predictive performance.
3. Causal effects of clearly identified economic shocks.

Dynamic Bayesian regression alone does not justify the third claim.

## Official data

Study period: 2009 Q1 to the latest verified complete quarter (initial cutoff:
2026 Q2). Data download begins in 2008 Q1 to provide differencing and lag history.
In practice the labour-force series start in 2009 Q1 for 25 countries (Eurostat's
break-corrected back series for the 2021 survey change mostly start in 2009), so the
first usable unemployment change, and every common EU-27 sample, starts in 2009 Q2.

| Quantity | Eurostat dataset | Filters | Units |
| --- | --- | --- | --- |
| Real GDP growth | `namq_10_gdp` | `B1GQ`, `CLV_PCH_PRE`, `SCA` | % change from previous quarter |
| Unemployment | `une_rt_q` | `Y15-74`, `T`, `PC_ACT`, `SA` | % of labour force |
| Labour-force participation | `lfsi_emp_q` | `ACT`, `Y15-74`, `T`, `PC_POP`, `SA` | % of resident population |

Sources: [GDP](https://doi.org/10.2908/namq_10_gdp),
[unemployment](https://doi.org/10.2908/une_rt_q),
[participation](https://doi.org/10.2908/lfsi_emp_q), via the
[Eurostat Statistics API](https://ec.europa.eu/eurostat/web/user-guides/data-browser/api-data-access/api-detailed-guidelines/api-statistics).

GDP growth is a quarterly percentage growth rate; unemployment and participation
are percentage levels. Changes in unemployment are measured in *percentage points*.
A one-percentage-point change in growth is not a one-percent rise in GDP levels.

The historical cohort contains the present-day EU-27 throughout, not a changing
country membership or an additional EU aggregate series. Eurostat uses `EL` for Greece.

## Run

Python 3.12 and Poetry 2.x, from `eu-okun-law/`. `poetry.toml` keeps the environment
project-local (`eu-okun-law/.venv`) and `poetry.lock` pins every dependency; CI installs
from the same lock.

```bash
poetry install --extras dev        # or --all-extras
poetry env info --path             # must print .../eu-okun-law/.venv
poetry run eu-okun fetch --end-quarter 2026-Q2 --output-dir data
poetry run eu-okun audit --output-dir data
poetry run pytest
poetry run ruff check src tests
poetry run mypy src
```

Poetry installs into whatever environment `VIRTUAL_ENV` names. If your shell sets it
(for example to a conda base environment), unset it first (`env -u VIRTUAL_ENV poetry
install ...`) and check `poetry env info --path`, or the packages land in that other
environment.

Extras: `report` (matplotlib, for the `review`, `bayes-report`, `bayes-checks` and
`predictive-report` figures), `bayes` (NumPyro, JAX, ArviZ, SciPy and h5netcdf, for the
hierarchical models, their NetCDF artifacts and the predictive comparison) and `dev`
(both, plus pytest, Ruff and mypy). The downloader needs internet access; all automated
tests are offline.

The command produces these untracked research inputs:

```text
data/
  raw/namq_10_gdp.json
  raw/une_rt_q.json
  raw/lfsi_emp_q.json
  processed/eu27_quarterly.csv
  snapshot_manifest.json
  coverage_audit.json
```

A provenance manifest records exact API URLs, download time, Eurostat update
metadata, response SHA-256, geographic coverage, missingness and quality flags.
The code does not interpolate or substitute zero for unavailable observations.
Save an immutable copy of the raw files/manifest for any published numerical result;
Eurostat historical data may be revised.

Live downloads run in CI on every pull request that touches the study (see the
workflows below). The snapshot used for the classical results review is frozen in
[`snapshots/eurostat-2026-10-15/`](snapshots/eurostat-2026-10-15/README.md) so that its
numbers can be regenerated after Eurostat revises the data.

## Analysis plan

Classical country-by-country distributed-lag regressions with HAC standard errors
and time-ordered holdouts (done, and independently reviewed), then hierarchical
partial-pooling Bayesian models with posterior rank uncertainty and shared shocks
(first stage done; see below), and a pre-registered predictive comparison of the two
(done; see below). The misfits found by the posterior predictive checks are addressed
by a registered amendment, and regime-specific and time-varying coefficients by another
(both done; see below). Only then attempt defensible structural identification
(distinguishing, for example, demand from productivity shocks). Investigate the
Netherlands and Luxembourg without assuming any institutional explanation in advance.

See [METHODS.md](METHODS.md), [ROADMAP.md](ROADMAP.md) and
[data/README.md](data/README.md).

## Licence

Project code/documentation use the repository's Apache 2.0 licence. Attribute
Eurostat and follow its reuse terms when distributing the data or results.

## Data-quality audit

`eu-okun audit --output-dir data` checks that raw source hashes match the manifest, every EU-27 country-quarter row exists exactly once, all rates are finite and within 0–100%, and records missing values and quality flags by country. It also counts usable consecutive-quarter Okun observations with and without participation; no regression is fitted. The generated `data/coverage_audit.json` records SHA-256 hashes for the manifest and processed CSV. Coverage alone cannot establish methodological comparability, particularly around the 2021 Labour Force Survey change. The audit retains Eurostat's flags but no estimation command uses them; section 2 of the [review](docs/CLASSICAL_RESULTS_REVIEW.md) documents where they occur (Spain and France follow a different unemployment definition from 2021; Germany's 2020 labour data are low-reliability; breaks are scattered across 2010-2024).

## Classical baseline (research stage)

After verifying a live snapshot and reviewing its audit, run:

```bash
eu-okun baseline --data-dir data --output results/ols_hac.json
```

This computes country-level ordinary least squares for unemployment-rate changes on current and 0/1/2/4 lags of GDP quarterly growth, optionally controlling for participation changes. Standard errors use an uncorrected Bartlett-kernel Newey–West HAC covariance with four lags, measured in calendar quarters like every other command; the output includes sample size, dates, conditioning and R-squared. The algorithm explicitly rejects rank-deficient/ill-conditioned designs and does not bridge missing calendar quarters. Models without enough complete observations are omitted, not fabricated. Concurrent GDP and participation terms mean these are descriptive associations and **not** a valid real-time forecasting experiment or causal shock responses. No empirical model is reported without source validation.

A path-scoped [live validation workflow](../.github/workflows/eu-okun-law-live.yml) retrieves actual Eurostat responses, performs the provenance audit and retains an artifact for review. Live workflow success does not by itself resolve methodological comparability or survey breaks.

## Cross-country coefficient sensitivity

After fetching and auditing the data, run:

```bash
eu-okun compare --data-dir data --output-dir results/comparison
```

The output contains `comparison.json`, `comparison.csv`, and `COMPARISON.md`. All 216 country/specification models share exactly the same calendar quarters across EU-27, all four GDP lag specifications and both participation-control settings. Complete cases are intersected before fitting. The GDP coefficient sum is evaluated with the full Newey–West HAC covariance matrix, including off-diagonal covariances. Reported 95% normal-approximation intervals are for each country's descriptive coefficient sum, *not* for pairwise country differences. The report shows all EU-27 countries alphabetically for the pre-specified two-lag uncontrolled model and specification sensitivity for PT, ES, NL and LU. Do not equate the coefficient sum with a structural GDP shock, and do not infer significant rankings from overlapping or non-overlapping country intervals.

The common sample includes 2020-2021. Those eight quarters supply most of the variation in GDP growth, so for many countries the published table mainly describes the pandemic period; see section 4 of the [review](docs/CLASSICAL_RESULTS_REVIEW.md) before quoting any coefficient from it.

## Macroeconomic regime sensitivity

Run `eu-okun regimes --data-dir data --output-dir results/regimes` to obtain `regimes.json`, `regimes.csv` and `REGIMES.md`. Windows: full sample (2009 Q1–2026 Q2), pre-pandemic (2009–2019), 2020–2021 excluded, and post-2021 (2022 Q1 onward). The post-2021 window uses GDP lag orders 0/1 only; longer lags require more observations. All windows enforce a common EU-27 complete-case sample within that window, exclude any cross-boundary input history, and report non-estimable regimes explicitly. These are descriptive stability checks, not formal structural-break tests or causal shock estimates.

## Formal pre/post stability and dependent country comparisons

Run `eu-okun infer --data-dir data --output-dir results/inference` on the audited source. Fully interacted pre-2020 versus post-2021 GDP-lag (K=1) regressions produce calendar-aware HAC Wald statistics and Holm-adjusted tests of GDP-sum change. Pandemic outcomes and their input history are excluded. PT/ES/NL/LU differences use synchronized circular time-block resampling (lengths 4/8/12 quarters; seed 20261015; 999 draws). Bootstrap percentile intervals and sign frequencies are descriptive, not null-centered tests or causal rankings.

### Small-sample inference for the post-2021 window

[SMALL_SAMPLE_PROTOCOL.md](SMALL_SAMPLE_PROTOCOL.md), registered before any bootstrap,
keeps the data, design, statistic and Holm family of these stability tests and replaces
only the normal reference distribution. It uses:

- null-imposed sieve (AR(1), wild) and moving-block bootstraps of the HAC t-statistic,
  with B = 999;
- a Monte Carlo size study (200 null data sets per test, nested bootstrap);
- bootstrap-t intervals for the post-2021 sums.

Results: [docs/SMALL_SAMPLE_INFERENCE.md](docs/SMALL_SAMPLE_INFERENCE.md).

```bash
poetry run eu-okun small-sample --data-dir snapshots/eurostat-2026-10-15 \
  --output-dir docs/small-sample --report docs/SMALL_SAMPLE_INFERENCE.md --review-dir docs/review
```

In brief:

- **Change tests.** None of the 14 published Holm-significant changes survives:
  - 0 of 54 under the sieve bootstrap and 0 under the block bootstrap;
  - again 0 in a repeat with B = 9999, which checks that the result is not an artefact
    of the p-value resolution.
- **Size of the published test.** Under each test's own fitted null, the normal
  reference rejects 22.8% of the time at a nominal 5% (51 of 54 tests above the
  binomial bound). The bootstrap test rejects 4.8%, which is adequate.
- **Post-2021 sums.** Without the participation variants, the sign is reportable in 4
  of 27 countries (BG and RO negative, DK and PL positive), against about 1.4 expected
  by chance.
- **Comparison.** This agrees with the hierarchical regime-shift model
  ([docs/BAYES_REGIMES.md](docs/BAYES_REGIMES.md)), which finds no credible
  post-2021 shift. The review's "weaker after 2021" list remains a descriptive pattern,
  not a tested result.
- **Run time.** The study takes about ten seconds, and a test regenerates the committed
  tables in the ordinary CI.

## Pseudo-out-of-sample forecast validation

Run `eu-okun forecast --data-dir data --output-dir results/forecast` to evaluate one-quarter-ahead unemployment-rate changes with expanding-window training. Models use lagged GDP and unemployment predictors only and are compared against zero-change, historical-mean and AR(1) benchmarks. The output includes per-origin forecasts, country metrics, and a machine-readable protocol. The design uses subsequently revised Eurostat values and assumes end-of-previous-quarter inputs were observable; therefore it is **not** a real-time or vintage-correct backtest. See [FORECAST_VALIDATION.md](FORECAST_VALIDATION.md). The pandemic targets dominate the published scores; the [review](docs/CLASSICAL_RESULTS_REVIEW.md) re-scores them without 2020-2021 (section 7).

## Classical results review

```bash
poetry run eu-okun review --data-dir snapshots/eurostat-2026-10-15 --output-dir docs/review --report docs/CLASSICAL_RESULTS_REVIEW.md --published-dir snapshots/ci-results-38754743850 --provenance "CI artifact eu-okun-empirical-baselines, workflow run 38754743850 (commit e73ee68)"
```

[docs/CLASSICAL_RESULTS_REVIEW.md](docs/CLASSICAL_RESULTS_REVIEW.md) is generated by this command from the frozen snapshot;
the CI outputs it compares against are frozen in
[snapshots/ci-results-38754743850/](snapshots/ci-results-38754743850/README.md), because
workflow artifacts expire after 30 days. its tables and figures are in [docs/review/](docs/review/). It adds no new statistical model. It recomputes every published estimate by an independent numerical route, re-estimates the existing specifications on pre-declared alternative samples (pandemic excluded, pre-2020, quality-flag screened, post-2021), and re-scores the existing forecasts. In brief, for that snapshot:

- The published numbers are computed correctly, and the archived CI outputs are reproduced exactly.
- Outside 2020-2021, unemployment-rate changes are negatively associated with GDP growth in most countries; the published full-sample table is strongly attenuated by the pandemic quarters.
- Post-2021 coefficients, country-pair differences and the value of GDP for forecasting are inconclusive or sensitivity-dependent.

The report separates verified results, sensitivity-dependent results, methodological limitations and unresolved questions. Every number in it, and every judgement that depends on a number, is computed from the outputs under decision rules stated in the report; the remaining text quotes Eurostat metadata or states data-independent limitations. Regenerate it rather than editing it. In CI, the estimates workflow runs the same review on the freshly downloaded data and archives it with the other results.

## Hierarchical Bayesian analysis

The model, priors, sensitivity set, convergence criteria and decision rules were
pre-registered in [BAYES_PROTOCOL.md](BAYES_PROTOCOL.md) and committed before any
model was fitted to Eurostat data. Country Okun coefficient sums are partially
pooled, with a common quarterly factor for cross-country dependence, country AR(1)
Student-t errors, and Eurostat-flagged observations masked out of the likelihood.

```bash
# Expensive (about an hour on a laptop CPU for all nine specifications)
poetry run eu-okun bayes-fit --data-dir snapshots/eurostat-2026-10-15 --output-dir results/bayes
# Cheap: tables, figures and report from the saved draws
poetry run eu-okun bayes-report --fits-dir results/bayes --output-dir docs/bayes   --report docs/HIERARCHICAL_BAYES.md --data-dir snapshots/eurostat-2026-10-15   --replication-dir docs/bayes/replication-linux
```

Sampling never runs in the ordinary pull-request checks. The
[EU Okun hierarchical Bayes](../.github/workflows/eu-okun-law-bayes.yml) workflow runs
it on demand, or when a pull request is labelled `bayesian-sampling`, and archives the
posterior draws. Results: [docs/HIERARCHICAL_BAYES.md](docs/HIERARCHICAL_BAYES.md).
They are descriptive conditional associations; partial pooling identifies no shock.

In brief, for the 2026-10-15 snapshot and 2009-2019:

- The association between GDP growth and unemployment changes is credibly negative
  for 15 countries in every fit accepted by either of two independent runs (together all
  nine specifications, including the unpooled one). Seven more (BE, DE, HU, IE, NL, PL,
  RO) are credibly negative in every fit except the unpooled one, which missed the R-hat
  threshold here (1.0106 against 1.01) and passed on Linux; so the count is 22 or 15
  depending on that fit. Their own data do show an association: OLS on the same quarters
  excludes zero for HU, IE, NL, PL and RO.
  Its size depends on the error model, more than on pooling.
- A second run on Linux reproduces the posterior medians within 0.004.
- The pre-registered model fails one predictive check (it understates the persistence
  of unemployment changes), and its common factor turns out to be correlated with
  average GDP growth, so in factor specifications the country coefficients describe
  the association beyond the common European cycle.
- Of the three earlier claims, "Spain large" is supported; "Netherlands delayed" and
  "Luxembourg weak" are not resolved under the pre-registered rules.

### Shared quarter effects, prior predictive checks and recovery

Amendment 1 of the protocol (section 9), registered before its fits, adds additive
shared quarter effects (sum to zero) with and without participation, prior predictive
checks, posterior recovery on synthetic data and ArviZ NetCDF artifacts
(`posterior.nc` in every fit directory: posterior, sample statistics, prior, prior
predictive, observed and constant data, pointwise log likelihood, with seeds, settings,
versions and snapshot hashes). Results: [docs/BAYES_MODEL_CHECKS.md](docs/BAYES_MODEL_CHECKS.md).

```bash
poetry run eu-okun bayes-fit --data-dir snapshots/eurostat-2026-10-15 --output-dir results/bayes   --specs time_effects,time_effects_participation,participation_no_common_factor --retry-warmup 3000
poetry run eu-okun bayes-export --fits-dir results/bayes --data-dir snapshots/eurostat-2026-10-15
poetry run eu-okun bayes-recovery --data-dir snapshots/eurostat-2026-10-15 --output-dir results/bayes-recovery
poetry run eu-okun bayes-checks --fits-dir results/bayes --data-dir snapshots/eurostat-2026-10-15   --output-dir docs/bayes-checks --report docs/BAYES_MODEL_CHECKS.md --recovery-dir results/bayes-recovery
```

In brief:

- All three new specifications converge. By PSIS-LOO, additive quarter effects are
  better supported than no shared effect (elpd +22.2, SE 9.3), and the pre-registered
  common factor with country loadings is better supported than additive effects
  (+23.7, SE 9.2); additive effects also under-reproduce the cross-country correlation
  (0.095 replicated against 0.168 observed).
- Under quarter effects `mu_theta` is -0.209 (90% interval -0.269 to -0.154), close to
  the primary -0.198; the quarter effects correlate -0.53 with average GDP growth, so
  they absorb part of the common GDP association.
- The priors are generous: under `primary`, 3.9% of prior-simulated quarterly changes
  exceed 5 pp, against a largest observed change of 3.0 pp.
- Recovery on synthetic data: 24, 25 and 23 of 27 true country sums lie inside their
  90% intervals (`primary`, `time_effects`, `gaussian_iid`); the primary model's
  factor values are under-covered (32 of 43), and the `gaussian_iid` recovery fit
  narrowly failed acceptance (max R-hat 1.0104 after the retry).
- An independent Linux run of the workflow gives the same four LOO verdicts and
  recovery counts within one country (tables in `docs/bayes-checks/replication-linux/`).

### Robustness of country differences and ranks

Amendment 2 of the protocol (section 10), registered before any cross-specification
pairwise or rank result was computed and before the K = 1 specification (`lags_1`) was
fitted. Results: [docs/BAYES_ROBUSTNESS.md](docs/BAYES_ROBUSTNESS.md).

```bash
poetry run eu-okun bayes-fit --data-dir snapshots/eurostat-2026-10-15 --output-dir results/bayes \
  --specs lags_1 --retry-warmup 3000
poetry run eu-okun bayes-robustness --fits-dir results/bayes --output-dir docs/bayes-robustness \
  --report docs/BAYES_ROBUSTNESS.md --pairs-csv docs/review/pairs_review.csv
```

In brief (pairwise probabilities from aligned joint draws; a difference survives when it
is credible, P >= 0.95, in the same direction in every accepted fit of the registered
set):

- 99 of 351 country pairs are credible in the primary fit, but only 59 survive every
  accepted specification (lag orders 1, 2 and 4, priors, pooling, error model, common
  factor, participation, pandemic exclusion); adding the quarter-effect fits leaves 59.
- No rank is established: the median 90% rank interval spans 15 places in the primary
  fit and 18 across specifications.
- Of the classical review's apparent differences among the four focus countries, Spain's
  more negative sum than Luxembourg's and the Netherlands' survives every specification;
  Luxembourg against the Netherlands does not. Spain against Portugal is borderline: it
  falls just below 0.95 in two fits here (0.947 and 0.939) and survives in an independent
  Linux run, where one of those fits narrowly failed acceptance; it is not a robust result.

### Dynamics that address the misfits

Amendment 3 of the protocol (section 11), registered before any of its fits, adds three
specifications:

- `dynamic_factor`: the common factor follows a stationary AR(1);
- `ardl`: the lagged unemployment change `psi[i] * y[i,t-1]` replaces the AR(1) error,
  and the comparable estimand is the long-run multiplier `theta_LR = theta / (1 - psi)`;
- `ardl_dynamic_factor`: both.

Results: [docs/BAYES_DYNAMICS.md](docs/BAYES_DYNAMICS.md).

```bash
poetry run eu-okun bayes-fit --data-dir snapshots/eurostat-2026-10-15 --output-dir results/bayes \
  --specs dynamic_factor,ardl,ardl_dynamic_factor --retry-warmup 3000
poetry run eu-okun bayes-dynamics --fits-dir results/bayes --data-dir snapshots/eurostat-2026-10-15 \
  --output-dir docs/bayes-dynamics --report docs/BAYES_DYNAMICS.md
```

In brief:

- All three specifications converge at the first attempt. The common factor is strongly
  persistent: `rho_f` is 0.92 (90% interval 0.86 to 0.96), and 0.91 with the lagged
  change.
- `ardl_dynamic_factor` fixes the persistence misfit in both runs (p = 0.102 here and
  0.130 on Linux; it replicates a lag-1 autocorrelation of 0.18 against 0.27 observed).
  `ardl` does not (p = 0.007 in both runs). `dynamic_factor` sits on the threshold: p is
  0.030 here and 0.052 on Linux, so its verdict is not stable.
- By PSIS-LOO on the 1098 cells common to all four fits, `dynamic_factor` (+15.5, SE
  5.8) and `ardl_dynamic_factor` (+26.2, SE 7.9) are better supported than `primary`;
  `ardl` is not distinguishable from it (+14.4, SE 7.3). These three verdicts are the same
  on Linux. Combining the two changes beats `ardl` here (+11.8, SE 5.7) but not on Linux
  (+10.7, SE 5.8): not stable.
- The country estimates change little on average: the mean is -0.200 in `primary`
  and -0.168 in `ardl_dynamic_factor`. The largest change is Spain's (-0.512 to -0.397).
  25 countries are credibly negative in `ardl_dynamic_factor` (24 in `primary`, plus
  Malta).
- On the long-run multiplier, "Luxembourg weak" holds in both ARDL fits. This follows
  from Luxembourg's negative lagged-change coefficient (about -0.4), which shrinks its
  long-run multiplier; on the short-run sum the probability is about 0.81. "Netherlands
  delayed" falls below its threshold in every new fit, except `dynamic_factor` on Linux
  (0.948 here). "Spain large" holds in all of them.
- The factor stays correlated with average GDP growth (median correlation -0.50 to
  -0.68 across the new fits), so `theta` remains an association beyond the common
  European cycle.
- An independent Linux run of the workflow reproduces the posterior medians within
  0.006, `rho_f`, the credibly negative sets and every other verdict. The tables are in
  `docs/bayes-dynamics/replication-linux/`.
- The new fits are reported alongside the pre-registered results and do not replace
  the primary specification.

### Regime-specific and time-varying coefficients

Amendment 4 of the protocol (section 12), registered before any of its fits, asks whether
the association differs between 2009-2019 and 2022-2026. It fits five specifications on
the pandemic-excluded windows:

- `regime_shift`: post-2021 shifts of each country's GDP coefficient sum, partially
  pooled, with intercept and error-scale shifts;
- the same with terms that use provisional GDP masked (`regime_shift_final_gdp`), and
  with the amendment 3 dynamics (`regime_shift_dynamic`, against
  `pandemic_excluded_dynamic`);
- `drifting_association`: a common random walk in the coefficient sum.

Results: [docs/BAYES_REGIMES.md](docs/BAYES_REGIMES.md).

```bash
poetry run eu-okun bayes-fit --data-dir snapshots/eurostat-2026-10-15 --output-dir results/bayes \
  --specs regime_shift,regime_shift_final_gdp,pandemic_excluded_dynamic,regime_shift_dynamic,drifting_association \
  --retry-warmup 3000
poetry run eu-okun bayes-regime-recovery --data-dir snapshots/eurostat-2026-10-15 \
  --output-dir results/bayes-regime-recovery
poetry run eu-okun bayes-regimes --fits-dir results/bayes --data-dir snapshots/eurostat-2026-10-15 \
  --output-dir docs/bayes-regimes --report docs/BAYES_REGIMES.md --recovery-dir results/bayes-regime-recovery
```

In brief:

- **No credible shift.** The average shift of the sum after 2021 is +0.028 (90%
  interval -0.007 to +0.072, P(> 0) = 0.90): no credible change. An average weakening
  larger than about 0.07 pp per pp is outside the interval. The recovery check detected
  a known shift of +0.15 on its synthetic data set.
- **Countries.** No evaluable country shows a credible shift. The shifts are pooled
  strongly (`tau_delta` about 0.03), so this mostly reflects pooling and low power, not
  evidence that every country is stable. After 2021 the sum is still credibly negative in 17
  of the 25 evaluable countries. Spain and France have no post-2021 terms (definition
  change) and are not evaluable.
- **Classical comparison.** This disagrees with the classical review's
  weaker-after-2021 set of 13 countries. The designs differ in lag order, sample start,
  pooling, the common factor and intercept shifts; which difference drives the
  disagreement is not tested.
- **Sensitivities.** With the amendment 3 dynamics no verdict changes. The
  provisional-GDP sensitivity failed the convergence criteria after the retry (R-hat
  1.012 for Spain's AR(1) coefficient), so it cannot be judged.
- **Drift.**
  - By PSIS-LOO the common random walk is better supported than a fixed coefficient in
    both runs (+14.2, SE 6.0 here; +13.6 on Linux).
  - Against the regime shift it is better supported here (+4.3, SE 1.8) but not on
    Linux (+3.5, SE 1.8), so that verdict is not stable.
  - Its period contrasts are not credible: 2014-2019 against 2009-2013 is +0.021
    (P = 0.92), and 2022-2026 against 2014-2019 is +0.004.
- **Regime shift.** By PSIS-LOO the regime shift is not distinguishable from one
  coefficient (+9.9, SE 5.8).
- **Replication.** An independent Linux run reproduces every other verdict, the shift
  medians within 0.003 and the recovery. `regime_shift_final_gdp` failed acceptance there
  too. The tables are in `docs/bayes-regimes/replication-linux/`.

## Predictive comparison

Pre-registered in [PREDICTIVE_PROTOCOL.md](PREDICTIVE_PROTOCOL.md) before any held-out
prediction was computed. Each model predicts the quarter's unemployment change from data
up to the previous quarter **plus the quarter's GDP growth** (a conditional prediction
on revised data, not a real-time forecast), re-estimated at every origin, and every
model is scored on the same cells.

```bash
# Expensive: 3 Bayesian specifications x 32 origins
poetry run eu-okun predictive-fit --data-dir snapshots/eurostat-2026-10-15 --output-dir results/predictive
# Cheap: classical predictions, scores, tests and report
poetry run eu-okun predictive-report --data-dir snapshots/eurostat-2026-10-15   --fits-dir results/predictive --output-dir docs/predictive --report docs/PREDICTIVE_COMPARISON.md
```

The refits run in the [EU Okun predictive comparison](../.github/workflows/eu-okun-law-predictive.yml)
workflow on demand, or when a pull request is labelled `predictive-sampling`. In brief,
for the 2026-10-15 snapshot (mean CRPS, panel Diebold-Mariano test on cross-country mean
loss differentials, Holm-adjusted over ten comparisons):

- **2016-2019 targets.** Given the quarter's GDP growth, every Okun model has a lower
  mean CRPS than no change. The hierarchical model's 13.5% advantage is the only decided
  comparison, and it is no longer decided once the quarters with failed refits are
  dropped. The registered hierarchical-versus-classical comparisons differ by 0.5-1.2%
  (not distinguishable).
- **2022-2026 targets.** No model using GDP growth clearly beats no change. The
  hierarchical models score 4-9% better than the classical regressions, but not at the
  5% familywise level (Holm p 0.05-0.13). Pooling alone sits on the threshold: Holm
  p 0.051 here and 0.048 in an independent Linux run, so that verdict is not robust.
- Predictive intervals are somewhat too wide (90% intervals cover 90-98% of outcomes),
  and all models over-predict unemployment changes in 2016-2019 and under-predict them
  after 2022.

## One-quarter-ahead forecasts: hierarchical Bayes against the classical models

Pre-registered in [BAYES_FORECAST_PROTOCOL.md](BAYES_FORECAST_PROTOCOL.md) before any fit
or score. Unlike the conditional comparison above, these are genuine one-quarter-ahead
forecasts in the PR #149 framework: the forecast for quarter t uses unemployment,
participation and GDP growth up to t-1 only (nothing dated t), with the framework's
origins (2018-Q1 to 2026-Q2), training rows and targets for every model. They are
**retrospective** (revised data), not real-time forecasts from historical vintages.

```bash
# Expensive: 3 hierarchical forecasting models x 34 origins
poetry run eu-okun forecast-bayes-fit --data-dir snapshots/eurostat-2026-10-15 --output-dir results/forecast-bayes
# Cheap: classical forecasts (exactly those of eu-okun forecast), scores, bootstrap, report
poetry run eu-okun forecast-report --data-dir snapshots/eurostat-2026-10-15 \
  --fits-dir results/forecast-bayes --output-dir docs/forecast-comparison --report docs/FORECAST_COMPARISON.md
```

The refits run in the [EU Okun predictive comparison](../.github/workflows/eu-okun-law-predictive.yml)
workflow with `study=forecast`, or when a pull request is labelled `forecast-sampling`.
Results: [docs/FORECAST_COMPARISON.md](docs/FORECAST_COMPARISON.md). In brief, for the
2026-10-15 snapshot (CRPS; synchronised circular block bootstrap; Holm over five
comparisons):

- **All 34 targets.** None of the five pre-registered comparisons is decided. The
  hierarchical AR(1) + GDP-lag model scores 15.3% better than its classical counterpart,
  mostly because the classical regressions forecast badly in 2020-Q3 and 2020-Q4 (mean
  CRPS difference -0.046 over all quarters, -0.013 without those two); persistence keeps
  the lowest MAE and RMSE.
- **Outside 2020-Q1..2022-Q2 (descriptive sensitivity).** The hierarchical models beat
  their classical counterparts by about 3% and the common factor helps further; these
  differences are distinguishable from sampling variation in that subset.
- **Convergence.** 31 of 102 refits miss the strict acceptance criteria after the retry
  (R-hat up to 1.017, ESS down to 246, no divergences; mostly the population scale
  parameter); their forecasts are scored as registered, and dropping their quarters
  changes no verdict.
- An independent Linux run of the refits gives the same five verdicts (tables in
  `docs/forecast-comparison/replication-linux/`).

## Can the relationship be read causally? (identification study)

[docs/IDENTIFICATION_STUDY.md](docs/IDENTIFICATION_STUDY.md) is the methodological
decision document. It formulates the identification problem, shows that a reduced-form
Okun coefficient is a weighted average of shock-conditional coefficients with unknown
weights, draws the causal graph, and evaluates recursive orderings, long-run and sign
restrictions and external instruments, with their testable and untestable assumptions
and the public data available. Only local projections on two externally identified
common shocks were judged defensible; they were pre-registered in
[IDENTIFICATION_PROTOCOL.md](IDENTIFICATION_PROTOCOL.md).

```bash
poetry install --extras "report structural"
poetry run eu-okun identification --data-dir snapshots/eurostat-2026-10-15 \
  --shocks-dir snapshots/external-shocks-2026-10-16 --output-dir docs/identification \
  --report docs/IDENTIFICATION_RESULTS.md
```

The shock series (Jarociński and Karadi 2020, ECB monetary policy; Känzig 2021, oil supply
news; both CC BY 4.0) are frozen with SHA-256 hashes and attribution in
[snapshots/external-shocks-2026-10-16](snapshots/external-shocks-2026-10-16/README.md).
Results: [docs/IDENTIFICATION_RESULTS.md](docs/IDENTIFICATION_RESULTS.md). In brief:

- Productivity shocks, labour-supply shocks and country-specific GDP shocks cannot be
  identified with public data for the EU-27.
- For the two common shocks, the shock-conditional Okun coefficients are not determined
  by 2009-2025 quarterly data: first stages are weak and every primary Anderson-Rubin
  set is unbounded, so demand- and supply-driven Okun coefficients cannot be compared.
- A monetary tightening moves GDP and unemployment on impact only; the primary oil
  series is predictable from lagged EU data and is labelled "identification doubtful".
- **Conclusion: the data do not justify a causal reading of the GDP-unemployment
  relationship. Every Okun coefficient of this study remains a descriptive association.**

### Monthly outcomes for the identified shocks (amendment)

[MONTHLY_IDENTIFICATION_PROTOCOL.md](MONTHLY_IDENTIFICATION_PROTOCOL.md) was committed and
pushed before any estimate. It asks whether the two shocks, kept at their native monthly
frequency, become informative with monthly unemployment and industrial production
(NACE B-D) as outcomes.

- **Sample.** Eurostat's metadata show that monthly unemployment is a trend, a moving
  average or interpolated between quarterly survey results in most countries. Those
  series put later months into a month's value. The primary sample therefore keeps the
  eight countries with direct monthly survey estimates, and the other groups are a
  sensitivity check.
- **Ratio.** It is `kappa_IP`, pp of unemployment per 1% of industrial production. It is
  not the GDP-based Okun coefficient.

```bash
poetry run eu-okun fetch-monthly --start-month 2000-01 \
  --output-dir snapshots/eurostat-monthly-2026-10-18      # already frozen; needs internet
poetry run eu-okun monthly-identification --data-dir snapshots/eurostat-monthly-2026-10-18 \
  --shocks-dir snapshots/external-shocks-2026-10-16 --quarterly-dir docs/identification \
  --output-dir docs/monthly-identification --report docs/MONTHLY_IDENTIFICATION_RESULTS.md
```

Results are in [docs/MONTHLY_IDENTIFICATION_RESULTS.md](docs/MONTHLY_IDENTIFICATION_RESULTS.md);
the run takes about 20 seconds. In brief:

- **Relevance does not improve.** The largest first-stage |t| is 1.21 with monthly
  industrial production, against 1.27 with quarterly GDP. All four primary ratio sets are
  unbounded.
- **Responses.** Unemployment is not distinguishable from zero at any horizon. The
  primary oil series gives industrial production *rising* after negative supply news,
  which is against theory, as with quarterly GDP.
- **Testable assumptions survive.**
  - Neither shock is predictable from lagged monthly EU data. The oil surprise was
    predictable only after summing to quarters.
  - No placebo test rejects.
- **Robustness.** The only bounded sets (the VAR-based oil news shock at 24 months, and
  monetary policy from 2001) contain zero.
- **The quarterly conclusion stands.** The bottleneck is weak instruments for real
  activity, not the frequency of the data.

### Country-specific GDP shocks: is a foreign-demand instrument feasible?

[SHIFT_SHARE_PROTOCOL.md](SHIFT_SHARE_PROTOCOL.md) was committed and pushed before any
computation. It asks two questions before any IV estimate is run:

- which trading partners make a shift-share foreign-demand instrument's exclusion
  restriction defensible;
- whether the resulting instrument can be strong enough.

**The instrument.** It combines each country's value added absorbed in partner p's final
demand (OECD TiVA 2025, 2005-2007) with partner p's quarterly GDP growth (OECD quarterly
national accounts).

```bash
poetry run eu-okun fetch-shift-share --output-dir snapshots/shift-share-2026-10-18   # needs internet
poetry run eu-okun shift-share --data-dir snapshots/shift-share-2026-10-18 \
  --eurostat-dir snapshots/eurostat-2026-10-15 --output-dir docs/shift-share \
  --report docs/SHIFT_SHARE_FEASIBILITY.md
```

Results are in [docs/SHIFT_SHARE_FEASIBILITY.md](docs/SHIFT_SHARE_FEASIBILITY.md); the run
takes about 5 seconds. In brief:

- **No go.** For distant, non-European demand (the only defensible set), the expected
  first-stage t is 0.19 against the registered threshold of 3.16. Even the most
  favourable calibration gives 0.77.
- **Why.**
  - Every country's distant exposure is dominated by the United States.
  - After quarter effects, which absorb the common EU cycle, the instrument varies by
    0.006 pp, against residual GDP-growth noise of 1.18 pp.
  - Ireland supplies 45% of the remaining variation.
- **Other partner sets.** Near extra-EU partners (P2: t = 0.31) and intra-EU partners
  (P3: t = 0.66) barely raise relevance, and they make the exclusion restriction
  indefensible.
- **Consequence.** No IV is estimated, and country-specific GDP shocks remain
  unidentified.

### Structural VAR restrictions: demand, technology and labour supply

[SIGN_RESTRICTIONS_PROTOCOL.md](SIGN_RESTRICTIONS_PROTOCOL.md) was committed and pushed
before any estimate. It sets up three things:

- **Model.** A pooled panel VAR(2) in GDP growth, deflator inflation and the changes in
  unemployment and participation, with country fixed effects.
- **Shocks.** Demand, technology and labour-supply shocks, set-identified by impact sign
  restrictions.
- **Inference.** Robust Bayesian (Giacomini and Kitagawa 2021): for every posterior draw,
  the whole identified set over admissible rotations, with no prior over rotations.

The recursive orderings are reported only as an orthogonalised, non-causal measure of
sensitivity to ordering.

```bash
poetry run eu-okun fetch-deflator --output-dir snapshots/eurostat-deflator-2026-10-18   # needs internet
poetry run eu-okun sign-restrictions --data-dir snapshots/eurostat-2026-10-15   --deflator-dir snapshots/eurostat-deflator-2026-10-18 --output-dir docs/sign-restrictions   --report docs/SIGN_RESTRICTIONS_RESULTS.md                                    # about 12 minutes
```

Results are in [docs/SIGN_RESTRICTIONS_RESULTS.md](docs/SIGN_RESTRICTIONS_RESULTS.md). In
brief:

- **Plausibility.** Every posterior draw admits the impact restrictions. Imposing them
  also at h = 1 leaves 1% of draws, so the data contradict that variant.
- **Demand against technology.** Both Okun-ratio sets are unbounded at 4 and 8 quarters,
  so the two cannot be distinguished, in every variant.
- **Labour supply against demand.** They differ at impact only by assumption. At 8
  quarters both unemployment responses include zero, and each shock's share of the
  forecast-error variance of the unemployment change spans 0-96%.
- **The uniform-rotation posterior** that sign-restriction studies usually report looks
  sharp: demand -0.43, technology -0.55. That sharpness comes from the prior over
  rotations, not from the data.
- **Recursive orderings (not causal)** move the "GDP shock" Okun ratio between -0.135 and
  -0.202 at 4 quarters and between -0.149 and -0.216 at 8 quarters.
- **Conclusion.** Restrictions the data cannot reject leave the productivity and
  labour-supply questions unanswered. Every Okun coefficient remains descriptive.

## Country heterogeneity: Portugal, Spain, the Netherlands and Luxembourg

[HETEROGENEITY_PROTOCOL.md](HETEROGENEITY_PROTOCOL.md) was committed and pushed before any
computation. It states hypotheses whose predictions can fail, and evaluates each of them
by the same rule in every country. It does not retrofit explanations to unusual
coefficients.

**Data.** Official series only, frozen in `snapshots/heterogeneity-2026-10-18/`:

- Eurostat national-accounts employment, domestic and resident, and hours;
- LFS temporary and permanent employees;
- temporary and part-time shares;
- OECD employment-protection indices.

```bash
poetry run eu-okun fetch-heterogeneity --output-dir snapshots/heterogeneity-2026-10-18   # needs internet
poetry run eu-okun heterogeneity --data-dir snapshots/eurostat-2026-10-15   --snapshot snapshots/heterogeneity-2026-10-18 --output-dir docs/heterogeneity   --report docs/HETEROGENEITY_CASE_STUDY.md                                     # about 30 seconds
```

Results are in [docs/HETEROGENEITY_CASE_STUDY.md](docs/HETEROGENEITY_CASE_STUDY.md). In
brief, with associations only and 2020-2021 excluded:

- **Luxembourg.** The measurement-mismatch hypothesis is not supported. Domestic
  employment, which includes cross-border commuters, is as flat as resident employment
  (elasticities +0.03 and +0.02). Output per hour absorbs Luxembourg's GDP fluctuations
  (elasticity +1.06).
- **The Netherlands.** One prediction is supported: it absorbs more of output
  fluctuations through output per hour than Spain does (+0.52 against −0.21). The
  delayed unemployment response is not distinguishable from Spain's except before 2020,
  and timing says nothing about temporary contracts.
- **Spain and Portugal.**
  - Spain's domestic-employment elasticity is about one (+1.11, 95% interval +0.78 to
    +1.28), and its output-per-hour elasticity is negative but not distinguishable from
    zero (-0.21, interval -0.39 to +0.03).
  - Portugal's hours respond as much as Spain's (+1.27 against +1.21), so the
    Spain-Portugal unemployment difference disappears in hours. Portugal publishes no
    seasonally adjusted employment in persons, so its hours-per-worker margin cannot be
    measured.
- **Temporary jobs.** The temporary-employment buffer is not determined in any country
  (Spain only before 2020).
- **Institutions, as correlation.** Stricter employment protection goes with a stronger
  unemployment response, the opposite of the registered prediction.
  - Six of the 18 registered correlations concern employment protection: two indicators
    in three variants. All six are negative.
  - None is decided in the primary (Bayesian) variant (Holm p 0.94 and 0.095).
  - Two are decided, both in other variants. These are correlations over at most 20
    countries.
- **No identified causal effect.** The report lists the hypotheses that cannot be tested
  reliably.

## Small-sample inference for the classical headline sums

[HEADLINE_INFERENCE_PROTOCOL.md](HEADLINE_INFERENCE_PROTOCOL.md) was committed and pushed
before any bootstrap or simulation. It keeps the published headline statistic (two-lag
sum, pandemic-excluded common sample, calendar-aware Newey-West t) and replaces only its
normal reference with null-imposed sieve and moving-block bootstraps (B = 9999), adds
bootstrap-t intervals, and runs a size study.

```bash
poetry run eu-okun headline-inference --data-dir snapshots/eurostat-2026-10-15 --grid docs/review/grid.csv --output-dir docs/headline-inference --report docs/HEADLINE_INFERENCE.md
```

Results are in [docs/HEADLINE_INFERENCE.md](docs/HEADLINE_INFERENCE.md). In brief:

- **The normal reference over-rejects.** Under each country's fitted null it rejects
  22.0% of the time at a nominal 5% (23 of 27 countries above the binomial range); the
  sieve bootstrap rejects 5.1%.
- **Unadjusted.** 20 of 27 sums are negative under both bootstraps (normal: 22; LV and
  RO no longer, LV is scheme-dependent).
- **After Holm over the 27 countries.** 1 sum survives under both bootstraps (normal: 21).
- **Caveat (diagnostic added after the run).** Imposing the null raises the residual
  persistence when a sum is large (median AR(1) +0.17 before 2020), so the registered
  p-values understate the evidence for large sums. The bootstrap-t intervals, from the
  unrestricted residuals, lie below zero in 19 countries.

**Post-hoc checks (not registered).** An external review (2026-10-20) pointed out that the
size study draws its null from the same restricted residuals. It also asked which error
rate the country count should control. [docs/NULL_RESIDUALS.md](docs/NULL_RESIDUALS.md)
repeats the headline and post-2021 studies with the error process taken from the
unrestricted residuals, keeping the registered designs, draws and seeds:

```bash
poetry run eu-okun null-residuals --data-dir snapshots/eurostat-2026-10-15 --headline-dir docs/headline-inference --small-sample-dir docs/small-sample --output-dir docs/null-residuals --report docs/NULL_RESIDUALS.md
```

- **Size on the less persistent null.**
  - The normal reference rejects 16.6% instead of 22.0%, still too often.
  - The registered bootstrap holds its size (5.1%).
  - A bootstrap that draws its errors from the unrestricted residuals over-rejects
    (8.0%). Its Holm count of 7 therefore does not replace the registered 1.
- **Error rate.**
  - With either scheme alone, Holm keeps 4.
  - Benjamini-Hochberg (false discovery rate) on the registered p-values keeps 19.
  - Benjamini-Yekutieli, valid under any dependence, keeps 1.
- **Post-2021.**
  - No change survives Holm under either error process.
  - The normal reference rejects 18.1% instead of 22.8%.

## Measurement noise or finance? Luxembourg's weak output-labour link

[GDP_MEASUREMENT_PROTOCOL.md](GDP_MEASUREMENT_PROTOCOL.md) was committed and pushed before
any of its data were frozen or any estimate was computed. It tests two rival explanations
of Luxembourg's pattern (output per hour moving one for one with output, labour input
barely moving): transitory noise in quarterly output, and the financial sector. Ireland
and Malta are secondary focus countries. Data: value added and hours by industry, total
and finance, frozen in `snapshots/gdp-measurement-2026-10-19/`.

```bash
poetry run eu-okun fetch-gdp-measurement --output-dir snapshots/gdp-measurement-2026-10-19   # needs internet
poetry run eu-okun gdp-measurement --data-dir snapshots/eurostat-2026-10-15 --snapshot snapshots/gdp-measurement-2026-10-19 --output-dir docs/gdp-measurement --report docs/GDP_MEASUREMENT.md
```

Results are in [docs/GDP_MEASUREMENT.md](docs/GDP_MEASUREMENT.md). One of eight
predictions is supported; under the interpretation rules fixed in advance, both
explanations fail for Luxembourg:

- **Noise: not established.** "Fail" is the protocol's label for "not supported", and
  for noise it means not established rather than ruled out.
  - Luxembourg's quarterly growth is less autocorrelated than the comparison group's
    (-0.17 against +0.19). The difference, -0.36 [-0.61, -0.02], excludes zero before
    adjustment but is not decided after Holm.
  - Its hours-output correlation does not rise at the year-on-year horizon (+0.18 quarter
    on quarter, +0.12 year on year), whereas elsewhere it rises by a median 0.36. That
    points away from transitory noise.
- **Finance.** Finance is 28% of Luxembourg's value added and 11% of its hours, yet
  removing it from output and hours barely changes the link (output per hour +1.00 in
  total, +0.95 outside finance).
- **Ireland.** Its quarterly growth is negatively autocorrelated (-0.31, the one
  supported prediction), but its year-on-year correlation rises like everyone else's.
  The test does not single Ireland out: in the comparison group Sweden is lower (-0.35),
  and Estonia (-0.25) and Denmark (-0.22) are close.

Not tested: Ireland's modified domestic demand (Irish CSO), annual accounts, error shared
by output and hours, sectoral composition beyond finance.

## Does the quarterly unemployment rate inherit monthly smoothing?

[SMOOTHING_PROTOCOL.md](SMOOTHING_PROTOCOL.md) was committed and pushed before any
comparison. In several countries the monthly rate is a trend (DE, NL), a centred moving
average (EE, PT) or benchmarked to moving annual averages (HR, LU). If the quarterly rate
were their average, that smoothing could produce the Bayesian persistence misfit and the
Dutch delayed response. No new data are needed.

```bash
poetry run eu-okun smoothing --data-dir snapshots/eurostat-2026-10-15 --monthly-dir snapshots/eurostat-monthly-2026-10-18 --review-dir docs/review --bayes-dir docs/bayes --output-dir docs/smoothing --report docs/SMOOTHING.md
```

Results are in [docs/SMOOTHING.md](docs/SMOOTHING.md):

- **Inheritance: the check does not discriminate.**
  - None of the six smoothed countries' quarterly rates is classed as the average of its
    monthly rate (correlation of changes: NL 0.91, DE 0.80, HR 0.60, LU 0.13).
  - The same rule also rejects 5 of the 8 direct-estimate countries, and NL's 0.91 ties
    Hungary and Romania.
  - The better evidence is Eurostat's metadata for `une_rt_m` (section 18.5). It
    benchmarks the monthly series to the quarterly survey figures, not the reverse.
- **Persistence (not determined).** Smoothed countries are, if anything, less persistent
  (+0.15 against +0.26 for direct estimates). Without them the misfit statistic rises from
  +0.27 to +0.30.
- **Timing (not determined).** Lag shares are about equal (+0.65 against +0.62).
- **What these tests show.** They compare 6 countries with 8, so they show no evidence
  that smoothing drives the other results, not evidence that it does not.
- **Luxembourg (descriptive, not registered).** Its quarterly unemployment changes are
  strongly negatively autocorrelated (-0.51), the signature of survey sampling noise.
  - The noise widens the intervals of its two-lag sum without biasing that sum.
  - In the ARDL models it produces the negative lagged-change coefficient (-0.43), which
    shrinks the long-run sum. Luxembourg's "weak" result on the long-run sum is therefore
    an artefact.

## Definition breaks and data vintages

[MEASUREMENT_PROTOCOL.md](MEASUREMENT_PROTOCOL.md) was committed and pushed before the
new release was downloaded or any estimate was computed.

- **The rule.** It documents the study's single definition-break rule
  (`review.flagged_change`) and where each analysis applies it.
- **Sensitivity A.** The registered headline inference is repeated, with its seeds, on
  the rows the rule keeps.
- **Sensitivity B.** The same inference is run on Eurostat's latest release.

```bash
poetry run eu-okun fetch --start-quarter 2008-Q1 --end-quarter 2026-Q2 --output-dir snapshots/eurostat-2026-10-22   # needs internet
poetry run eu-okun measurement --frozen snapshots/eurostat-2026-10-15 --new snapshots/eurostat-2026-10-22 --grid docs/review/grid.csv --output-dir docs/measurement --report docs/MEASUREMENT.md
```

Results are in [docs/MEASUREMENT.md](docs/MEASUREMENT.md):

- **Under the rule** (all post-2021 changes for ES and FR dropped, and a few break
  quarters elsewhere):
  - 17 sums are negative under both bootstraps, against 20;
  - 2 remain after Holm, against 1;
  - 16 remain under Benjamini-Hochberg, against 19;
  - Bulgaria, Germany and Poland lose significance under one scheme;
  - Spain moves from -0.79 to -0.93, and France from -0.30 to -0.41.
- **Latest release.**
  - The labour-force series are byte-identical (no release since 2026-09-10).
  - GDP is revised in 134 of 1998 cells, in 6 countries, by at most 0.80 pp.
  - No count or country classification changes, and no sum moves by more than 0.042.
  - Eurostat publishes no historical vintages of these country series, so this bounds
    only the revisions since the freeze.

## The EU as a whole: a labour-force-weighted aggregate

[AGGREGATE_PROTOCOL.md](AGGREGATE_PROTOCOL.md) was committed and pushed before its data
were frozen or any estimate was computed.

- **Data.** A new snapshot, `snapshots/eu-aggregate-2026-10-21`, adds unemployed persons
  and rates (for labour-force weights) and the official `EU27_2020` series.
- **Estimates.**
  - The labour-force-weighted mean of the country sums, against the unweighted mean.
  - The Okun sum of the EU aggregate itself.
- **Tests.** Both differences are tested two-sided, with Holm.

```bash
poetry run eu-okun fetch-eu-aggregate --output-dir snapshots/eu-aggregate-2026-10-21   # needs internet
poetry run eu-okun aggregate --data-dir snapshots/eurostat-2026-10-15 --snapshot snapshots/eu-aggregate-2026-10-21 --case-study docs/heterogeneity/elasticities.csv --cce docs/cce/countries.csv --output-dir docs/aggregate --report docs/AGGREGATE.md
```

Results are in [docs/AGGREGATE.md](docs/AGGREGATE.md):

- **Estimates.**
  - Unweighted mean -0.29 [-0.33, -0.19].
  - Labour-force-weighted mean -0.33 [-0.40, -0.21]. DE, FR, IT, ES and PL hold 65% of
    the weight.
  - The EU aggregate -0.45 [-0.57, -0.25].
- **Tests.**
  - Weighting changes the mean by -0.05 [-0.09, +0.00].
  - The aggregate differs from the weighted mean by -0.12 [-0.22, -0.00].
  - Both have Holm p 0.10, so neither is decided. 3 of 8 sensitivity tests are decided.
- **Data check.** The labour-force-weighted country rate matches the official EU rate
  within 0.07 pp.

## A common-correlated-effects mean-group panel

[CCE_PROTOCOL.md](CCE_PROTOCOL.md) was committed and pushed before any CCE estimate was
computed. It applies Pesaran's (2006) CCE mean-group estimator to the headline sample.
Each country's two-lag regression is augmented with the cross-country averages of GDP
growth (lags 0-2) and of the unemployment change. P1 tests the CCE mean sum with the
mean-group t statistic. P2 tests its difference from the plain mean group with a
synchronised block bootstrap. Holm is applied over the two.

```bash
poetry run eu-okun cce --data-dir snapshots/eurostat-2026-10-15 --case-study docs/heterogeneity/elasticities.csv --bayes docs/bayes/country_estimates.csv --output-dir docs/cce --report docs/CCE.md
```

Results are in [docs/CCE.md](docs/CCE.md):

- **P1, supported.** The CCE mean sum is -0.17 [-0.22, -0.08]: the association survives
  removing common shocks.
- **P2, supported.** It is less negative than the plain mean group (-0.29) by +0.11
  [+0.06, +0.17], which is close to the Bayesian factor model.
- **Sensitivity.** Both hold in all six settings: lags 1 and 4, leave-one-out averages,
  block 8, without IE and LU, and before 2020.
- **Countries.**
  - CCE intervals exclude zero in 8 countries, against 19 with OLS.
  - Spain stays at -0.51, while the Netherlands moves from -0.28 to +0.07.
  - The rank correlation with the Bayesian medians is +0.74.

## Asymmetry between expansions and contractions

[ASYMMETRY_PROTOCOL.md](ASYMMETRY_PROTOCOL.md) was committed and pushed before any
asymmetric estimate was computed. It adds the negative part of GDP growth, with lags, to
the headline regression. Only GDP was inspected beforehand: the counts of negative-growth
quarters, 320 in all. No new data are needed.

- **Primary.** A common extra response to negative growth, pooled over the 27 countries.
  Each country keeps its own intercept and Okun slopes.
- **Secondary.** The mean of the country-by-country asymmetries, over the 23 countries
  with at least 8 negative-growth quarters.
- **Inference.** Synchronised block bootstrap over quarters, with Holm over the two
  tests.

```bash
poetry run eu-okun asymmetry --data-dir snapshots/eurostat-2026-10-15 --case-study docs/heterogeneity/elasticities.csv --output-dir docs/asymmetry --report docs/ASYMMETRY.md
```

Results are in [docs/ASYMMETRY.md](docs/ASYMMETRY.md):

- **Pooled.** Unemployment responds by -0.25 per point of positive growth and -0.35 per
  point of negative growth. The difference is -0.10 [-0.19, +0.03] (Holm p 0.26, not
  determined).
- **Mean group.** -0.10 [-0.38, +0.11], not determined.
- **Sensitivity.**
  - The pooled difference is negative in all seven settings: lags 1 and 4, a mean-growth
    threshold, block 8, without IE and LU, without EL, and before 2020.
  - Neither test is decided in any setting.
  - The mean group with block 8 is labelled unreliable, because 6.8% of its draws are
    degenerate.
- **Reading.** Contractions cluster in 2009-2013 and in the euro-area crisis countries,
  so even a decided difference could reflect the shocks rather than the labour market.

## The core robustness grid without the participation control

Half of the review's 24 core specifications per country control for the change in
participation, which responds to output and shares its denominator with the unemployment
rate. [CORE_GRID_PROTOCOL.md](CORE_GRID_PROTOCOL.md) was committed and pushed before the
re-classification: the same verdict rule on the 12 estimates without the control, after
reproducing the published verdicts. No estimate is recomputed.

```bash
poetry run eu-okun core-grid --grid docs/review/grid.csv --robustness docs/review/robustness.csv --output-dir docs/core-grid --report docs/CORE_GRID.md
```

Results are in [docs/CORE_GRID.md](docs/CORE_GRID.md):

- **Robust negative:** 18 countries, against 17 published; 22 without the full sample,
  against 21.
- **Changed:** Romania only, from specification-dependent to robust negative. The focus
  countries keep their verdicts (Spain and the Netherlands robust, Portugal
  specification-dependent, Luxembourg sign-unstable).
- The manuscript and the Medium article now use these verdicts and quote the published
  count alongside.

## Okun's law in hours, with and without the pandemic

[HOURS_PROTOCOL.md](HOURS_PROTOCOL.md) was committed and pushed before any full-sample
elasticity was computed. It extends the case study's estimator to every country with hours
data. It also tests the job-retention explanation of the pandemic attenuation: hours should
keep following output in 2020-2021 while unemployment does not, and the adjustment should
move to hours per worker. The pandemic-excluded hours elasticities were already in the
case study and had been seen, so they are reported as descriptive. No new data are needed.

```bash
poetry run eu-okun hours --data-dir snapshots/eurostat-2026-10-15 --snapshot snapshots/heterogeneity-2026-10-18 --case-study docs/heterogeneity/elasticities.csv --output-dir docs/hours --report docs/HOURS.md
```

Results are in [docs/HOURS.md](docs/HOURS.md):

- **Point estimates.**
  - With the pandemic, the mean hours elasticity rises from +0.43 to +0.74 (retention
    ratio 1.70). The mean unemployment elasticity shrinks from -0.29 to -0.19 (0.67).
  - Hours per worker take up more of the adjustment (+0.34), output per hour less (-0.30).
    Persons respond as usual (-0.04).
- **Registered tests: all not determined.** The registered tests of P1, P2a and P2b could
  not have been decided. 3.3% of the resamples contain no pandemic quarter, and in those
  the differences are exactly zero, so the p-value cannot fall below 0.067.
- **Amendment (registered after that result, before it was computed).** The amendment holds
  the number of pandemic quarters fixed in every resample.
  - It supports P1 (Holm p 0.004) and P2a (0.024).
  - P2b (fewer persons adjusting) and P3 (the cross-country correlation) are not
    determined.
  - P1 holds in all four sensitivity settings, and P2a in three.
- **Descriptive.** Hours intervals lie above zero in 10 of 26 countries without the
  pandemic and in 15 with it. The rank correlation between the unemployment and hours
  responses is +0.55.

## Publication: manuscript, Medium article and reproduction

[publication/](publication/README.md) holds the final outputs:

- the LaTeX manuscript ([publication/manuscript/](publication/manuscript/));
- the Medium article generated from the same values ([publication/medium/article.md](publication/medium/article.md));
- the data, experiment, environment and value manifests ([publication/manifest/](publication/manifest/));
- the audit of every prior output ([publication/AUDIT.md](publication/AUDIT.md)).

Every number in the text is read by code from the committed result tables, and the build
fails on an undefined value. Reproduction steps, one command per line, are in
[publication/README.md](publication/README.md). In short:

```bash
poetry run eu-okun publication --check
poetry run eu-okun publication --build
```

Then, from `publication/manuscript/`:

```bash
latexmk -pdf main.tex
```

The `publication` job of the
[EU Okun empirical estimates](../.github/workflows/eu-okun-law-estimates.yml) workflow
checks the generated files and compiles the PDF. It uploads the PDF as an artifact.
Nothing is published automatically.
