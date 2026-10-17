# Okun's law revisited: an EU-27 empirical study

> Status: research in progress. Live Eurostat acquisition, data audits and the classical
> (OLS/HAC) descriptive analyses run in CI, and their results have been independently
> verified and stress-tested in [docs/CLASSICAL_RESULTS_REVIEW.md](docs/CLASSICAL_RESULTS_REVIEW.md).
> Pre-registered hierarchical Bayesian estimates ([docs/HIERARCHICAL_BAYES.md](docs/HIERARCHICAL_BAYES.md))
> and a pre-registered predictive comparison ([docs/PREDICTIVE_COMPARISON.md](docs/PREDICTIVE_COMPARISON.md))
> are available. There are no country rankings and no causal effects.

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

## Pseudo-out-of-sample forecast validation

Run `eu-okun forecast --data-dir data --output-dir results/forecast` to evaluate one-quarter-ahead unemployment-rate changes with expanding-window training. Models use lagged GDP and unemployment predictors only and are compared against zero-change, historical-mean and AR(1) benchmarks. The output includes per-origin forecasts, country metrics, and a machine-readable protocol. The design uses subsequently revised Eurostat values and assumes end-of-previous-quarter inputs were observable; therefore it is **not** a real-time or vintage-correct backtest. See [FORECAST_VALIDATION.md](FORECAST_VALIDATION.md). The pandemic targets dominate the published scores; the [review](docs/CLASSICAL_RESULTS_REVIEW.md) re-scores them without 2020-2021 (section 7).

## Classical results review

```bash
eu-okun review --data-dir snapshots/eurostat-2026-10-15   --output-dir docs/review --report docs/CLASSICAL_RESULTS_REVIEW.md
```

[docs/CLASSICAL_RESULTS_REVIEW.md](docs/CLASSICAL_RESULTS_REVIEW.md) is generated by this command from the frozen snapshot; its tables and figures are in [docs/review/](docs/review/). It adds no new statistical model. It recomputes every published estimate by an independent numerical route, re-estimates the existing specifications on pre-declared alternative samples (pandemic excluded, pre-2020, quality-flag screened, post-2021), and re-scores the existing forecasts. In brief, for that snapshot:

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
  nine specifications, including the unpooled one), and for 7 more
  (BE, DE, HU, IE, NL, PL, RO) only when estimates borrow strength across countries.
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
  strongly (`tau_delta` about 0.03). After 2021 the sum is still credibly negative in 17
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
