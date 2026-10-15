# Okun's law revisited: an EU-27 empirical study

> Status: research in progress. Live Eurostat acquisition, data audits and the classical
> (OLS/HAC) descriptive analyses run in CI, and their results have been independently
> verified and stress-tested in [docs/CLASSICAL_RESULTS_REVIEW.md](docs/CLASSICAL_RESULTS_REVIEW.md).
> There are no country rankings, Bayesian estimates or causal effects yet.

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

Python 3.12 or later, from `eu-okun-law/`:

```bash
poetry install --extras dev
poetry run eu-okun fetch --end-quarter 2026-Q2 --output-dir data
poetry run eu-okun audit --output-dir data
poetry run pytest
poetry run ruff check src tests
poetry run mypy src
```

The `dev` extra includes matplotlib for the review figures; outside development,
install the `report` extra (`pip install -e ".[report]"`) to run `eu-okun review`.

Or use `python -m pip install -e ".[dev]"` and then run `eu-okun fetch`.
The downloader needs internet access; all automated tests are offline.

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

## Planned analysis

Start with classical country-by-country distributed-lag regressions, HAC
standard errors and genuine time-ordered holdouts. Compare those results with
hierarchical partial-pooling Bayesian models, posterior country-rank
uncertainty, shared-shock effects and regime-change sensitivity. Only then
attempt defensible structural identification (distinguishing, for example,
demand from productivity shocks). Investigate the Netherlands and Luxembourg
without assuming any institutional explanation in advance.

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
