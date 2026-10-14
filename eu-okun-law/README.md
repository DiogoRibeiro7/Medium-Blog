# Okun's law revisited: an EU-27 empirical study

> Status: research in progress. The current version implements Eurostat acquisition
> data audits and descriptive regression code, not verified empirical country rankings, Bayesian estimates, or causal effects.

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

This initial change has passed synthetic offline tests only. A live Eurostat
download and review of country coverage, statuses and breaks are the next gate.

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

`eu-okun audit --output-dir data` checks that raw source hashes match the manifest, every EU-27 country-quarter row exists exactly once, all rates are finite and within 0–100%, and records missing values and quality flags by country. It also counts usable consecutive-quarter Okun observations with and without participation; no regression is fitted. The generated `data/coverage_audit.json` records SHA-256 hashes for the manifest and processed CSV. Coverage alone cannot establish methodological comparability, particularly around the 2021 Labour Force Survey change. A live fetch has not yet been verified in CI.

## Classical baseline (research stage)

After verifying a live snapshot and reviewing its audit, run:

```bash
eu-okun baseline --data-dir data --output results/ols_hac.json
```

This computes country-level ordinary least squares for unemployment-rate changes on current and 0/1/2/4 lags of GDP quarterly growth, optionally controlling for participation changes. Standard errors use an uncorrected Bartlett-kernel Newey–West HAC covariance with four lags; the output includes sample size, dates, conditioning and R-squared. The algorithm explicitly rejects rank-deficient/ill-conditioned designs and does not bridge missing calendar quarters. Models without enough complete observations are omitted, not fabricated. Concurrent GDP and participation terms mean these are descriptive associations and **not** a valid real-time forecasting experiment or causal shock responses. No empirical model is reported without source validation.

A path-scoped [live validation workflow](../.github/workflows/eu-okun-law-live.yml) retrieves actual Eurostat responses, performs the provenance audit and retains an artifact for review. Live workflow success does not by itself resolve methodological comparability or survey breaks.
