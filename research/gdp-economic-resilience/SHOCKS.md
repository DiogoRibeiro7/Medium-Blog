# Historical shock analysis

Companion to the [research protocol](RESEARCH_PROTOCOL.md). Uses the existing
World Bank WDI and Eurostat A10 pinned inputs, with no additional acquisition.
Run from this directory with Python 3.12+ (standard library only):

```bash
python -m unittest discover -s tests -v
python historical_shocks.py --fetch
python historical_shocks.py
```

## Predefined windows

| Episode | Onset | Five-year fit | Ten-year fit | Anchor | Primary comparison |
| --- | --- | --- | --- | --- | --- |
| Global financial crisis | 2008 | 2003–2007 | 1998–2007 | 2007 | 2008–2010 |
| COVID-19 pandemic | 2020 | 2015–2019 | 2010–2019 | 2019 | 2020–2022 |
| Energy-price shock | 2022 | 2015–2019 | 2010–2019 | 2021 | 2022–2024 |

The three-year window is identical in length. All paths additionally extend
through 2025 for recovery reporting, subject to observed coverage. 2025
Eurostat estimates may be provisional or missing; flags are retained in the
path table. Follow-up windows overlap subsequent shocks and have different
lengths, so recovery is not attributed solely to the initial episode.

## Mechanical benchmarks and outcomes

For each country, metric, or published Eurostat A10 sector, fit OLS to log
levels over the exact predeclared five or ten annual observations. Project
the estimated slope from the observed anchor level (not the fitted intercept).
The third benchmark holds the anchor flat. Missing or nonpositive fit or
anchor values make the trend benchmark unavailable; never switch silently
to a shorter training window. For 2022 the fit ends in 2019 but the anchor is
2021, so the projection conditions on the pandemic-era level and does not
assume a pandemic-free 2021.

For actual level Y(t) and benchmark B(t):

- Annual gap: 100 × [Y(t)/B(t) − 1].
- Signed cumulative shortfall: 100 × sum[B(t) − Y(t)] / sum[B(t)] in the
  fixed three-year window. Negative shortfall means outperformance.
- Worst trend gap and maximum drop below the anchor over that same window.
- First observed return to the anchor after an observed drop, through 2025.
  Distinguish no drop, recovered, right-censored, missing follow-up and
  missing anchor. A missing year before an apparent recovery prevents
  identification of the first return year. Duration is from episode onset,
  not necessarily the year of the first drop.

The country-level comparisons report both real GDP and real GDP per capita;
the latter is the main outcome for pre-shock structure associations. Eurostat
uses published real GVA (chain-linked 2020 values) for TOTAL, ten disjoint A10
sectors and manufacturing C as a separate diagnostic overlay. Never sum
chain-linked levels or treat manufacturing as an additional independent sector.

Associations: join WDI pre-onset agriculture, industry, services, manufacturing
shares of GDP and broad three-sector HHI to the primary five-year trend
shortfall for real GDP per capita. Report unweighted descriptive Pearson
correlations separately for each episode with explicit pairwise sample counts
and included country IDs. No imputation, pooling, significance claims or causal
effects. The overlapping sector shares are not independent covariates.

## Deliverables

- `wdi_shock_paths.csv`: country GDP and GDP-per-capita paths.
- `wdi_shock_metrics.csv`: fixed-window losses and recovery summaries.
- `eurostat_shock_paths.csv`: published sector real GVA paths and original flags.
- `eurostat_shock_metrics.csv`: sector-level losses and recovery summaries.
- `pre_shock_structure.csv`: country-level feature/outcome pairs, including missing values.
- `structure_loss_correlations.csv`: fifteen descriptive feature/episode correlations.
- `summary.json`: source commits, windows, coverage and table sizes.

The workflow tests synthetic cases, verifies data hashes, reruns the analysis
offline, compares output bytes and uploads the result artifact.

## Limits of interpretation

The forecast paths are transparent reference scenarios, **not identified
causal counterfactuals**. Trend extrapolation may be wrong because of secular
slowdowns, policy changes, inflation dynamics or other shocks. The episode
names identify time windows rather than independently estimated treatment
effects. The energy-price shock overlaps the pandemic recovery and changes
in monetary policy. Sector shares partly reflect price levels. Realized
shortfalls cannot establish that a service sector is obsolete, that
manufacturing causes resilience, or that a country's economic policy caused
the observed outcome. Test alternative benchmarks before interpreting any
ranking. No new source licensing or canonical data changes are needed.
