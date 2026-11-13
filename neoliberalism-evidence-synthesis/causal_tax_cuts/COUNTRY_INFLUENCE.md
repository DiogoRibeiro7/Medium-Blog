# Country-deletion influence for the tax-cut event study

## Question and design

Our inequality estimates come from a small number of reforming countries.
Are they stable when any one country is removed? A single large economy or
unusual reform episode could otherwise dominate the reported mean.

The new `country_influence.py` performs a **leave-one-country-out refit**,
one country at a time. Each deletion removes the country's *entire
1965–2015 observation history*, whether that country supplied treated
episodes, donors, or both. It then calls the same full balanced-cohort
event-time matching code from `event_study.py`, with the same lag, horizon,
donor and event eligibility criteria.

For each event time `k`, let `\widehat{\delta}(k)` be the full-sample
matched contrast and `\widehat{\delta}_{(-c)}(k)` the estimate from a
complete refit excluding country `c`. Report:

\[
 \Delta_c(k)=\widehat{\delta}_{(-c)}(k)-\widehat{\delta}(k).
\]

The influence summary records:

- The minimum and maximum re-estimated contrast across successful country
  deletions, and the largest absolute deviation from the full estimate.
- The most influential omitted country, breaking numerical ties
  alphabetically.
- Whether deleting a country **reverses** the contrast's sign (strictly
  positive to negative, or negative to positive).
- How many treated reform events survive each deletion, because country
  omission can change both the donor pools and the matched treatment cohort.
- Any deletion that cannot support the declared minimum event count; such
  entries retain a named status and **blank numerical estimates**.

All event-time values are retained, including the four pre-reform placebo
leads, so readers can see whether a country's influence differs before and
after the reform date.

## Why this is not a confidence interval

The interval between the smallest and largest leave-one-country-out estimate
is an **influence range, not a probability interval**. It has no 95% coverage
interpretation. Country deletion changes the treatment sample and donor
composition, and multiple reform events share donor countries. A naive
bootstrap of event-level rows would not preserve these dependencies.

With 18 countries and only 14 contributing matched reform events in the
first outcome vintage, formal inference requires careful attention to
cluster size, few-treated-cluster problems, and cross-country exposure.
See MacKinnon and Webb,
[*The wild bootstrap for few (treated) clusters*](https://doi.org/10.1111/ectj.12107),
for the limitations of common cluster-based procedures in such settings.
This repository does **not** claim to implement the methods in that paper.

## Reproducibility and output

```bash
python -m causal_tax_cuts.country_influence \
  --panel data/tax-study/panel.csv --outcome top1_share_pct \
  --lags 5 --horizon 5 --min-events 5 \
  --output data/tax-study/country-influence
```

Generated files:

1. `country_deletions.csv`: each omitted country and event time, its
   re-estimated mean, shift from the full estimate, matched event count,
   treated-country count, pretrend slope and explicit success/failure flag.
2. `country_influence_summary.csv`: per-horizon full estimate, deletion
   extrema, maximum influence, sign reversal count, surviving sample range
   and the number of successful/failed deletions.
3. `country_influence_audit.json`: input panel, specification and country
   roster, full-cohort counts, failed countries and a no-inference warning.

CI obtains the same authors' public reform CSV and original WID bulk files
as the previous research steps. It publishes the diagnostic outputs as
short-lived artifacts, and the first verified real-data summary is frozen
separately after the live run succeeds.

This checks fragility of **one major-tax-cut inequality contrast**. It does
not identify the causal effect of neoliberalism on the distribution of wages
or on housing affordability. The next separate empirical step is acquiring
and validating real macroeconomic covariates for pre-reform balance.
