# Statistical robustness and matched-sample sensitivity

This phase evaluates how much the study's descriptive results depend on
methodological choices. It does not fit an independent causal model or
assign sampling probabilities to a selected panel of countries.

## Wealth and GDP

Reuse the four prespecified windows in the [baseline](README.md):
1996–2020, 1995–2020, 2000–2020 and 1996–2019. Convert endpoint percentage
changes to annualized log growth, 100 × log(1 + growth/100) / elapsed years,
so different durations are comparable. Missing observations and nonpositive
endpoints remain unavailable.

For every window report three samples: all available countries, available
countries excluding Bulgaria, and the intersection of available countries
across all four windows. The fixed common sample separates a window effect
from a change in country composition. The Bulgaria exclusion is explicit
because the independently reviewed CWON source has an anomalous 1996
baseline; the observation is neither deleted from the raw data nor corrected.
Cyprus remains unavailable. Report mean and median annualized GDP-minus-wealth
growth gaps, Pearson and Spearman correlations, and the identities of
countries with GDP growth and wealth decline. A separate country table
distinguishes divergence in every requested window from divergence in every
available window.

## Historical shocks

Reuse the three benchmark definitions and episodes from [SHOCKS.md].
For each episode and GDP metric, compare all three benchmark pairs on the
intersection of countries with complete losses for all three. Report the
median absolute difference in estimated cumulative shortfall, strict sign
reversals, Pearson and rank correlations. These are benchmark sensitivities,
not causal effect estimates.

For each preshock feature (agriculture, industry, services, manufacturing,
three-sector HHI), correlate the feature with real GDP per capita shortfall
under each benchmark. Hold the feature-specific sample fixed across all
three benchmarks. Repeat for EU27 members only, excluding non-EU comparators.
Report included country identities and missing counts. Do not pool repeated
countries across episodes. A sign reversal means strictly opposite signs;
zero is neither positive nor negative.

## Sectoral granularity

Compare matched-country concentration rankings for 2000, 2010, 2019, 2021
and 2024: WDI three broad sectors versus four with manufacturing split out;
Eurostat A10 versus available A64 sections A–T. Each pair uses complete
concentration observations from both classifications and excludes the
Eurostat EU aggregate. A64 retains the reconciliation restriction in the
original module. Different HHI classifications have different mathematical
lower bounds, so their absolute levels are not directly comparable.
Rank agreement is the main granularity diagnostic, not proof of robustness
to every possible sector definition.

## Influence and interpretation

All correlation tables report Pearson r, Spearman r with average ranks
for ties, and the minimum and maximum Pearson correlation after omitting
each country in turn. Retain the omitted country identities, number of
defined leave-one-out values, and a strict sign-reversal indicator.
A constant input or a sample smaller than three has undefined correlation.
Leave-one-out ranges are influence diagnostics, **not confidence intervals**.
The selected countries share shocks, trading relationships and policies;
neither independence nor random sampling is assumed. No p-values or
causal claims are made. These checks do not resolve omitted-variable bias,
nonstationarity, wealth-accounting limitations or missing exposure data.

## Reproduce

From this directory, Python 3.12 or newer, standard library only:

```bash
python -m unittest discover -s tests -v
python robustness.py --fetch
python robustness.py
```

The script reuses verified registry contracts rather than downloading any
new data from the original providers. The workflow executes offline replay
and matched-sample assertions before uploading the following tables:

| File | Expected rows |
| --- | ---: |
| wealth_window_sensitivity.csv | 12 |
| wealth_country_stability.csv | 33 |
| shock_benchmark_sensitivity.csv | 18 |
| structure_correlation_sensitivity.csv | 90 |
| classification_sensitivity.csv | 10 |

The summary records source commits, output dimensions and countries with
GDP growth but wealth decline in all four requested windows. All empirical
claims in the manuscript must be traced to verified output values. No new
data or source licences are required.
