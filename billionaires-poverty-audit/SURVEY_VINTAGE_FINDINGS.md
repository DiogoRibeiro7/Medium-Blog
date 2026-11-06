# Survey-vintage sensitivity: reproduced numerical findings

Evidence status: **descriptive exploratory citizenship comparison**, not the original 64-country residence chart and not a causal effect.

## Reproduction

- Workflow: [Reproduce billionaire-poverty exploratory figures, run 42203993729](https://github.com/DiogoRibeiro7/Medium-Blog/actions/runs/42203993729)
- PR under test: [#544](https://github.com/DiogoRibeiro7/Medium-Blog/pull/544)
- Head commit: `235fc5596d9484448ddd3194dbd7bb0a45eb635f`
- GitHub Actions job `reproduce-figures`: **success**, including the original statistical tests, five survey-vintage tests, 8 figure pairs, and artifact upload.
- Artifact ID: `13494724547`; GitHub-reported ZIP digest: `sha256:10e8bd507f927dad710318fbf2bf881292e6c54fba949679673b894f65a508ef`.
- Artifact expiration: 6 December 2026. Re-run the workflow or use the committed code and SHA-256-locked panel to regenerate results after expiration.

The numerical values below are exactly the **rounded values printed in the successful job log**, not independently transcribed full-precision JSON values. Their source is the executable `survey_vintage_audit.py` output; the complete JSON and CSV are included in the temporary Actions artifact.

## Survey-age restrictions

Coefficient units: poverty percentage points per tenfold increase in billionaires per 10 million people **by citizenship**, conditional on log PPP GDP per capita, with country-level OLS and HC3 inference.

| Maximum poverty-vs-GDP observation lag | Country sample (n) | Billionaire coefficient | HC3 p |
| ---: | ---: | ---: | ---: |
| 2 years | 35 | −9.497 | 0.028 |
| 3 years | 43 | −7.406 | 0.111 |
| 4 years | 49 | −9.789 | 0.054 |
| 5 years | 54 | −7.912 | 0.119 |

The poverty observations span **2020–2025**, and the maximum within-row poverty/GDP observation gap is five years. Four lag thresholds have enough countries to fit the prespecified regression.

The direction is negative across these four restrictions, but inference is not stable. Only the n=35 restriction has an unadjusted individual-model HC3 p-value below 0.05. **This must not be presented as a robust discovery or as four independent hypothesis tests**: the nested samples overlap, and many specifications were examined. The p-values are diagnostic, not family-wise corrected publication claims.

The lag restrictions change **which countries** enter the regression. Consequently, the difference between −9.497 and −7.912 cannot be interpreted as the effect of old surveys. It mixes timing, geographic selection, leverage and development-level composition.

## Full-sample explanatory comparison

All three models use the same 54 countries.

| Model | In-sample R² |
| --- | ---: |
| Billionaire density only | 0.5471 |
| Log PPP GDP per capita only | 0.8049 |
| Log PPP GDP plus billionaire density | 0.8113 |
| Incremental R² for billionaires after GDP | **0.0064** |

The incremental value is `0.8113 - 0.8049`, using displayed precision. This is an **in-sample descriptive fit difference**; it is not a causal contribution, out-of-sample prediction gain or evidence that GDP is exogenous.

## Interpretation for the Medium article

Supported:

1. A strong raw association does not identify whether billionaire prevalence reduces poverty.
2. Development level is a substantially stronger single-predictor description within this 54-country exploratory sample.
3. The GDP-adjusted billionaire association remains negative, but its estimated magnitude and HC3 significance depend on survey-vintage sample restrictions.
4. The 2021-PPP US$30/day threshold is meant for cross-country PPP comparison; this does not eliminate survey methodology, local consumption-basket and temporal-comparability limitations.

Not supported:

- The claim that one more billionaire, more billionaire concentration, or a tenfold density increase **causes** poverty to fall.
- The opposite claim that the descriptive association must be spurious or necessarily positive after every adjustment.
- An exact reconstruction of the original 64-country chart or Forbes March 2026 **residence** counts.
- Treating these secondary citizenship counts as the historically frozen March Forbes values. Independently published March top-ten counts already disagree with the November API for the US, Russia and Italy.
- Calling the n=35 model “significant” without disclosing the other restrictions and their shared, nonrandom selection.

## Remaining release gates

1. Resolve original chart country selection, residence/citizenship attribution and the full March 2026 country-count vintage, or expressly publish this as a distinct exploratory citizenship study.
2. Review the full-precision `audit.json`, including HC3 confidence intervals and the continuous survey-lag model, before any claim that draws on them.
3. Verify every published chart and caption against the frozen input and generated figure hashes.
4. Place the final long-form manuscript in the private `research-articles-medium` repository; keep code, data, statistical evidence and figures in this public companion.
