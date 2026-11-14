# Five-year WDI matching support and balance: 14 November 2026

This report evaluates which historical tax-cut reforms can be matched on
**observed pre-treatment** World Bank controls and whether those selected
donors show better pre-reform balance. It is **not an adjusted causal effect**.
The previous exploratory five-year top-income-share contrast of +0.795
percentage points is unchanged and is not independently validated here.

## Data and eligibility

- Original joined outcome panel: **853** country-years, including **29**
  observed major tax-cut event-years, across 18 countries (1965–2015).
- World Bank source: the same four official WDI CSV ZIP archives, **last
  updated 11 November 2026**, independently archived by SHA-256 in the
  [machine-readable audit](macro_balance_2026-11-14.json).
- Design: five pre-reform treatment-history years; balanced observed WID
  outcomes `t-5..t+5`; macro matching uses only **GDP-per-capita growth,
  CPI inflation, and trade openness at `t-5..t-1`**.
- Future donor tax cuts (`t..t+5`) are excluded as in the older descriptive
  estimator, which itself raises selection concerns.

| Event-eligibility result | Count |
| --- | ---: |
| Original reform events in observed WID outcome panel | **29** |
| Excluded: incomplete history or reform at `t-1` | 7 |
| Excluded: no history-eligible comparison donors | 1 |
| Excluded: treated country's required WDI lags missing | 1 |
| Excluded: no macro-complete eligible donor | 0 |
| Supported reform events under these rules | **20** |
| Distinct treated countries among supported events | **14** |

Of the 20 supported reforms, the history-only rule provides **184**
candidate donor-event pairs. Complete pre-period WDI data retain **174**;
nearest-five ranking uses **94** donor-event pairs. These are pairs of
reform events and donor countries, *not* independent observations. Two very
early supported events have only two macro-complete donors; nearest-five
ranking cannot improve their balance.

## Observed pre-reform balance

Each variable uses the same supported-event cohort and same donor-complete
reference pool before and after selecting up to five nearest donors.
The table shows the average **absolute standardised pre-reform mean gap**;
smaller means more balanced on that specific observed covariate.

| WDI covariate | All eligible macro donors | Nearest up-to-five | Events with reduced absolute gap |
| --- | ---: | ---: | ---: |
| GDP per capita growth | 0.885 | **0.696** | 14 / 20 |
| CPI inflation | 1.388 | **1.102** | 14 / 20 |
| Trade openness / GDP | 1.480 | **1.320** | 13 / 20 |

Those gaps remain substantial. The controls are only proxy measures, and
matching three time-averaged series does not eliminate omitted confounding.

More importantly, **pre-reform income-share slope balance worsens**:

| Outcome diagnostic | All eligible macro donors | Nearest up-to-five |
| --- | ---: | ---: |
| Mean absolute top-1% income-share slope gap (percentage points/year) | **0.198** | **0.218** |

This does not establish a treatment effect or automatically discredit the
matching rule. It shows that selecting on these three macro summaries can
make the counterfactual *worse* on a separate, relevant outcome diagnostic.
We must report both results. Neither a smaller macro SMD nor a non-significant
placebo would prove exchangeability or parallel trends.

## Reproducibility and research gate

The official WDI ZIP source hashes and joined panel SHA-256 are in the
companion [frozen JSON](macro_balance_2026-11-14.json). The exact eligibility
reasons, donor IDs, per-event distances, pretrend slopes and all standardised
covariate gaps are reproducible from the checked GitHub Actions artifacts:
`world-bank-macro-coverage` and `pre-reform-macro-balance`.

This audit recommends **no adjusted ATT yet**. Before interpreting a matched
contrast we should test sensitivity to donor limits and lag specifications,
quantify observable common support, and address future-treatment-based donor
censoring, clustered dependence and unobserved political/institutional
confounding. The full published-paper estimator has not been replicated.
