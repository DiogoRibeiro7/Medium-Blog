# Lagged macro matching: eligibility and balance, not an effect estimate

This diagnostic continues the verified World Bank WDI acquisition. It asks a
narrower question than the main tax-cut article: **with the observed outcome
windows, which events have comparable controls, and how much pre-reform
balance improves when donors are selected using actual macroeconomic data?**
No new causal estimate, standard error, confidence interval or p-value is
computed; the existing unadjusted estimates are unchanged.

## Inputs and pre-declared rules

Use the full 853-row joined WID/tax-cut/WDI panel produced by `world_bank.py`.
`load_panel(..., covariates=())` first loads the **entire** income-share and
treatment history; the separate WDI reader preserves absent controls as
`None`, not zero. No row is dropped based on GDP growth, inflation or trade
openness after a tax cut, and the event year is not a macro matching feature.

For each of the 29 recorded tax-cut events at time `t`:

1. Require five observed pre-reform binary treatment years, with no reform at
   `t-1`, plus observed income-share outcomes for **every year** `t-5..t+5`.
2. The *history-only* donor candidates must have exactly the same prior five
   reform indicators, a complete outcome window, and no new major tax cut
   from `t` through `t+5`. The last rule is inherited from the exploratory
   event-study design and is **post-baseline future-exposure selection**;
   it can bias a causal comparison.
3. Require GDP-per-capita growth, CPI inflation and trade openness at **all
   five pre-reform years** for the treated country and separately each donor.
   Missing one pre-reform value excludes that donor, not the whole historical
   country panel. Unemployment is omitted because the original WDI coverage
   begins only in 1991.
4. For the treated country and each macro-complete donor, compute one mean
   over five pre-reform years for each of the three controls, giving a
   three-dimensional selection profile. Only these lagged features enter
   ranking, **never a treatment-year or post-treatment outcome or control**.
5. Standardise each feature over the treated country plus that event's
   eligible donor candidates. Rank donors by Mahalanobis distance using
   `0.75 × covariance + 0.25 × identity` as a fixed covariance shrinkage
   rule; use up to the five nearest donors and deterministic alphabetical
   tie-breaking. No donors are invented when a cohort is unsupported.

## Fair before/after balance comparison

For each *matched* event and each WDI variable, the pre-ranking reference is
the mean of **all macro-complete, history-eligible donors**. The post-ranking
reference is the mean of up to five selected donors. Thus donor *selection*
changes, but treatment events and the complete-macro reference pool are held
fixed. Report event-level standardised mean gaps:

\[
SMD_{e,k}^{\mathrm{pool}} =
\frac{\overline X_{\mathrm{treated},e,k} -
\overline X_{\mathrm{pool},e,k}}
{\mathrm{SD}(\overline X_{\mathrm{donor},e,k})},
\quad
SMD_{e,k}^{\mathrm{selected}} =
\frac{\overline X_{\mathrm{treated},e,k} -
\overline X_{\mathrm{selected},e,k}}
{\mathrm{SD}(\overline X_{\mathrm{donor},e,k})}.
\]

The denominator is the *same* donor-pool population standard deviation in
both columns. When it is zero, SMD is undefined (`null`), not an artificial
zero; the original and donor means remain visible. Summaries average absolute
gaps across *evaluable* events, not over all country-years.

As an **independent outcome diagnostic**, fit an ordinary linear slope to
each country's *observed* top-1% income share at `t-5..t-1`. Compare the
treated slope with the pool mean and selected donor mean slopes, in percentage
points per year. The top-income-share slopes are **not** used for ranking.
Better WDI balance need not imply better pre-reform outcome trends.

## Reproduce

Build the WID/author event panel first, then the [World Bank joined panel](MACRO_CONTROLS.md).
From `neoliberalism-evidence-synthesis/`:

```bash
python -m causal_tax_cuts.macro_matching \
  --panel data/wdi-source/outcome_panel_with_macro.csv \
  --provenance data/wdi-source/world_bank_provenance.json \
  --lags 5 --max-horizon 5 --max-donors 5 \
  --output-dir data/tax-study/macro-balance
```

The output contains `macro_event_eligibility.csv` (all reform events and
mutually exclusive attrition reasons), `macro_pretrend_balance.csv`
(three rows per matched reform, including undefined cases), and
`macro_balance_audit.json` (aggregate counts, summaries, WDI source ZIP
SHA-256s, joined-panel SHA-256, and interpretation warnings). None of these
outputs is a treatment-effect estimate. On PRs, the official source workflow
publishes them as the seven-day `pre-reform-macro-balance` artifact.

The first checked results are in
[`results/MACRO_MATCHING_RESULTS_2026-11-14.md`](results/MACRO_MATCHING_RESULTS_2026-11-14.md).
The next methodological task is a sensitivity grid over lag length, donor
cap, coverage and outcome-pretrend balance, followed by proper identification
and uncertainty design *before* publishing any adjusted ATT.
