# Dynamic tax-cut pathways: a protocol and feasibility audit

This is a **design gate before causal estimation**. The earlier
[donor-support audit](DONOR_SUPPORT.md) measured how many comparator
country–reform pairs were removed because those countries later enacted
their own major cuts. It did not examine the *treated countries' own*
subsequent exposure. The current exploratory matching code **keeps**
later major cuts in the treated pathway while **excluding** comparator
paths once those countries experience a cut.

That asymmetry makes it essential to state the intervention of interest
before interpreting any post-reform contrast. This audit creates explicit
observed paths and counts without estimating any new treatment effect.

## Two distinct candidate comparisons

An eligible initiation is a coded major cut at annual time `t`, with no
cut at `t-1` and the exact same five pre-reform event-history years in the
baseline comparator pool.

| Candidate comparison | Initiating arm | Comparator arm | Later policy changes |
| --- | --- | --- | --- |
| **Initiation at `t` with natural follow-up** | Major cut at `t` | No major cut at `t` | Subsequent cuts may occur in **both** arms; no later-cut exclusion. Not automatically an intention-to-treat effect in an observational study. |
| **One coded major cut versus none through `h`** | Major cut at `t`; no *additional* cuts in `t+1..t+h` | No major cut in `t..t+h` | Both arms become **nonadherent** when they deviate. A causal per-protocol analysis would require defensible selection adjustment, potentially artificial censoring with sequential weights. |

The existing matched contrast is **neither comparison exactly**: it
retains treated repeat cuts (as in natural follow-up) but restricts
comparators to no future cut (as in the sustained comparison).

Both candidate strategies are defined only by the author's annual
`taxcut_2SD` event. They **do not specify a homogeneous tax rate change,
taxpayer incidence, legislation date, or all other fiscal interventions**.
This is a serious consistency limitation, not a detail solved by matching.

## What is and is not conditioned on

We deliberately preserve the previously audited eligible cohort:

- Fully observed **WID top-1% shares from `t-5` through `t+5`** for
  initiators and baseline comparator candidates, without interpolation.
- Identical previous five years of **binary major-cut history**, no
  cut at `t-1` for the initiator.
- Complete **three WDI controls only at `t-5..t-1`** for eligibility:
  GDP-per-capita growth, CPI inflation and trade openness.

This source-based screening selects 20 initiating reform events and
239 baseline candidate comparator–event pairs. We do **not** change that
cohort when WDI data from `t` or later are absent.

For each treated and comparator path at every horizon `h=0..5`, record:

1. Whether a coded major cut occurred in `t+h` and how many occurred
   since the focal `t`.
2. Whether the path has stayed compatible with its **sustained**
   comparison rule. Treated paths deviate after their first *additional*
   cut in `t+1..t+h`; comparators deviate after any cut in `t..t+h`.
   A comparator cut at `t` is *baseline nonadherence*, not a later
   follow-up event.
3. Whether all **three WDI controls** are observed at the previous
   calendar year, `t+h-1`; and whether every such annual triple from
   `t-1` through `t+h-1` is complete.

The third item is **only a feasibility check for sequential adjustment**.
Annual WDI quantities at `t+h-1` may be affected by earlier policy;
their release dates also may not precede the next annual policy choice.
Blindly adding them as regressors would not resolve time-varying
confounding and could introduce bias. The recorded event year is coarse
and does not establish within-year causal order.

## Reproduction and audit artifacts

Run after the [official WDI join](MACRO_CONTROLS.md), from
`neoliberalism-evidence-synthesis/`:

```bash
python -m causal_tax_cuts.dynamic_exposure \
  --panel data/wdi-source/outcome_panel_with_macro.csv \
  --provenance data/wdi-source/world_bank_provenance.json \
  --frozen-reference causal_tax_cuts/results/donor_support_2026-11-15.json \
  --output-dir data/tax-study/dynamic-exposure
```

The command writes:

- `dynamic_policy_paths.csv`: one row per observed initiator or
  comparator and horizon; future exposures, adherence and lagged WDI
  availability are separate fields.
- `dynamic_reform_recurrence.csv`: one row per eligible initiating
  reform, with the first and number of additional coded major cuts.
- `dynamic_event_horizons.csv`: paired strategy-adherence counts at
  each horizon, retaining the baseline comparator denominator.
- `dynamic_exposure_audit.json`: aggregate counts, design definitions,
  identification barriers, joined-panel SHA-256 and WDI archive hashes.

Every comparator's horizon risk set must **exactly** reconcile to the
merged donor-support audit. The frozen comparison of the prior risk-set
counts is enforced **only when the panel SHA-256 is identical**; a
different source vintage is labelled `source_vintage_changed`.

The networked source-probe GitHub Action uploads these files as the
seven-day `tax-cut-dynamic-exposure` artifact. The first real-data
audit is frozen in
[`results/DYNAMIC_EXPOSURE_RESULTS_2026-11-16.md`](results/DYNAMIC_EXPOSURE_RESULTS_2026-11-16.md).

## Decisions required before a causal estimate

1. **Specify the intervention and estimand.** A one-time coded reform
   and a sustained ban on further coded major cuts are different
   interventions. An initiation comparison that allows naturally
   occurring follow-up exposure is different again.
2. **Establish the treatment process and calendar timing.** Annual
   indicators cannot show whether covariates were observed before a
   particular law or whether other tax changes happened simultaneously.
   Obtain legislative timing, policy intensity and complementary tax
   measures, where possible.
3. **Find sequential confounders with support.** Pre-decision macro
   coverage is necessary but not enough. Political composition,
   international common shocks, capital openness and fiscal context
   matter; assess joint overlap and risk of unstable weights *before*
   modeling exposure/censoring probabilities.
4. **Choose a credible independent unit and inference design.** The
   same country can recur across multiple event-specific comparator
   paths. Those are not 239 independent countries. Country-level
   serial correlation, interference and the roughly 18-country panel
   constrain uncertainty estimation.
5. **Address observed-outcome selection.** This diagnostic deliberately
   requires WID values through `t+5` at baseline even for shorter
   horizons. That restriction may select on future outcome availability.

No propensity scores, inverse-probability weights, causal estimates or
confidence intervals are calculated in this phase. The remaining
identification gaps are explicit stop conditions for a causal claim.

## Primary methods references

- Imai, Kim & Wang (2023), [*Matching Methods for Causal Inference with
  Time-Series Cross-Sectional Data*](https://imai.fas.harvard.edu/research/files/tscs.pdf),
  *American Journal of Political Science*. Their time ordering of
  covariates, treatment and outcomes requires careful justification.
- Hernán and colleagues, [*Electronic medical records can be used to
  emulate target trials of sustained treatment strategies*](https://pmc.ncbi.nlm.nih.gov/articles/PMC5847447/).
  Its artificial-censoring/weighting logic motivates a **possible**
  design, not a valid automatic transfer of assumptions to tax policy.
