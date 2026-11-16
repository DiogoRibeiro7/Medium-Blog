# Dynamic policy-path feasibility — 16 November 2026

This is an observation of subsequent major-cut policy histories among
the same 20 baseline eligible reforms in the
[15 November donor-support audit](DONOR_SUPPORT_RESULTS_2026-11-15.md).
It does **not** estimate an ATT or validate a target-trial emulation.
The detailed protocol is in
[`../DYNAMIC_EXPOSURE.md`](../DYNAMIC_EXPOSURE.md); the exact summary
and original-source hashes are in
[`dynamic_exposure_2026-11-16.json`](dynamic_exposure_2026-11-16.json).

## Unchanged vintage and cohort

- The same **853-row** WID/major-cut joined outcome panel, **29**
  observed reform years and **20** macro/donor-eligible initiating
  reforms. All comparator candidates have complete observed WID
  outcomes through `t+5` and required WDI in `t-5..t-1`.
- Full joined-panel SHA-256:
  `8dfcdd2bb7ea8c378436d5415a70ae0a83f8f1b431f4e46c70039bf578072321`.
- **239** pre-censor comparator–reform pairs, with time-specific
  no-new-cut risk sets exactly reconciled to the previous audit.
- The four official WDI archives report a **2026-11-11** source update
  date; all four source checksums are preserved in the frozen JSON.

## The asymmetric observed follow-up matters

Five of the 20 initiating reforms are followed by an additional
`taxcut_2SD` major cut within five years:

| Initial reform | First additional coded major cut |
| --- | --- |
| Canada, 1972 | 1973 |
| United States, 1982 | 1987 |
| Australia, 1987 | 1988 |
| New Zealand, 1987 | 1988 |
| Sweden, 1990 | 1991 |

The original exploratory matching procedure retains all five of those
treated paths but excludes comparator paths after a comparator cut.
This is **not symmetric** if the intended intervention is exactly
one major cut versus no major cuts.

## Descriptive retention under two *hypothetical* sustained policies

Here the initiating arm is allowed a cut at `t` but no *additional*
major cut; the comparator arm is allowed **no** cut from `t` onward.
The entries below count observed paths compatible with those event
histories. They do not adjust for why a country changed policy.

| Horizon | Initiating reforms with no repeat cut | Comparator–event pairs with no cut through horizon | Pairs with both paths compatible | Compatible pairs also complete in sequential lagged WDI |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 20 | 227 | 227 | 227 |
| 1 | 16 | 212 | 176 | 176 |
| 2 | 16 | 197 | 168 | 168 |
| 3 | 16 | 184 | 159 | 159 |
| 4 | 16 | 179 | 154 | 154 |
| 5 | 15 | 174 | 144 | 144 |

At the five-year horizon, **15 of 20** initiating reforms remain
compatible with a "single coded major cut" path. The paired-compatible
count is **144 of the original 239** baseline comparator–reform pairs.
These pairs repeatedly draw on the same small set of study countries
and must not be interpreted as independent treatment assignments.

Every path that remained compatible with both sustained policies had
all three WDI variables observed at each prior annual decision year
through horizon five. That is a **coverage result**, not evidence that
the data contain all the sequential confounders required for causal
identification. Annual values can be measured too late or be affected
by previous cuts.

## What remains unestablished

The current observational contrast resembles neither a clean baseline
initiation comparison with natural follow-up in *both* groups nor a
sustained-strategy comparison that adjusts for deviations in *both*
groups. A per-protocol-style design might use artificial censoring with
inverse-probability weights, but doing so now could be badly biased:

1. The binary major-cut flag is not a unique, uniform intervention.
2. The political and fiscal determinants of subsequent tax decisions
   have not been measured or adjusted.
3. Baseline and longitudinal **joint** covariate overlap have not been
   demonstrated; there are few independent countries.
4. The existing baseline cohort is selected for complete WID outcomes
   in the future `t+5` window.
5. Annual WDI lags are a feasibility screen, not validated
   pre-legislation measurements.

The next research phase is **intervention definition, timing validation,
joint support and confounder measurement**, not another model fitted to
the matched outcome differences.
