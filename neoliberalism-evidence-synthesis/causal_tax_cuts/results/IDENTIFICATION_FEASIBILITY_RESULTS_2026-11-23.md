# Policy-specific causal identification feasibility — 23 November 2026

This is a **source-reproduced research design audit**, not a new estimate
of the effects of tax reform. The unit is the *original coded reform onset*
in the unchanged 18-country Hope–Limberg/WID/WDI study, not an independently
dated statute. The 20 baseline-eligible onsets and their historical
donor-exposure rules are unchanged.

## Gate results

| Diagnostic | Verified count |
|---|---:|
| Original author flags / descriptive run onsets | **30 / 25** |
| Onsets previously eligible for donor analysis | **20** |
| Eligible onsets with provision-level statutory context | **5/20** |
| Documented eligible onsets inside joint WDI donor hull at h=0 | **1/5** |
| Documented eligible onsets inside joint WDI donor hull at h=3 | **1/5** |
| Documented eligible onsets inside joint WDI donor hull at h=5 | **0/5** |
| All eligible onsets inside joint WDI donor hull at h=5 | **2/20** |
| Documented eligible onsets with complete five-year DPI histories | **4/5** |
| Documented eligible onsets followed by another major cut by h=5 | **1/5** |
| Flag-to-law causal mapping and composite effective date verified | **0/20** |
| Policy-specific effects identified by present data/design | **0/20** |

The **two** onsets satisfying the existing *three-variable finite-sample*
year-five donor-hull criterion are **Finland 1986 and Germany 2008**.
Neither has been documented in the statutory evidence register. Their
positive WDI-hull result says nothing by itself about omitted political,
fiscal, outcome-history or time-varying covariate support.

### Five legally reviewed eligible cases

| Original onset | Official statutory context | Strict joint WDI hull h=0 / h=3 / h=5 | Further 2-SD cut during follow-up | Full DPI five-year treated history |
|---|---|---|---|---|
| UK 1979 | Finance (No. 2) Act 1979 | No / No / No | No | No (only four DPI pre-years) |
| US 1982 | Economic Recovery Tax Act 1981 | No / No / No | Yes (USA 1987) | Yes |
| UK 1988 | Finance Act 1988 | **Yes / Yes / No** | No | Yes |
| Germany 2001 | Steuersenkungsgesetz 2000 | No / No / No | No | Yes |
| Netherlands 2001 | Wet inkomstenbelasting 2001 | No / No / No | No | Yes |

The pre-reform WID top-1% slope gaps relative to **macro-selected**
donors, in percentage points per year, are respectively **+0.0878,
+0.1632, +0.0070, −0.0716 and −0.2696**. None is a test of
counterfactual parallel trends. The four-year loss of joint support
for UK 1988 must not be converted into a causal effect for h≤3.

## Interpretation and next work

The existing study has no jointly supported, legally documented
five-year policy-specific treatment comparison *under its current
three-WDI-mean donor interpolation check*. More fundamentally, none of
the legal examples establishes **which component** of the composite
author index caused its binary flag or precisely **when** the relevant
population experienced that composite treatment. Exchanging the annual
binary label for a statute's tax-year boundary would be unsupported.

- **Research priority A:** Audit primary-law components, announcement and
  effective dates for Finland 1986 and Germany 2008, then reassess
  comparators, pretrends, political confounders and policy overlap on a
  *new, prospectively specified policy-specific design*.
- **Research priority B:** Seek genuine within-country identification
  (e.g., threshold-based statutory exposure) only if detailed
  administrative or income-tax microdata can support it. Do not claim
  these data have already been obtained.

The point is **not** that no tax reform can be studied causally. The
present limited sample, recurrent composite exposure, future donor
censoring and lack of validated policy assignment do not support
presenting the exploratory +0.795-percentage-point historical matched
contrast as a new causal ATT, much less a causal effect of
"neoliberalism" as a whole.

## Reproducibility

The Python audit cross-validates **ten input CSV/JSON source files**
and a shared joined-panel SHA-256
`8dfcdd2bb7ea8c378436d5415a70ae0a83f8f1b431f4e46c70039bf578072321`.
It writes all **20 event gates** and **140 event–horizon observations**,
plus a frozen deterministic source-hash manifest:
[identification_feasibility_2026-11-23.json](identification_feasibility_2026-11-23.json).

See [the methods and explicit decision criteria](../IDENTIFICATION_FEASIBILITY.md).
The live WID/WDI/DPI source workflow uploads the complete event-level
audit as a seven-day `tax-cut-identification-feasibility`
artifact. It records all raw input SHA-256 hashes and checks two
derived numeric-column fingerprints rounded solely **for source
identity**, preserving the full-precision inputs for analysis.
Across two live CI runs, the final event and horizon CSVs were
byte-identical; only seven Mahalanobis distances (up to 4.44e-16) and
nineteen donor-hull reconstruction residuals (up to 4.63e-16)
varied in binary floating-point serialization. Scientific classifications,
donor sets, cohort and source WID/WDI SHA-256 were unchanged.
Any substantive result difference on the same original panel fails
the new reproducibility check.
