# Okun's law in hours, for every country, with and without the pandemic (protocol)

Registered on 2026-10-20 and committed and pushed before any quantity marked *new* below
was computed. Changes after the first computation go in the change log at the end, with
reasons.

## 1. Why, and what was already seen

An external review (2026-10-19) asked for an hours-based Okun's law for every country, not
only the four case studies. The manuscript also explains the pandemic attenuation of the
unemployment coefficient by job-retention schemes that "kept unemployment from following
output", but it does not test that explanation.

**Already computed and seen before this protocol was written.**

- The case study (HETEROGENEITY_PROTOCOL.md) estimated the two-lag elasticity to GDP
  growth of total hours, hours per worker, domestic employment, output per hour and the
  unemployment-rate change, for every country with data, **on the pandemic-excluded sample
  only**. They are committed in `docs/heterogeneity/elasticities.csv`.
- While scoping this study, the hours elasticities and their intervals for all 26
  countries were printed and read. The largest are Portugal's and Spain's, and the
  smallest is Luxembourg's.
- The classical full-sample and pandemic-excluded sums of the unemployment-rate change are
  published (`docs/review/grid.csv`). Including the pandemic moves them towards zero in
  most countries.

So the per-country hours elasticities without the pandemic, and any comparison built only
from them, are **not blind**. Section 6 reports them as descriptive.

**New: not computed before this protocol.** Every elasticity on the full sample, which
includes the pandemic quarters, for hours, hours per worker, domestic employment and
output per hour, and with them every test in section 5.

## 2. Data

No new data. The inputs are the same as the case study's:

- GDP growth and the unemployment rate from `snapshots/eurostat-2026-10-15`;
- total hours worked (`namq_10_a10_e`, `THS_HW`, `SCA`) and domestic employment in persons
  (`namq_10_a10_e`, `THS_PER`, `SCA`) from `snapshots/heterogeneity-2026-10-18`;
- both snapshots are audited before use.

**Coverage.**

- Hours: 26 countries (all but BE).
- Persons, and so hours per worker: 20 countries (no BE, CZ, EL, FR, MT, PL or PT).

The country sets are read from the data, not typed.

**Concepts.** Hours and persons are national-accounts, domestic concepts. The
unemployment rate is a Labour Force Survey concept for residents. The 2021 survey redesign
falls inside the pandemic window, and it affects the unemployment-rate change on the full
sample but not hours.

## 3. Measures and estimator

The measures and estimator are those of the case study (HETEROGENEITY_PROTOCOL.md,
sections 3-4), unchanged:

- `h = 100 Δ ln H` (hours);
- `e_dom = 100 Δ ln E` (persons);
- `hpw = h − e_dom` (hours per worker);
- `prod = g − h` (output per hour);
- `du` (the unemployment-rate change, pp).

The identity `g = prod + hpw + e_dom` holds exactly, so the elasticities of the three
margins sum to one in every sample.

**Elasticity.** OLS of `m[t]` on a constant and `g[t], g[t−1], g[t−2]`, with quarters
2009-Q3 to 2026-Q2. β is the sum of the three slopes.

**Samples.**

- *Excluded*: rows whose window [t−2, t] avoids 2020-Q1..2021-Q4. This is the case study's
  primary sample.
- *Full*: every row, pandemic windows included.

**Uncertainty.** A synchronised circular block bootstrap over quarters, as in the case
study.

- Whole quarters are resampled, keeping every country, measure and sample of a quarter
  together.
- The full and excluded estimates of each draw therefore share one resample, and their
  differences have a valid joint distribution.
- Block length 4, B = 1999, seed **20261026**.
- 95% percentile intervals. Two-sided bootstrap p-value `2 min(share ≤ 0, share ≥ 0)`, as
  in the case study.

**Reproduction check.** The excluded-sample point estimates of `h`, `hpw`, `e_dom`,
`prod` and `du` must equal those in `docs/heterogeneity/elasticities.csv` (to 1e-9) for
every country. Otherwise the run stops.

## 4. The explanation being tested

**Explanation.** Job-retention schemes kept workers employed while their hours fell with
output. If so, then on the full sample:

- labour input measured in hours keeps following output;
- the unemployment rate does not;
- the adjustment that remains moves from the number of workers to hours per worker.

**What a confirmation would not show.**

- It would not show that the schemes *caused* this. Paid absence and temporary closures
  with continued pay produce the same pattern: people stay employed while their hours fall.
- It would not separate the schemes from the 2021 survey redesign's effect on measured
  unemployment.

The tests show whether the data have the pattern that the explanation requires. Nothing
is identified.

## 5. Registered tests

C_h is the set of countries with hours (26) and C_p the set with persons (20). Means are
unweighted over countries.

**P1, hours keep their association and unemployment does not.**

- Retention ratio of measure m: `R_m = mean_{C_h} β_m(full) / mean_{C_h} β_m(excluded)`.
- Statistic: `D = R_h − R_du`, with both ratios over the same 26 countries.
- **Prediction:** `D > 0`.

**P2, adjustment moves from workers to hours per worker.**

- Change in measure m: `Δ_m = mean_{C_p} β_m(full) − mean_{C_p} β_m(excluded)`.
- **P2a:** `Δ_hpw > 0`.
- **P2b:** `Δ_e_dom < 0`.

**P3, across countries, the pandemic attenuates unemployment more where hours per worker
take up more of the adjustment.**

- For each country in C_p:
  - `a_i = β_du,i(full) − β_du,i(excluded)`, positive when unemployment moves towards zero
    (β_du is negative);
  - `b_i = β_hpw,i(full) − β_hpw,i(excluded)`.
- Statistic: the Spearman correlation of `a` and `b`.
- **Prediction:** positive.
- p-value: permutation, 9999 permutations, seed 20261026, two-sided, with the case
  study's implementation (`heterogeneity.correlations`).
- **This is a correlation across 20 countries.** The changes of a country are driven by
  the same pandemic quarters, so they describe how each country's labour market moved in
  those quarters, not a mechanism.

**Verdicts.** The family is P1, P2a, P2b and P3, Holm-adjusted together.

- **Supported:** adjusted p < 0.05 with the predicted sign.
- **Contradicted:** adjusted p < 0.05 with the opposite sign.
- **Not determined:** otherwise.

## 6. Reported without a test

These are descriptive. The excluded-sample quantities were seen (section 1).

- **Per country, every country with hours:** β_h on both samples with intervals, and
  β_du on both samples for comparison.
- **Counts:**
  - the number of countries whose hours interval lies above zero, on each sample;
  - the number whose hours elasticity moves towards zero when the pandemic is included,
    against the same count for unemployment with this estimator.
- **Rankings:** the Spearman correlation across C_h between `−β_du` and `β_h` on the
  excluded sample. Both are estimated on the same quarters, so their estimation errors
  are correlated within a country; the correlation is reported with that caveat.
- **Focus countries:** the positions of ES, PT, NL and LU in the hours ranking.
- **Retention ratios** `R_h` and `R_du`, each with its interval.
- **Mean changes** of all three margins (`prod`, `hpw`, `e_dom`), which sum to zero by
  the identity.

## 7. Sensitivity (fixed in advance)

- **Lag order:** K = 1 and K = 4 instead of 2.
- **Bootstrap:** block length 8.
- **Countries:** without IE and LU. Their GDP is dominated respectively by multinational
  accounting and by cross-border commuters, and Irish GDP rose in 2020.

Each setting reports the four statistics and their verdicts under the same rule, with
Holm within the setting. The primary verdicts are those of section 5. A sensitivity
result that differs is reported, not substituted.

## 8. Use in the manuscript (fixed in advance)

- **Classical section, pandemic paragraph.** The job-retention sentence is kept only if
  P1 and P2a are supported. Otherwise it is rewritten to say that the data do not show
  that pattern. In both cases the verdicts of P1-P3 are quoted.
- **Hours elasticities.** The per-country hours elasticities, on both samples, are added
  as a figure.
- **Discussion.** The hours-based Okun's law is removed from "standard analyses not done".
- **Medium article.** Its pandemic paragraph says what the test found.

## 9. Outputs

`eu-okun hours` writes:

- `docs/hours/countries.csv`;
- `docs/hours/tests.csv`;
- `docs/hours/sensitivity.csv`;
- `docs/hours/summary.json`;
- the report `docs/HOURS.md`.

A test regenerates them, and checks the reproduction of the case-study estimates and the
margin identity.

## Change log

- **2026-10-20, after the first computation: a flaw in the registered tests, and an
  amendment registered before it was computed.**
  - *Registered result.* All four tests are not determined. The point estimates of P1 and
    P2a have the predicted sign; P2b is close to zero.
  - *The flaw.* A resample that contains no pandemic row gives identical full and
    excluded estimates, so the differences of P1, P2a and P2b are exactly zero. With
    block length 4, 3.3% of the 1999 resamples contain none (7.9% with block 8). The
    two-sided p-value of section 3 counts them, so it cannot fall below 0.067 (0.157 with
    block 8). P1, P2a and P2b could not have been supported at any effect size. P3 is a
    permutation test and is not affected.
  - *What stands.* The registered verdicts stay the primary result, and section 8 applies
    to them. In section 8, "say that the data do not show that pattern" is read as "say
    that the registered test does not establish it", because saying the data show no such
    pattern would misreport point estimates that have the predicted sign.
  - *Amendment (secondary, labelled as after the registered result).* The same four
    statistics, Holm family and verdict rule, with a **stratified** synchronised circular
    block bootstrap, so that every resample holds the observed number of pandemic rows:
    - the strata are the rows whose window [t−K, t] touches 2020-Q1..2021-Q4 (10 rows
      for K = 2) and all other rows from 2009-Q3 to 2026-Q2;
    - each stratum is resampled separately, in time order, with the same block length.
      The non-pandemic stratum treats 2019-Q4 and the first quarter after the pandemic
      window as adjacent;
    - one generator with seed 20261026 draws the non-pandemic stratum, then the pandemic
      stratum;
    - P3 is unchanged;
    - the same sensitivity settings apply;
    - with only ten pandemic rows, resampling within that stratum is coarse, and the
      intervals are approximate.
  - *Use.* The amended verdicts are reported next to the registered ones, labelled as
    amended after the registered result was seen. They do not replace the registered
    verdicts in section 8.
- **2026-10-20, after the first computation: an implementation correction.** No rule
  changed.
  - *The problem.* In a resample without a pandemic row, the full and excluded estimates
    are equal in exact arithmetic. The implementation computed them in separate matrix
    products of different length. In 2 or 3 of the 66 such resamples per test, it
    returned differences of about 1e-16 instead of zero.
  - *Why it matters.* The sign of that rounding differs between Windows and Linux. CI on
    Linux gave a registered p-value of P1 of 0.076, against 0.075 on Windows.
  - *The correction.* In those resamples the full-sample draws are now set to the
    excluded-sample draws, as exact arithmetic requires.
  - *Effect on the registered p-values.*
    - Primary: P1 goes from 0.075 to 0.076 and P2b from 0.681 to 0.682. P2a is
      unchanged.
    - Sensitivity settings: some registered p-values move by at most 0.004.
    - No verdict changes. The amendment, whose resamples always contain the pandemic
      rows, is not affected.
