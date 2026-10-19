# Does the quarterly unemployment rate inherit monthly smoothing? (protocol)

Registered on 2026-10-19, before any of the quantities below was computed. The only data
work beforehand was reading the vintages of the two frozen series. No comparison of the
quarterly and monthly unemployment rates, and no autocorrelation or lag share by method
group, has been computed.

## 1. Question

Every quarterly result in the study uses Eurostat's quarterly unemployment rate
(`une_rt_q`). The monthly identification protocol (MONTHLY_IDENTIFICATION_PROTOCOL.md,
section 2) classified how each country produces its *monthly* rate (`une_rt_m`), from
Eurostat's reference metadata:

| Group | Monthly method | Countries |
| --- | --- | --- |
| D | Direct monthly Labour Force Survey estimates | AT, CZ, EL, FI, HU, IT, RO, SE |
| T | A trend published as "seasonally adjusted" | DE, NL |
| M | Three-month moving average under the middle month | EE, PT |
| B | Benchmarked to moving annual averages of survey results | HR, LU |
| I | Quarterly survey results interpolated monthly | BE, BG, CY, ES, FR, LT, LV, PL, SK |
| A, U | Ambiguous or not stated | DK; IE, MT, SI |

If the quarterly rate is the average of the monthly rate, the smoothing of groups T, M and
B passes into every quarterly result. That could produce three things already reported:

- the persistence misfit of the Bayesian model: an observed lag-1 autocorrelation of
  quarterly unemployment changes of 0.27, against about 0.06 in replicated data;
- the Dutch "delayed" response (NL is in group T);
- Luxembourg's weak quarterly association (LU is in group B), because a moving annual
  average spreads any response over more than the two lags of the headline.

An external review (2026-10-19) raised the first two. This study checks the mechanism and
tests the first two consequences. The third is reported descriptively.

## 2. Data

No new download. The frozen snapshots `snapshots/eurostat-2026-10-15` (`une_rt_q`,
updated 2026-09-10) and `snapshots/eurostat-monthly-2026-10-18` (`une_rt_m`, seasonally
adjusted, updated 2026-10-16). The monthly vintage is a month newer. Revisions of the
seasonal adjustment between them can break exact equality, so section 3 does not require
it.

## 3. Inheritance check (Q1)

For each country and quarter from 2009-Q1 to 2026-Q2 with the quarterly rate and all three
monthly rates, `m_q` is the mean of the three monthly rates. Reported per country:

- the mean and the largest absolute difference `|u_q − m_q|`;
- the correlation of the quarterly changes of `u_q` and `m_q`.

A country's quarterly rate is classed as the **average of its monthly rate** when the
correlation of the changes is at least 0.95 and the mean absolute difference is at most
0.1 pp. The classification decides only whether the consequences below can be read as
inherited; it is not a test.

## 4. Consequences (tests)

**Groups.** The smoothed group S = T ∪ M ∪ B = {DE, NL, EE, PT, HR, LU}. The comparison
group is D (direct monthly estimates; 8 countries). Groups I, A and U are left out of the
primary comparison.

**P1, persistence.** The country lag-1 autocorrelation of the quarterly unemployment
change is computed exactly as in the Bayesian predictive check (`bayes_fit._masked_stats`):

- the primary Bayesian sample, 2009-Q2 to 2019-Q4;
- flag-masked terms removed;
- pairs of consecutive unmasked quarters only.

Prediction: the mean autocorrelation of S exceeds that of D.

**P2, timing.** For each country, the classical headline regression is used: two lags,
pandemic-excluded common sample, from `review.sample_designs`. The lag share is
`(b1 + b2) / theta`. It is defined only where `theta` is negative with a 95% HAC interval
below zero; elsewhere the country is dropped. Prediction: the mean lag share of S exceeds
that of D.

**Inference.**

- An exact permutation test of the difference in group means: every assignment of the
  pooled countries to groups of the observed sizes is enumerated, and the test is
  one-sided in the predicted direction.
- Countries are the units, so the test makes no assumption about dependence over time.
- Holm over P1 and P2.

**Verdict.**

- *Supported*: the Holm-adjusted p-value is at most 0.05.
- *Contradicted*: the one-sided test in the opposite direction has a Holm-adjusted p-value
  of at most 0.05.
- *Not determined*: otherwise.

## 5. Descriptive additions

- **The misfit statistic.** The observed mean autocorrelation over all countries in the
  Bayesian sample, recomputed, and again without S. Both are set against the primary
  model's replicated mean.
- **Luxembourg.**
  - Its lag-1 autocorrelation and the inheritance check.
  - Its classical sum with two and with four lags of GDP growth, from the committed
    `docs/review/grid.csv` (pandemic-excluded).
  - This bears on, but does not test, whether annual-average smoothing attenuates its
    two-lag sum.
- **Sensitivity.**
  - The comparison group D ∪ I, for P1 and P2 (own Holm family).
  - P1 on the pandemic-excluded sample from 2009 to 2026 instead of 2009-2019.

## 6. Interpretation (fixed in advance)

- **Persistence.** If P1 is supported, the persistence misfit is at least partly a product
  of how smoothed countries produce their series, not only of labour-market dynamics. The
  manuscript says so.
- **Timing.** If NL is classed as an average of its monthly rate and P2 is supported, the
  Dutch delayed response cannot be separated from the trend method. The manuscript
  qualifies the Dutch claim accordingly.
- **Neither supported.** The smoothing remains a documented caveat, without evidence that
  it drives either result.
- **Status.** None of these is a statement about labour markets; they are statements
  about the data.

## 7. Outputs and reproduction

`eu-okun smoothing` writes `docs/smoothing/` and `docs/SMOOTHING.md`. The committed tables
are regenerated by a test from the frozen snapshots.

## Change log
