# Measurement noise or finance? Luxembourg's weak output-labour link (protocol)

Registered on 2026-10-19, before any of the quantities below was computed. The only data
work beforehand was checking which Eurostat series exist, in which units and for which
countries (section 2). No series used here was downloaded into a snapshot, and no
regression, correlation or autocorrelation involving them was computed.

## 1. Starting point

The case study (HETEROGENEITY_PROTOCOL.md; docs/HETEROGENEITY_CASE_STUDY.md) found that in
Luxembourg, with 2020-2021 excluded:

- measured output per hour moves almost one for one with measured GDP (elasticity +1.06);
- domestic and resident employment barely move (+0.03 and +0.02);
- the classical Okun sum is sign-unstable.

The commuter explanation was not supported. An external review (2026-10-19) pointed out
that this pattern is also what measurement error in quarterly GDP would produce, and the
manuscript now lists it as an untested rival explanation. Ireland and Malta also have
among the weakest associations. This study tests two explanations that can each fail:

- **H1, transitory measurement noise.** Quarterly output growth contains noise that is
  unrelated to labour input and averages out over time.
- **H2, financial value added.** The financial sector's value added moves with things
  other than labour input, such as asset prices and fees, and it is large in Luxembourg.

Neither is a causal claim, and other explanations remain (section 8).

## 2. Data

A new snapshot, `snapshots/gdp-measurement-2026-10-19`, downloaded from the Eurostat
Statistics API and frozen with SHA-256 hashes:

| Key | Dataset | Filters | Use |
| --- | --- | --- | --- |
| `va_clv` | `namq_10_a10` | B1G, SCA, CLV20_MEUR, nace_r2 TOTAL and K | volumes |
| `va_cp` | `namq_10_a10` | B1G, SCA, CP_MEUR, nace_r2 TOTAL and K | weights |
| `hours` | `namq_10_a10_e` | EMP_DC, SCA, THS_HW, nace_r2 TOTAL and K | labour input |

- **Unemployment** comes from the frozen quarterly snapshot `snapshots/eurostat-2026-10-15`
  (`une_rt_q`), as everywhere else in the study.
- **Coverage, checked beforehand.** Value added at current and chain-linked prices exists
  for all 27 countries from 2008-Q1. Hours exist for 26 (not Belgium) and finance hours
  for LU, IE and MT among others. Countries without the data are dropped from the
  statistic that needs it and listed.
- **Vintages.** `namq_10_a10` was updated on 2026-10-19 and `namq_10_a10_e` on
  2026-10-15. Revisions between them and the frozen GDP series are not reconciled.
- **Value added, not GDP.** Gross value added excludes taxes less subsidies on products.
  Every comparison below stays within value added, so that the finance comparison is not
  mixed with the GDP-GVA difference. A descriptive table compares GVA growth with the
  frozen GDP growth.

## 3. Measures

For country i and quarter t (all growth rates in per cent):

- `g_t = 100 Δ ln V_t`, with V total value added in chain-linked volumes.
- `gK_t = 100 Δ ln VK_t`, with VK the value added of finance and insurance (K).
- **Non-financial value added.** Its growth is
  `gN_t = (g_t − w_{t−1} gK_t) / (1 − w_{t−1})`, with `w_t` the share of K in total value
  added at current prices. Chain-linked volumes are not additive, so this weighted
  subtraction is the primary method. The difference of chain-linked levels, `V − VK`, is a
  sensitivity.
- **Hours.** `h_t = 100 Δ ln H_t` (all industries) and `hN_t = 100 Δ ln (H_t − HK_t)`
  (outside finance). Hours are additive.
- **Year on year.** `G_t = 100 (ln V_t − ln V_{t−4})` and `Y_t = 100 (ln H_t − ln H_{t−4})`.
- **Unemployment.** `du_t`, the quarterly change in the unemployment rate, in pp.

## 4. Sample and statistics

**Sample.** Rows run from 2009-Q3 to 2026-Q2. A row is kept only if every level it uses
lies outside 2020-Q1..2021-Q4: levels t−3 to t for a regression with two lags of growth,
t−4 to t for a year-on-year change, t−2 to t for an autocorrelation of growth. This is
stricter than the case study's window [t−2, t] by one quarter, following the
review's point about the level behind the longest lagged change.

**Statistics, per country:**

- `rho_i`: the first-order autocorrelation of `g`, over rows with both `g_t` and `g_{t−1}`.
- `c_q,i = corr(h_t, g_t)` and `c_y,i = corr(Y_t, G_t)`; their difference
  `dc_i = c_y,i − c_q,i`.
- `b_i`: the two-lag sum of `h` on `g` (OLS of `h_t` on a constant, `g_t`, `g_{t−1}` and
  `g_{t−2}`), and `bN_i`, the same of `hN` on `gN`; their difference `db_i = bN_i − b_i`.
- `theta_i`: the two-lag sum of `du` on `g`, and `thetaN_i`, of `du` on `gN`; their
  difference `dtheta_i = thetaN_i − theta_i`.

**Comparison.** For a focus country f and a statistic s, the test statistic is
`s_f − median_{j ≠ f, j outside LU, IE, MT} s_j`. The comparison group excludes all three
focus countries.

## 5. Hypotheses and predictions

**H1, transitory measurement noise** (focus countries LU, IE and MT):

- **P1.1** `rho_f` is lower than in the comparison group. Noise in quarterly levels makes
  measured growth negatively autocorrelated.
- **P1.2** `dc_f` is higher than in the comparison group. Year-on-year changes average
  out level noise, so the correlation of hours with output should rise more where noise
  is larger.

**H2, financial value added** (focus country LU):

- **P2.1** `db_LU` is higher than in the comparison group: removing finance from both
  output and hours strengthens the hours elasticity more in Luxembourg.
- **P2.2** `dtheta_LU` is lower (more negative) than in the comparison group: the Okun
  sum becomes more negative when output excludes finance.

That gives eight predictions: P1.1 and P1.2 for three countries, and P2.1 and P2.2 for
Luxembourg.

## 6. Inference and verdicts

- **Bootstrap.** A synchronised circular block bootstrap over the rows' quarters: blocks
  of 4 consecutive quarters are drawn, every country and every statistic of a quarter is
  kept together, and every statistic is recomputed in each replicate. Each row carries its
  own lags, so lags are never broken. `B = 1999`, seed 20261025.
- **Reported.** 95% percentile intervals and the two-sided bootstrap p-value
  `2 min(share ≤ 0, share ≥ 0)` of each test statistic.
- **Multiplicity.** Holm over the eight predictions.
- **Verdict.**
  - *supported*: the interval excludes zero in the predicted direction and the Holm p-value
    is at most 0.05;
  - *contradicted*: the interval excludes zero in the opposite direction and the Holm
    p-value is at most 0.05;
  - *not determined*: otherwise.
- **Interpretation, fixed in advance.**
  - P1.1 and P1.2 both supported for a country: its quarterly output data show the
    signature of transitory noise. The manuscript says that measurement noise is
    consistent with the data, not that it is proven.
  - P2.1 or P2.2 supported for Luxembourg: the financial sector accounts for part of the
    weak link.
  - Neither: both explanations fail their tests, and Luxembourg's weak link remains
    unexplained.
  - Supported predictions are patterns consistent with a mechanism; none is a causal
    effect.

## 7. Sensitivity and descriptive additions

- **Sensitivity** (each with its own Holm family): non-financial value added from the
  difference of chain-linked levels; block length 8.
- **Descriptive:**
  - the share of finance in value added and in hours (2009-2019 means);
  - Luxembourg's non-financial output-per-hour elasticity (`gN − hN` on `gN`, two-lag sum),
    against its total;
  - the correlation between GVA growth and the frozen GDP growth, by country.

## 8. Not tested

- **Ireland's modified domestic demand.** It is published by the Irish Central Statistics
  Office, not Eurostat, and is not used here.
- **Annual national accounts.** Year-on-year changes of quarterly data are used instead,
  so the sample and vintage stay the same.
- **Measurement error shared by output and hours,** for example hours estimated from
  output, cannot be detected by these tests.
- **Sectoral composition beyond finance** is not tested.

## 9. Outputs

`eu-okun fetch-gdp-measurement` (snapshot) and `eu-okun gdp-measurement` (tables in
`docs/gdp-measurement/`, report `docs/GDP_MEASUREMENT.md`). The committed tables are
regenerated by a test from the frozen snapshot.

## Change log
