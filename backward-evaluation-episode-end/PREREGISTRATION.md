# Pre-registration: an end-of-episode check for the backward training-window rule

Written and committed on 9 November 2026, **before** the test data are downloaded and before
any analysis code exists. The only contact with the test series so far is the availability
probe in `probe/` (commit `085de46`), which recorded first and last months and observation
counts, not values. The data are fetched by `fetch.py` in a separate GitHub Actions commit
after this one, and the analysis code is committed after that.

## Background (known before this test)

The backward rule fits the forecasting model on the most recent block of a series (the anchor),
walks back block by block, and stops at the first block where the anchor model's log loss is
worse than the block's own cross-validated model with a one-sided Newey–West z statistic above
`z_crit`. It then trains on the anchor plus the blocks that passed (companion folder
`../backward-evaluation`).

Two earlier results shaped this design and are therefore not evidence for it:

1. In the simulations of `../backward-evaluation-pooling`, the rule's damage after temporary
   episodes was concentrated **after the episode had ended but while it was still inside the
   anchor block**. After a 24-period volatility burst the plain rule gained +0.046 nats per
   forecast in the first 60 periods and lost −0.025 afterwards; after a two-period crash it lost
   −0.194 in the first 60 periods. Once an episode has ended, the most recent data differ from
   the episode, so the end of an episode is observable in real time even though its duration at
   the start is not.
2. On ten US monthly series, Student-t(5) errors reduced the rule's losses (mean gain over the
   expanding window −0.0066 with Student-t against −0.0263 with Gaussian errors, eight test
   series, March–December 2020 not scored; industrial production +0.0031 against −0.0625).
   Pooling over window lengths had the largest mean gain (+0.0319 Student-t, +0.0635 Gaussian)
   and the fixed pool with weight 0.25 the largest among fixed pools (+0.0434 Gaussian).

Those ten US series (UNRATE, PAYEMS, HOUST, W875RX1, DPCERA3M086SBEA, WPSFD49207, GS10, M2SL,
INDPRO, CPIAUCSL) are the **development set**: they are reported, labelled in-sample, and do not
count towards any verdict.

## Hypothesis

> Adding an end-of-episode check to the backward rule removes most of the rule's losses after
> temporary episodes while keeping most of its gains after lasting changes, and the checked rule
> forecasts at least as well as the two robust defaults, pooling over window lengths and a fixed
> 0.25 pool.

## The rule under test

Notation: origin `o` (the first month forecast), block length `L`, recent span `r`.

At every re-selection origin (every 12 months), run the backward walk exactly as before.

- If the walk does not stop, train on all data (the expanding window), as the plain rule does.
- If it stops at block `j`, so that the backward window is `[o − jL, o)` and the rejected block is
  `[o − (j+1)L, o − jL)`, apply the **end-of-episode check**:
  - the *change model* is fitted on the backward window without its last `r` rows,
    `[o − jL, o − r)`;
  - the *past model* is fitted on the rejected block `[o − (j+1)L, o − jL)`;
  - both score the last `r` rows `[o − r, o)` one step ahead; with `d_t` the change model's log
    loss minus the past model's, `z_end = mean(d) / se(d)` (Newey–West standard error, as in the
    walk);
  - if `z_end > c` the past model forecasts the most recent rows significantly better than the
    model of the change: **the change has ended**, and the rule trains on the expanding window;
  - otherwise the rule trains on the backward window, as the plain rule does.

Primary settings: `r = 12` months (the time since the previous decision), `c = 1.645` (one-sided
5%). The check only ever switches the plain rule's choice to the expanding window, and only on
evidence that the most recent data look like the rejected past again; when the change is new and
the change model has not yet seen it, `z_end` is near zero and the backward window is kept.

The window chosen at a re-selection origin is held fixed for the next 12 origins; models are
refitted on that window at every origin.

## Schemes

All with the same model class at each origin:

- `E`: expanding window;
- `B`: plain backward rule;
- `C`: backward rule with the end-of-episode check (**the rule under test**);
- `P0.25`: linear pool `0.25 f_B + 0.75 f_E` whenever the walk stops (`E` otherwise);
- `AveW`: equal-weight pool of predictive densities over windows from `L` to all data in steps of
  12 months (Pesaran and Pick, 2011);
- `PC0.25`: linear pool `0.25 f_C + 0.75 f_E` whenever `C` uses the backward window (secondary,
  not part of any criterion).

Gain of a scheme over `E` = mean log loss of `E` minus mean log loss of the scheme (nats per
forecast; positive is better).

## Criteria and verdict (primary analysis)

Over the ten test series, with `G(·)` the gain over `E`:

- **H1, losses removed:** over the series where `B` loses to `E`, `Σ G(C) ≥ ½ Σ G(B)`.
- **H2, gains kept:** over the series where `B` gains on `E`, `Σ G(C) ≥ ½ Σ G(B)`.
- **H3, competitive:** `mean G(C) ≥ max(mean G(AveW), mean G(P0.25))`.

Verdict: **supported** if H1, H2 and H3 hold; **repairs the rule but does not beat the defaults**
if H1 and H2 hold and H3 fails; **not supported** otherwise. If `B` loses on no test series, H1 is
not testable and the verdict is *supported* if H2 and H3 hold and *not supported* otherwise.

## Test series (fixed now; chosen from the probe by availability only)

Inclusion rule applied to the probe: monthly; seasonally adjusted or without a seasonal pattern;
first observation no later than 1975; last observation no earlier than January 2024; one primary
source and at most one fallback, tried in this order.

Group A, real activity:

1. Japan, harmonised unemployment rate: FRED `LRHUTTTTJPM156S` (fallback FRED `LRUNTTTTJPM156S`);
   monthly change, percentage points.
2. Canada, harmonised unemployment rate: FRED `LRHUTTTTCAM156S` (fallback Statistics Canada
   vector 2062815); monthly change, percentage points.
3. United Kingdom, index of production (B–E, CVM, SA): ONS `K222` in `DIOP` (fallback FRED
   `GBRPROINDMISMEI`); `100 Δ log`.
4. Germany, production of total industry (SA): FRED `DEUPROINDMISMEI`; `100 Δ log`.
5. Japan, production of total industry (SA): FRED `JPNPROINDMISMEI` (fallback DBnomics
   `OECD/MEI/JPN.PRINTO01.IXOBSA.M`); `100 Δ log`.

Group B, interest rates and exchange rates:

6. United Kingdom, 10-year government bond yield: FRED `IRLTLT01GBM156N`; monthly change, pp.
7. Germany, 10-year government bond yield: FRED `IRLTLT01DEM156N`; monthly change, pp.
8. Canada, 10-year government bond yield: FRED `IRLTLT01CAM156N`; monthly change, pp.
9. Yen per US dollar: FRED `EXJPUS`; `100 Δ log`.
10. Swiss francs per US dollar: FRED `EXSZUS`; `100 Δ log`.

If a series cannot be downloaded from either source, it is dropped and the criteria are computed
over the rest; the article reports the drop.

## Settings (all analyses)

- Real data: AR(3) by maximum likelihood; transformed data from January 1960 (or the first
  available month); forecast origins from **January 1990** to the last month; one step ahead;
  blocks `L = 60` months; gap 12 months; at most 8 blocks back; `z_crit = 2.498`
  (`Φ⁻¹(1 − 0.05/8)`); re-selection every 12 months; 5-fold within-block cross-validation; fixed
  seeds.
- **Primary analysis:** Student-t errors with 5 degrees of freedom; scoring leaves March–December
  2020 out of the averages (not out of estimation or of the decisions).
- Secondary: the same with Gaussian errors (attribution: the check versus the error
  distribution); scoring all months; `r = 24`; `c = 0` and `c = 2.498`.
- Development set: the ten US series above, origins from January 1985 (as in the earlier study),
  both error distributions.

## Secondary evidence (reported, not part of the verdict)

**S1, simulations.** The eight scenarios of `../backward-evaluation-pooling/sims.py`: AR(1) with
φ = 0.5, σ = 1, 720 rows, change at row 480, forecasts for rows 480–719; regimes (volatility
1→2, 2→1; mean 0→1.5; φ 0.2→0.8), episodes reverting after 24 rows (volatility 3; mean 3), a
crash and rebound (shocks of −12 and +6 σ), and no change; blocks 60, gap 6, `r = 12`,
`c = 1.645`. Gaussian models: 300 replications, all schemes. Student-t(5) models: 300
replications, all schemes except `AveW`. For each model class: across the three episodes, does
`C` remove at least half of `B`'s losses; across the four regimes, does `C` keep at least half of
`B`'s gains; and the share of decisions in which `C` uses the backward window, by time since the
change, against `B`.

**S2, the real-versus-financial split on new data.** The earlier study found, exploratorily, that
strong rejections persisted in all 8 cases for prices, money and interest rates and in 3 of 9 for
real activity. The premise test of `../backward-evaluation-pooling/premise.py` (AR(3), Gaussian,
60-month blocks, gap 12; rejection strength `R_k` = z of block `k`'s model on block `k − 1`;
persistence `P_k = E[k−1, k+1] − E[k, k+1]`) is run on the ten test series. Predictions: the share
of triples with `R > 2.498` that persisted (`P > 0`) is higher in group B than in group A, and the
Spearman correlation of `R` and `P` is higher in group B than in group A. Reported with the counts
and a one-sided Fisher exact test for the shares; with this few triples the test has low power and
the result is read as a check of direction.

**S3, timing.** For each test series, the decisions in which the check switched the plain rule to
the expanding window, listed with their dates.
