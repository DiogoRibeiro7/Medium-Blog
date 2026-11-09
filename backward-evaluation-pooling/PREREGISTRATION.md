# Pre-registration: an evidence-weighted pool for the backward window rule

Written and committed **before** the eight test series below were downloaded. The commit that
adds this file precedes the commit in which the data workflow adds `data/fred_monthly.csv`.
Nothing in the rule, the criteria or the list of series is to be changed after the test data
are seen. Any analysis not listed here is reported as exploratory.

## Background

The backward window rule (companion project `backward-evaluation`) fits a model on the most
recent block of a series, walks back block by block, and stops at the first block where that
model's log loss is worse than the block's own cross-validated model with a one-sided
z statistic above `z_crit`. It then trains on the blocks that passed.

On US industrial production the rule lost to an expanding window because, after the 2008-09
recession and after COVID-19, the most recent block was itself an unusual episode: the rule
correctly found that the past differed from it and wrongly trained on the episode alone. The
companion article ended with a conjecture, stated before any test:

> a pool that gives the backward window a weight proportional to how strongly the last row
> rejects the older blocks would keep most of the rule's gains on inflation and remove most of
> its losses on industrial production.

**Known before this test.** In the first study the strongest rejections came in the episodes
(industrial production, Gaussian errors: z = 5.8 in January 2010 and z = 9.1 in January 2022),
and the rule's useful rejection on inflation was moderate (z = 3.1 at the last origin). Read
literally, this in-sample evidence already points against the conjecture, since a weight that
rises with the strength of rejection gives the most weight to the short window in exactly those
episodes. Industrial production and CPI inflation are therefore reported separately as the
in-sample case and do not count towards the verdict.

## The rule under test

- **Evidence-weighted pool (EWP).** At each re-selection origin run the backward walk (blocks of
  `block_len` rows, `gap` rows dropped next to the anchor, at most 8 blocks back,
  `z_crit = Φ⁻¹(1 − 0.05/8) = 2.498`). If the walk stops, let `z*` be the z statistic of the
  block at which it stopped. The forecast is the linear pool

  `w · f_B + (1 − w) · f_E`,  `w = Φ(z* − z_crit)`,

  where `f_B` is the Gaussian forecast from the backward window and `f_E` the forecast from the
  expanding window. At the threshold the two get equal weight; the weight on the backward window
  rises towards 1 as the evidence grows. If the walk does not stop, both components coincide.
  `w` is held fixed between re-selections (every 12 origins), like the window.

## Comparators

- `E`: expanding window.
- `B`: the plain backward rule (`w = 1` whenever the walk stops).
- `P_c`: fixed pools `c · f_B + (1 − c) · f_E` with `c ∈ {0.25, 0.5, 0.75}`.
- `AveW`: equal-weight pool of forecasts over window lengths from `block_len` to all data in steps
  of 12 (after Pesaran and Pick, 2011).

## Scores and criteria

Primary score: mean log loss (negative log predictive density, nats per forecast). Also reported:
90% coverage (PIT inside [0.05, 0.95]) and squared error of the pooled mean.

Gain of a scheme S on a series: `G(S) = mean log loss(E) − mean log loss(S)`.

- **C1, gains kept.** Over the test series where `G(B) > 0`: `Σ G(EWP) ≥ ½ Σ G(B)`.
- **C2, losses removed.** Over the test series where `G(B) < 0`: `Σ G(EWP) ≥ ½ Σ G(B)`
  (the loss shrinks by at least half).
- **C3, the weighting matters.** The mean of `G(EWP)` across test series is at least the mean of
  `G(P_c)` for the best fixed `c` in `{0.25, 0.5, 0.75}`, chosen in hindsight.

The conjecture is declared **supported** if C1, C2 and C3 all hold on the eight test series under
the primary scoring, and **not supported** otherwise. Each criterion is reported separately.

## The premise: does a strong rejection predict a lasting change?

The conjecture assumes that the strength of a rejection says something about whether the change
will last. This is tested directly on each test series with the train-period × test-period
matrix (AR(3), Gaussian, blocks of 60 months, gap 12):

- for consecutive blocks `k − 1, k, k + 1`, the **rejection strength** `R_k` is the z statistic of
  block `k`'s model on block `k − 1` against block `k − 1`'s own cross-validated model;
- the **persistence** `P_k = E[k − 1, k + 1] − E[k, k + 1]` is positive when block `k`'s model
  forecasts block `k + 1` better than block `k − 1`'s model does (the change persisted) and
  negative when the older model is better (it reverted).

Reported: the Spearman correlation between `R` and `P` over all such triples pooled across the
eight series, and the share of triples with `R > 2.498` in which `P > 0`. Under the conjecture's
premise the correlation is positive and most strong rejections persist. A sensitivity analysis
with 36-month blocks is reported as well.

## Test data

Eight monthly US series from FRED, none used in the first study, downloaded by `fetch.py`
(latest vintage), from January 1960 or the first month after:

| FRED id | Series | Transformation |
| --- | --- | --- |
| UNRATE | Unemployment rate | first difference (percentage points) |
| PAYEMS | Nonfarm payrolls | 100 × log change |
| HOUST | Housing starts | 100 × log change |
| W875RX1 | Real personal income excluding transfers | 100 × log change |
| DPCERA3M086SBEA | Real personal consumption expenditures | 100 × log change |
| WPSFD49207 | PPI, finished goods | 1200 × log change |
| GS10 | 10-year Treasury yield | first difference (percentage points) |
| M2SL | M2 money stock | 100 × log change |

In-sample, reported separately: INDPRO (100 × log change) and CPIAUCSL (1200 × log change).

## Settings

- Real data: AR(3) with Gaussian errors by least squares; blocks of 60 months; gap 12;
  forecast origins from January 1985 to the last month; one step ahead.
- **Primary scoring** leaves March-December 2020 out of the averages (not out of training).
  Secondary scoring uses all months.
- Simulations (secondary evidence, same criteria computed across scenarios): AR(1) with
  φ = 0.5, σ = 1; 720 rows; change at row 480; forecasts for rows 480-719; blocks of 60, gap 6;
  300 replications per scenario. Lasting regimes: volatility rises (σ 1 → 2), volatility falls
  (σ 2 → 1), level shift (μ 0 → 1.5), persistence rises (φ 0.2 → 0.8). Temporary episodes, which
  revert after 24 rows: a volatility burst (σ 1 → 3), a level episode (μ 0 → 3), and a crash and
  rebound (shocks of −12σ and +6σ in two consecutive rows). Plus no change.
- Robustness (secondary): Student-t(5) errors for E, B, EWP, P_0.5 and AveW on all ten series.
