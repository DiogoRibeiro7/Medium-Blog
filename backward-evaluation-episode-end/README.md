# An end-of-episode check for the backward training-window rule

Companion code for the article *An End-of-Episode Check for Backward Training Windows, Tested on Ten Series from Five Economies*.

The backward rule fits a forecasting model on the most recent block of a series, walks back block by
block, and trains only on the blocks that the recent model still explains
(`../backward-evaluation`). It works after lasting changes and fails after temporary episodes,
because an episode stays inside the most recent block for years after it has ended. This folder
tests one repair, written down before the test data were downloaded: when the walk stops, check
whether the most recent year looks like the rejected past again; if it does, the change has ended
and the rule trains on all the data.

## The pre-registration chain

| commit | what |
| --- | --- |
| `e2e8fc1` | availability probe for candidate series (records dates and counts, never values) |
| `085de46` | probe log, written by the probe workflow |
| `a727b3b` | [`PREREGISTRATION.md`](PREREGISTRATION.md), `fetch.py` and the data workflow |
| `e474332` | test data, downloaded and committed by the data workflow |
| `be380f8` | the same CSV stored as a plain file instead of a Git LFS pointer (identical SHA-256) |
| `59d7c61` | analysis code, committed before the test data were read |

## Files

- `PREREGISTRATION.md`: the rule, the schemes, the criteria H1–H3 and the verdict rule, the ten test
  series, the settings, and the secondary evidence S1–S3.
- `probe/probe.py`, `probe/probe_log.tsv`: the availability probe and its log.
- `fetch.py`, `data/`: the ten test series (`international_monthly.csv`, `sources.json`,
  `fetch_log.txt` with SHA-256 of every response). The development set (ten US series) is read from
  `../backward-evaluation-pooling/data`.
- `episode.py`: the end-of-episode check, the selection, a fast path for Gaussian least-squares
  autoregressions (cumulative sums) and a generic path for any model (Student-t here), and the
  criteria. The walk, the losses and the models come from `../backward-evaluation`; the linear pool
  from `../backward-evaluation-pooling`.
- `forward.py`: the pre-registered test on the ten series and the in-sample development set.
- `premise.py`: S2, the real-activity versus rates-and-currencies split.
- `sims.py`: S1, simulations with Gaussian and Student-t models (scenarios imported from
  `../backward-evaluation-pooling/sims.py`).
- `post.py`: descriptive summaries derived from the saved results (decisions, the period after
  COVID-19, attribution).
- `figures.py`, `figures/`: the five figures of the article.
- `tests/`: unit tests (direction of the check on clear cases, agreement of the fast and generic
  paths, verdict logic).

## Run

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python forward.py --workers 2          # about 5 minutes on two cores
python premise.py
python sims.py --model gaussian --reps 300 --workers 2
python sims.py --model t --reps 300 --workers 2    # about 20 minutes on two cores
python post.py
python figures.py
```

The workflow `.github/workflows/backward-evaluation-episode-end.yml` reruns everything and checks that
the regenerated summaries match the committed ones within tolerance
(`../backward-evaluation-pooling/check_results.py`).

## Results

**Pre-registered verdict (Student-t(5) errors, March–December 2020 not scored): not supported.**

- H1 fails: on the five test series where the plain rule loses to the expanding window, the check
  removed 19% of its losses (total gain −0.0523 → −0.0425 nats per month).
- H2 holds: on the five where it gains, the check kept all of it (+0.1427 → +0.1436).
- H3 fails: mean gain +0.0101 for the checked rule against +0.0177 for pooling over window lengths
  (+0.0090 plain rule, +0.0075 fixed 0.25 pool).
- Gaussian errors (secondary): the check removed 46% of the losses; not supported either.

Secondary evidence:

- S1 simulations: with Gaussian models the check removed 79% of the plain rule's losses after
  episodes and kept 99% of its gains after lasting changes; with Student-t models, 31% and 99.5%.
- After COVID-19 (forecasts from January 2021), mean gain over the expanding window across the ten
  test series: Gaussian −0.139 (plain) → −0.068 (checked); Student-t −0.0098 → +0.0005
  (`results/post_summary.txt`).
- S2: Spearman(R, P) +0.356 (p = 0.010) for rates and exchange rates against +0.025 (p = 0.86) for
  real activity; strong rejections persisted in 3 of 3 against 4 of 8 (one-sided Fisher p = 0.21).
- Development set (US, in-sample): with Gaussian errors the check removed 93% of the losses
  ("repairs the rule but does not beat the defaults"); with Student-t errors 38%.

## Data

FRED (Federal Reserve Bank of St. Louis): OECD Main Economic Indicators series `LRHUTTTTJPM156S`,
`LRHUTTTTCAM156S`, `DEUPROINDMISMEI`, `JPNPROINDMISMEI`, `IRLTLT01GBM156N`, `IRLTLT01DEM156N`,
`IRLTLT01CAM156N`; Board of Governors of the Federal Reserve System H.10 series `EXJPUS`, `EXSZUS`.
Office for National Statistics: index of production `K222` (dataset `DIOP`), Open Government Licence.
Latest vintages, downloaded on 9 November 2026; not real-time data.
