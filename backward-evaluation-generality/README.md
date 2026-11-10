# Backward training windows beyond economics: electricity load and Wikipedia traffic

Companion code for the article *Training-Window Rules Beyond Economics: Electricity Load, Wikipedia
Traffic and Four Model Families*.

Three earlier studies in this repository (`../backward-evaluation`, `../backward-evaluation-pooling`,
`../backward-evaluation-episode-end`) tested a backward training-window rule on monthly macroeconomic
and financial series with autoregressions. This folder asks whether their three main findings hold
on **daily** data from two other domains, with **four model families**: 25 European countries' daily
electricity load (Open Power System Data, 2015–2020) and 304 English Wikipedia articles' daily page
views (2015–2026), forecast one day ahead by an autoregression, a seasonal linear model, additive
Holt–Winters and LightGBM, each with Gaussian and Student-t predictive distributions.

## The pre-registration chain

| commit | what |
| --- | --- |
| `d5f687e` | availability probe (records columns, dates, counts and canonical titles, never values) |
| `8719ed8` | probe logs, written by the probe workflow |
| `5445ea3` | [`PREREGISTRATION.md`](PREREGISTRATION.md), `fetch.py` and the data workflow |
| `6132fbd` | data, downloaded by the data workflow with the pre-registered inclusion rules applied |
| `12d1216` | analysis code and the compute workflow, committed before the data were read |
| `78ea08e` | compute workflow fix (its first run computed every shard but failed at the final `git pull`); figures script |
| `4773e5c` | results and figures, committed by the compute workflow |

## Files

- `PREREGISTRATION.md`: the findings F1–F3, the data and inclusion rules, models, settings, verdict
  rules and secondary analyses.
- `probe/`: the availability probe and its logs.
- `fetch.py`, `data/`: `electricity_daily.csv`, `wikipedia_daily.csv.gz`, `inclusion.txt` (every
  candidate series and the rule that included or excluded it) and `fetch_log.txt` (SHA-256 of every
  response).
- `models.py`: designs and the four families (numba Holt–Winters filter; LightGBM with out-of-fold
  residual scale; Student-t regression for the linear families).
- `run.py`: one shard of one domain: window decisions every 91 days with the seasonal linear design,
  then every family on every window needed, one-step forecasts, log losses by 91-day period.
- `aggregate.py`: combines the shards, computes F1–F3 per cell and the verdicts, and the secondary
  analyses.
- `post.py`: descriptive summaries derived from the aggregated results.
- `figures.py`, `figures/`: the eight figures of the article.
- `tests/`: unit tests on synthetic data.

## Run

The full computation (329 series × 4 families × 2 distributions, refits every 91 days) takes about
three CPU-hours. The workflow `.github/workflows/backward-evaluation-generality.yml` splits it into 17
shards, aggregates them and commits the results.

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python run.py --domain elec --workers 4
for k in $(seq 0 15); do python run.py --domain web --shard $k --of 16 --workers 4; done
python aggregate.py
python post.py
python figures.py
```

## Results

Verdicts (pre-registered, 8 cells = 2 domains × 4 families; `results/summary.txt`):

- **F1, pooling over window lengths beats the backward rules: travels (8 of 8 cells).** With Student-t
  distributions, the mean of `G(AveW) − G(B)` ranges from +0.0013 (electricity, AR) to +0.1660
  (web, GBM) nats per day; pooling beat the plain rule in 72–100% of series per cell.
- **F2, the end-of-episode check repairs the plain rule under Gaussian models: does not travel
  (0 of 8).** It removed 13–49% of the plain rule's losses on web traffic and on electricity with
  the seasonal and boosting models, and made them larger on electricity with AR and ETS.
- **F3, heavy tails do much of the check's work: does not travel (2 of 8).** In 5 of 8 cells the
  plain rule did *worse* with Student-t than with Gaussian distributions; with Student-t scoring
  the walk rejected the past far more often (electricity 42% of decisions against 13%; web 79%
  against 64%).

Other results: the seasonal linear model had the lowest log loss of the four families in both
domains; Student-t distributions lowered every family's log loss by 0.16–0.34 nats per day; the
fixed 0.25 pool beat the expanding window on average in all 8 cells, pooling over windows in 7.
The second (canonical) run reproduced the first run's summary exactly except for the LightGBM web
cell, which differs in the fourth decimal (CPU-dependent floating point).

## Data

Open Power System Data, time series package 2020-10-06 (Wiese et al., 2019), ENTSO-E Transparency
Platform load data. Wikimedia pageviews API (agent `user`, all access), English Wikipedia,
2015-07-01 to 2026-10-31.
