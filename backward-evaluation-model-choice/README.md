# Choosing a forecasting model by backward evaluation: a pre-registered test

Companion code for the article *Backward Evaluation as a Model-Selection Rule: A Pre-Registered Test
Against the Forward Backtest on Daily Web Traffic*.

The earlier folders in this repository (`../backward-evaluation`, `../backward-evaluation-pooling`,
`../backward-evaluation-episode-end`, `../backward-evaluation-generality`) used backward evaluation,
fitting a model on recent data and scoring it on the past, to decide **how much history** to train
on. This folder tests the other use: deciding **which model** to deploy. At every decision date (every
91 days) each candidate is scored in two ways, and the procedure deploys the candidate with the
lowest score for the next 91 days:

- **FWD**, the forward backtest: the candidate's one-day-ahead log loss over the last 364 days, as
  it was actually deployed (expanding window, refitted every 91 days).
- **BWD**, backward evaluation: the candidate fitted once on the last 364 days and scored, one day
  ahead, on everything before a 28-day gap. **BWD1** scores only the year before the gap; **HYB**
  averages the FWD and BWD scores.

Comparators: **ENS** (equal-weight pool of the candidates' predictive densities), **FIX** (the
candidate that did best on the development set) and **ORACLE** (the best candidate of each 91-day
period, in hindsight). Candidate sets: **A**, four model families with Student-t distributions
(autoregression of order 7, a seasonal linear model, additive Holt–Winters, LightGBM; the models of
`../backward-evaluation-generality/models.py`); **B**, autoregressions of orders 1, 7, 14 and 28;
**A8**, the four families with Gaussian and Student-t distributions.

Data: a test set of **245 English Wikipedia articles** from seven lists fixed before any page-view
value was downloaded (US states, cities, Solar System bodies, programming languages, musical
instruments, sports, mammals; daily views, July 2015 to October 2026), and an in-sample development
set: the 25 electricity-load series and 304 Wikipedia articles of `../backward-evaluation-generality`.

## The pre-registration chain

| commit | what |
| --- | --- |
| `a58d924` | availability probe for the test lists (canonical titles and day counts, never values) |
| `d545f5d` | probe log, written by the probe workflow |
| `4b0531e` | [`PREREGISTRATION.md`](PREREGISTRATION.md), `fetch.py` and the data workflow |
| `f4f78d1` | test data, downloaded by the data workflow with the pre-registered inclusion rules applied |
| `1714511` | analysis code and the compute workflow, committed before any test value was read |
| `a96d047` | results, committed by the compute workflow |
| `3a479ab` | figures, exploratory follow-up (`post.py`) and this README |

## Files

- `PREREGISTRATION.md`: hypotheses H1–H3, data and inclusion rules, candidates, procedures, verdict
  rules and secondary analyses.
- `probe/`: the availability probe and its log.
- `fetch.py`, `data/`: `wikipedia_test_daily.csv.gz`, `inclusion.txt` (every candidate article and the
  rule that included or excluded it) and `fetch_log.txt` (SHA-256 of every response).
- `choice.py`: one shard of one data set: deployed one-day-ahead log losses of every candidate,
  backward scores at every decision date, the choices of every procedure and their losses.
- `aggregate.py`: combines the shards, computes H1–H3 and the secondary analyses.
- `post.py`: exploratory follow-up, not pre-registered (disagreements, level drift, decision year).
- `figures.py`, `figures/`: the five figures of the article.
- `tests/`: unit tests on synthetic data.

## Run

About two CPU-hours. The workflow `.github/workflows/backward-evaluation-model-choice.yml` splits the
computation into 17 shards, aggregates them and commits the results.

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python choice.py --domain dev_elec --workers 4
for d in dev_web test_web; do for k in $(seq 0 7); do python choice.py --domain $d --shard $k --of 8 --workers 4; done; done
python aggregate.py
python post.py
python figures.py
```

## Results

Pre-registered verdicts on the 245 test articles (`results/summary.txt`; log loss in nats per day,
one-sided paired t-tests across articles):

- **H1, backward evaluation picks better models than the forward backtest (set A): reversed.**
  `L(FWD) − L(BWD) = −0.0249` (se 0.0018): the backward choice was worse, in all seven lists.
- **H2, the advantage is larger among model families than among autoregression orders: reversed.**
  The backward choice lost more among families (−0.0249) than among orders (−0.0079).
- **H3, pooling the candidates beats both selection procedures: supported.** The pool beat the
  backward choice by 0.044 and the forward choice by 0.019 nats per day (7 and 6 of 7 lists).

Other results: the forward backtest's scores correlated with the next quarter's losses about three
times as strongly as the backward scores (mean Spearman 0.52 against 0.17, set A); always deploying
the seasonal linear model (FIX) did as well as choosing by the forward backtest; the backward score
penalised the families with many parameters tied to the recent year (seasonal linear, LightGBM) far
more than the adaptive ones (Holt–Winters), and its picks did better than the forward backtest's
only in decisions taken in 2020. A local rerun of one test shard (31 articles) reproduced the
workflow's losses and choices exactly.
