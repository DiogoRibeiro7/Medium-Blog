# The success specification decides which forecast wins: a pre-registered test on M5 food sales

Companion code for an article by Diogo Ribeiro on writing down what success means for a forecast before building
it. The test takes one demand-planning problem (weekly orders of 13,777 food products x store series from the M5
Walmart data), builds nine candidate forecasters, and ranks them under six definitions of success: the default a
team gets from its tools (daily absolute error at the grain of the data) and five clauses written for the planners
who use the forecast (the unit and timing of the decision, the cost of an error, the number of bad weeks they
tolerate, and how much the forecast may move between runs).

`PLAN.md` is the pre-registration, written before any sales figure was read.

## Results in one paragraph

The default picks the gradient-boosted model trained on absolute error, and that model loses on all four planner
clauses (H1 supported, 4 of 4 established): its weekly totals are 29% below sales, and 62% below in the most
intermittent series, because a model trained for the daily median predicts zero on most days of a slow product.
Choosing by the default captures 35% of the cost improvement over the planners' four-week average that the best
candidate offers (registered as "also reported"). Under the full specification as registered (lowest cost among
models no worse than the four-week average on bad weeks and on revisions), no candidate qualifies, because none
revises less than the four-week average, so the incumbent stays and H2 is not testable. Accuracy at the planner's
week and stability are only weakly related across the nine candidates (Kendall's tau 0.08; H3, a negative tau, not
supported). Descriptive analyses added after the results: the stability allowance at which the specification
switches to a model (19.5% more revision than the incumbent), blends of the weekly model with the four-week average
that meet the full specification, and the Poisson floor of the bad-week clause by sales volume.

## Files

- `PLAN.md`: the pre-registration and the one change after it (rounding before the bad-week threshold).
- `fetch_data.py`: downloads the M5 archive, checks its SHA-256 and extracts the three files used.
- `success_core.py`: data loading, the candidates, the LightGBM features and the metrics.
- `forecast.py`: runs the nine candidates at the 66 origins (14 warm-up, 52 test); writes `data/forecasts.npz`.
- `evaluate.py`: the registered test; `results/summary.txt` holds every registered number.
- `details.py`: descriptive analyses added after the results (`results/details.txt`); not part of the registered
  test.
- `figures.py`, `figures/`: the five figures of the article.
- `tests/`: unit tests on synthetic data with the M5 layout.

## Run

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python fetch_data.py --data data
python forecast.py --raw data/raw --out data          # about 20 minutes on 2 cores
python evaluate.py --raw data/raw --data data --out results
python details.py --raw data/raw --data data --out results
python figures.py --results results --out figures
```

`.github/workflows/forecast-success-specification.yml` runs the same steps and checks that the winners and verdicts match the committed summary. The data
(about 360 MB unpacked) and the forecasts are not committed.

## Data

M5 forecasting competition data (Walmart unit sales, prices and calendar, 29 January 2011 to 22 May 2016), as
mirrored in `Nixtla/m5-forecasts` on GitHub. Makridakis, S., Spiliotis, E. and Assimakopoulos, V. (2022), M5
accuracy competition: Results, findings, and conclusions, International Journal of Forecasting 38(4), 1346-1364.
