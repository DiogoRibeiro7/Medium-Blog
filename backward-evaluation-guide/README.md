# Reading the backward-evaluation matrix: a diagnostic, not a rule

Companion code for the article *Reading the Backward-Evaluation Matrix to Diagnose How
Forecasting Models Fail Across Periods*, the last part of the backward-evaluation series.

The earlier folders (`../backward-evaluation`, `../backward-evaluation-pooling`,
`../backward-evaluation-episode-end`, `../backward-evaluation-generality`,
`../backward-evaluation-model-choice`) built the train-period × test-period matrix and tested
rules that act on it automatically, to choose a training window or a model. Those rules lost to
pooling and to the forward backtest. This folder is about reading the matrix instead, with one
new tool: every cell's excess log loss is split into

- **level**, what a constant shift of the forecasts, fitted on the scored block, removes;
- **spread**, what a constant rescaling of the predictive distribution then removes;
- **structure**, what neither removes: predictable variation the transferred model misses.

## The module

`bwdiag.py` is a single file (NumPy and SciPy only) meant to be copied into other projects.

```python
from bwdiag import LinearModel, lag_rows, transfer_matrix

X, z, t = lag_rows(series, 3)                     # AR(3) rows; any features in time order work
M = transfer_matrix(X, z, lambda: LinearModel(df=5), block_len=60, gap=12)
M.excess, M.level, M.spread, M.structure          # k x k matrices (train block x test block)
print(M.describe())                               # today's model on every earlier block
M.regime_costs(6); M.regimes(2)                   # contiguous regimes and how much they explain
```

- Models: anything with `fit(X, z)` and `predict(X) -> (mean, scale)`, Gaussian or Student-t
  (a finite `df` attribute). `LinearModel` (least squares or Student-t maximum likelihood) and
  `RegressorModel` (any scikit-learn style regressor, scale from out-of-fold residuals) are included.
- Models that are not row-based: `matrix_from_forecasts(z, blocks, forecast, own_losses, ...)`.
- Production: `monitor(X, z, factory, window, step, gap)` compares today's model with the window
  before it at regular intervals, using only data available at each date.
- `repair(z, mean, scale, df)` is the decomposition for any set of forecasts.

## Files

- `bwdiag.py`, `tests/`: the module and its unit tests.
- `sims.py`: simulated series with one known change (level, spread, dynamics, a one-block episode,
  none); `results/sims_summary.txt`.
- `macro.py`: US CPI inflation and industrial production growth (the first study's data), AR(3)
  with Student-t and Gaussian errors; `results/macro_summary.txt`, `results/macro_cells.csv`.
- `daily.py`, `aggregate.py`: the 25 electricity series and the 304 + 245 Wikipedia articles of
  the earlier studies; five models (autoregression, seasonal linear, Holt–Winters, LightGBM and a
  level-normalised LightGBM); matrices, monitoring and the forward check; `results/daily_*`.
- `PLAN.md`: the plan for the forward check, committed before the full computation.
- `figures.py`, `figures/`: the nine figures of the article.

## Run

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python sims.py --reps 100 && python macro.py           # about two minutes
python daily.py --domain elec --workers 4
for k in $(seq 0 11); do python daily.py --domain dev_web --shard $k --of 12 --workers 4; done
for k in $(seq 0 9); do python daily.py --domain test_web --shard $k --of 10 --workers 4; done
python aggregate.py && python figures.py
```

The daily part takes a few CPU-hours; `.github/workflows/backward-evaluation-guide.yml` runs it as
23 shards and commits the results.

## Results

- Simulations: the decomposition reads a level shift as level (0.63 of 0.62 nats), a doubling of
  the shock scale as spread (0.31 of 0.30), a change in persistence as structure plus spread
  (0.24 + 0.19 of 0.45) and a one-block episode as spread plus level; cells within a regime are
  at the noise floor (about 0.01 per repair).
- CPI inflation splits into two regimes at August 1981; today's model fails on 1976–81 by level.
  Industrial production splits at 2016 under Gaussian errors (the 2020 shock) and at 1986 under
  Student-t errors (the Great Moderation).
- Daily data, today's model on earlier years (245 test articles): Holt–Winters 0.06, AR 0.37,
  seasonal linear 1.14 (0.97 level), LightGBM 1.78 (1.30 level) nats/day; level-normalised
  LightGBM 0.14.
- Forward check (PLAN.md, T1): the level-normalised LightGBM beats LightGBM by 0.108 nats/day
  (se 0.008), on 239 of 245 test articles and in 7 of 7 lists (supported); 282 of 304 development
  articles; 25 of 25 electricity series. It still loses to the seasonal linear model (by 0.043).
- Monitoring shows the 2025 fall in Wikipedia traffic as a level change with flat structure.
