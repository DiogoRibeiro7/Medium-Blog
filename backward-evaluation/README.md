# Testing time-series models against the past: the train-period × test-period matrix

Computational companion to a Medium article. The usual backtest fits a model on one
stretch of a series and scores it on a later stretch. This project fits the model on
every block of the series and scores it on **every other block, earlier ones included**.
The result is a matrix; the usual backtest uses only the cells just above its diagonal.
The cells below the diagonal ("backward" evaluation: train on the present, test on the
past) answer a different question: *how far back does today's model still work, and what
kind of change separates the present from the past?*

The project is repository-first: the article is written from the numbers these scripts
produce.

## What is computed

1. **The matrix** (`backward_eval.transfer_matrix`). The series is cut into consecutive
   blocks aligned to its end. A model is fitted on each block and scored one step ahead on
   every row of every other block, skipping rows within `gap` rows of the training block.
   Each column is benchmarked against the block's own model, estimated by random K-fold
   cross-validation inside the block (valid for autoregressions with uncorrelated errors,
   Bergmeir, Hyndman and Koo 2018). A cell is the mean excess loss over that benchmark.
   Scores: log loss (negative log predictive density), squared error (as a ratio) and the
   share of outcomes inside the central 90% interval.
2. **The backward rule** (`backward_eval.backward_window`). Fit a model on the most recent
   block, walk back block by block, and stop at the first block where that model is worse
   than the block's own model with a one-sided z statistic (Newey-West standard error)
   above a threshold. Train on the blocks that passed. Default threshold: Bonferroni at
   5% over the blocks it may examine (z > 2.50 for eight blocks).
3. **Forward tests of the rule** (`schemes.py`). The rule is judged the ordinary way:
   one-step forecasts scored forward, against expanding and rolling windows, an
   equal-weight pool over window lengths (after Pesaran and Pick 2011), forward
   validation of the window length (after Pesaran and Timmermann 2007), and a
   reverse-ordered CUSUM-of-squares window (our implementation of the idea in
   Pesaran and Timmermann 2002).

## Run

```bash
python -m pip install -r requirements.txt
python fetch.py --out data          # FRED series (run by the data workflow; snapshot committed)
python matrix_sims.py --reps 500    # part 1: results/matrix_sims.npz, matrix_summary.txt  (~1 min)
python rule_sims.py --reps 300      # part 2: results/rule_sims.csv.gz, rule_windows.csv, rule_summary.txt (~8 min, 2 cores)
python application.py               # part 3: results/application_*.csv, application_summary.txt (~1 min)
python figures.py                   # figures/fig1_*.png ... fig7_*.png
python -m pytest -q tests
python check_results.py results/matrix_summary.txt results/rule_summary.txt results/application_summary.txt
```

`example_your_model.py` shows the same tools on a gradient-boosting model with twelve
lags (about 2–3 minutes; not part of CI).

## Using it on your own model

```python
from backward_eval import ResidualGaussian, transfer_matrix, backward_window

# rows in time order: X[r] features known before target z[r]
res = transfer_matrix(X, z, model_factory, block_len=120, gap=12)
E = res.excess("nlpd")              # excess log loss, train block (rows) x test block (columns)
C = res.mean_matrix("cover90")       # coverage of the 90% interval
last_row = E[-1, :-1]                # today's model on every past block, oldest first
choice = backward_window(X, z, len(z), model_factory, block_len=120, max_back=8, z_crit=2.5, gap=12)
```

A model is any object with `fit(X, z)` and `predict(X) -> (mean, scale)`; set a finite
`df` attribute for Student-t predictive distributions. `ResidualGaussian` wraps any
scikit-learn regressor.

Reading the matrix:

- **flat** - what the model learns is stable; use all the history;
- **values growing with distance from the diagonal** - gradual drift; the growth rate sets
  how often to retrain;
- **blocks** - regimes; do not train across the boundary;
- **a single row and column standing out ("a cross")** - one unusual period (a recession,
  a pandemic), not a regime; do not let it alone define the training window;
- **asymmetry** `E[i, j] != E[j, i]` - the change has a direction. Under log loss the side
  that loses more is the one whose test period has the larger variance or the larger
  regressor variance (for a pure variance change the excess is `(r - 1 - log r) / 2`,
  with `r` the ratio of test to training variance). Squared error does not see variance
  changes; coverage does.

## Data

`data/fred_monthly.csv`, downloaded by `fetch.py` through `.github/workflows/backward-evaluation-data.yml`
(log in `data/fetch_log.txt`, sources in `data/sources.json`). Latest-vintage FRED series:
INDPRO (industrial production, Federal Reserve G.17), CPIAUCSL (CPI-U, BLS), UNRATE (not
used in the article). Industrial production growth is 100 × the monthly log change; inflation
is 1200 × the monthly log change of the CPI. These are revised data, not the real-time
vintages a forecaster would have had.

## Files

- `backward_eval.py` - matrix, losses, models (`ARGaussian`, `ARStudentT`, `ResidualGaussian`), backward rule, reverse CUSUM of squares, population formulas
- `schemes.py` - forward comparison of window schemes (fast least-squares path and a general slow path)
- `dgp.py` - AR(1) processes with breaks and drift
- `matrix_sims.py`, `rule_sims.py`, `application.py`, `figures.py` - the three parts and the figures
- `tests/` - unit tests (loss formulas against SciPy, least squares against `lstsq`, matrix and rule behaviour on simulated series)
