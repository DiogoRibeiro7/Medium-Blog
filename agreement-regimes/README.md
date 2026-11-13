# The p95 rule after a change of regime, and forward in time

Companion code for a follow-up to `../second-count-prediction`. That study tested a rule for the smallest random
loss of rows a second count can catch nine times in ten at a 5% alarm budget: about twice the 95th percentile of
how far log(delivered / outside count) strays from its twelve-month median on clean months. The rule missed on
one pair, Citi Bike files against NYC DOT's table, whose large deviations are confined to 2014-2015, and the
article conjectured that computing the rule on the months since the last level shift would fix it.

This folder tests that conjecture on the seven other pairs of that study, with the split at the largest level
shift chosen by a fixed rule, and adds a forward-in-time test: predict from the previous 36 months, observe
over the next 24. The hypotheses and every setting are in `PLAN.md`, committed before anything it defines was
computed. No data are fetched; the input is `../second-count-prediction/results/corrected/ratios.csv`.

## Files

- `PLAN.md`: the pre-registration.
- `feed/detect.py`: the second-count-prediction detector, prediction rule and loss grid, copied unchanged (a
  unit test checks the copy byte for byte).
- `feed/regimes.py`: the split, the segment and the forward windows.
- `evaluate.py`: the test; writes `results/summary.txt`, `results/pairs.json`, `results/windows.csv` and
  `results/curves.csv`.
- `figures.py`, `figures/`: the figures of the article.
- `tests/`: unit tests.

## Run

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python evaluate.py --ratios ../second-count-prediction/results/corrected/ratios.csv --out results
python figures.py
```

`.github/workflows/agreement-regimes.yml` runs the same steps and commits the results and figures.
