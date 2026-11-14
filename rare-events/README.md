# Rare events: fix the threshold, not the data

Reproduction code for the Medium article arguing that, when the event of interest is rare, rebalancing the training data (undersampling, oversampling, SMOTE or balanced class weights) does not make a classifier better at telling events from non-events; it moves the predicted probabilities away from the real risks, and the only useful thing it does, flagging more cases, is done more cleanly by keeping the data as they are and choosing the decision threshold from the costs of the two errors. It covers what each kind of rebalancing does to the fitted model, why a rebalanced model at a cutoff of 0.5 is an ordinary model at a lower cutoff in disguise, how to undo the shift when someone else has already rebalanced, and how to set the threshold for the decision the model has to support.

## Scripts

- `rare.py` — the methods, written from scratch in NumPy/SciPy: logistic regression by Newton–Raphson with case weights, an offset and an optional ridge penalty (intercept not penalised); random undersampling and oversampling to a 50:50 balance; SMOTE (it reproduces imbalanced-learn's draws for the same seed); balanced class weights; prior correction of probabilities to a new event rate; AUC, Brier score, log loss, calibration intercept and slope; confusion counts, the cost-based threshold 1 / (1 + R), expected cost and net benefit.
- `data.py` — the two data sets and their design matrices.
- `check_rare.py` — checks: on the Default data, the logistic fits equal Tables 4.1 and 4.3 of *An Introduction to Statistical Learning* (2nd edition) and statsmodels; on the Caravan data, the confusion counts of the book's lab at cutoffs 0.5 and 0.25; weighted logistic regression equals statsmodels with frequency weights; the ridge fit equals scikit-learn; AUC, Brier score and log loss equal scikit-learn; SMOTE equals imbalanced-learn 0.14.2 with the same seed; on a large simulated case-control sample, undersampling changes only the intercept, by the logarithm of the sampling ratio.
- `01_default.py` — credit-card default (333 defaults among 10,000 customers): the five ways of training, what each does to the coefficients and the probabilities, which cutoff on the original model each rebalanced model at 0.5 amounts to, and out-of-fold discrimination, calibration and expected cost for four cost ratios (figures 1 and 2).
- `02_caravan.py` — who buys caravan insurance (348 buyers among 5,822 customers, 85 regressors): ridge-penalised logistic regression, the same five ways of training, out-of-fold discrimination, calibration, expected cost and net benefit (figure 3).
- `03_simulation.py` — a logistic model with known probabilities, event rates of 0.5%, 2% and 10%, training samples of 5,000 and 20,000, 200 samples per cell, each model evaluated on 100,000 test cases: calibration, error of the predicted probabilities against the true ones, and excess cost over the best possible decisions for three cost ratios (figures 4 and 5).
- `common.py` — data download and caching, figure style.
- `check_reproduction.py` — compares every number of a fresh run with the committed `results/*.txt`.

Each script writes its numbers to `results/` and its figures to `figures/`.

## Run

```bash
pip install -r requirements.txt
python check_rare.py           # also writes results/00_checks.txt
python 01_default.py
python 02_caravan.py           # about 3 minutes
python 03_simulation.py        # about 15 minutes on 2 cores; --reps 20 for a quick run
```

A GitHub Actions workflow (`.github/workflows/rare-events.yml`) reruns everything on each change and checks that every printed number matches the committed `results/*.txt`.

## Data

Downloaded on first run from the Rdatasets mirror (https://github.com/vincentarelbundock/Rdatasets) and cached in `data/` (not committed):

- `ISLR::Default` — simulated credit-card data for 10,000 customers: default, balance, income and student status (James, Witten, Hastie and Tibshirani 2021).
- `ISLR::Caravan` — 5,822 customers of an insurance company, 85 socio-demographic and product-ownership variables and whether they bought caravan insurance; from the CoIL Challenge 2000 (van der Putten and van Someren 2004).

## Contact

Diogo Ribeiro — dfr@esmad.ipp.pt — ORCID 0009-0001-2022-7072
