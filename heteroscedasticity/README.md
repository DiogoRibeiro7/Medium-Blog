# Heteroscedasticity: standard errors, weights and prediction intervals

Reproduction code for the Medium article on regressions whose error variance is not constant. It is organised around four decisions: whether a test for heteroscedasticity should choose the standard errors (it should not), which robust covariance or wild bootstrap to trust when a few points have high leverage, when weighting by an estimated variance function pays and why its standard errors still need to be robust, and how a variance model improves prediction intervals and the targeting of manual reviews.

The tests for detecting heteroscedasticity are covered by the R package [heteroTests](https://github.com/DiogoRibeiro7/heteroTests); this folder starts where those tests stop.

## Scripts

- `het.py` — the estimators, written from scratch in NumPy/SciPy: least squares with classical, HC0, HC1, HC2, HC3, HC4, HC4m and HC5 covariances (the formulas of R's sandwich package); the Koenker (studentised) Breusch–Pagan and White tests; restricted and unrestricted wild bootstrap tests of one coefficient with Rademacher, Mammen and Webb weights (t statistic with an HC covariance, vectorised); weighted least squares; feasible GLS with exponential and power variance functions; and linear quantile regression by linear programming (HiGHS).
- `check_het.py` — checks: classical and HC0–HC3 standard errors and WLS equal statsmodels; HC0 and HC4 standard errors and HC4 p-values on `sandwich::PublicSchools` equal the output printed by sandwich 3.1-3 (vignette and examples, after Cribari-Neto 2004); the tests equal statsmodels; the vectorised wild bootstrap equals a loop and, with n = 10, agrees within Monte Carlo error with the exact p-value from all 1,024 sign vectors; quantile regression equals statsmodels QuantReg; FGLS recovers a known variance function.
- `01_public_schools.py` — school spending and income by US state (`sandwich::PublicSchools`): twelve p-values for one coefficient, and Alaska's leverage (figure 1).
- `02_house_prices_pretest.py` — `wooldridge::hprice1` in levels and logs: the variance tests pass in logs while the HC3 standard errors are up to 39% larger, and HC4/HC5 explode at a leverage of 0.84 (figure 2).
- `03_simulation_inference.py` — size and power of 13 tests of a slope (classical, HC0–HC5, HC4m, two pretest rules, three wild bootstraps) for three designs (uniform, lognormal, real household incomes), four variance functions and five sample sizes, 2,000 samples per cell; the pretest's conditional behaviour; a digest of the cells (figures 3 and 4).
- `04_wealth_wls.py` — net financial assets of single-person households (`wooldridge::k401ksubs`): OLS, WLS with 1/income weights, FGLS with power and exponential variance functions, classical and HC3 standard errors against a pairs bootstrap of the whole procedure (figure 5).
- `05_simulation_wls.py` — OLS, fixed weights and estimated weights when the weights are right, wrong or pointless: RMSE and coverage with classical and HC3 standard errors (figure 6).
- `06_ames_intervals.py` — seven 90% prediction intervals for house prices (`modeldata::ames`, normal sales) over 200 fit/calibrate/test splits: coverage within price, age and variance groups, interval score, and the share of bad valuations caught by a review budget ranked by leverage, by the variance model or by quantile-regression width (figures 7 and 9).
- `07_simulation_intervals.py` — the same intervals with a known truth: exact conditional coverage by income decile under constant variance, variance proportional to x, heavy-tailed errors and a variance shape the model cannot follow (figure 8).
- `common.py` — data download and caching, figure style.
- `check_reproduction.py` — compares every number of a fresh run with the committed `results/*.txt`.

Each script writes its numbers to `results/` and its figures to `figures/`.

## Run

```bash
pip install -r requirements.txt
python check_het.py               # also writes results/00_checks.txt
python 01_public_schools.py
python 02_house_prices_pretest.py
python 03_simulation_inference.py     # about 4 minutes on 2 cores; --reps 200 for a quick run
python 04_wealth_wls.py
python 05_simulation_wls.py
python 06_ames_intervals.py           # about 40 seconds
python 07_simulation_intervals.py     # about 40 seconds on 2 cores
```

A GitHub Actions workflow (`.github/workflows/heteroscedasticity.yml`) reruns everything on each change and checks that every printed number matches the committed `results/*.txt`.

## Data

Downloaded on first run from the Rdatasets mirror (https://github.com/vincentarelbundock/Rdatasets) and cached in `data/` (not committed):

- `sandwich::PublicSchools` — per-capita public-school spending and income, US states, 1979 (Greene, Econometric Analysis).
- `wooldridge::hprice1` and `wooldridge::k401ksubs` — data sets of Wooldridge, Introductory Econometrics.
- `modeldata::ames` — the Ames, Iowa housing data of De Cock (2011).

## Contact

Diogo Ribeiro — dfr@esmad.ipp.pt — ORCID 0009-0001-2022-7072
