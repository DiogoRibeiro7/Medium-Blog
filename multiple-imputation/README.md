# Multiple imputation on real data

Reproduction code for the Medium article on multiple imputation: missingness mechanisms, Rubin's rules, why accurate single imputations give the wrong answer, how many imputations to run, predicting when the missingness changes at deployment, and a delta-adjustment sensitivity analysis on NHANES.

## Scripts

- `mi.py` — multiple imputation by chained equations written from scratch, following the R package mice: Bayesian linear regression (`norm`), predictive mean matching (`pmm`, type-1 matching, 5 donors), logistic regression (`logreg`) and random-forest donors (`rf`, after mice's method of that name); delta adjustment; Rubin's rules with Barnard–Rubin degrees of freedom and the fraction of missing information.
- `check_mi.py` — checks the pooling formulas, coverage on simulated MAR data, and agreement with `statsmodels`' MICE on the NHANES regression.
- `00_population.py` — describes the CPS population and how far the Mincer equation is from the population regression.
- `01_cps_mechanisms.py` — 1,000 samples of 1,000 men from the March 1988 CPS; 40% of education deleted under MCAR, MAR on experience, MAR on wages and MNAR on education itself; thirteen methods (complete cases, six single imputations, five multiple imputations) compared on bias, standard errors, coverage and imputation error. `--population model` regenerates wages from the fitted Mincer equation so the analysis model is correct.
- `figures_01.py` — figures 1 and 2 from the experiment-1 summaries.
- `02_number_of_imputations.py` — 200 repeated MI runs per m on one sample; Monte Carlo error of the pooled SE against von Hippel's rule.
- `03_prediction_shift.py` — gradient boosting and linear models trained with informative missingness, tested under the same mechanism, under MCAR, and with nothing missing.
- `04_nhanes_real.py` — NHANES 2009–2012 adults: missing-data patterns, complete cases against MICE, number of imputations from a pilot run, delta adjustment for SBP and the poverty ratio.

Each script writes its numbers to `results/` and its figure to `figures/`. The article text is in `article.md`.

## Run

```bash
pip install -r requirements.txt
python check_mi.py
python 00_population.py
python 01_cps_mechanisms.py --population model   # ~13 min on 2 cores
python 01_cps_mechanisms.py --population real    # ~13 min on 2 cores
python figures_01.py
python 02_number_of_imputations.py               # ~2 min
python 03_prediction_shift.py                    # ~3 min
python 04_nhanes_real.py                         # ~6 min
```

## Data

- `AER::CPS1988` via the Rdatasets mirror (https://github.com/vincentarelbundock/Rdatasets): 28,155 men from the March 1988 Current Population Survey (Bierens & Ginther 2001). No missing values, which is why it serves as the population for the deletion experiments.
- `NHANES::NHANESraw` from the NHANES R package repository (https://github.com/ProjectMOSAIC/NHANES): the 2009–2010 and 2011–2012 NHANES samples. The analysis uses participants aged 20–79 and ignores the survey weights, so its estimates describe the sample, not the US population.

Both are downloaded on first run and cached in `data/` (not committed).

## Contact

Diogo Ribeiro — dfr@esmad.ipp.pt — ORCID 0009-0001-2022-7072
