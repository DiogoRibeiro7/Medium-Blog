# Measurement error in regression

Reproduction code for the Medium article on measurement error: attenuation and leakage into correlated covariates, the outside information every correction needs (replicates, validation samples, instruments, reliability coefficients), what happens when each correction's assumption fails, and why a predictive model should be corrected only for the measurement it will be deployed on.

## Scripts

- `me.py` — the estimators, written from scratch in NumPy: corrected least squares (Fuller's method of moments, one or several error-prone covariates), regression calibration from replicates or from a validation subsample, SIMEX with quadratic and rational extrapolants, two-stage least squares with a replicate as the instrument, and multiple imputation of the true value from a validation subsample (outcome included), plus naive OLS/logistic fits and a bootstrap helper.
- `check_me.py` — large-sample checks: naive attenuation and leakage match theory, every correction recovers the truth under classical error, regression calibration with replicates equals the moment correction, the SIMEX path matches its exact form, 2SLS and the logistic fit match reference formulas.
- `01_nhanes_replicates.py` — three systolic readings per adult in NHANES 2009–2012: reliability, the first-reading effect and how it biases the age slope of blood pressure (figure 1).
- `02_davis_validation.py` — self-reported and measured weight and height (carData::Davis): non-classical self-report error, the swapped row 12, and a simulated 25% internal validation design (figure 2).
- `03_wage_instruments.py` — the return to schooling with IQ and KWW as noisy measures of ability (wooldridge::wage2): OLS, 2SLS in both directions and a reliability sensitivity curve (figure 3).
- `04_bfi_scales.py` — Big Five scales (psych::bfi): alpha and omega, and a driver regression of education on the scales corrected for unreliability (figure 4).
- `05_simulation.py` — 300 samples of 1,000 per scenario on NHANES covariates: six error structures (classical, correlated replicates, heteroscedastic, Berkson, systematic, differential) at reliabilities 0.9, 0.7 and 0.5, nine estimators, linear and logistic outcomes, and prediction on new data with the naive and the corrected model.
- `figures_05.py` — figures 5–8 from the simulation summaries, the SIMEX example, and the population quantities used in the text.

Each script writes its numbers to `results/` and its figure to `figures/`. The article text is in `article.md`.

## Run

```bash
pip install -r requirements.txt
python check_me.py
python 01_nhanes_replicates.py
python 02_davis_validation.py
python 03_wage_instruments.py
python 04_bfi_scales.py
python 05_simulation.py        # ~30 min on 2 cores; --reps 30 for a quick run
python figures_05.py
```

A GitHub Actions workflow (`.github/workflows/measurement-error.yml`) reruns everything on each change and checks that the numbers match the committed `results/*.txt`.

## Data

Downloaded on first run and cached in `data/` (not committed):

- `NHANES::NHANESraw` from the NHANES R package repository (https://github.com/ProjectMOSAIC/NHANES): NHANES 2009–2010 and 2011–2012, adults aged 20–79, unweighted.
- From the Rdatasets mirror (https://github.com/vincentarelbundock/Rdatasets): `carData::Davis` (Davis 1990), `wooldridge::wage2` (Blackburn & Neumark 1992) and `psych::bfi` (SAPA project).

## Contact

Diogo Ribeiro — dfr@esmad.ipp.pt — ORCID 0009-0001-2022-7072
