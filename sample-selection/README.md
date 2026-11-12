# Sample selection: what identifies a Heckman correction

Reproduction code for the Medium article arguing that a Heckman selection correction without a credible exclusion restriction is identified by the normal distribution alone, and that it can then be worse than no correction at all. It covers what an exclusion restriction buys, how to check one, what the correction does when there is none, and what to report instead.

## Scripts

- `sel.py` — the estimators, written from scratch in NumPy/SciPy: probit by Newton–Raphson; Heckman's two-step estimator with the corrected covariance matrix; full-information maximum likelihood with an analytic gradient (also with ρ held fixed, for profiles and sensitivity curves, and from several starting values); a series control function (polynomial in the inverse Mills ratio); diagnostics for the collinearity of the inverse Mills ratio with the outcome regressors; Manski's worst-case bounds on a quantile.
- `check_sel.py` — checks: the probit equals statsmodels; two-step and ML estimates, standard errors, σ, ρ and log-likelihoods on `sampleSelection::Mroz87` equal the output printed by the R package sampleSelection 1.2-16 for Wooldridge's and Greene's specifications, including the second, higher maximum of Greene's likelihood reported in the package vignette; a large simulated sample recovers the truth.
- `01_mroz_exclusions.py` — married women's wages (Mroz 1987): five exclusion restrictions, their strength, the collinearity of the correction term, the profile likelihood in ρ, tests of each exclusion, Manski bounds (figure 1).
- `02_randhie_two_part.py` — medical spending in the RAND Health Insurance Experiment with Cameron and Trivedi's specification (no exclusion restriction): two-part model against two-step and ML, local maxima, a bootstrap, and two placebos that keep the residuals' shape but remove selection (figure 3).
- `03_simulation_exclusion.py` — 64 designs (four error distributions, two values of ρ, four exclusion strengths, two selected shares), 500 samples of 1,000 each: bias, RMSE, coverage and local maxima for least squares, two-step, ML and a series estimator with and without an excluded variable (figures 4 and 5).
- `04_mills_linearity.py` — how close the inverse Mills ratio is to a straight line where the data are, in the two real cases and for a normal index (figure 2).
- `05_mroz_sensitivity.py` — what to report without a credible exclusion: a sensitivity curve in ρ for the return to education and the offered wage of non-workers, and Manski bounds against every model's implied median (figure 6).
- `common.py` — data download and caching, figure style.
- `check_reproduction.py` — compares every number of a fresh run with the committed `results/*.txt`.

Each script writes its numbers to `results/` and its figures to `figures/`.

## Run

```bash
pip install -r requirements.txt
python check_sel.py                   # also writes results/00_checks.txt
python 01_mroz_exclusions.py
python 02_randhie_two_part.py         # about 6 minutes on 2 cores
python 03_simulation_exclusion.py     # about 20 minutes on 2 cores; --reps 50 for a quick run
python 04_mills_linearity.py
python 05_mroz_sensitivity.py
```

A GitHub Actions workflow (`.github/workflows/sample-selection.yml`) reruns everything on each change and checks that every printed number matches the committed `results/*.txt`.

## Data

Downloaded on first run from the Rdatasets mirror (https://github.com/vincentarelbundock/Rdatasets) and cached in `data/` (not committed):

- `sampleSelection::Mroz87` — 753 married women in the Panel Study of Income Dynamics, work and wages in 1975 (Mroz 1987).
- `sampleSelection::RandHIE` — the RAND Health Insurance Experiment, as used by Cameron and Trivedi (2005, section 16.6).

## Contact

Diogo Ribeiro — dfr@esmad.ipp.pt — ORCID 0009-0001-2022-7072
