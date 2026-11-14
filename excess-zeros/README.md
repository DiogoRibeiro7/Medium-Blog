# Excess zeros: overdispersion versus zero-inflation

Reproduction code for the Medium article arguing that excess zeros relative to a Poisson model are usually what overdispersion looks like, that a negative binomial model already explains most of them, and that zero-inflated models are often chosen for reasons that have little to do with the zeros. It covers where excess zeros come from, what the usual tests do, where a zero-inflated model's better fit actually comes from, and how to choose a count model for the decision it has to support.

## Scripts

- `counts.py` — the estimators, written from scratch in NumPy/SciPy: Poisson (with sandwich standard errors); negative binomial (NB2), with a constant dispersion or a dispersion that depends on regressors; zero-inflated Poisson and NB2; hurdle Poisson and NB2 (logit for zero versus positive, zero-truncated count); Vuong's test with the AIC and BIC corrections; fitted probabilities of every count. Maximum likelihood with analytic gradients and a Newton polish; the zero-inflated and hurdle models start from more than one point.
- `data.py` — the three data sets and their design matrices.
- `check_counts.py` — checks: on the NMES 1988 data, the Poisson, NB, hurdle and zero-inflated fits equal the output printed in the vignette of the R package pscl 1.5.9 (coefficients and standard errors to six or three decimals, log-likelihoods, AIC, BIC, expected zeros); on the bioChemists data, the Poisson, NB, zero-inflated and zero-truncated fits equal statsmodels; analytic gradients equal numerical ones; large simulated samples recover the truth.
- `01_three_datasets.py` — physician office visits (NMES 1988), articles of PhD biochemists, doctor visits in Australia (1977–78): six models, observed and expected zeros, Vuong tests, where each model's gain over the NB comes from, the zero-inflated model's share of structural zeros, decision quantities by model (figures 2, 3 and 4).
- `02_simulation.py` — five data-generating processes (NB; NB whose dispersion depends on a regressor; Poisson-lognormal; zero-inflated NB; hurdle NB), two sample sizes, 500 samples each: what the Vuong test, AIC and BIC choose, what the zero-inflated model reports as structural zeros, and the errors of three decision quantities for every model (figures 5, 6 and 7).
- `03_mechanism.py` — why overdispersion produces zeros: P(Y = 0) against the mean for the Poisson, NB and Poisson-lognormal, and observed against expected zeros in the three data sets (figure 1).
- `common.py` — data download and caching, figure style.
- `check_reproduction.py` — compares every number of a fresh run with the committed `results/*.txt`.

Each script writes its numbers to `results/` and its figures to `figures/`.

## Run

```bash
pip install -r requirements.txt
python check_counts.py          # also writes results/00_checks.txt
python 01_three_datasets.py
python 02_simulation.py         # about 15 minutes on 2 cores; --reps 50 for a quick run
python 03_mechanism.py
```

A GitHub Actions workflow (`.github/workflows/excess-zeros.yml`) reruns everything on each change and checks that every printed number matches the committed `results/*.txt`.

## Data

Downloaded on first run from the Rdatasets mirror (https://github.com/vincentarelbundock/Rdatasets) and cached in `data/` (not committed):

- `AER::NMES1988` — 4,406 people aged 66 and over covered by Medicare, US National Medical Expenditure Survey 1987–88 (Deb and Trivedi 1997).
- `pscl::bioChemists` — articles published in the last three years of the PhD by 915 biochemistry graduate students (Long 1990).
- `AER::DoctorVisits` — doctor visits in the past two weeks, 5,190 adults, Australian Health Survey 1977–78 (Cameron and Trivedi 1986).

## Contact

Diogo Ribeiro — dfr@esmad.ipp.pt — ORCID 0009-0001-2022-7072
