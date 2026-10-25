# Mixed-effects models on three public data sets

Reproduction code for the Medium article on mixed-effects (multilevel) models: partial pooling, within/between confounding, standard errors under clustering, and conditional versus marginal effects in a logistic GLMM.

## Scripts

- `lmm.py` — linear mixed model fitted by profiled REML/ML (lme4's parameterisation), one grouping factor, any number of random effects per group. `check_lmm.py` cross-checks it against `statsmodels.MixedLM` on the radon and High School and Beyond models.
- `glmm.py` — random-intercept logistic regression by adaptive Gauss–Hermite quadrature (Laplace when `n_agq=1`).
- `01_radon_pooling.py` — Minnesota radon: complete, no and partial pooling; scikit-learn's `TargetEncoder` against empirical Bayes shrinkage; 200 repeated hold-out splits; leave-counties-out; counties above the EPA action level.
- `02_hsb_within_between.py` — High School and Beyond: ICC, within/between/random-intercept SES slopes, the Mundlak correction, standard errors under four models, Raudenbush & Bryk's intercepts-and-slopes-as-outcomes model.
- `03_coverage_simulation.py` — 2,000 simulated data sets per number of schools (10 to 160), calibrated on High School and Beyond: coverage of nominal 95% intervals for a school-level effect and a student-level slope. `--figure-only` redraws the figure from `results/coverage_sim.csv.gz` after a full run.
- `04_contraception_glmm.py` — 1988 Bangladesh Fertility Survey: GLMM, GEE and plain logistic regression; conditional versus population-averaged effects.

Each script writes its numbers to `results/` and its figure to `figures/`.

## Run

```bash
pip install -r requirements.txt
python 01_radon_pooling.py        # ~1 min
python 02_hsb_within_between.py   # seconds
python 03_coverage_simulation.py  # ~5 min on 2 cores
python 04_contraception_glmm.py   # seconds
```

## Data

All three data sets are read from the Rdatasets mirror (https://github.com/vincentarelbundock/Rdatasets) and cached in `data/`:

- `HLMdiag::radon` — 919 homes in 85 Minnesota counties (Price, Nero & Gelman 1996; Gelman & Hill 2007).
- `mlmRev::Hsb82` — 7,185 students in 160 schools, High School and Beyond 1982 (Raudenbush & Bryk 2002, ch. 4).
- `mlmRev::Contraception` — 1,934 women in 60 districts, 1988 Bangladesh Fertility Survey (Huq & Cleland 1990).

## Contact

Diogo Ribeiro — dfr@esmad.ipp.pt — ORCID 0009-0001-2022-7072
