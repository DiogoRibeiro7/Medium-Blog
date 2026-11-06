# How much follow-up does a mixture cure model need?

Computational companion to a Medium article in preparation. Claim: a mixture cure model estimates the
fraction of units that never have the event (the "cure" or never-recover fraction) only when follow-up
reaches past the tail of the event times of the others. With shorter follow-up, the estimate is an
extrapolation set by the chosen latency distribution, and the data alone only bound the fraction from above.

## Run

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python simulate.py         # results/sim_raw.csv, sim_cells.csv, sim_summary.txt (about 20 minutes on 2 cores)
python application.py      # results/application.csv, application_summary.txt (EU industry busts)
python figures.py          # figures/fig1_*.png ... fig6_*.png
python check_results.py results/sim_summary.txt results/application_summary.txt   # compare with the committed versions, within tolerance
```

## What is computed

- **Models** (`models.py`): univariate mixture cure models with lognormal or Weibull latency and their
  no-cure counterparts, fitted by maximum likelihood (BFGS); 90% Wald intervals for the cure fraction on the
  logit scale; the Kaplan-Meier estimator; the Kaplan-Meier plateau (the nonparametric estimate of the cure
  fraction, and an upper bound for it); the Maller-Zhou alpha_n statistic for sufficient follow-up.
- **Simulation** (`simulate.py`): staggered entry over a calendar window of length tau, so individual
  follow-up is uniform on (0, tau). The latency distribution is scaled so its 95th percentile is 10, and the
  follow-up ratio R = tau / 10 runs from 0.25 to 3. Grid: cure fraction 0.2 or 0.5; latency lognormal
  (sigma 0.5 or 1.0) or Weibull (shape 1.5); n = 250, 1,000 or 4,000; 200 datasets per cell. For each
  dataset: bias and failure (error above 0.10) of the Kaplan-Meier plateau and of cure models with each
  latency family; coverage and width of the Wald intervals; whether AIC prefers a cure model; whether the
  Maller-Zhou statistic declares follow-up sufficient; the gap between the two latency families; the gap
  between the best cure and the best no-cure model at twice the longest follow-up.
- **Application** (`application.py`): the same quantities for busts in EU industries by depth band, from
  `../recovery-after-busts/results/episodes_all.csv`, with a stable upper bound for the never-recover share
  (Kaplan-Meier at the last time with at least 10 busts at risk).

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt
