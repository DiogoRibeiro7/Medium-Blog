# Selection, dependence, historical coverage and measurement

Companion code for the Medium article *Selection, Dependence, Historical Coverage and Measurement: Four Limits on What a Large Dataset Can Estimate*.

One script prints every number in the article and draws every figure. All data are simulated under stated assumptions; published figures from the cited papers (Meng 2018, Bradley et al. 2021, Ginsberg et al. 2009) appear only as inputs to arithmetic checks and are labelled as such in the output.

## What the script covers

| Section | Question |
|---------|----------|
| 1 | Selection. Meng's identity checked exactly on a finite population; RMSE and interval coverage of random versus selected samples when units with Y = 1 are 10% more likely to be recorded; bias-adjusted effective sample sizes, with checks against Meng (2018) and Bradley et al. (2021). |
| 2 | Dependence. Effective sample size of the mean of an AR(1) series (approximation, exact sum, Monte Carlo) and coverage of the naive interval; Kish design effect for clustered records, with a simulation check. |
| 3 | Frequency versus horizon. Brownian motion with drift: precision of the drift and volatility estimators for 5, 20 and 80 years sampled from annually to every five minutes; years of data needed for the drift to be distinguishable from zero. |
| 4 | Overlapping windows. How many independent decades 91 rolling ten-year windows from a century of annual returns are worth. |
| 5 | Screening. Largest correlation between a 128-week target and one million unrelated candidate series, white noise versus smooth AR(1), with Bartlett's approximation to the effective sample size of a correlation. |

## Run

From this folder:

```bash
python -m pip install -r requirements.txt
python large_data_short_history.py > results/output.txt
```

Python 3.13, under half a minute on a laptop. The script writes the three figures to `figures/` and prints the numbers; `results/output.txt` is the run the article quotes. All randomness comes from one seeded generator, so a rerun with the pinned versions reproduces the output.

## Assumptions

The parameters are illustrative, not estimates: a binary outcome with mean 0.5 and a 10% difference in recording propensity; AR(1) innovations and clustered errors that are Gaussian; a drift of 6% and volatility of 20% a year with Gaussian increments and no microstructure noise; independent annual returns for the overlapping-window exercise; AR(1) candidate series with φ = 0.95 as a stand-in for smooth weekly series. Section 3 draws the drift and volatility estimators from their exact distributions for the figure, and confirms the result with 300 directly simulated five-minute paths, each aggregated to every coarser grid.
