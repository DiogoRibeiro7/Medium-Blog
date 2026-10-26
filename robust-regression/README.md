# Robust regression on real data

Reproduction code for the Medium article on robust regression: M-, S-, MM- and least-trimmed-squares estimators, breakdown and efficiency, why deletion rules built on least squares miss clustered errors, robust fits as an audit of large real data sets, and what happens when the outliers are the quantity of interest (insurance claims).

## Scripts

- `rob.py` — the estimators, written from scratch in NumPy: Huber and bisquare M-estimation by IRLS (MAD scale, H1 or sandwich covariance), least trimmed squares (FAST-LTS with concentration steps), the S-estimator (FAST-S, bisquare ρ, 50% breakdown) and the MM-estimator (S start, fixed S scale, bisquare with 95% efficiency, sandwich standard errors), following the defaults of R's `robustbase::lmrob`.
- `check_rob.py` — checks `rob.py` against robustbase's published MM results for `starsCYG` and `stackloss`, against `statsmodels` RLM for Huber M-estimation, and checks that LTS, S and MM recover the true coefficients with 40% bad leverage points.
- `00_functions.py` — ψ and weight functions (figure 2) and asymptotic efficiencies at the normal model.
- `01_contamination.py` — 300 samples of 500 diamonds per scenario: prices typed with an extra zero, carats with a slipped decimal point (1% to 40% of rows), and t-distributed errors with 5, 3, 2 and 1 degrees of freedom; thirteen estimators compared on bias, RMSE, coverage and detection of the corrupted rows.
- `figures_01.py` — figures 3, 4 and 5 from the experiment-1 summary.
- `02_classic_cases.py` — Belgian telephone calls and the CYG OB1 star cluster: least-squares diagnostics against robust fits, and the outlier map (figure 1).
- `03_large_real.py` — the full diamonds data set (with its physically impossible rows) and the robustbase NOx data: least squares against MM, and the outlier map of 53,920 diamonds (figure 6).
- `04_claims.py` — 4,624 motor-insurance claims: Gamma GLM, least squares and robust fits on the raw and log scales, with and without Duan's smearing, over 100 train/test splits (figure 7).
- `05_software.py` — MM-estimation in Python three ways (this code, statsmodels `RLMDetSMM`, statsmodels `RLM` started from an S-estimate) against robustbase on `starsCYG` and `stackloss`.

Each script writes its numbers to `results/` and its figure to `figures/`. The article text is in `article.md`.

## Run

```bash
pip install -r requirements.txt
python check_rob.py
python 00_functions.py
python 01_contamination.py      # ~45 min on 2 cores; --reps 50 for a quick run
python figures_01.py
python 02_classic_cases.py
python 03_large_real.py         # ~2 min
python 04_claims.py             # ~6 min
python 05_software.py
```

A GitHub Actions workflow (`.github/workflows/robust-regression.yml`) reruns everything on each change and checks that the numbers match the committed `results/*.txt`.

## Data

All data sets come from the Rdatasets mirror (https://github.com/vincentarelbundock/Rdatasets), are downloaded on first run and cached in `data/` (not committed):

- `ggplot2::diamonds` — 53,940 diamonds with price, carat, cut, colour, clarity and dimensions.
- `MASS::phones` — Belgian international telephone calls, 1950–1973 (Rousseeuw & Leroy 1987).
- `robustbase::starsCYG` — 47 stars of the CYG OB1 cluster (Rousseeuw & Leroy 1987).
- `robustbase::NOxEmissions` — 8,088 hourly NOx measurements next to a Swiss motorway.
- `datasets::stackloss` — Brownlee's stack-loss data, used only in the checks.
- `insuranceData::dataCar` — 67,856 one-year motor policies from 2004–05 (de Jong & Heller 2008).

## Contact

Diogo Ribeiro — dfr@esmad.ipp.pt — ORCID 0009-0001-2022-7072
