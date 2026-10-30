# Dependent Data: Autocorrelation, Effective Sample Size, and Uncertainty

Companion study for *A Thousand Observations Are Not a Thousand Replicates: Autocorrelation, Effective Sample Size, and the Block Bootstrap*, article #10 in the [series index](../ARTICLES.md).

This project examines inference and forecast evaluation when observations are temporally dependent. All numerical results arise from **seeded synthetic data**, not from patients, companies, devices, or measured operational systems.

## Reproduction

Python 3.12 is verified in the article-specific CI workflow. From this directory:

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v
    python analysis.py
    python analysis.py --quick

The full command regenerates eight numbered publication PNGs and ten CSV tables, using a fixed seed of 20261030. The quick command writes separate files with the suffix _quick so CI does not replace the publication artifacts. The notebook [dependent_data_uncertainty.ipynb](dependent_data_uncertainty.ipynb) can be executed from a fresh kernel; its validation cell runs the small simulation in an isolated temporary directory.

## Experimental design

The principal model is a strictly stationary Gaussian autoregression:

\[
X_t=\phi X_{t-1}+\sqrt{1-\phi^2}\,\varepsilon_t,\qquad
\varepsilon_t\overset{\mathrm{iid}}{\sim}N(0,1),\quad |\phi|<1,
\]

where the initial observation is drawn from its stationary N(0,1) distribution. Thus all scenarios have marginal variance one, allowing the dependence structure to vary without confounding it with scale.

For a mean of n observations, the exact variance is

\[
\operatorname{Var}(\bar X)=\frac{1}{n}
\left[1+2\sum_{k=1}^{n-1}\left(1-\frac{k}{n}\right)\phi^k\right].
\]

For n=240 and \(\phi=0.8\), the finite-sample variance inflation is 8.8333, implying a mean-estimation precision equivalent to 27.17 independent unit-variance observations. The asymptotic long-run variance is 9.

The repeated-coverage experiment uses 400 independent paths per correlation value, 190 bootstrap resamples per path, a declared Bartlett HAC lag window of 13, and overlapping non-circular bootstrap blocks of length 13. It estimates coverage of five nominal-95% intervals: IID normal, IID bootstrap, Bartlett HAC normal, moving-block basic bootstrap, and an unattainable known-covariance Gaussian oracle. An additional experiment examines HAC lag and block-length sensitivity.

Separately, a one-lag forecasting exercise contains a known shift in the conditional mean at observation 800 of 1200. Random mixed-regime holdout, chronological pre-change holdout and genuine post-change forecast evaluation **target different populations**. This demonstration must not be read as an apples-to-apples comparison of algorithms.

## Figures

| Figure | Question |
| --- | --- |
| [1](figures/fig1_dependent_series.png) | How does temporal structure change when marginal variance is fixed? |
| [2](figures/fig2_autocorrelation.png) | What does the sample ACF reveal and fail to reveal? |
| [3](figures/fig3_effective_sample_size.png) | How many independent-observation equivalents remain for the sample mean? |
| [4](figures/fig4_sampling_distribution.png) | How does correlation change the sampling law of an average? |
| [5](figures/fig5_interval_coverage.png) | What proportion of nominal 95% intervals actually cover zero? |
| [6](figures/fig6_hac_bandwidth.png) | How sensitive is Bartlett HAC to the chosen maximum lag? |
| [7](figures/fig7_block_length.png) | How sensitive is the moving-block bootstrap to block length? |
| [8](figures/fig8_forecast_validation.png) | Why does the validation split define the forecasting question? |

## Machine-readable evidence

| Result file | Contents |
| --- | --- |
| results/series_realizations.csv | Four controlled sample paths |
| results/acf_diagnostics.csv | Empirical and population autocorrelations |
| results/effective_sample_size.csv | Exact variance inflation, effective sample size and large-n approximation |
| results/mean_sampling.csv | Repeated means under positive and negative correlation |
| results/coverage_replicates.csv | Per-path coverage and width for five interval methods |
| results/coverage_summary.csv | Empirical coverage, width and binomial Monte Carlo standard error |
| results/hac_bandwidth.csv | Covariance-estimator bias and coverage by truncation lag |
| results/block_length.csv | Block-bootstrap coverage and width by block length |
| results/forecast_validation.csv | Replicated RMSEs in three different holdout regimes |
| results/summary.csv | Reproducible headline metrics and experimental settings |

## Limits and interpretation

- Effective sample size here is **parameter-specific**: it matches the variance of the sample mean under the unit-variance model. It is not a count of independent subjects or a general information measure.
- Normal-based HAC inference is asymptotic. Lag-window choice matters; fixed n=240 and strong positive persistence are challenging.
- Moving-block bootstrap assumes an approximately stationary, sufficiently weakly dependent process and requires suitable block-length growth with sample size. Resampling across regime changes can invalidate it.
- The oracle uses the **known population autocovariance** and is a benchmark, not a deployable estimator. Its seeded empirical coverage differs from nominal because of Monte Carlo variation.
- Reported coverage proportions use a finite number of outer replications. Bootstrap intervals also have inner Monte Carlo error from 190 resamples.
- Negative correlations may improve the precision of a mean and make independence-based intervals unnecessarily wide.
- The forecast example is intentionally nonstationary and **is not used to justify** stationarity-based HAC or block-bootstrap intervals.
- Correlation can also arise through clustering, spatial spillovers, common shocks, and overlapping windows. The demonstration analyzes serial dependence only.

Read the [long-form manuscript](article.md) for complete derivations, interpretation and primary references.
