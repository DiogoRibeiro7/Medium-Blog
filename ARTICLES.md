# Articles Developed in This Chat

This is a chronological index of the five long-form articles developed in
this conversation for `Medium-Blog`. It is **not** a complete
catalogue of the repository's articles or historical notebooks.

| # | Article | Main topics | Figures | Reproducible companion |
| --- | --- | --- | ---: | --- |
| 1 | [The Queue Behind the Forecast: Why Good Demand Predictions Still Produce Bad Waiting-Time Decisions](queueing-theory-for-data-scientists/article.md) | Queueing theory, arrival and service variability, utilization, waiting-time tails, capacity decisions, Monte Carlo simulation | 7 | [Study README](queueing-theory-for-data-scientists/README.md) · [Python analysis](queueing-theory-for-data-scientists/analysis.py) · [Notebook](queueing-theory-for-data-scientists/queueing_theory.ipynb) |
| 2 | [The Tail Is Not a Rounding Error: Extreme Value Theory for Data Scientists](extreme-value-theory-for-data-scientists/article.md) | Peaks over threshold, generalized Pareto and generalized extreme-value models, return levels, threshold diagnostics, uncertainty, dependence | 8 | [Study README](extreme-value-theory-for-data-scientists/README.md) · [Python analysis](extreme-value-theory-for-data-scientists/analysis.py) · [Notebook](extreme-value-theory-for-data-scientists/extreme_value_theory.ipynb) |
| 3 | [The Alarm Is Not the Change: Sequential Change-Point Detection for Data Scientists](sequential-change-detection-for-data-scientists/article.md) | Sequential stopping times, Shewhart, CUSUM, EWMA, detection delay, finite-horizon false-alarm calibration, autocorrelation | 8 | [Study README](sequential-change-detection-for-data-scientists/README.md) · [Python analysis](sequential-change-detection-for-data-scientists/analysis.py) · [Notebook](sequential-change-detection-for-data-scientists/sequential_change_detection.ipynb) |
| 4 | [A Factory Is Not a Histogram: Industrial Statistics for Data Scientists](industrial-statistics-for-data-scientists/article.md) | Rational subgrouping, statistical process control, capability indices, crossed gauge R&R, measurement misclassification, guard bands, factorial experiment | 8 | [Study README](industrial-statistics-for-data-scientists/README.md) · [Python analysis](industrial-statistics-for-data-scientists/analysis.py) · [Notebook](industrial-statistics-for-data-scientists/industrial_statistics.ipynb) |
| 5 | [A Good Fit Is Not an Identified Model: Inverse Problems for Data Scientists](inverse-problems-for-data-scientists/article.md) | Identifiability, singular-value decay, truncated SVD, Tikhonov regularization, GCV, Gaussian inverse uncertainty, sensor placement | 8 | [Study README](inverse-problems-for-data-scientists/README.md) · [Python analysis](inverse-problems-for-data-scientists/analysis.py) · [Notebook](inverse-problems-for-data-scientists/inverse_problems.ipynb) |

## Reproducibility

Each article has its own directory, dependency requirements, tests,
publication figures, and machine-readable experimental results. The analyses
use **synthetic data** to isolate specific statistical questions; their
numerical findings are not claims about a real organization or deployment.

To reproduce an article, enter its directory and follow the commands in
its linked study README. The manuscripts are prepared for Medium; inclusion
in this index does not imply that they have already been published there.
