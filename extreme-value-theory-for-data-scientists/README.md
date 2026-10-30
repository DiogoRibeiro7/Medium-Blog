# Extreme Value Theory for Data Scientists: The Tail Is Not a Rounding Error

Reproducible scientific companion to the Medium-ready manuscript
[`article.md`](article.md). This is a **synthetic experiment with a known
population**, not an empirical analysis of a real company, climate record, or
financial market. The aim is to isolate the extrapolation problem, verify the
mathematical calculations, and distinguish marginal tail probabilities from
temporally clustered extremes.

## Questions

1. How badly can whole-distribution Gaussian and lognormal models estimate
   very high quantiles when the upper tail has another shape?
2. How can generalized Pareto peaks-over-threshold fitting recover a known
   tail, and how sensitive is it to the threshold?
3. What information do diagnostic plots and iid bootstrap intervals provide?
4. How does the GEV block-maxima route differ from threshold exceedances?
5. Why can two time series have **identical marginal distributions**, yet
   different probabilities of at least one extreme within a block?

## Exact data-generating experiment

- Each training event is a positive magnitude.
- 90% of events come from 10 times a Beta(2, 5) body and lie below 10.
- 10% lie above 10, with excess distributed GPD(shape=0.30, scale=3).
- The threshold 10 is **known by construction**. In a real application,
  threshold choice would be uncertain.
- Full run: 40,000 iid training observations, 120,000 independent holdout
  observations, 180 whole-sample iid bootstrap resamples, and 400 nonoverlapping
  blocks of size 100 for GEV fitting.
- A separate 60,000-observation experiment uses independent Pareto moving
  maxima to illustrate clustering. It has **the same one-point marginal CDF**
  as its iid comparator but a different temporal dependence structure.
- Quick run: smaller independent simulations, saved with a _quick suffix.

All pseudo-random streams use the SeedSequence namespace convention with
master seed 20261029. The GPD parameter uses SciPy's genpareto shape
convention; the GEV fit reports theoretical xi = -scipy_genextreme_c.

## Reproduce

Use Python 3.12 in this directory:

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v
    python analysis.py
    python analysis.py --quick

The full command creates the versioned evidence listed below. The quick command
does **not** overwrite the publication artifacts. The notebook
[`extreme_value_theory.ipynb`](extreme_value_theory.ipynb) executes a reduced,
annotated experiment and is also checked in a fresh kernel by CI.

## Figures

| Figure | Publication artifact | Interpretation |
| --- | --- | --- |
| 1 | `figures/fig1_body_and_tail.png` | Ordinary observations and the logarithmic survival tail |
| 2 | `figures/fig2_far_tail_quantiles.png` | Exact versus Gaussian, lognormal, and POT high-quantile predictions |
| 3 | `figures/fig3_mean_excess.png` | Mean-excess diagnostic across thresholds |
| 4 | `figures/fig4_threshold_stability.png` | Shape stability and declining exceedance counts |
| 5 | `figures/fig5_gpd_fit_diagnostics.png` | GPD Q-Q and excess-survival checks |
| 6 | `figures/fig6_return_level_uncertainty.png` | Conditional iid bootstrap uncertainty for a far return level |
| 7 | `figures/fig7_block_maxima_gev.png` | GEV fit to 100-event nonoverlapping block maxima |
| 8 | `figures/fig8_dependence_clusters.png` | Equal marginals but different block-level extreme probabilities |

## Data tables

| File | Contents |
| --- | --- |
| `results/summary.csv` | Seed, synthetic truth, fitted parameters, bootstrap interval, dependence diagnostics |
| `results/model_quantiles.csv` | Five high quantiles, exact population values, three estimated models |
| `results/threshold_diagnostics.csv` | GPD shape, scale, exceedances and mean excess at eight thresholds |
| `results/bootstrap_return_levels.csv` | 180 fixed-threshold bootstrap estimates of the p=0.9999 quantile |
| `results/dependence_diagnostics.csv` | Five block sizes, empirical/exact risk for iid and dependent series |
| `analysis.py` | Typed data generators, estimators, seed isolation and eight plot generators |
| `tests/test_analysis.py` | Distribution identities, invariants, reproducibility and artifact tests |

No raw synthetic observations are committed because each draw is reproducible
from the fixed seed, versioned code and dependency versions. All stored CSVs
are *full-run* results. Any files ending in _quick are disposable smoke outputs.

## Interpretive limits

The exact GPD tail was chosen **on purpose**; it makes the comparison
diagnostic, not evidence that EVT outperforms all alternatives on real data.
The body-tail splice is artificial. The regular iid bootstrap is conditional
on an exact fixed threshold, stationary iid sampling, and the assumed GPD
family; it excludes threshold-selection and structural model uncertainty.
The moving-maximum demonstration is a separate constructed dependence model,
not evidence about a particular production platform.

The mean-excess and threshold-stability plots are diagnostics, not
identification theorems. High return levels do not imply reliable calendar
times without an event rate and a validated dependence/nonstationarity model.
See the manuscript's original-source references for the conditions under
which asymptotic GEV and GPD approximations are justified.
