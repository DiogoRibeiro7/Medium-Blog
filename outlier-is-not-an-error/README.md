# An Outlier Is Not an Error — reproducible companion

Article #31 in the `Medium-Blog` long-form collection. The 11,000+ word [editorial manuscript is maintained privately](https://github.com/DiogoRibeiro7/research-articles-medium/blob/main/themes/statistics-for-data-scientists/outlier-is-not-an-error/article.md). This public companion develops reproducible experiments covering robust location, heavy tails, interval coverage, regression contamination, leverage, multivariate masking, minority-group alerts, tail-risk deletion, and distribution-shift calibration.

**Everything here is synthetic.** No proprietary files or real-world error labels are used. The deliberately shifted population in a simulation is *not* asserted to consist of bad records. Figures and statistics are original experimental outputs.

## Environment and execution

Python 3.12 is the tested environment. Install dependencies in your virtual environment:

```bash
python -m pip install -r requirements.txt
```

The four numerical libraries are pinned to the versions used for the archived evidence: NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8 and scikit-learn 1.8.0. Notebook execution dependencies have compatible bounded ranges. This controls known numerical-version drift, though bit-for-bit floating-point reproduction may still vary across platforms or BLAS implementations.

From this directory, regenerate the complete canonical evidence:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg python analysis.py
```

The deterministic seed is `20261102`. The script writes `results/{study}_replicates.csv`, `results/{study}_summary.csv`, and `figures/{figure}.png` for each of nine studies. Each summary contains the mean over independent outer repetitions, between-repetition standard deviation, Monte Carlo standard error, ±1.96-MCSE limits, and an analytic reference value where supplied. These are *simulation Monte Carlo errors*, not observational confidence intervals.

For a short, non-overwriting smoke test:

```bash
python analysis.py --quick
```

Quick mode creates `*_quick.csv` and `*_quick.png` variants; it does not alter the published canonical results. It uses three independent outer repetitions for most studies and sixteen for interval coverage (full mode uses 24 or 30, and 500 for coverage). Some quick-mode sample sizes, per-repetition inner repetitions, or ensemble sizes are smaller by design.

If you want to use a different random seed in **full** mode, specify a separate output directory: `python analysis.py --seed 123 --output /tmp/outlier-experiment-123`. The command rejects a noncanonical full seed targeting the article directory so that an exploratory run cannot silently replace the archived results.

To rebuild the checked-in, deliberately *unexecuted* notebook from its deterministic builder:

```bash
python notebook_builder.py
```

Open `outlier_is_not_an_error.ipynb` and run all cells from a fresh kernel. The notebook executes all nine quick studies in a temporary folder, displays all nine figures, and leaves the committed evidence untouched.

Run the numerical tests and artifact consistency checks:

```bash
python -m unittest discover -s tests -v
```

To additionally execute the notebook in a fresh kernel as CI does:

```bash
RUN_NOTEBOOK_EXECUTION=1 python -m unittest discover -s tests -v
```

The dedicated [GitHub Actions workflow](../.github/workflows/outlier-is-not-an-error.yml) is triggered only by changes under this study directory or its own workflow file. It installs the declared scientific dependencies, validates all archived evidence and source contracts, executes the notebook with `nbclient`, and runs quick reproduction. `MPLCONFIGDIR` is set to a writable temporary location in CI; set one locally if your home configuration directory is read-only.

## Evidence inventory

| # | Study / CSV basename | Figure | Outer repetitions (full) | Question |
| ---: | --- | --- | ---: | --- |
| 1 | `contamination` | `fig1_contamination_and_location.png` | 30 | How far do mean, median, trim, and fixed-scale Huber location move under a shifted mixture? |
| 2 | `tails` | `fig2_heavy_tail_sample_size.png` | 30 | How does the 75th-percentile absolute location error scale with sample size under normal, t3, and Cauchy tails? |
| 3 | `intervals` | `fig3_coverage_and_estimands.png` | 500 | What coverage do a Student-t *mean* interval and a binomial-sign *median* interval achieve for their different targets? |
| 4 | `regression` | `fig4_vertical_outliers_regression.png` | 30 | What happens to intercept and slope error when outcomes receive large random vertical shifts? |
| 5 | `leverage` | `fig5_leverage_is_not_influence.png` | 30 | Can equal predictor leverage yield radically different Cook's D and leave-one-out slope shifts? |
| 6 | `multivariate` | `fig6_robust_multivariate_distance.png` | 24 | How does a shifted cluster mask itself in classical covariance compared with MCD? |
| 7 | `minority` | `fig7_legitimate_minority_flagged.png` | 24 | Does a fixed Isolation Forest review budget disproportionately flag a legitimate minority? |
| 8 | `tail_risk` | `fig8_tail_deletion_changes_risk.png` | 30 | How much do top-1% deletion and capping change a lognormal mean, q99, and ES99? |
| 9 | `shift` | `fig9_thresholds_under_shift.png` | 30 | What does a fixed 1% reference threshold do under tail and variance shift, and what is lost in recalibration? |

The `tails` CSV metrics are explicitly named `q75_abs_error_sample_mean`, `q75_abs_error_median` and `q75_abs_error_trimmed_10pct`. Their `mean` summary field is the **across-outer-replication average of that 75th-percentile statistic**, not a sample or population mean. In particular, the Cauchy population mean does not exist. Under the normal model the analytic benchmark is `norm.ppf(.875)/sqrt(n)`, not `norm.ppf(.75)/sqrt(n)`.

There are **18 canonical CSVs** (raw independent replication observations plus one summary per study), **9 original PNGs**, **one executable notebook**, and the analysis, builder, tests, and workflow. CSV `theory` fields are blank when no analytic benchmark is provided; blank is not zero.

## Selected results to cross-check

All figures and prose are based on the checked-in `*_summary.csv` values. Roundoff is expected in displayed prose.

- Study 1: At 10% shifted-mixture probability, the population mean is 1.0 and the sample mean averages about 0.983. The median averages 0.128 and the fixed-scale Huber location 0.173. These estimate different targets.
- Study 2: Under Cauchy sampling, the sample average's outer-averaged 75th-percentile absolute distance from location zero remains about 2.35 at 800 observations; the median's is about 0.062. The Cauchy population mean is undefined.
- Study 3: At 2% probability of a +20 component, the Student-t interval covers its mean target 78.2% of the time, and the sign interval covers its separate median target 97.0% of the time.
- Study 4: At 8% random vertical response contamination, average absolute intercept errors are OLS 1.492, Huber 0.098, and Theil–Sen 0.240. Since shifts are independent of X, the population slope remains 2; don't interpret absolute slope error as asymptotic slope bias.
- Study 5: On-line and off-line high-leverage rows both have mean hat leverage 0.493, but mean Cook's D is 0.191 versus 68.290.
- Study 6: At 20% displaced-mixture share, classical Mahalanobis screening flags 0% of shifted rows, while selected MCD screening flags 100%, under this deliberately separated two-dimensional generator.
- Study 7: With 1% legitimate minority prevalence and 5% top-score review budget, 95.8% of valid minority rows are flagged on average. That is **not** a real-world false-error rate.
- Study 8: Dropping the top 1% of fully valid lognormal outcomes produces a mean ratio 0.899 and 99% expected-shortfall ratio 0.549 relative to the unaltered population.
- Study 9: Fixed-reference alerts rise from about 1.0% under normal background to 8.64% under 1.5× standard deviation; recalibrating the budget lowers planted shifted-observation recall from 99.2% to 87.3%.

## Interpretation and source constraints

The estimator called `huber_location` uses a *known unit scale* and clips its score at 1.345; it is not a robust joint scale estimator. The Theil–Sen study uses SciPy's one-predictor slope plus its separate intercept convention. The MCD study uses scikit-learn MinCovDet `support_fraction=.7`; chi-square screening on fitted distances is a model-dependent heuristic. Isolation Forest's top-5% alert rule is a rank-based review-budget choice. The simulated minority is defined as *valid*, and the deliberately shifted cluster used for multivariate component TPR is not a confirmed measurement error. The sign interval assumes continuous IID sampling. Study 9's threshold recalibration deliberately uses its current background to expose the review-budget/recall tradeoff; a deployment would require independent calibration and review-label governance.

The `minority` evidence includes a per-replicate **score cutoff**, which is an experimentally varying quantity; the exact 5% alert budget is a design constraint, not a variable Monte Carlo outcome. In the `shift` evidence, `sd_1.5` denotes a *standard deviation* of 1.5 (variance 2.25). Fixed-threshold background alert rates have no claimed analytic `theory` value in the CSV, while recalibrated rates have a 0.01 rank-budget reference by construction. A blank theory field does not imply a zero event rate.

The full editorial discussion and historical method source audit are in the private manuscript. The public code, pinned dependencies, tests, original figures, and archived evidence remain available here for verification. Don't use any of these synthetic numerical rates as forecasts for an unobserved system.
