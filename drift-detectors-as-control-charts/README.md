# Drift detectors as control charts

Public reproducibility companion for the Medium article *Drift Detectors as Control Charts: The Phase I, Multiplicity and Autocorrelation Problems Their Defaults Inherit*, by Diogo Ribeiro. The editorial manuscript is kept in the private [research-articles-medium](https://github.com/DiogoRibeiro7/research-articles-medium/tree/main/drift-detectors-as-control-charts) repository; this folder holds the code, the printed numbers and the figures. Every number quoted in the article is printed by `drift_numbers.py`, and the committed `output.txt` is what it prints.

The thesis: the drift detectors used to monitor models in production are control charts. Page-Hinkley is Page's CUSUM run on deviations from a self-starting mean; DDM is a p-chart whose centre line is the lowest point the running error rate has reached; ADWIN and KSWIN are two-sample tests repeated over sliding windows; the batch scores of Evidently and NannyML (KS, PSI, normed Wasserstein) are one chart per column per window, with a threshold that is either a significance level, a fixed score, or mean + 3 sd of ten reference chunks. Each default threshold therefore assumes what a Shewhart chart assumes (known parameters, one test, independent observations, the right units), and each inherits the problems the statistical-process-control literature has documented when those assumptions fail: estimated limits (Phase I), many charts at once (multiplicity) and dependent data (autocorrelation). The script measures how far each default is from its nominal behaviour on simulated in-control data and on two public data sets, then replaces the defaults with designs built to a stated false-alarm budget and measures what the budget costs in detection delay.

## What the script does

1. **Data.** Downloads and checks both data sets; Elec2 is taken from two independent copies and compared.
2. **The translation.** Checks numerically that river's Page-Hinkley signals at the same time as a CUSUM of deviations from the running mean; prints the defaults of the installed detectors.
3. **Stream detectors under the null.** Page-Hinkley, ADWIN, KSWIN and DDM on simulated in-control streams: signals per 10,000 observations against the scale of the data, against AR(1) autocorrelation, and for DDM against runs in the errors; where DDM's reference point sits.
4. **Batch scores under the null.** PSI (decile bins and Evidently's bins) and Evidently's normed Wasserstein against window size, with independent and AR(1) data; the two-sample KS test under AR(1); NannyML's default alert threshold (mean + 3 sd over 10 reference chunks) and a decomposition of its excess; many columns at once.
5. **Bike Sharing.** A gradient-boosting demand model fitted to 2011 (out-of-fold residuals) and applied to 2012, when demand grew: input-drift scores by day, week and month; river's detectors on the hourly residuals of 2011 and on two in-control surrogates; daily Shewhart and CUSUM charts on the AR(1)-whitened daily mean residual with guaranteed limits (spcbudget's `guaranteed_k`; CUSUM decision intervals by the parametric and nonparametric bootstrap of Gandy and Kvaløy 2013); ADWIN and Page-Hinkley calibrated to the same budget on block-bootstrap surrogates; leave-one-quarter-out in-control replays and injected shifts; detection of the 2012 change.
6. **Elec2.** A logistic regression fitted to the first 28 days and left fixed: the autocorrelation of its errors, the daily error rate against the binomial, DDM/ADWIN/Page-Hinkley with the real order, the order shuffled inside each day and fully shuffled; budgeted daily charts; the reference window in which three columns are constant.
7. **The budgets side by side.**

## Running

Python 3.13 with the pinned versions in `requirements.txt` (which installs the `spcbudget` module of the [statistical-process-control](../statistical-process-control) companion from a pinned commit).

```
pip install -r requirements.txt
python -m unittest discover -s tests      # checks of drifttools.py (CUSUM ARL against textbook values, Page-Hinkley against river)
python drift_numbers.py > output.txt      # about four minutes; writes figure_data.json
python figures.py
```

A GitHub Actions workflow (`.github/workflows/drift-detectors-as-control-charts.yml`) runs the tests, reruns the script on every change and checks that the output matches the committed `output.txt`.

## Files

- `drift_numbers.py`: every number in the article.
- `drifttools.py`: simulated null streams; replicas of Evidently's PSI and normed Wasserstein scores and of NannyML's default threshold; the CUSUM average run length by the Markov-chain method of Brook and Evans (1972), for normal or empirical input; CUSUM decision intervals with a guaranteed in-control ARL.
- `figures.py`: the five figures in `figures/`.
- `tests/test_drifttools.py`: checks of the helpers.

## Replicated library behaviour

The stream detectors are river's own (version 0.26.1). The batch scores and thresholds are replicated from the source code, read at these commits:

- Evidently [`34771fe`](https://github.com/evidentlyai/evidently/tree/34771fe41c08ca074d96a52aaf4a45cd640cbaf5): `_get_default_stattest` in `src/evidently/legacy/calculations/stattests/registry.py` (KS at p < 0.05 for a numerical column when the reference has at most 1,000 rows, normed Wasserstein >= 0.1 above that); `wasserstein_distance_norm.py` (distance divided by `max(np.std(reference), 0.001)`); `psi.py` and `utils.get_binned_data` (Sturges equal-width bins on the pooled data, empty bins filled); `drift_share = 0.5` in `src/evidently/presets/drift.py`.
- NannyML [`4c86ee3`](https://github.com/NannyML/nannyml/tree/4c86ee350dc541cb244f965cf555b4b2a004ae99): `StandardDeviationThreshold` in `nannyml/thresholds.py` (mean + 3 x `np.nanstd`), the univariate-drift defaults in `nannyml/drift/univariate/calculator.py` (upper threshold only), reference-chunk scoring in `nannyml/drift/univariate/methods.py`, and `DefaultChunker` (10 chunks) in `nannyml/chunk.py`.

## Data

Downloaded by `drift_numbers.py` from pinned commits of public GitHub mirrors, checked against SHA-256 hashes (computed after converting CRLF line endings to LF), and cached in `data/`, which is not committed.

- **Bike Sharing** (Fanaee-T 2013, UCI Machine Learning Repository, [doi:10.24432/C5W894](https://doi.org/10.24432/C5W894), CC BY 4.0). `hour.csv`: 17,379 hourly counts of Capital Bikeshare rentals in Washington, DC, 2011-2012, with weather. Two mirrors serve identical files: [udacity/deep-learning-v2-pytorch @83d69f3](https://github.com/udacity/deep-learning-v2-pytorch/tree/83d69f31047c0daa0d24c500eb3ba850c3380f89/project-bikesharing/Bike-Sharing-Dataset) and [udacity/deep-learning @dd93380](https://github.com/udacity/deep-learning/tree/dd93380579bb192ab0a778e75596eec68141b4cc/first-neural-network/Bike-Sharing-Dataset). Introduced in Fanaee-T, H. and Gama, J. (2014), "Event labeling combining ensemble detectors and background knowledge", *Progress in Artificial Intelligence* 2(2-3), 113-127, doi:10.1007/s13748-013-0040-3.
- **Elec2** (Harries 1999; the normalized version distributed with MOA, as on [OpenML 151](https://www.openml.org/d/151)). 45,312 half-hourly records of the New South Wales electricity market, 7 May 1996 to 5 December 1998; the label says whether the NSW price rose or fell against its 24-hour moving average. Two independent copies, compared by the script: `elec.csv` from [scikit-multiflow/streaming-datasets @f1eb9f0](https://github.com/scikit-multiflow/streaming-datasets/tree/f1eb9f068c3450233721c3beac6fb0e880e9e7d7) (label 1 = UP, no date or day columns) and `elec2_data.dat`/`elec2_label.dat` from [vlosing/driftDatasets @01d08de](https://github.com/vlosing/driftDatasets/tree/01d08dec716870a36ea793e26e7e896ba921fe5d/realWorld/Elec2) (with the day of week; label 0 = UP). The original release states no licence; it is used here, as in the drift-detection literature, for research.
