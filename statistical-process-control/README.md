# Statistical process control in four parts

Public reproducibility companion for a four-part series on Medium by Diogo Ribeiro. The editorial manuscripts are kept in the private [research-articles-medium](https://github.com/DiogoRibeiro7/research-articles-medium/tree/main/statistical-process-control) repository (Parts [1](https://github.com/DiogoRibeiro7/research-articles-medium/blob/main/statistical-process-control/part1/article.md), [2](https://github.com/DiogoRibeiro7/research-articles-medium/blob/main/statistical-process-control/part2/article.md), [3](https://github.com/DiogoRibeiro7/research-articles-medium/blob/main/statistical-process-control/part3/article.md) and [4](https://github.com/DiogoRibeiro7/research-articles-medium/blob/main/statistical-process-control/part4/article.md)); this folder holds the code, the printed numbers and the figures. The thesis of the series: the in-control average run length of 370 belongs to a three-sigma Shewhart chart whose mean and standard deviation are *known*. Every chart in use has limits estimated from a finite Phase I sample, so its actual in-control ARL (the conditional ARL) is a random variable fixed by those estimates; its distribution is skewed, a majority of charts built the textbook way false-alarm more often than advertised, and the 370 is at best an average over a population of charts nobody owns. Every number quoted in the articles is printed by the script in the corresponding folder, and the committed `output.txt` is what the script prints.

- **Part 1 — where the 370 comes from, with the numbers recomputed** ([`part1/`](part1)). The normal tail against the Chebyshev and unimodal bounds Shewhart could appeal to; the geometric run length of a chart with known parameters; exact average run lengths of the Western Electric zone tests by Markov chain (checked by simulation) and Nelson's eight tests by simulation; the British action and warning limits; three-sigma limits under eight distributions for individual values and means of five; and a first look at estimated limits — the conditional in-control ARL of X-bar and individuals charts built from 20 to 1000 Phase I subgroups, the widening that restores the average false-alarm probability, and the correlation that shared estimates put between alarms.
- **Part 2 — the mechanism, from one chart to a production line** ([`part2/`](part2)). The conditional ARL of one chart derived exactly by conditioning, its sensitivity to the estimate of sigma, three estimators compared, and the condition under which its mean is infinite; machines charted in parallel, with Sidak limits and each machine's own Phase I or a pooled one; short runs after a changeover, fixed limits against a self-starting (Q) chart and the shift it absorbs, and sigma pooled across products; stations in series, with a cause-selecting chart whose regression is estimated in Phase I; and Hotelling's T^2 with an estimated mean vector and covariance matrix, by Imhof's method, for 2 to 50 sensors.
- **Part 3 — two public production lines, audited** ([`part3/`](part3)). A line cannot be certified in control, so Part 3 measures the realized alarm rate: limits built from a Phase I stretch the way a practitioner would build them, run over the rest of the record, and set beside a reference for independent normal data with the same Phase I size. On a four-hour, 1 Hz multistage continuous-flow line: held readings, autocorrelation and the semivariogram, four textbook chart designs, the spread across Phase I windows (against independent-normal and AR(1) references), one variable on three parallel machines, and the regression adjustment of stage 2 for stage 1 estimated in a window and used outside it. On the SECOM semiconductor line (1,567 runs, 590 sensors): an individuals chart per sensor, line-level alarms and Šidák limits, a decomposition of the excess into distribution shape, estimation and drift, and PCA-based T^2 and SPE against a normal reference with the same correlation matrix.
- **Part 4 — designing a line's charts around a false-alarm budget** ([`part4/`](part4)). Limits that guarantee the in-control ARL with a stated probability instead of on average (exact for normal data, checked by simulation, with the parametric and nonparametric bootstrap), what happens to each guarantee under four non-normal distributions, a joint guarantee for a line of K charts against Šidák limits and per-chart guarantees, the in-control data a distribution-free guarantee needs, and the methods applied to the two Part 3 lines: a budget of one false alarm per hour for the continuous-flow line (with injected faults) and one per 100 runs for SECOM (fixed, rolling and adaptive calibrated thresholds, and BH on conformal p-values).

## Running

Python 3.13 with the pinned versions in `requirements.txt`.

```
pip install -r requirements.txt
cd part1
python history_numbers.py > output.txt   # about a minute; no downloads
python figures.py
cd ../part2
python mechanism_numbers.py > output.txt  # about five minutes; no downloads; writes figure_data.json
python figures.py
cd ../part3
python audit_numbers.py > output.txt      # under a minute after the downloads; writes figure_data.json
python figures.py
cd ../part4
python budget_numbers.py > output.txt     # about a minute after the downloads; writes figure_data.json
python figures.py
cd ..
python -m unittest discover -s tests      # tests of the spcbudget module
```

A GitHub Actions workflow (`.github/workflows/statistical-process-control.yml`) runs the module tests, reruns each part on every change and checks that the output matches the committed `output.txt`.

## The spcbudget module

[`spcbudget/`](spcbudget) collects the tools Part 4 uses, in one file with no dependencies beyond NumPy and SciPy. Install it with `pip install "git+https://github.com/DiogoRibeiro7/Medium-Blog#subdirectory=statistical-process-control"`, or copy the file.

- `guaranteed_k(m, alpha, eps)`: the limit width for an individuals (or, with `nu`, an X-bar) chart whose conditional false-alarm probability is at most `alpha` with probability `1 - eps`, exact for normal data; `guarantee_probability` gives that probability for any width.
- `bootstrap_guaranteed_k(x, alpha, eps, rng, kind=...)`: the same target by parametric or nonparametric bootstrap of the Phase I data (Gandy and Kvaløy 2013).
- `line_guaranteed_k(m, K, line_alpha, eps, rng)`: one width for K charts that guarantees the false-alarm probability of the whole line; `sidak_alpha` and `sidak_k` split a line budget across charts.
- `min_calibration_size(alpha, eps)` and `calibrated_threshold(scores, alpha, eps)`: distribution-free thresholds from order statistics of in-control scores, with a guarantee conditional on the calibration set (Wilks 1941).
- `conformal_pvalues` and `benjamini_hochberg`: conformal p-values and the BH step-up procedure.
- `AdaptiveThreshold(alpha, gamma, window)`: the adaptive conformal threshold of Gibbs and Candès (2021), which holds a long-run alarm frequency on any data stream.

## Data for Part 1

None. Part 1 recomputes published numbers from their definitions (normal and other distribution functions, Markov chains, and seeded simulation of Phase I samples).

## Data for Part 2

None. Every chart is built from a simulated in-control Phase I sample and its conditional false-alarm probability is computed exactly given its estimates.

## Data for Part 3

Downloaded by `audit_numbers.py` from pinned commits of public GitHub mirrors, checked against SHA-256 hashes (computed after converting CRLF line endings to LF, so that mirrors differing only in line endings agree), and cached in `part3/data/`, which is not committed. Two independent mirrors of each data set give identical files.

- **Multi-stage continuous-flow manufacturing process** (Liveline Technologies; posted on Kaggle in 2020, recorded on 6 March 2019). One file, `continuous_factory_process.csv`: 14,088 readings at 1 Hz over about four hours, from three machines in parallel feeding a combiner, 15 measurement locations after stage 1, two machines in series, and the same 15 locations after stage 2, with setpoints. Mirrors: [LeonardoMdSACode @97c447c](https://github.com/LeonardoMdSACode/Multi-stage_continuous-flow_manufacturing_process/tree/97c447cfc87b4183609710513805843f0cfbe8b8) and [hjeffreywang @52e34be](https://github.com/hjeffreywang/Advanced-Prediction-of-Multi-stage-continuous-flow-manufacturing-process/tree/52e34be815907b6700d2c2975e8f1aca4b508775) (whose `notes_on_dataset.txt` documents the column layout). Use is subject to the terms of the original Kaggle release.
- **SECOM** (McCann and Johnston, 2008; UCI Machine Learning Repository, [doi:10.24432/C54305](https://doi.org/10.24432/C54305), CC BY 4.0). Two files, `secom.data` (1,567 runs × 590 sensors, missing as NaN) and `secom_labels.data` (pass/fail and time stamp, 19 July to 17 October 2008). Mirrors: [Aylin1 @c6e3cc0](https://github.com/Aylin1/SECOM_Project/tree/c6e3cc0a06b60155aafb6c06bb7cb39b7cf96d14) and [halinh1512 @a4aff01](https://github.com/halinh1512/Predictive-Model-Semiconductor-SECOM/tree/a4aff0133c2e2dce5311721fdb3874f7c3b956ba).

## Data for Part 4

The same three files as Part 3, downloaded and checked the same way by `budget_numbers.py` and cached in `part4/data/`.

## Licence

Apache-2.0. Contact: dfr@esmad.ipp.pt
