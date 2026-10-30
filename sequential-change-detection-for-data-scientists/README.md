# The Alarm Is Not the Change: Sequential Change-Point Detection for Data Scientists

Reproducible companion to the Medium-ready long-form manuscript
[`article.md`](article.md). This is an explicitly **synthetic, controlled
experiment** about online alarms, not a retrospective evaluation of real
patient, retail, infrastructure or financial data. The estimand is the
probability of a sequential alarm within a well-defined observation window.
It is not a claim to know the true change time from an algorithm's alarm.

This study extends, without duplicating, the older
[nonparametric/offline change-point notebook](../change_point/change_point.ipynb)
already present in Medium-Blog.

## Questions

1. What does an online alarm mean, and why is its time not the change time?
2. How do upward Shewhart, Gaussian mean CUSUM, EWMA and Gaussian variance
   CUSUM behave when they have **matched finite-horizon null risk**?
3. How do persistent mean shifts, isolated spikes and variance changes differ?
4. Why must detection rates account for alarms that occur *before* a known
   simulated change?
5. What happens if thresholds calibrated under iid noise are deployed against
   correlated data that have the same marginal Gaussian distribution?

## Experimental design

| Choice | Full publication experiment |
| --- | --- |
| Randomness | `numpy.random.SeedSequence` namespaces, master seed `20261030` |
| Sampling | Gaussian mean-zero, unit-variance baseline; 200 observations per stream |
| True simulated change | Zero-based index 80, with observations 80–199 post-change |
| Calibration | 5,000 independent iid-null streams; empirical 90th percentile of each detector's **maximum across the 200 observations** |
| Holdout | 6,000 additional iid-null streams for validation; 6,000 separate stationary Gaussian AR(1) null streams with φ=0.6 |
| Alternatives | 5,000 independent streams per scenario: mean shifts +0.5, +1, +2; one-time +5 spike; post-change standard deviation multiplied by 1.7 |
| Main detection endpoint | Probability of first alarm in the 50 observations after the true change, **conditional on no alarm before change** |
| Early endpoint | Same conditional event in the first 5 observations |
| Uncertainty | Descriptive 95% Wilson intervals for binomial event fractions |

The calibration **targets** a 10% chance of *at least one* alarm in 200
independent null observations. It does **not** match detectors by average
run length (ARL), control a false-discovery rate, or guarantee a 10% risk
for arbitrarily long monitoring. The uncalibrated AR(1) counterexample
deliberately violates iid assumptions while preserving N(0,1) marginals.

All detectors use one-sided upper thresholds, and their initial states
are known zero, not estimated warm-up values. We do not study downward
shifts, uncertain baseline estimation, missing data, non-Gaussian nulls,
multiple-sensor monitoring, or operational outcomes.

## Reproduce

Run from this directory with Python 3.12:

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v
    python analysis.py
    python analysis.py --quick

The full analysis writes all eight publication PNGs and six CSV tables.
The quick analysis has smaller independent runs and uses `_quick` suffixes,
so it does not overwrite publication evidence. The executable, annotated
notebook [`sequential_change_detection.ipynb`](sequential_change_detection.ipynb)
uses a quick run and is executed in a fresh kernel by CI.

## Figures

| Figure | Publication image | Question |
| --- | --- | --- |
| 1 | `figures/fig1_three_types_of_change.png` | What does a lasting change look like versus a pulse or variance change? |
| 2 | `figures/fig2_statistic_alarm_paths.png` | How do different alarm statistics evolve on the same mean-shift stream? |
| 3 | `figures/fig3_horizon_calibration.png` | What exactly does finite-horizon threshold calibration match? |
| 4 | `figures/fig4_mean_shift_detection.png` | How does sensitivity change as a persistent mean shift grows? |
| 5 | `figures/fig5_detection_by_delay.png` | Which methods alarm sooner, without dropping undetected runs? |
| 6 | `figures/fig6_spike_vs_persistence.png` | How do alarms react to a one-time spike versus accumulation? |
| 7 | `figures/fig7_autocorrelation_miscalibration.png` | What fails when the null is autocorrelated? |
| 8 | `figures/fig8_variance_change_detection.png` | Why must a scale change be monitored separately from a mean shift? |

## Published tables

| Artifact | Evidence |
| --- | --- |
| `results/summary.csv` | Thresholds, iid null risk and key conditional detection endpoints by method |
| `results/calibration.csv` | Calibration counts, independent validated false-alarm fractions and 95% Wilson bounds |
| `results/scenario_metrics.csv` | 20 method-by-scenario rows; prechange survival, 5/50-event detection, conditional median delay |
| `results/dependence_null.csv` | 8 null method-by-process comparisons, with Wilson intervals |
| `results/detection_by_delay.csv` | 200 cumulative 50-event delay rows, retaining undetected survivors in the denominator |
| `results/sample_path.csv` | The 200 values and four evolving statistics for the illustrative single paths |

The committed tables come from the full simulation. Raw random streams,
generated quick-mode outputs and caches are not committed. The eight
figures and six CSVs can be recreated from the fixed seed and code.

## Interpretive constraints

An alarm is not a retrospective estimate of change location. An estimated
delay among **detected** runs is selected on detection; always read it
alongside the fraction detected before the deadline. Conditioning on
prechange survival is necessary in this controlled experiment, because
an earlier false alarm makes the path ineligible for a new first alarm.
This conditional population is not the same as an unconditional real-world
deployment success rate.

The alternative models are simple and known by construction. The variance
CUSUM has a designed Gaussian variance alternative, which explains its
strong result when that alternative occurs. A field deployment needs
baseline estimation, dependence-aware calibration, missingness rules,
reset/refractory policies and investigation procedures. See the manuscript
for derivations, limitations and original-source references.
