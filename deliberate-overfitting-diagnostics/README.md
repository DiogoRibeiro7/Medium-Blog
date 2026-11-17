# Deliberate overfitting as a diagnostic

Companion code for the Medium article *Using Deliberate Overfitting to Debug Pipelines and Estimate Noise Floors* (Diogo Ribeiro, November 2026).

A common piece of modelling advice says: after feature engineering, overfit a model on purpose; if it can memorise the training data the signal is there, and if it cannot the problem lies upstream. This folder tests that rule and the three jobs a deliberately overfit model can actually do: show that a pipeline can learn at all, measure the noise floor when rows repeat, and, once calibrated against shuffled labels, act as a test for signal.

## Data

Both data sets are public and downloaded by `fetch.py`, which checks their SHA-256 digests.

* **UCI Adult** (Becker and Kohavi, 1996): 48,842 rows of the 1994 Current Population Survey, label "income above $50K". The mirror combines the UCI training and test files and has already removed the trailing period that the UCI test file puts on its labels.
* **California housing** (Pace and Barry, 1997): 20,640 block groups of the 1990 census, in the version with ocean proximity used in Géron's *Hands-On Machine Learning*. The target is the log of the median house value.
* **Simulation** (`common.py`): ten standard-normal features, a fixed nonlinear rule of five of them, and each label flipped with probability eta. Because the rule is deterministic and the flips are independent of the features, the Bayes error is exactly eta.

## Scripts

Run in this order (about 40 minutes on two cores):

```
python fetch.py
python exp1_memorization.py      # memorisation grid, detection power, feature-set ranking
python exp2_noise_floor.py       # duplicate floor, Cover-Hart bounds, learning curves, residual variance
python exp3_bugs.py              # six planted pipeline bugs against seven checks
python summarize.py              # every number quoted in the article -> results/summary.txt
python figures.py                # figures/
python -m unittest discover -s tests -v
```

`exp1_memorization.py` and `exp2_noise_floor.py` take an optional argument naming the parts to run (`python exp1_memorization.py c` runs only the feature-set ranking).

| File | What it holds |
|---|---|
| `results/exp1a_memorisation_grid.csv` | training and test accuracy of four memorising models, and the effort each needed, by noise rate and for real versus shuffled labels |
| `results/exp1b_detection_power.csv` | 60 data sets per noise rate: memorisation rule, shuffle-calibrated leaves test, held-out permutation test |
| `results/exp1c_feature_set_ranking.csv` | UCI Adult feature sets ranked by training fit and by held-out accuracy |
| `results/exp2a_adult_duplicate_floor.csv` | plug-in duplicate floor, overfit-tree training error and held-out errors as Adult columns are added |
| `results/exp2b_cover_hart.csv` | leave-one-out 1-NN error and the Cover-Hart lower bound against the known Bayes error |
| `results/exp2c_learning_curves.csv`, `exp2c_asymptotes.csv` | learning curves and power-law asymptotes (simulation, Adult) |
| `results/exp2d_residual_variance_*.csv` | nearest-neighbour residual-variance estimates (simulation, California housing) |
| `results/exp3_detection_matrix.csv`, `exp3_check_details.csv` | which check flags which planted bug, with the underlying statistics |
| `results/summary.txt` | every number quoted in the article |

## Reproducibility

Library versions are pinned in `requirements.txt`. Seeds are fixed in every script. The workflow `.github/workflows/deliberate-overfitting-diagnostics.yml` downloads the data, runs the tests on the committed results, reruns the planted-bug and feature-set experiments and runs the tests again on the fresh output. Small differences in the last digit of an accuracy are possible across machines; the tests check the qualitative results (who memorises what, which check flags which bug, the floor identities), not the last digit.
