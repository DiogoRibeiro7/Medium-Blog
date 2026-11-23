# Prevalence from an imperfect test: the Santa Clara serosurvey, worked through to two decisions

Companion code for a Medium article that applies a problem-first Bayesian workflow to a real survey. In April 2020, 50 of 3,330 people in Santa Clara County tested positive for SARS-CoV-2 antibodies (Bendavid et al. 2020). The workflow runs from beliefs about the test, through the update and its checks, to two decisions:
- **Decision 1.** Could the survey support acting on the published prevalence?
- **Decision 2.** How should a fixed number of tests be split between validating the test and surveying people?

## What it finds

All numbers below are from `output.txt`.

**Could the positives all be false?** With specificity known only from 399 correct of 401 negatives, 3,330 uninfected people would give 50 or more positives 7.0% of the time.

**The posterior, four ways.** Unweighted prevalence, with the published range (2.49% to 4.16%, weighted) translated to 1.33% at its bottom:
- **Version 1 validation, pooled.** Median 1.06%, 95% interval 0.13% to 1.89%. P(below 0.5%) 14.0%.
- **Version 2 validation, pooled** (3,308/3,324, 130/157). Median 1.24%, 95% interval 0.67% to 1.86%, close to the journal version's 0.7% to 1.8%.
- **Hierarchical, prior scale 0.3.** Median 1.26%, 95% interval 0.18% to 2.26%. P(below 0.5%) 9.3%; P(at least 1.33%) 43.8%.
- **Hierarchical, prior scale 0.2.** 0.28%, 1.21% and 2.02% at 2.5%, 50% and 97.5%, against Gelman and Carpenter's 0.3%, 1.2% and 2.1%.

**The pooling check.** The 13 specificity studies disagree more than one shared specificity allows (posterior predictive p below 0.001). Under pooling, the 1,102-sample study had a 0.8% chance of the zero false positives it reported.

**Engine check.** On 300 surveys simulated from the model's priors, 90% intervals cover 90.3% (KS p = 0.88).

**Decision 1.**
- **The claim.** Under the defensible models, the chance that prevalence reached the bottom of the published range is 38% to 44%.
- **The best number to act on.** It depends on which error costs more. With the hierarchical model at scale 0.3, it runs from 0.53% (10th percentile) to 1.86% (90th).

**Decision 2.** The budget is 3,853 tests, with specificity 99.5% and sensitivity 83% in the survey's own lab.
- **The best split.** 800 to 1,300 known negatives (21% to 34% of tests), against Santa Clara's 401, gives a 90% interval 6% to 19% narrower.
- **Low prevalence.** At a true prevalence of 0.5%, no split clears half the truth in more than about 20% of surveys. Even 30,824 tests do it in 64%.

**Other labs against your own lab.** Validation done in other labs hits a floor set by how much labs differ:
- **Between-lab sd 0.73** (the estimate under prior scale 0.3). The 95% interval cannot get narrower than 1.82 points, however many samples those labs test.
- **The same lab.** 400 known negatives tested there give 1.78 points, and 3,200 give 1.21.

## Data

Published counts, entered in `prevalence_core.py`. Nothing is downloaded or committed.
- **Version 1 data** (11 April 2020). 50/3,330; specificity 399/401; sensitivity 103/122.
- **Version 2 data** (27 April 2020). 13 specificity studies and 3 sensitivity studies, as transcribed by Gelman and Carpenter (2020) in [bob-carpenter/diagnostic-testing](https://github.com/bob-carpenter/diagnostic-testing/tree/eb625dbb845df3f42204836c6a105e2f23f70bed/src/specificity-santa-clara) (commit `eb625db`). A test checks the pooled totals against the published 3,308/3,324 and 130/157.

## Reproduce

```
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v          # the methods on cases with a known answer
python prevalence_numbers.py > output.txt        # every number in the article; about four minutes
python figures.py                                # figures/fig1 to fig5
```

The workflow `.github/workflows/imperfect-test-prevalence.yml` runs all of this and fails if `output.txt` changes.

## Method (`prevalence_core.py`)

- **Model.** p = *π* × sensitivity + (1 − *π*) × (1 − specificity). Uniform priors on prevalence; for the pooled model, uniform priors on specificity and sensitivity.
- **Integration.** The posterior of *π* is computed on a grid (fine below 5%) by averaging the survey likelihood over draws of (specificity, sensitivity) given the validation data alone. The survey depends on the hyperparameters only through the survey lab's specificity and sensitivity, so this is exact up to Monte Carlo error, for the pooled and the hierarchical model alike.
- **Hierarchical model.** As in Gelman and Carpenter (2020): logit specificities and sensitivities are Normal(*μ*, *σ*), with *μ* ~ Normal(4, 2) and *σ* ~ half-Normal(0, *τ*). The hyperposterior is computed on a grid, with each study's likelihood integrated by 60-point Gauss–Hermite quadrature. The survey lab's rates are draws for a new study.
- **Checks.** A posterior predictive check of pooling (chi-square-type statistic), and posterior quantiles of the truth on simulated surveys (Cook, Gelman and Rubin 2006).
- **Decision 2.** 300 simulated surveys per split. Specificity and sensitivity are integrated with stratified quantiles of their Beta posteriors.

## Files

- **`prevalence_core.py`**: the data, the model, the grid posterior, the hierarchical model, the checks and the design simulation.
- **`prevalence_numbers.py`**: all numbers, printed in order, and the arrays the figures use (`results/`, not committed).
- **`figures.py`**: the five figures.
- **`tests/test_core.py`**: 11 tests. Among them: the published totals, a perfect test giving the Beta posterior, the journal interval, Gelman and Carpenter's strong-prior result, the quadrature against Monte Carlo, the pooling check and calibration.
