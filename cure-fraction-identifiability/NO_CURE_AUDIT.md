# No-cure control: when a mixture cure model invents a cured population

This is a computational extension of the [mixture-cure identifiability study](README.md),
not a new empirical claim about EU industry recoveries. The original study
simulates positive cure fractions (0.2 and 0.5). This control reverses the
question: **if everyone eventually experiences the event, how often could the
existing diagnostic falsely suggest a nonzero cure fraction?**

The experiment was defined as an independent grid with new random-number seeds
before running the simulations. The code reuses the original data generator,
maximum-likelihood fits, and AIC comparison without changing their parameters
or any of the original study's files.

## Experimental design

| Parameter | Values |
| --- | --- |
| True cured fraction | 0.0 (no permanently event-free subjects) |
| Susceptible latency | Lognormal, sigma 0.5; lognormal, sigma 1.0; Weibull, shape 1.5 |
| Sample size | 250, 1,000 |
| Largest follow-up / susceptible latency 95th percentile, R | 0.25, 0.5, 1.0, 2.0 |
| Replicates | 100 per cell; 24 cells; **2,400 independent datasets** |
| RNG | Independent `numpy.random.default_rng` streams, cell seeds from 98000 |
| Censoring | Staggered-entry administrative censoring, as in the original study |

Three distinct outcomes are measured for each dataset:

1. **AIC cure preference:** the best cure model has lower AIC than the best
   no-cure model, even though the data-generating process has no cured subjects.
2. **Combined diagnostic pass:** AIC favors cure *and* lognormal and Weibull
   cure estimates differ by less than 0.05, using the earlier paper's rule.
3. **False cure report:** combined diagnostic passes *and* the fitted cure
   fraction from the best cure model exceeds 0.10. The >0.10 threshold is the
   original study's consequential-error cutoff, repurposed as a practical
   reporting boundary. It is **not** a preregistered clinical decision rule.

The original diagnostic does not itself include the >0.10 reporting cutoff.
Accordingly the second and third outcomes are both shown, rather than
presenting their false-positive rates as interchangeable. All-fit convergence
is tracked independently. Each cell's uncertainty interval is a two-sided
95% Wilson binomial interval over 100 Monte Carlo replicates; these express
**simulation error**, not uncertainty from future real-world data.

## Results from the archived run

| Diagnostic across 2,400 datasets | False-positive frequency |
| --- | ---: |
| AIC selects a cure model | 9.4% |
| AIC plus family agreement passes | 4.3% |
| Combined rule reports cure fraction >0.10 | 1.9% (46 / 2,400; Wilson 95%: 1.4–2.5%) |

The aggregate proportion is an **equal-cell average over this chosen grid**,
not a deployment-wide error probability. It is dominated by the conditions in
the simulation and should never be treated as a calibrated significance level.

The worst observed cell was lognormal latency with sigma=0.5, n=250,
and R=0.25: **24 of 100 runs falsely reported cure** (24%, Wilson 95%:
16.7–33.2%). Only 2.5% of subjects had an observed event on average in that
cell. The analogous n=1,000 cell still gave 17 false reports in 100 datasets.
Across all six R=0.25 cells, the false-report rate was 7.0%; across six R=2
cells it was 0 in 600 datasets. That zero is not proof of a zero underlying
error rate.

All four likelihood fits converged in 95.3% of datasets (2,288/2,400). Of 46
false reports, 45 occurred when all four fits converged. Thus simply dropping
nonconvergent runs would not remove the short-follow-up problem.

The main interpretation is asymmetric: cross-family agreement and AIC provide
a useful warning system, but **agreement is not sufficient to establish a
population cure fraction when follow-up is very short**. Both models may
agree on the wrong extrapolated tail. This strengthens the case for treating
non-identifiability as an explicit conclusion, and for collecting longer
follow-up before reporting a never-event proportion.

## Reproduction

From `cure-fraction-identifiability/` after installing `requirements.txt`:

```bash
python -m pytest -q tests
python no_cure_audit.py --reps 100 --workers 4
python plot_no_cure.py
python check_results.py results/no_cure_summary.txt
```

`python no_cure_audit.py --summary-only` regenerates the cell table and report
from the committed raw CSV without re-running optimization. The following
files are separate from the original simulation:

| Artifact | Purpose |
| --- | --- |
| `no_cure_audit.py` | Reproducible experiment and metric definitions |
| `tests/test_no_cure_audit.py` | Zero-cure, rule, and Wilson-interval contract tests |
| `results/no_cure_raw.csv` | 2,400 per-dataset fitted outcomes |
| `results/no_cure_cells.csv` | 24 cell summaries and uncertainty intervals |
| `results/no_cure_summary.txt` | Human-readable frozen numerical baseline |
| `figures/fig7_no_cure_false_discovery.png` | Generated diagnostic figure |
| `plot_no_cure.py` | Figure reproduction script |

## Scope and limitations

- The data-generating process belongs to one of the two fitted latency families;
  misspecified nonparametric tails, covariates, informative censoring, and
  time-dependent cure fractions are not covered.
- Only three latency shapes, two sample sizes, and four follow-up ratios are
  included. The extreme short-follow-up condition has very few observed events.
- One hundred replicates per cell reveal large failure rates, but a small
  error rate cannot be estimated precisely. The overall Wilson interval pools
  deliberately heterogeneous design cells and is descriptive only.
- Optimizer convergence is recorded, not proof of global maximum likelihood.
  Fitting a nonzero cure fraction is not a test of biological or economic cure.
- Figure 7 and this note describe **synthetic** data; neither should be read
  as revising the empirical EU application. The article manuscript remains in
  the private `research-articles-medium` repository.
