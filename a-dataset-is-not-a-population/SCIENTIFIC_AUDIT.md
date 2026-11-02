# Article #34 — Scientific and numerical audit

**Status:** Internal reproducibility and assumption audit, 2 November 2026.
All nine studies are synthetic mathematical demonstrations. This audit is not
an external peer review or evidence about actual survey populations.

## Reproducibility verification

The canonical archive contains nine `*_replicates.csv` and nine
`*_summary.csv` files at seed **20261102**, with **300 repetitions per study**.
`tests/test_analysis.py` independently calls the source simulation and summary
writer in a temporary directory, then checks every archived CSV row and
summary field for exact equality. This audit was run after retrieving the
committed archives from the PR branch. All 18 comparisons passed.

The fresh-kernel notebook check separately exercises quick synthetic runs
from an unrelated working directory, without changing archival evidence.
It is a smoke test, not the full publication rerun.

Summary intervals below are `mean signed error ± 1.96 × empirical SD / √300`.
These intervals describe uncertainty in Monte Carlo averages, **not** design
confidence intervals for an unknown population quantity. In the `predictive`
study the reported signed error is squared-error risk **above 0.25 Bayes
noise risk**, rather than estimator bias for a population mean.

## Numerical audit

The values are calculated from the committed 300-repetition summary CSVs.

| Experiment | Principal contrast | Mean signed error or excess risk |
| --- | --- | ---: |
| SRS | n=120 of N=1000, sample mean | +0.03160 |
| Disproportionate strata | Unweighted / poststratified | +1.49336 / +0.00087 |
| Outcome-dependent response | Respondents / oracle-IPW / misspecified-IPW | +0.77661 / +0.00818 / +0.38087 |
| Undercoverage | Frame mean against full-population mean | −1.01362 |
| Voluntary panel | Opt-in / probability sample | +1.51730 / +0.00220 |
| Joint mismatch | Marginal calibration / joint calibration | −1.86623 / −0.00282 |
| Unequal probabilities | Unweighted / raw-IPW / capped-IPW | +1.09264 / +0.00328 / +0.62820 |
| Predictive shift | Sampled-test / target-test / target-weighted target-test excess risk | +0.99870 / +4.98071 / +1.11691 |
| Population drift | Historical / updated stable composition / changed mechanism | −1.99959 / +0.00234 / −1.39766 |

### Study-specific interpretation and limitations

1. **SRS:** The fixed finite population and sample design make the sample mean
   design-unbiased under repeated SRSWOR, but the realised **300-draw** average
   error is +0.03160. Its 95% *Monte Carlo* interval is +0.00830 to +0.05489,
   excluding zero. This is a possible finite simulation fluctuation; do not
   describe the realised sample as proving zero bias or as falsifying the
   design-based theorem. The experiment does not yet record per-draw survey
   confidence intervals or their coverage.
2. **Stratification:** Known 800:200 population strata are sampled 80:80.
   Unweighted samples overrepresent the high-outcome stratum. In this exact
   fixed-allocation design, inverse inclusion weights sum to known N, making
   poststratified, HT/N and Hájek numerically identical. This does **not**
   establish equivalence of these estimators under arbitrary sample designs.
3. **Nonresponse:** The oracle weights use response probabilities depending
   on the latent outcome shock. These are deliberately unavailable in a real
   respondent-only analysis. The misspecified model omits that shock; neither
   the simulation nor its favourable oracle result identifies missingness
   without unverifiable assumptions.
4. **Undercoverage:** Twenty percent of the finite population has zero frame
   inclusion probability and a shifted mean. More repetitions of sampling
   from the restricted frame reduce simulation error, not the structural gap
   to the full-population target. Corrective extrapolation would require
   information absent from this experiment.
5. **Voluntary panels:** The opt-in mechanism depends on X, which predicts Y.
   The smaller probability sample is nearly unbiased, although its observed
   replicate variation is larger. Panel size and response counts should not
   be generalized to real online surveys.
6. **Calibration:** Two marginal auxiliary shares already match exactly; a
   *marginal* rake would leave weights unchanged. The unobserved joint shares
   differ substantially. Joint poststratification is possible only because
   the complete four-cell target counts are supplied externally. No iterative
   raking algorithm is claimed to have been fitted here.
7. **Weight trimming:** Selection probabilities are known in a synthetic
   Poisson design. The ratio-weighted Hájek estimator is not universally
   finite-sample design-unbiased; its near-zero average error is empirical.
   A 90th-percentile weight cap removes some variability but introduces a
   large positive bias in this design. The cap is deliberately arbitrary,
   not a universal policy recommendation. The current archive does not yet
   retain per-replicate effective sample sizes.
8. **Predictive shift:** A *constant* predictor is purposefully misspecified.
   Training and target X distributions are N(−1,1) and N(+1,1), while
   Y|X remains invariant. Density ratio `exp(2X)` improves a target-weighted
   constant fit; target error remains above Bayes risk because a constant
   cannot represent X and the weights have finite-sample variability. This
   is a comparison of population-specific **risk**, not an estimate of label
   prevalence. No concept shift is present in this study.
9. **Drift:** Known target group shares fix compositional change under stable
   group-conditional means, but the same weighting fails when one conditional
   mean also rises. The shifted mechanism is specified, not inferred from
   unlabeled target data.

## Publication checks still required

- A full 8,000+ word original manuscript, with all nine figures referenced
  and numerical statements tied explicitly to archived summaries.
- An explicit section on sampling frames, inclusion mechanisms, estimands,
  positivity, design variance, and overlap.
- Optional supplementary experiment or derivation for the SRS finite-
  population correction and survey interval coverage; the existing code
  does **not** establish nominal coverage.
- Weight concentration diagnostics where effective sample size is discussed;
  do not invent them from the current aggregate summaries.
- Verify all bibliographic metadata and quote-free interpretations against
  the original sources. The links below support methods, not these synthetic
  numeric results.
- Review CI status and figures visually before declaring ready to merge.

## Primary methodological references checked

- Horvitz, D. G. and Thompson, D. J. (1952), *A Generalization of Sampling
  Without Replacement from a Finite Universe*, JASA **47**(260), 663–685.
  DOI: https://doi.org/10.1080/01621459.1952.10483446
- Deville, J.-C. and Särndal, C.-E. (1992), *Calibration Estimators in Survey
  Sampling*, JASA **87**(418), 376–382.
  DOI: https://doi.org/10.1080/01621459.1992.10475217
- Bethlehem, J. (2010), *Selection Bias in Web Surveys*, *International
  Statistical Review* **78**(2), 161–188.
  DOI: https://doi.org/10.1111/j.1751-5823.2010.00112.x
- Elliott, M. R. and Valliant, R. (2017), *Inference for Nonprobability
  Samples*, *Statistical Science* **32**(2), 249–264.
  DOI: https://doi.org/10.1214/16-STS598
- Sugiyama, M., Krauledat, M. and Müller, K.-R. (2007), *Covariate Shift
  Adaptation by Importance Weighted Cross Validation*, JMLR **8**, 985–1005.
  https://jmlr.org/papers/v8/sugiyama07a.html
- Kish, L. (1965), *Survey Sampling*, Wiley. The original edition and later
  reprints must not be conflated in the manuscript.

The separate Hájek (1971) bibliographic entry in `RESEARCH_PLAN.md` remains
**unverified** and should not be added to the final article until checked.
