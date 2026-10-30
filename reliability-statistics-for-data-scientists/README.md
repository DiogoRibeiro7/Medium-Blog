# A Machine Does Not Fail at Its Mean

Reproducible companion to **article #8** in the
[chronological article index](../ARTICLES.md).
Read the manuscript:
[A Machine Does Not Fail at Its Mean: Reliability Statistics, Censoring, and Maintenance Decisions](article.md).

The study is a deliberately inspectable, fully **synthetic** industrial
reliability example. Its failure ages, administrative censoring windows,
cost ratios, competing events, and policy outcomes do **not** come from any
factory, employer, device fleet, or real maintenance system.

## Questions addressed

1. Why does mean time to failure fail to describe age-specific risk?
2. What information does a right-censored asset contribute?
3. How do Kaplan–Meier and Nelson–Aalen estimators handle the changing risk set?
4. Why do the Weibull parameters change if we fit only observed failures?
5. How should we calculate failure probability over the *next* 300 hours,
   conditional on surviving to the current age?
6. When is a preventive age-replacement policy economically worthwhile?
7. How should we evaluate a selected replacement rule in independent renewals?
8. Why does 1 − Kaplan–Meier overstate observable cause-specific risk when
   another failure mechanism competes for the first event?

## Reproduce

Run from this directory, using Python 3.12:

```bash
python -m pip install -r requirements.txt
python analysis.py
python -m unittest discover -s tests -v
```

For a smaller, independent smoke run that keeps full results unchanged:

```bash
python analysis.py --quick
```

The code uses only NumPy, SciPy and Matplotlib for the scientific experiment.
The notebook additionally uses Jupyter's kernel and execution dependencies.
The scoped GitHub Actions workflow verifies the unit tests, fresh-kernel
notebook execution, and quick study. All pseudorandom generators have
separate fixed streams derived from seed 20261030.

## Research design

The main asset sample is generated from a Weibull lifetime distribution
with shape **2.30** and scale **1,400 operating hours**. Each of **1,200**
independent components has an independent administrative censoring
time uniformly drawn from 550 to 2,100 hours. We observe the lesser of
failure and censoring time, plus an indicator of which occurred.
Only **637** observed records are confirmed failures. Every fitting routine
uses observed time and event status alone.

The correct observed-data log likelihood is

\[
\ell(\beta,\eta)
=\sum_i\delta_i\log f(t_i;\beta,\eta)
+\sum_i(1-\delta_i)\log S(t_i;\beta,\eta).
\]

It is optimized in log-parameter coordinates with an analytic gradient.
The deliberately incorrect comparison discards censored observations and
fits a Weibull distribution to the complete cases. This is not an
appropriate general estimator of the original lifetime law.

Kaplan–Meier, Nelson–Aalen and Aalen–Johansen are implemented from first
principles. The study uses independent censoring by construction.
Greenwood uncertainty bands are pointwise approximate, not simultaneous
coverage statements.

### Maintenance policy

If a component survives to planned replacement age \(\tau\), it incurs
synthetic preventive cost **220**. Otherwise it fails before \(\tau\)
and incurs corrective cost **1,150**. All renewed components are as-good-as-new.
The renewal–reward cost per operating hour is

\[
J(\tau)=
\frac{220S(\tau)+1150[1-S(\tau)]}
{\int_0^\tau S(u)\,du}.
\]

The denominator is expected cycle length, not \(\tau\) itself. An infinite
replacement age means run to failure. The decision excludes downtime,
labor scheduling, spare-parts lead time, inspection cost, partial repairs
and multi-component dependence.

### Competing risks

The second, independent synthetic experiment generates two latent Weibull
failure ages, with shapes (2.2, 1.3), scales (1,450, 2,150), and a separate
administrative censoring time. We observe only the first event.
Aalen–Johansen estimates the cumulative probability of cause 1 or cause 2
as the *first realized event*. Treating cause 2 as ordinary censoring in a
cause-1 Kaplan–Meier fit produces a different, hypothetical net-risk curve.
The latent independent causes are an explicit simulation assumption,
not something generally identified from observed competing-risk data.

## Selected full-run results

| Quantity | Synthetic estimate |
| --- | ---: |
| Main sample, observed failures | 637 / 1,200 (53.1%) |
| Generating Weibull shape and scale | 2.300, 1,400.0 hours |
| Censor-aware Weibull shape and scale | 2.429, 1,400.4 hours |
| Failures-only Weibull shape and scale | 2.518, 1,046.9 hours |
| True mean lifetime | 1,240.3 hours |
| True probability of failure during next 300 hours, conditional on survival to age 1,000 | 31.8% |
| Censor-aware fitted conditional probability | 32.5% |
| Failures-only fitted conditional probability | 56.6% |
| True optimal replacement age | 675.5 hours |
| Censor-aware fitted replacement age | 674.5 hours |
| Failures-only fitted replacement age | 504.8 hours |
| True rate at true optimal age | 0.5924 cost units/hour |
| True rate under censor-aware fitted policy | 0.5924 cost units/hour |
| True rate under failures-only fitted policy | 0.6215 cost units/hour |
| Run-to-failure rate | 0.9272 cost units/hour |
| Cause-1 true cumulative incidence at approximately 1,804 hours | 54.3% |
| Cause-1 Aalen–Johansen at approximately 1,804 hours | 53.9% |
| Naïve 1 − KM treating cause 2 as censored | 78.4% |

The differences are from one seeded synthetic sample, not estimated savings
or reliability statistics for an actual manufacturing organization.
The true generating law appears as an *oracle benchmark* available only
because this is a simulation.

## Eight original figures

| # | Figure | What it shows |
| --- | --- | --- |
| 1 | [Survival and hazards](figures/fig1_survival_and_hazards.png) | Weibull shape changes the hazard despite a common scale |
| 2 | [Censoring and KM](figures/fig2_censoring_and_km.png) | Risk-set accounting versus failures-only survival |
| 3 | [Likelihood and hazard](figures/fig3_likelihood_and_hazard.png) | Likelihood contours, fitted parameter locations and resulting hazards |
| 4 | [Censor sensitivity](figures/fig4_censoring_sensitivity.png) | Independent censoring windows change complete-case bias |
| 5 | [Conditional failure probability](figures/fig5_conditional_failure_risk.png) | Probability over an interval versus instantaneous hazard |
| 6 | [Age-replacement cost](figures/fig6_age_replacement_cost.png) | Renewal–reward cost under competing replacement ages |
| 7 | [Renewal validation](figures/fig7_renewal_validation.png) | Independent Monte Carlo confirms model-defined cost rates |
| 8 | [Competing risks](figures/fig8_competing_risks.png) | Cumulative incidence versus hypothetical net failure risk |

All figures are generated by [analysis.py](analysis.py); none are
adapted from third-party plots.

## Ten machine-readable source tables

| File | Contents |
| --- | --- |
| [summary.csv](results/summary.csv) | Parameter estimates, policy ages, evaluated cost rates and CIF at the chosen horizon |
| [observed_lifetimes.csv](results/observed_lifetimes.csv) | First 500 observed event/censor pairs, with synthetic *oracle-only* latent lifetime |
| [survival_estimates.csv](results/survival_estimates.csv) | True, KM, censor-aware and failures-only survival |
| [hazard_shapes.csv](results/hazard_shapes.csv) | Hazard and survival under three Weibull shapes |
| [likelihood_surface.csv](results/likelihood_surface.csv) | Censor-aware likelihood diagnostics on a parameter grid |
| [censor_sensitivity.csv](results/censor_sensitivity.csv) | Repeated fitted parameters across six censoring regimes |
| [conditional_risk.csv](results/conditional_risk.csv) | Fitted/true conditional failure risk by current age |
| [maintenance_curve.csv](results/maintenance_curve.csv) | Model-implied age-replacement cost-rate curves |
| [maintenance_policies.csv](results/maintenance_policies.csv) | Theoretical and Monte Carlo rates for five policies |
| [competing_risks.csv](results/competing_risks.csv) | Aalen–Johansen, generating-law CIF and net-risk comparison |

## Interpretation and limitations

- Censoring here is independent of failure time. Informative sensor removal,
  fleet retirement, warranty changes, or location-based attrition need
  conditional modeling or sensitivity analysis.
- Right-censoring is not an observed zero failure age. Only the survival
  contribution through the censoring time is known.
- The synthetic generating distribution is Weibull. A real system must
  test this assumption and examine mixture populations, operating regimes,
  covariates, repair history, time-varying loads, and calibration.
- The estimated cost optimum assumes renewal of an as-good-as-new
  independent component. A repairable asset with imperfect repairs has a
  different decision problem.
- Competing-risk cumulative incidence differs from 1 − cause-specific KM.
  Hypothetical removal of the competing cause needs additional causal
  assumptions; our simulation fixes independent latent failure times.
- The paper does not fit a Cox model, estimate dynamic treatment effects,
  or provide a certified operational maintenance recommendation.

## Primary research references

- [Kaplan & Meier (1958), nonparametric estimation from incomplete observations](https://doi.org/10.1080/01621459.1958.10501452).
- [NIST, Weibull distribution formulas](https://www.itl.nist.gov/div898/handbook/apr/section1/apr162.htm).
- [NIST, maximum likelihood estimation for censored life data](https://www.itl.nist.gov/div898/handbook/apr/section4/apr412.htm).
- [Barlow & Hunter (1960), optimum preventive maintenance policies](https://doi.org/10.1287/opre.8.1.90).
- [Aalen & Johansen (1978), transition-matrix estimation with censored observations](https://www.jstor.org/stable/i412608).
