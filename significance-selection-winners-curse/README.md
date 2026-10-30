# Significance Selection, Power, and the Winner's Curse

Companion to *A Significant Result Is Not a Replicable Effect: Statistical Power, Type S and Type M Errors, and the Winner's Curse* (article #11 in the [series index](../ARTICLES.md)).

The study is entirely **synthetic**. It separates the probability of statistical significance, the sign and magnitude of selected effects, independent replication, multiplicity, correctly specified Bayesian shrinkage, and confidence-interval coverage after reporting decisions. No clinical, corporate or scientific publication dataset was used.

## Reproduction

The article-specific GitHub Actions workflow tests Python 3.12. From this directory:

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v
    python analysis.py
    python analysis.py --quick

The full run writes eight publication figures and ten CSV tables to [figures](figures/) and [results](results/). The quick run writes separate _quick-suffixed files and never overwrites the publication evidence. The [notebook](significance_selection.ipynb) executes independently from a fresh kernel and checks the quick study in a temporary directory.

## Mathematical experiments

### Statistical power and the significance filter

Every unselected effect estimate obeys

\[
\widehat\theta\sim N(\theta,s^2),\quad
Z=\widehat\theta/s\sim N(\delta,1),\quad \delta=\theta/s.
\]

We reject a two-sided point null at level \(\alpha=.05\) when \(|Z|>c\), where \(c=\Phi^{-1}(1-\alpha/2)\). Power, Type S sign errors, Type M absolute exaggeration, and confidence-interval coverage conditional on significance use **exact Gaussian tail integrals**, not Monte Carlo approximations. Type S and Type M are undefined for a true effect of zero.

The main discovery experiment uses \(\theta=0.3,\ s=0.3,\ \delta=1\). We draw 180,000 independent candidate studies. Only significant original estimates are recorded as discoveries. Each candidate receives a separately drawn replication effect, which is then examined **without a replication reporting filter**. We distinguish signed replication mean, average absolute magnitude and replication significance rate.

### Multiple comparisons

We simulate 240 independent hypothesis families per null fraction, each with 1,000 tests. The fractions of true nulls are 0.80, 0.95 and 0.99. Under alternatives, the standardized mean is 2.5; under nulls it is zero. The test statistics and p-values are independent within each family by construction. We compare:

- Independent unadjusted two-sided tests at 0.05.
- Bonferroni familywise correction at \(0.05/m\).
- Benjamini–Hochberg step-up false-discovery-rate control at target \(q=.05\).

The false discovery proportion (FDP) for a family is \(V/\max(R,1)\), with zero when no hypotheses are rejected. Its expectation across families estimates FDR. This is different from the ratio of aggregated false and total discoveries.

### Shrinkage after selection

The hierarchical experiment assumes a genuinely known prior \(\theta_j\sim N(0,\tau^2)\) with \(\tau=.2\) and Gaussian observation noise with standard deviation \(s=.3\). We filter reports by \(|\widehat\theta|/s>c\) and compare the selected raw estimate with the normal-prior posterior mean

\[
\mathbb E(\theta\mid\widehat\theta)
=\frac{\tau^2}{\tau^2+s^2}\widehat\theta.
\]

We vary the *assumed* prior scale to show risk sensitivity. This is an explicitly **oracle-prior** demonstration, not a claim that automatic shrinkage always improves arbitrary effect estimates.

### Conditional confidence-interval coverage

The usual interval \(\widehat\theta\pm cs\) covers the true \(\theta\) with unconditional probability \(1-\alpha=.95\). We calculate its conditional coverage given \(|Z|>c\) exactly. Because selection is made using the same statistic, nominal unconditional coverage need not hold in the reported subset.

## Figures

| Figure | Purpose |
| --- | --- |
| [1](figures/fig1_significance_filter.png) | Original and significance-conditioned estimation distributions |
| [2](figures/fig2_power_design.png) | Exact power versus sample size for several true effect sizes |
| [3](figures/fig3_type_s_and_type_m.png) | Type S wrong-sign risk and Type M magnitude exaggeration |
| [4](figures/fig4_winners_curse.png) | Discovery probability and reported effect inflation |
| [5](figures/fig5_discovery_vs_replication.png) | Significant originals versus unfiltered independent replications |
| [6](figures/fig6_many_tests.png) | Multiplicity: false discoveries and alternative detection |
| [7](figures/fig7_selection_and_shrinkage.png) | Known-prior Gaussian shrinkage, selection and misspecified-prior sensitivity |
| [8](figures/fig8_post_selection_intervals.png) | Conditional interval coverage among statistically significant reports |

## Result tables

| CSV | Contents |
| --- | --- |
| results/power_design.csv | Exact power for each sample size and effect size |
| results/selection_errors.csv | Exact power, Type S, Type M and selected absolute magnitude |
| results/discovery_replication.csv | Sampled individual significant originals and corresponding independent replications |
| results/replication_summary.csv | Full-population selection and replication diagnostics |
| results/multiplicity_replicates.csv | FDP, discoveries, power and any false discovery per simulated family |
| results/multiplicity_summary.csv | Expected FDP (FDR), mean counts, mean detection and FWER across families |
| results/shrinkage_sensitivity.csv | Conditional MSE and MAE versus assumed prior standard deviation |
| results/shrinkage_selected.csv | Sampled observed and latent effect pairs with oracle-prior posterior means |
| results/conditional_coverage.csv | Exact unconditional and significance-conditioned coverage |
| results/summary.csv | Main analytic and simulated numeric findings |

## Limits

- The normal-test analysis assumes a known standard error and a correctly specified independent Gaussian estimate. It is not a derivation for unknown variances, small-sample t tests, observational causal identification, or dependent units.
- Significance is only a predeclared two-sided test against the point null. No optional stopping, unrecorded specification search or researcher degrees of freedom are present; those processes can introduce additional selection.
- Type S and Type M describe the subset passing the filter for a **fixed nonzero effect**, not the proportion of nulls among significant findings.
- The replication result assumes the same true effect and standard error for each independent replication. Real replication may change populations, treatments, measurement quality or effect heterogeneity.
- Benjamini–Hochberg FDR control here relies on independent p-values. Its standard guarantee is not asserted for arbitrary correlated tests or adaptive hypothesis specification.
- FDR is an expectation across *families* of tests, not a posterior probability that a particular result is false.
- The shrinkage prior is known and correct by simulation design. It would need to be estimated and validated in an actual application; prior misspecification can change the risk.
- Conditional coverage differs from unconditional confidence-interval coverage because of reporting selection. Neither statement alone gives a Bayesian posterior probability about a fixed parameter.
- Full and quick runs have different Monte Carlo counts. Interpret finite-replication differences accordingly.

For derivations, substantive interpretation and foundational references, read the [article](article.md).
