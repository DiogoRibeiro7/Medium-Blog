# Structural VAR restrictions: pre-registered protocol

An amendment to the identification study, written, committed and pushed before any VAR
was estimated. Changes after the first estimate are listed in the change log at the end,
with reasons.

**Data work before this protocol was written.** Only the units and the EU-27 coverage of
the quarterly GDP deflator (Eurostat `namq_10_gdp`, `PD_PCH_PRE_NAC`, `SCA`) were checked.
It covers 2008-Q1 to 2026-Q2 for all 27 countries.

## 1. Why, and what is asked

**What the study already did with VAR restrictions.**
[docs/IDENTIFICATION_STUDY.md](docs/IDENTIFICATION_STUDY.md), section 4:

- rejected recursive orderings and long-run restrictions;
- judged sign restrictions "feasible but not defensible", for three reasons:
  1. the shocks are only set-identified;
  2. the usual uniform prior over rotations shapes the reported responses (Baumeister
     and Hamilton 2015);
  3. some signs are debatable.

Of these strategies, only external instruments were estimated.

**What this amendment adds.** It completes tasks 4 and 7 of the identification
milestone: structural VAR restrictions, and sensitivity to identification choices.

- **Sign restrictions, with inference that does not depend on the rotation prior.** It
  uses the robust Bayesian approach of Giacomini and Kitagawa (2021). For every posterior
  draw of the reduced form, it reports the whole identified set: the range over every
  rotation consistent with the restrictions. This removes reason 2. Reasons 1 and 3 are
  reported, not removed.
- **The recursive orderings.** The 24 orderings are shown only to measure how much an
  arbitrary ordering moves the answer. **They are never read causally.**

**Questions.**

1. **Demand against technology.** Under the restrictions below, are the shock-conditional
   Okun ratios of demand and technology shocks distinguishable at 4 and 8 quarters?
2. **Labour supply against demand.** The impact signs are imposed, so at impact the two
   shocks differ by assumption. Do the restrictions survive the data, does the
   distinction persist beyond impact, and how much of unemployment's forecast-error
   variance does each shock explain?
3. **Ordering.** How much do the recursive orderings move the orthogonalised "GDP shock"
   Okun ratio?

## 2. Model

**Variables.** `x[i,t] = (g, pi, du, dp)'`, quarterly, for country i:

- `g`: real GDP growth (`namq_10_gdp`, `CLV_PCH_PRE`, `SCA`);
- `pi`: GDP deflator growth (`PD_PCH_PRE_NAC`, `SCA`);
- `du`: the change in the unemployment rate;
- `dp`: the change in the participation rate.

`g`, `du` and `dp` come from the frozen snapshot `snapshots/eurostat-2026-10-15`. The
deflator is frozen in a new snapshot with a SHA-256 manifest.

**Reduced form.** A pooled panel VAR(2) with country fixed effects (the within
estimator):

```text
x[i,t] = c_i + A_1 x[i,t-1] + A_2 x[i,t-2] + eta[i,t],    E[eta eta'] = Sigma
```

- **Homogeneity.** The dynamics and covariances are common to all countries, which is a
  strong assumption; the robustness checks drop Ireland, whose GDP is dominated by
  multinational accounting.
- **Rows.** Every country-quarter whose `x[t]` and lags are all observed, and whose window
  [t-2, t] avoids 2020-Q1..2021-Q4.

**Posterior.**

- A flat prior gives the Normal-inverse-Wishart posterior:
  - `Sigma ~ IW(S, n - k - N)`, with S the residual cross-product, n rows, k = 8 slope
    parameters per equation and N fixed effects;
  - `vec(A) | Sigma ~ N(vec(A_hat), Sigma ⊗ (X'X)^-1)`, with X the demeaned lags.
- 1000 posterior draws are retained, seed 20261018.
- Draws whose companion matrix has an eigenvalue of modulus ≥ 1 are discarded and
  counted.

## 3. Identification

Structural shocks are `eps = B^-1 eta`, with `B = chol(Sigma) Q` and Q orthonormal.

**Sign restrictions (S1-S3) on impact (h = 0).**

| Shock | g | pi | du | dp | Rationale |
| --- | --- | --- | --- | --- | --- |
| **Demand** | + | + | − | free | Output and prices move together; firms hire |
| **Technology** | + | − | − | free | Output up, prices down; more is produced and unemployment falls |
| **Labour supply** | free | − | + | + | More people enter the labour force before finding jobs, which raises unemployment and participation and lowers prices |

- The fourth shock is left unrestricted.
- Every pair of identified shocks differs in at least one restricted sign.
- **The restrictions are not testable**, except jointly: a reduced-form draw for which no
  rotation satisfies them is evidence against them.
- The technology restriction on `du` is debatable: in sticky-price models, hours or
  employment can fall after a productivity improvement (Galí 1999). It is dropped in a
  robustness check.

**Inspiration and limits.**

- The scheme follows the logic of Foroni, Furlanetto and Lepetit (2018), who use the
  response of unemployment to separate labour-supply from other supply shocks.
- It is not their exact scheme: they also use wages, and this study has no harmonised
  quarterly wage series in its frozen data.

**Admissible rotations.**

- For each posterior draw, M = 20,000 rotations are drawn from the Haar distribution (QR
  of a Gaussian matrix with a sign normalisation).
- A rotation is admissible if its columns can be assigned, with signs, to the three
  restricted shocks so that every restriction holds. Every valid assignment of a rotation
  is kept.
- The identified set of a target is approximated by its minimum and maximum over the
  admissible rotations. This is an inner approximation: the true set can only be wider.
  A check with M = 40,000 on the first 100 posterior draws reports how much the bounds
  move.

## 4. Targets

All targets are for one-standard-deviation structural shocks, at h = 0, 4 and 8 quarters.

- **Cumulative responses:**
  - of GDP, `Y[t+h] - Y[t-1]` (%);
  - of the unemployment rate (pp) and of the participation rate (pp);
  - for each identified shock.
- **Shock-conditional Okun ratio** `kappa_k(h)` = cumulative unemployment response /
  cumulative GDP response, for demand and technology at h = 4 and 8.
  - In a draw where the admissible GDP responses at h do not all share one sign, the
    ratio's identified set contains a pole and is unbounded. That draw is recorded with
    bounds (−inf, +inf).
- **Forecast-error variance share of unemployment** at h = 8 due to each identified
  shock. It is bounded in [0, 1].

## 5. Summaries

**Robust Bayes (prior-free over rotations).** For each target:

- the **set of posterior means** `[mean of lower bounds, mean of upper bounds]`;
- the **robust credible region** `[5th percentile of lower bounds, 95th percentile of
  upper bounds]`. It contains the identified set with posterior probability of at least
  90% (Bonferroni). Infinite bounds propagate, so the region is unbounded when more than
  5% of draws are.

**Plausibility.** The share of posterior draws with at least one admissible rotation is
the posterior probability that the identified set is non-empty.

**Contrast, labelled prior-dependent.** The single-prior posterior that keeps one uniform
admissible rotation per draw: the median with 68% and 90% bands. This is what most
sign-restriction studies report.

**Recursive orderings, labelled orthogonalised and not causal.**

- For each of the 24 orderings of `(g, pi, du, dp)`, the "GDP shock" is the
  orthogonalised innovation of the `g` equation.
- The report gives the posterior median of its Okun ratio at h = 4 and 8, and the range
  across orderings.

## 6. Decision rules

- **Question 1.** Demand- and technology-driven Okun ratios are **distinguishable under
  S1-S3** at h only if both robust credible regions are bounded and do not overlap.
  Otherwise they are **not distinguishable**.
- **Question 2.**
  - The impact distinction is reported as **imposed**.
  - The distinction **persists** at h = 8 if the robust credible region of the cumulative
    unemployment response excludes zero with the impact sign for both shocks.
  - The variance shares are reported as sets.
- **Plausibility.** A non-empty probability below 0.5 is reported as the data
  contradicting the restrictions for most reduced forms.
- **Labels.** Every result is labelled "set-identified under S1-S3". Recursive-ordering
  results are labelled "orthogonalised, not causal". No reduced-form or orthogonalised
  response is labelled causal.

## 7. Robustness (fixed in advance)

| Variant | Change |
| --- | --- |
| Technology without the unemployment sign | T: g+, pi− only |
| Labour supply with output up | L: g+ added on impact |
| Restrictions at h = 0 and 1 | The same signs on the period responses at h = 1 |
| Four lags | VAR(4) |
| Without Ireland | IE dropped from estimation |
| Pandemic included | Every row, including windows touching 2020-2021 |

## 8. Not covered

- Point identification.
- Restrictions on wages, hours or productivity, whose quarterly series are not in the
  frozen data.
- Country-specific VARs.
- Shocks identified outside the VAR (see IDENTIFICATION_PROTOCOL.md).

## References

- Baumeister, C. and Hamilton, J. D. (2015). *Econometrica* 83(5).
- Foroni, C., Furlanetto, F. and Lepetit, A. (2018). Labor supply factors and economic
  fluctuations. *International Economic Review* 59(3).
- Galí, J. (1999). Technology, employment, and the business cycle. *American Economic
  Review* 89(1).
- Giacomini, R. and Kitagawa, T. (2021). Robust Bayesian inference for set-identified
  models. *Econometrica* 89(4).
- Rubio-Ramírez, J. F., Waggoner, D. F. and Zha, T. (2010). Structural vector
  autoregressions: theory of identification and algorithms for inference. *Review of
  Economic Studies* 77(2).

## Change log

- **During implementation, before any estimate on real data: clarifications.**
  - *Inverse-Wishart.* The degrees of freedom are n - k - N, with scale S, so the
    posterior mean of Sigma is S / (n - k - N - 5).
  - *Uniform-prior contrast.* It takes, in each draw, the first admissible rotation in
    sampling order (with its first valid assignment). Haar sampling makes this a draw
    from the uniform distribution over admissible rotations.
  - *Ratio bounds.* A draw's ratio set is bounded only if the cumulative GDP response at
    h has the same strict sign for every admissible rotation.
  - *Robust region.* It uses order statistics (the "lower" 5th percentile of the lower
    bounds and the "higher" 95th percentile of the upper bounds), never interpolation.
    Infinite bounds therefore propagate, and the region can only be wider.
  - *Convergence check.* It reports the relative increase in set width when the rotations
    are doubled, for the finite, non-ratio targets of the first 100 draws.
  - *Deflator vintage.* The deflator snapshot comes from Eurostat's 2026-10-16 update of
    `namq_10_gdp`, one day later than the frozen GDP growth (2026-10-15). Revisions
    between the two are not reconciled.
  - *CI.* The full reproduction (7 specifications × 1000 draws × 20,000 rotations) runs in
    a dedicated job of the estimates workflow, not in the ordinary offline test suite.
