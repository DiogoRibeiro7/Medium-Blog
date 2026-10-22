# Diagnostics of the reported Bayesian results (protocol)

Registered on 2026-10-22 and committed and pushed before any fit, refit or prior
simulation below was run. Changes after the first computation go in the change log at
the end, with reasons. The models, priors, samples and decision rules of
BAYES_PROTOCOL.md are not changed; every result here is reported alongside them.

## 1. Question, and what was already seen

An external assessment (2026-10-22) asked for three things before the posterior
differences are treated as definitive:

- whether the PSIS-LOO approximation is reliable;
- whether a few observations drive the results;
- whether the priors are calibrated, given that the prior predictive check produces
  quarterly changes far larger than any observed.

It also noted that some model-comparison verdicts differ between runs.

**Already known before this protocol was written**, from the committed reports and
their Linux replications:

- **Pareto k.** Each fit in a registered comparison has 1 to 10 cells with Pareto
  k > 0.7 in the local run. `primary` has 10 of 1124. No comparison is labelled
  unreliable locally, because the registered label needs more than 1% of the cells.
- **Pareto k in the Linux runs.** `primary` has 14 of 1124 such cells, and 13 of 1098 in
  the amendment 3 cell set. This labels unreliable one comparison of section 9.2
  (`time_effects` against `primary`) and the three amendment 3 comparisons with
  `primary`. The reports compare verdicts between runs but not these labels, so this
  difference is not reported anywhere yet. It was found while designing this study.
- **Verdicts that differ between runs.** Two PSIS-LOO verdicts differ between the local
  and Linux runs:
  - `ardl_dynamic_factor` against `ardl`: +11.8 (SE 5.7) locally, +10.7 (SE 5.8) on
    Linux;
  - `drifting_association` against `regime_shift`: +4.3 (SE 1.8) locally, +3.5 (SE 1.8)
    on Linux.
  The persistence verdict of `dynamic_factor` also differs (p = 0.030 and 0.052).
- **Prior predictive check.** Under the registered priors, 3.9% of simulated changes
  in `primary` exceed 5 pp in absolute value. The median of the largest simulated change
  is 21.7 pp; the largest observed change is 3.0 pp. No decomposition of these extremes
  has been computed.
- **Not yet computed.** No exact leave-one-out refit, no fit with a cell left out of
  the likelihood, no fit with another seed and no fit with a narrower prior scale.

## 2. Scope

**The registered PSIS-LOO comparisons**, each on its registered cell set:

| Set | Comparison (A against B) | Cells |
| --- | --- | --- |
| Section 9.2 | `time_effects` / `no_common_factor` | every unmasked cell of the primary window |
| Section 9.2 | `time_effects_participation` / `participation_no_common_factor` | as above |
| Section 9.2 | `time_effects` / `primary` | as above |
| Section 9.2 | `time_effects_participation` / `time_effects` | as above |
| Section 11.4 | `dynamic_factor` / `primary` | common to the four fits of amendment 3 |
| Section 11.4 | `ardl` / `primary` | as above |
| Section 11.4 | `ardl_dynamic_factor` / `primary` | as above |
| Section 11.4 | `ardl_dynamic_factor` / `ardl` | as above |
| Section 12.5 | `regime_shift` / `pandemic_excluded` | common to the pair |
| Section 12.5 | `drifting_association` / `pandemic_excluded` | as above |
| Section 12.5 | `drifting_association` / `regime_shift` | as above |
| Section 12.5 | `regime_shift_dynamic` / `pandemic_excluded_dynamic` | as above |

These involve 13 fits: `primary`, `no_common_factor`, `time_effects`,
`time_effects_participation`, `participation_no_common_factor`, `dynamic_factor`, `ardl`,
`ardl_dynamic_factor`, `pandemic_excluded`, `regime_shift`, `drifting_association`,
`pandemic_excluded_dynamic` and `regime_shift_dynamic` (corrected; see the change log).

**The persistence verdicts of section 11.4:** `primary`, `dynamic_factor`, `ardl` and
`ardl_dynamic_factor`.

**The two estimands quoted in the manuscript:**

- the mean country sum `mu_theta` in `primary`;
- the mean long-run sum (the per-draw mean of the 27 `theta_LR`) in
  `ardl_dynamic_factor`.

For each of them, the set of credibly negative countries (P(< 0) >= 0.95) is reported
as well.

**Common settings.** Every fit and refit uses:

- the frozen snapshot `snapshots/eurostat-2026-10-15`;
- the specification's own registered design;
- the sampler settings of BAYES_PROTOCOL.md section 6, with the acceptance criteria and
  the single retry (3000 warm-up iterations);
- the registered seed 20261016, unless a section says otherwise.

## 3. Run-to-run stability

**Runs.** Each of the 13 fits is refitted with four further seeds: 20261030, 20261031,
20261032 and 20261033. Together with the local and Linux runs, that gives six runs.

**Computed in each run, by the registered procedures:**

- the 12 PSIS-LOO verdicts;
- the four persistence verdicts;
- the two estimands;
- the number of credibly negative countries in `primary` and `ardl_dynamic_factor`;
- the number of cells with Pareto k > 0.7 in each fit, with the registered reliability
  label.

The Linux run contributes the values its committed summaries contain. A fit that is not
accepted after the retry is reported, and the comparisons that need it are left out of
that run.

**Rule.**

- A verdict is **stable** if it is the same in every run in which it can be computed.
- Otherwise it is **run-dependent**: it is reported with its range and is not treated as
  established.
- For each comparison, the range of the ratio elpd difference / SE over the runs is
  also reported.

## 4. Reliability of PSIS-LOO: exact leave-one-out

**Cells.** For each of the 13 fits, every cell in one of its comparison sets that has
Pareto k > 0.7 in the local run or in any of the four seed runs.

**Refit.** For each such cell, the fit is refitted with that cell's likelihood term
removed:

- The cell's value is kept as data. A later quarter's AR(1) or lagged-change term still
  conditions on it.
- This is the leave-one-term-out posterior that PSIS-LOO approximates.

**Exact value.** The cell's exact elpd is the log of the mean, over the refit's draws,
of the cell's predictive density.

**Corrected comparisons.** Each comparison is recomputed in the local run, on its
registered cell set, with the exact values in place of the PSIS values for those cells.
The verdict rule is the registered one: better supported when the difference exceeds
twice its standard error.

**Reported.**

- Each cell's PSIS and exact values.
- Each comparison's PSIS and corrected difference and SE, and whether its verdict
  changes.

A comparison whose corrected value needs a refit that is not accepted after the retry
is reported as **not resolved**.

## 5. Influential observations

The refitted cells of section 4 are the observations that PSIS identifies as
influential. For `primary` and `ardl_dynamic_factor`:

**Inventory.** Each refitted cell's country, quarter, observed change and Pareto k, and
the number of the 13 fits in which it exceeds 0.7.

**One cell at a time.** From the refits of section 4:

- the change in the estimand's median;
- any change in the set of credibly negative countries.

**All at once.** One further fit, with every refitted cell of that specification removed
from the likelihood together.

**Rule.** A result is **sensitive to influential cells** if, in the joint fit or in any
single-cell refit:

- the estimand's median falls outside its registered 90% interval; or
- the set of credibly negative countries changes.

The persistence p-value of the joint fit is reported as information.

## 6. Prior calibration

**(a) Decomposition.** First, the registered prior predictive check of `primary` (1000
draws, seed 20261018) is repeated with one component neutralised at a time:

- the GDP terms (`theta` and `b`) set to zero;
- Gaussian innovations in place of Student-t;
- every `sigma` fixed at its prior centre, 0.3 pp;
- Ireland's cells left out of the summary. Ireland has the largest quarterly GDP swings
  in the panel.

For each variant, the report gives the share of simulated changes beyond 5 pp and the
median largest simulated change. This shows which part of the prior produces the
extremes. It is descriptive.

**(b) The calibrated prior scale.**

- **Ladder.** `prior_scale` takes the values 1, 0.75, 0.5, 0.35 and 0.25. It multiplies
  every prior scale that the registered `wide_priors` specification multiplies.
- **Unchanged.** The prior of `nu`, Gamma(2, 0.1), and the centre of `log_sigma` do not
  change, as in `wide_priors`.
- **Criterion.** The registered prior predictive check is applied at each step. A step
  passes when at most 1% of simulated changes exceed 5 pp in absolute value. This is the
  threshold that the model-checks report applies.
- **Choice.** The calibrated scale is the widest step that passes. It is chosen
  separately for `primary` and `ardl_dynamic_factor`, from the prior predictive alone,
  before any calibrated fit.
- **If no step passes,** 0.25 is used, and the report says that the criterion is not met.

**(c) Calibrated fits.** `primary` and `ardl_dynamic_factor` are refitted with their
calibrated scale. Reported, next to the registered fits:

- the estimand (median and 90% interval);
- the set of credibly negative countries;
- the persistence p-value;
- PSIS-LOO `ardl_dynamic_factor` against `primary`, on the cells of section 11.4.

**Rule.** A result is **prior-sensitive** if the calibrated estimand's median falls
outside its registered 90% interval, or if the set of credibly negative countries
changes.

The registered priors remain the primary priors. Section 9.3 of BAYES_PROTOCOL.md fixed
them, and this study does not change them.

## 7. Interpretation and use in the manuscript (fixed in advance)

**Established.** A PSIS-LOO verdict is established only if:

- it is stable (section 3); and
- its corrected verdict is the same (section 4).

**Statements in the manuscript.**

- **LOO preferences.** The manuscript states a preference only if it is established.
  Otherwise it gives the difference and says that the preference is not established.
- **"Preferred by PSIS-LOO."** The two such sentences of section 5 about the amended
  model are rewritten to match the results of sections 3 and 4.
- **Persistence.** A verdict that is not stable is described as on its threshold.
- **Priors.** The priors paragraph reports the decomposition and the calibrated result
  in one or two sentences. It replaces "reported rather than tuned" only by adding to it.
- **Influence.** Influential cells, and any result that is sensitive to them, are
  stated in the checks paragraph.
- **Limitations.** The discussion's limitations and the article are updated in one
  sentence each.

**Not part of this study:**

- no new structural model;
- no new country hypothesis;
- no change to the registered priors, samples or primary specification.

## 8. Outputs

- **`eu-okun bayes-diagnostics-fit`** runs every fit and refit above into
  `results/bayes-diagnostics/`. It can resume, and it can run several fits at once.
- **`eu-okun bayes-diagnostics`** writes:
  - `docs/bayes-diagnostics/stability.csv`;
  - `docs/bayes-diagnostics/exact_loo.csv`;
  - `docs/bayes-diagnostics/comparisons.csv`;
  - `docs/bayes-diagnostics/influence.csv`;
  - `docs/bayes-diagnostics/prior_calibration.csv`;
  - `docs/bayes-diagnostics/summary.json`;
  - the report `docs/BAYES_DIAGNOSTICS.md`.
- **Tests:**
  - On a small synthetic design, the exact value agrees with PSIS for a cell with low
    Pareto k.
  - Removing a cell's term keeps its value as conditioning data.
  - The report regenerates from the committed summary.

## Change log

- **After the seed fits started, before any result was read: 13 fits, not 14.**
  Section 2 listed `participation` among the fits that the registered comparisons
  involve. It enters none of them: section 9.2 computes its PSIS-LOO but compares it
  with no other fit. The comparisons therefore involve 13 fits, and the seed, exact and
  influence stages use those 13. `participation` is not refitted. Section 2 now reads 13,
  and the paragraph that listed the fits is corrected.
